
#pragma once

#ifdef HAVE_RDMA

#include <infiniband/verbs.h>
#include <rdma/rdma_cma.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <poll.h>
#include <unistd.h>
#include <cstring>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <stdexcept>
#include <iostream>

#include "transport.h"
#include "log.h"

// ==================== RDMA 常量 ====================
static constexpr int RDMA_MAX_WR = 128;          // 最大工作请求数
static constexpr int RDMA_MAX_SGE = 1;            // 每个WR的最大SGE数
static constexpr int RDMA_CQ_SIZE = 256;          // 完成队列大小
static constexpr int RDMA_MAX_INLINE = 256;       // 最大内联数据大小
static constexpr size_t RDMA_BUF_SIZE = 4 * 1024 * 1024;  // 4MB 收发缓冲区
static constexpr int RDMA_POLL_TIMEOUT_MS = 5000; // CQ轮询超时（毫秒）

// ==================== RDMA QP 握手信息 ====================
struct RdmaQPInfo {
    uint16_t lid;       // 本地标识符
    uint32_t qpn;       // QP编号
    uint32_t psn;       // 包序列号
    uint32_t rkey;      // 远程内存注册key
    uint64_t vaddr;     // 远程虚拟地址
    union ibv_gid gid;  // GID（RoCE需要）
};

// ==================== RDMA 连接 ====================
struct RdmaConnection {
    struct ibv_qp *qp = nullptr;
    
    // 发送缓冲区
    void *send_buf = nullptr;
    struct ibv_mr *send_mr = nullptr;
    
    // 接收缓冲区
    void *recv_buf = nullptr;
    struct ibv_mr *recv_mr = nullptr;
    
    // 连接元数据
    RdmaQPInfo local_info;
    RdmaQPInfo remote_info;
    
    // TCP 辅助通道 socket（用于握手和辅助通信）
    int aux_sock = -1;
    
    // 连接状态
    bool is_valid = false;
    int node_id = -1;
    
    ~RdmaConnection() {
        // 资源释放由 RdmaConnectionPool 管理
    }
};

// ==================== RDMA 连接池 ====================
class RdmaConnectionPool {
private:
    struct NodeConnections {
        std::deque<std::shared_ptr<RdmaConnection>> idle_conns;
        std::mutex mutex;
        std::string ip;
        int port = 0;
        int active_count = 0;
    };
    
    std::unordered_map<int, NodeConnections> node_pools;
    std::mutex pools_mutex;
    
    // RDMA 设备上下文（共享）
    struct ibv_context *ib_ctx = nullptr;
    struct ibv_pd *pd = nullptr;
    struct ibv_cq *send_cq = nullptr;
    struct ibv_cq *recv_cq = nullptr;
    uint8_t ib_port = 1;
    int gid_index = -1;
    
    // 创建 TCP 辅助通道连接
    int create_tcp_connection(const std::string& ip, int port) {
        int sock = socket(AF_INET, SOCK_STREAM, 0);
        if (sock < 0) return -1;
        
        struct timeval timeout;
        timeout.tv_sec = 5;
        timeout.tv_usec = 0;
        setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
        setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
        
        int flag = 1;
        setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));
        
        struct sockaddr_in addr;
        memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port);
        inet_pton(AF_INET, ip.c_str(), &addr.sin_addr);
        
        if (connect(sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            close(sock);
            return -1;
        }
        
        return sock;
    }
    
    // 创建 QP
    struct ibv_qp* create_qp() {
        if (!pd || !send_cq || !recv_cq) return nullptr;
        
        struct ibv_qp_init_attr qp_init_attr;
        memset(&qp_init_attr, 0, sizeof(qp_init_attr));
        qp_init_attr.send_cq = send_cq;
        qp_init_attr.recv_cq = recv_cq;
        qp_init_attr.qp_type = IBV_QPT_RC;
        qp_init_attr.cap.max_send_wr = RDMA_MAX_WR;
        qp_init_attr.cap.max_recv_wr = RDMA_MAX_WR;
        qp_init_attr.cap.max_send_sge = RDMA_MAX_SGE;
        qp_init_attr.cap.max_recv_sge = RDMA_MAX_SGE;
        qp_init_attr.cap.max_inline_data = RDMA_MAX_INLINE;
        
        return ibv_create_qp(pd, &qp_init_attr);
    }
    
    // 将 QP 转换到 INIT 状态
    int modify_qp_to_init(struct ibv_qp *qp) {
        struct ibv_qp_attr attr;
        memset(&attr, 0, sizeof(attr));
        attr.qp_state = IBV_QPS_INIT;
        attr.port_num = ib_port;
        attr.pkey_index = 0;
        attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE;
        
        return ibv_modify_qp(qp, &attr,
            IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
    }
    
    // 将 QP 转换到 RTR 状态
    int modify_qp_to_rtr(struct ibv_qp *qp, const RdmaQPInfo &remote) {
        struct ibv_qp_attr attr;
        memset(&attr, 0, sizeof(attr));
        attr.qp_state = IBV_QPS_RTR;
        attr.path_mtu = IBV_MTU_1024;
        attr.dest_qp_num = remote.qpn;
        attr.rq_psn = remote.psn;
        attr.max_dest_rd_atomic = 1;
        attr.min_rnr_timer = 12;
        
        attr.ah_attr.dlid = remote.lid;
        attr.ah_attr.sl = 0;
        attr.ah_attr.src_path_bits = 0;
        attr.ah_attr.port_num = ib_port;
        
        // RoCE: 使用 GRH
        if (gid_index >= 0) {
            attr.ah_attr.is_global = 1;
            attr.ah_attr.grh.dgid = remote.gid;
            attr.ah_attr.grh.flow_label = 0;
            attr.ah_attr.grh.hop_limit = 1;
            attr.ah_attr.grh.sgid_index = gid_index;
            attr.ah_attr.grh.traffic_class = 0;
        }
        
        int flags = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | 
                    IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | 
                    IBV_QP_MIN_RNR_TIMER;
        
        return ibv_modify_qp(qp, &attr, flags);
    }
    
    // 将 QP 转换到 RTS 状态
    int modify_qp_to_rts(struct ibv_qp *qp) {
        struct ibv_qp_attr attr;
        memset(&attr, 0, sizeof(attr));
        attr.qp_state = IBV_QPS_RTS;
        attr.timeout = 14;
        attr.retry_cnt = 7;
        attr.rnr_retry = 7;
        attr.sq_psn = 0;
        attr.max_rd_atomic = 1;
        
        return ibv_modify_qp(qp, &attr,
            IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | 
            IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC);
    }
    
    // 通过 TCP 辅助通道交换 QP 信息
    bool exchange_qp_info(int sock, const RdmaQPInfo &local, RdmaQPInfo &remote) {
        // 发送本地 QP 信息
        if (send(sock, &local, sizeof(RdmaQPInfo), 0) != sizeof(RdmaQPInfo)) {
            return false;
        }
        // 接收远程 QP 信息
        if (recv(sock, &remote, sizeof(RdmaQPInfo), MSG_WAITALL) != sizeof(RdmaQPInfo)) {
            return false;
        }
        return true;
    }
    
    // 获取本地 LID 和 GID
    bool get_local_info(RdmaQPInfo &info, struct ibv_qp *qp) {
        struct ibv_port_attr port_attr;
        if (ibv_query_port(ib_ctx, ib_port, &port_attr)) {
            return false;
        }
        
        info.lid = port_attr.lid;
        info.qpn = qp->qp_num;
        info.psn = 0;  // 使用固定 PSN
        
        if (gid_index >= 0) {
            if (ibv_query_gid(ib_ctx, ib_port, gid_index, &info.gid)) {
                return false;
            }
        } else {
            memset(&info.gid, 0, sizeof(info.gid));
        }
        
        return true;
    }
    
    // 建立完整的 RDMA 连接
    std::shared_ptr<RdmaConnection> create_rdma_connection(const std::string& ip, int port, int node_id) {
        auto conn = std::make_shared<RdmaConnection>();
        conn->node_id = node_id;
        
        // 1. 建立 TCP 辅助通道
        conn->aux_sock = create_tcp_connection(ip, port);
        if (conn->aux_sock < 0) {
            LOG_DEBUG_STREAM("[RDMA] 无法建立到 " << ip << ":" << port << " 的TCP辅助通道");
            return nullptr;
        }
        
        // 2. 创建 QP
        conn->qp = create_qp();
        if (!conn->qp) {
            LOG_DEBUG_STREAM("[RDMA] 创建QP失败");
            close(conn->aux_sock);
            return nullptr;
        }
        
        // 3. 分配和注册发送/接收缓冲区
        conn->send_buf = malloc(RDMA_BUF_SIZE);
        conn->recv_buf = malloc(RDMA_BUF_SIZE);
        if (!conn->send_buf || !conn->recv_buf) {
            LOG_DEBUG_STREAM("[RDMA] 缓冲区分配失败");
            goto cleanup;
        }
        
        conn->send_mr = ibv_reg_mr(pd, conn->send_buf, RDMA_BUF_SIZE,
            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
        conn->recv_mr = ibv_reg_mr(pd, conn->recv_buf, RDMA_BUF_SIZE,
            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        
        if (!conn->send_mr || !conn->recv_mr) {
            LOG_DEBUG_STREAM("[RDMA] 内存注册失败");
            goto cleanup;
        }
        
        // 4. QP 转 INIT
        if (modify_qp_to_init(conn->qp)) {
            LOG_DEBUG_STREAM("[RDMA] QP转INIT失败");
            goto cleanup;
        }
        
        // 5. 获取本地信息并交换 QP 元数据
        if (!get_local_info(conn->local_info, conn->qp)) {
            LOG_DEBUG_STREAM("[RDMA] 获取本地QP信息失败");
            goto cleanup;
        }
        conn->local_info.rkey = conn->recv_mr->rkey;
        conn->local_info.vaddr = (uint64_t)conn->recv_buf;
        
        if (!exchange_qp_info(conn->aux_sock, conn->local_info, conn->remote_info)) {
            LOG_DEBUG_STREAM("[RDMA] QP信息交换失败");
            goto cleanup;
        }
        
        // 6. QP 转 RTR
        if (modify_qp_to_rtr(conn->qp, conn->remote_info)) {
            LOG_DEBUG_STREAM("[RDMA] QP转RTR失败");
            goto cleanup;
        }
        
        // 7. QP 转 RTS
        if (modify_qp_to_rts(conn->qp)) {
            LOG_DEBUG_STREAM("[RDMA] QP转RTS失败");
            goto cleanup;
        }
        
        // 8. 预投递接收请求
        {
            struct ibv_sge sge;
            sge.addr = (uint64_t)conn->recv_buf;
            sge.length = RDMA_BUF_SIZE;
            sge.lkey = conn->recv_mr->lkey;
            
            struct ibv_recv_wr recv_wr, *bad_wr;
            memset(&recv_wr, 0, sizeof(recv_wr));
            recv_wr.wr_id = 0;
            recv_wr.sg_list = &sge;
            recv_wr.num_sge = 1;
            
            if (ibv_post_recv(conn->qp, &recv_wr, &bad_wr)) {
                LOG_DEBUG_STREAM("[RDMA] 预投递接收请求失败");
                goto cleanup;
            }
        }
        
        conn->is_valid = true;
        LOG_DEBUG_STREAM("[RDMA] 成功建立到节点 " << node_id << " (" << ip << ":" << port << ") 的RDMA连接");
        return conn;
        
    cleanup:
        if (conn->send_mr) ibv_dereg_mr(conn->send_mr);
        if (conn->recv_mr) ibv_dereg_mr(conn->recv_mr);
        if (conn->qp) ibv_destroy_qp(conn->qp);
        if (conn->send_buf) free(conn->send_buf);
        if (conn->recv_buf) free(conn->recv_buf);
        if (conn->aux_sock >= 0) close(conn->aux_sock);
        return nullptr;
    }
    
    // 销毁 RDMA 连接
    void destroy_connection(std::shared_ptr<RdmaConnection> conn) {
        if (!conn) return;
        
        if (conn->qp) {
            ibv_destroy_qp(conn->qp);
            conn->qp = nullptr;
        }
        if (conn->send_mr) {
            ibv_dereg_mr(conn->send_mr);
            conn->send_mr = nullptr;
        }
        if (conn->recv_mr) {
            ibv_dereg_mr(conn->recv_mr);
            conn->recv_mr = nullptr;
        }
        if (conn->send_buf) {
            free(conn->send_buf);
            conn->send_buf = nullptr;
        }
        if (conn->recv_buf) {
            free(conn->recv_buf);
            conn->recv_buf = nullptr;
        }
        if (conn->aux_sock >= 0) {
            close(conn->aux_sock);
            conn->aux_sock = -1;
        }
        conn->is_valid = false;
    }

public:
    RdmaConnectionPool() = default;
    
    // 初始化 RDMA 设备
    bool init(struct ibv_context *ctx, struct ibv_pd *protection_domain,
              struct ibv_cq *scq, struct ibv_cq *rcq,
              uint8_t port, int gid_idx) {
        ib_ctx = ctx;
        pd = protection_domain;
        send_cq = scq;
        recv_cq = rcq;
        ib_port = port;
        gid_index = gid_idx;
        return true;
    }
    
    // 注册节点
    void register_node(int node_id, const std::string& ip, int port) {
        std::lock_guard<std::mutex> lock(pools_mutex);
        auto result = node_pools.try_emplace(node_id);
        NodeConnections& nc = result.first->second;
        nc.ip = ip;
        nc.port = port;
    }
    
    // 获取 RDMA 连接
    std::shared_ptr<RdmaConnection> acquire(int node_id) {
        std::string ip;
        int port = 0;
        
        {
            std::lock_guard<std::mutex> lock(pools_mutex);
            auto it = node_pools.find(node_id);
            if (it == node_pools.end()) return nullptr;
            
            NodeConnections& nc = it->second;
            std::lock_guard<std::mutex> nc_lock(nc.mutex);
            
            // 尝试复用空闲连接
            while (!nc.idle_conns.empty()) {
                auto conn = nc.idle_conns.front();
                nc.idle_conns.pop_front();
                
                if (conn && conn->is_valid) {
                    nc.active_count++;
                    LOG_DEBUG_STREAM("[RDMA Pool] 复用连接到节点 " << node_id 
                        << ", idle=" << nc.idle_conns.size()
                        << ", active=" << nc.active_count);
                    return conn;
                }
                // 无效连接，销毁
                destroy_connection(conn);
            }
            
            ip = nc.ip;
            port = nc.port;
        }
        
        // 锁外创建新连接
        if (!ip.empty() && port > 0) {
            auto conn = create_rdma_connection(ip, port, node_id);
            if (conn) {
                std::lock_guard<std::mutex> lock(pools_mutex);
                auto it = node_pools.find(node_id);
                if (it != node_pools.end()) {
                    std::lock_guard<std::mutex> nc_lock(it->second.mutex);
                    it->second.active_count++;
                    LOG_DEBUG_STREAM("[RDMA Pool] 创建新连接到节点 " << node_id
                        << ", active=" << it->second.active_count);
                }
            }
            return conn;
        }
        return nullptr;
    }
    
    // 释放连接回池
    void release(int node_id, std::shared_ptr<RdmaConnection> conn, bool valid = true) {
        if (!conn) return;
        
        std::lock_guard<std::mutex> lock(pools_mutex);
        auto it = node_pools.find(node_id);
        if (it == node_pools.end()) {
            destroy_connection(conn);
            return;
        }
        
        NodeConnections& nc = it->second;
        std::lock_guard<std::mutex> nc_lock(nc.mutex);
        
        if (!valid || !conn->is_valid) {
            destroy_connection(conn);
            if (nc.active_count > 0) nc.active_count--;
            return;
        }
        
        nc.idle_conns.push_back(conn);
        if (nc.active_count > 0) nc.active_count--;
    }
    
    // 关闭所有连接
    void close_all() {
        std::lock_guard<std::mutex> lock(pools_mutex);
        for (auto& kv : node_pools) {
            std::lock_guard<std::mutex> nc_lock(kv.second.mutex);
            for (auto& conn : kv.second.idle_conns) {
                destroy_connection(conn);
            }
            kv.second.idle_conns.clear();
            kv.second.active_count = 0;
        }
        node_pools.clear();
    }
};

// ==================== RDMA Transport 实现 ====================
class RdmaTransport : public ITransport {
private:
    TransportConfig config;
    
    // RDMA 设备资源
    struct ibv_context *ib_ctx = nullptr;
    struct ibv_pd *pd = nullptr;
    struct ibv_cq *send_cq = nullptr;
    struct ibv_cq *recv_cq = nullptr;
    
    // 连接池
    RdmaConnectionPool conn_pool;
    
    // 活跃连接映射：conn_id -> RdmaConnection
    // 这里使用 conn_id（虚拟fd）来兼容 ITransport 接口
    std::unordered_map<int, std::shared_ptr<RdmaConnection>> active_conns;
    std::mutex active_conns_mutex;
    std::atomic<int> next_conn_id{10000};  // 虚拟连接ID起始值
    
    // 服务端相关
    int server_sockfd = -1;  // TCP 辅助通道服务端 socket
    std::atomic<bool> server_running{false};
    
    // 获取下一个虚拟连接ID
    int get_next_conn_id() {
        return next_conn_id.fetch_add(1);
    }
    
    // 将 RdmaConnection 注册为活跃连接，返回虚拟 conn_id
    int register_active_conn(std::shared_ptr<RdmaConnection> conn) {
        int conn_id = get_next_conn_id();
        std::lock_guard<std::mutex> lock(active_conns_mutex);
        active_conns[conn_id] = conn;
        return conn_id;
    }
    
    // 获取活跃连接
    std::shared_ptr<RdmaConnection> get_active_conn(int conn_id) {
        std::lock_guard<std::mutex> lock(active_conns_mutex);
        auto it = active_conns.find(conn_id);
        if (it != active_conns.end()) return it->second;
        return nullptr;
    }
    
    // 移除活跃连接
    std::shared_ptr<RdmaConnection> remove_active_conn(int conn_id) {
        std::lock_guard<std::mutex> lock(active_conns_mutex);
        auto it = active_conns.find(conn_id);
        if (it == active_conns.end()) return nullptr;
        auto conn = it->second;
        active_conns.erase(it);
        return conn;
    }
    
    // RDMA Send 操作
    int rdma_send(std::shared_ptr<RdmaConnection> conn, const void* data, size_t size) {
        if (!conn || !conn->is_valid || !conn->send_mr) return -1;
        if (size > RDMA_BUF_SIZE) {
            LOG_DEBUG_STREAM("[RDMA] 发送数据超过缓冲区大小: " << size << " > " << RDMA_BUF_SIZE);
            return -1;
        }
        
        // 复制数据到注册的发送缓冲区
        memcpy(conn->send_buf, data, size);
        
        struct ibv_sge sge;
        sge.addr = (uint64_t)conn->send_buf;
        sge.length = size;
        sge.lkey = conn->send_mr->lkey;
        
        struct ibv_send_wr send_wr, *bad_wr;
        memset(&send_wr, 0, sizeof(send_wr));
        send_wr.wr_id = 0;
        send_wr.sg_list = &sge;
        send_wr.num_sge = 1;
        send_wr.opcode = IBV_WR_SEND;
        send_wr.send_flags = IBV_SEND_SIGNALED;
        if (size <= RDMA_MAX_INLINE) {
            send_wr.send_flags |= IBV_SEND_INLINE;
        }
        
        if (ibv_post_send(conn->qp, &send_wr, &bad_wr)) {
            LOG_DEBUG_STREAM("[RDMA] ibv_post_send 失败");
            return -1;
        }
        
        // 等待发送完成
        struct ibv_wc wc;
        int polls = 0;
        while (polls == 0) {
            polls = ibv_poll_cq(send_cq, 1, &wc);
            if (polls < 0) {
                LOG_DEBUG_STREAM("[RDMA] ibv_poll_cq(send) 失败");
                return -1;
            }
        }
        
        if (wc.status != IBV_WC_SUCCESS) {
            LOG_DEBUG_STREAM("[RDMA] Send 完成错误: " << ibv_wc_status_str(wc.status));
            return -1;
        }
        
        return 0;
    }
    
    // RDMA Recv 操作（等待接收完成）
    int rdma_recv(std::shared_ptr<RdmaConnection> conn, void* data, size_t max_size, size_t &actual_size) {
        if (!conn || !conn->is_valid || !conn->recv_mr) return -1;
        
        // 等待接收完成
        struct ibv_wc wc;
        int polls = 0;
        int timeout_count = 0;
        while (polls == 0) {
            polls = ibv_poll_cq(recv_cq, 1, &wc);
            if (polls < 0) {
                LOG_DEBUG_STREAM("[RDMA] ibv_poll_cq(recv) 失败");
                return -1;
            }
            if (polls == 0) {
                timeout_count++;
                if (timeout_count > RDMA_POLL_TIMEOUT_MS * 1000) {
                    LOG_DEBUG_STREAM("[RDMA] 接收超时");
                    return -1;
                }
                // 短暂让出 CPU
                std::this_thread::yield();
            }
        }
        
        if (wc.status != IBV_WC_SUCCESS) {
            LOG_DEBUG_STREAM("[RDMA] Recv 完成错误: " << ibv_wc_status_str(wc.status));
            return -1;
        }
        
        actual_size = wc.byte_len;
        if (actual_size > max_size) actual_size = max_size;
        
        // 复制数据
        memcpy(data, conn->recv_buf, actual_size);
        
        // 重新投递接收请求
        struct ibv_sge sge;
        sge.addr = (uint64_t)conn->recv_buf;
        sge.length = RDMA_BUF_SIZE;
        sge.lkey = conn->recv_mr->lkey;
        
        struct ibv_recv_wr recv_wr, *bad_wr;
        memset(&recv_wr, 0, sizeof(recv_wr));
        recv_wr.wr_id = 0;
        recv_wr.sg_list = &sge;
        recv_wr.num_sge = 1;
        
        if (ibv_post_recv(conn->qp, &recv_wr, &bad_wr)) {
            LOG_DEBUG_STREAM("[RDMA] 重新投递接收请求失败");
            return -1;
        }
        
        return 0;
    }
    
    // 初始化 RDMA 设备
    bool init_rdma_device() {
        // 获取设备列表
        int num_devices;
        struct ibv_device **dev_list = ibv_get_device_list(&num_devices);
        if (!dev_list || num_devices == 0) {
            LOG_DEBUG_STREAM("[RDMA] 未找到RDMA设备");
            return false;
        }
        
        struct ibv_device *ib_dev = nullptr;
        if (!config.ib_dev.empty()) {
            // 查找指定设备
            for (int i = 0; i < num_devices; i++) {
                if (config.ib_dev == ibv_get_device_name(dev_list[i])) {
                    ib_dev = dev_list[i];
                    break;
                }
            }
            if (!ib_dev) {
                LOG_DEBUG_STREAM("[RDMA] 未找到指定设备: " << config.ib_dev);
                ibv_free_device_list(dev_list);
                return false;
            }
        } else {
            // 使用第一个可用设备
            ib_dev = dev_list[0];
            LOG_DEBUG_STREAM("[RDMA] 使用设备: " << ibv_get_device_name(ib_dev));
        }
        
        // 打开设备上下文
        ib_ctx = ibv_open_device(ib_dev);
        ibv_free_device_list(dev_list);
        if (!ib_ctx) {
            LOG_DEBUG_STREAM("[RDMA] 打开设备失败");
            return false;
        }
        
        // 设置端口和 GID 索引
        ib_port = (uint8_t)config.ib_port;
        gid_index = config.gid_index;
        
        // 自动检测 GID 索引
        if (gid_index < 0) {
            struct ibv_port_attr port_attr;
            if (ibv_query_port(ib_ctx, ib_port, &port_attr) == 0) {
                if (port_attr.link_layer == IBV_LINK_LAYER_ETHERNET) {
                    gid_index = 0;  // RoCE 默认使用 GID 0
                }
            }
        }
        
        // 创建保护域
        pd = ibv_alloc_pd(ib_ctx);
        if (!pd) {
            LOG_DEBUG_STREAM("[RDMA] 创建保护域失败");
            return false;
        }
        
        // 创建完成队列
        send_cq = ibv_create_cq(ib_ctx, RDMA_CQ_SIZE, nullptr, nullptr, 0);
        recv_cq = ibv_create_cq(ib_ctx, RDMA_CQ_SIZE, nullptr, nullptr, 0);
        if (!send_cq || !recv_cq) {
            LOG_DEBUG_STREAM("[RDMA] 创建完成队列失败");
            return false;
        }
        
        // 初始化连接池
        conn_pool.init(ib_ctx, pd, send_cq, recv_cq, ib_port, gid_index);
        
        LOG_DEBUG_STREAM("[RDMA] 设备初始化成功, port=" << (int)ib_port << ", gid_index=" << gid_index);
        return true;
    }

public:
    RdmaTransport(const TransportConfig& cfg) : config(cfg) {
        if (!init_rdma_device()) {
            throw std::runtime_error("RDMA设备初始化失败，请检查RDMA硬件环境");
        }
    }
    
    ~RdmaTransport() override {
        stop_server();
        
        // 关闭连接池
        conn_pool.close_all();
        
        // 清理活跃连接
        {
            std::lock_guard<std::mutex> lock(active_conns_mutex);
            active_conns.clear();
        }
        
        // 释放 RDMA 资源
        if (send_cq) { ibv_destroy_cq(send_cq); send_cq = nullptr; }
        if (recv_cq) { ibv_destroy_cq(recv_cq); recv_cq = nullptr; }
        if (pd) { ibv_dealloc_pd(pd); pd = nullptr; }
        if (ib_ctx) { ibv_close_device(ib_ctx); ib_ctx = nullptr; }
    }
    
    void register_node(int node_id, const std::string& ip, int port) override {
        conn_pool.register_node(node_id, ip, port);
    }
    
    int acquire_connection(int node_id) override {
        auto conn = conn_pool.acquire(node_id);
        if (!conn) return -1;
        return register_active_conn(conn);
    }
    
    void release_connection(int node_id, int conn_id, bool valid) override {
        auto conn = remove_active_conn(conn_id);
        if (conn) {
            conn_pool.release(node_id, conn, valid);
        }
    }
    
    void close_connection(int conn_id) override {
        auto conn = remove_active_conn(conn_id);
        // 连接从活跃映射中移除后，shared_ptr 引用计数归零时自动释放
        // 但 RDMA 资源需要显式释放
        if (conn && conn->aux_sock >= 0) {
            close(conn->aux_sock);
            conn->aux_sock = -1;
        }
    }
    
    int start_server(int port) override {
        // RDMA 模式下仍然使用 TCP 辅助通道接受连接
        // 客户端先通过 TCP 连接，完成 RDMA 握手后使用 RDMA 通信
        server_sockfd = socket(AF_INET, SOCK_STREAM, 0);
        if (server_sockfd < 0) {
            LOG_DEBUG_STREAM("[RDMA] 创建TCP辅助通道服务端socket失败");
            return -1;
        }
        
        int opt = 1;
        setsockopt(server_sockfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
        
        struct sockaddr_in addr;
        memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = INADDR_ANY;
        addr.sin_port = htons(port);
        
        if (bind(server_sockfd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            LOG_DEBUG_STREAM("[RDMA] bind 失败");
            close(server_sockfd);
            server_sockfd = -1;
            return -1;
        }
        
        if (listen(server_sockfd, 1024) < 0) {
            LOG_DEBUG_STREAM("[RDMA] listen 失败");
            close(server_sockfd);
            server_sockfd = -1;
            return -1;
        }
        
        server_running.store(true);
        LOG_DEBUG_STREAM("[RDMA] 服务端启动成功, port=" << port);
        return server_sockfd;
    }
    
    void stop_server() override {
        server_running.store(false);
        if (server_sockfd >= 0) {
            shutdown(server_sockfd, SHUT_RDWR);
            close(server_sockfd);
            server_sockfd = -1;
        }
    }
    
    int accept_client() override {
        if (server_sockfd < 0) return -1;
        
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(server_sockfd, (struct sockaddr*)&client_addr, &client_len);
        if (client_fd < 0) return -1;
        
        // 对于 RDMA 模式，接受 TCP 连接后需要完成 RDMA 握手
        // 但由于服务端的 handle_client 需要处理多种消息类型（包括非RDMA客户端），
        // 这里直接返回 TCP fd，上层逻辑通过 send_raw/recv_raw 透明处理
        // 对于需要 RDMA 的长连接场景，可以在后续升级为 RDMA 连接
        return client_fd;
    }
    
    void send_raw(int conn_id, const void* data, size_t size) override {
        // 尝试查找 RDMA 活跃连接
        auto conn = get_active_conn(conn_id);
        if (conn && conn->is_valid) {
            // 使用 RDMA Send 发送
            // 对于大消息需要分片发送
            const char* ptr = (const char*)data;
            size_t remaining = size;
            while (remaining > 0) {
                size_t chunk = std::min(remaining, RDMA_BUF_SIZE);
                if (rdma_send(conn, ptr, chunk) != 0) {
                    throw std::runtime_error("RDMA send 失败");
                }
                ptr += chunk;
                remaining -= chunk;
            }
            return;
        }
        
        // 回退到 TCP（用于服务端接受的 TCP 辅助通道连接）
        const char* buf = (const char*)data;
        size_t total = 0;
        while (total < size) {
            ssize_t n = ::send(conn_id, buf + total, size - total, 0);
            if (n <= 0) {
                throw std::runtime_error("RDMA transport: TCP send 失败");
            }
            total += n;
        }
    }
    
    void recv_raw(int conn_id, void* data, size_t size) override {
        // 尝试查找 RDMA 活跃连接
        auto conn = get_active_conn(conn_id);
        if (conn && conn->is_valid) {
            // 使用 RDMA Recv 接收
            char* ptr = (char*)data;
            size_t remaining = size;
            while (remaining > 0) {
                size_t actual_size = 0;
                if (rdma_recv(conn, ptr, remaining, actual_size) != 0) {
                    throw std::runtime_error("RDMA recv 失败");
                }
                ptr += actual_size;
                remaining -= actual_size;
            }
            return;
        }
        
        // 回退到 TCP（用于服务端接受的 TCP 辅助通道连接）
        char* buf = (char*)data;
        size_t total = 0;
        while (total < size) {
            ssize_t n = ::recv(conn_id, buf + total, size - total, MSG_WAITALL);
            if (n <= 0) {
                throw std::runtime_error("RDMA transport: TCP recv 失败");
            }
            total += n;
        }
    }
};

#endif // HAVE_RDMA
