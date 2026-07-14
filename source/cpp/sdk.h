#pragma once

#include "cluster.h"
#include <chrono>
#include <future>
#include <sys/time.h>
#include <poll.h>
#include <thread>
#include <vector>

// SDK Client：通过通信中间层与远程集群进行交互的客户端。
// 提供向量添加、K 近邻查询、分区、保存加载等远程操作接口，
// 支持 TCP 和 RDMA 两种传输模式，并具备自动路由发现和多端点负载均衡能力。
template <typename T>
class ClusterSDKClient
{
public:
    using DistT = DistTOf<T>;
private:
    struct QueryEndpoint
    {
        std::string ip;
        int port;
    };

    std::string host_ip;
    int host_port;
    int sockfd;
    bool connected;
    std::vector<QueryEndpoint> query_endpoints;
    std::vector<QueryEndpoint> all_work_endpoints;  // 全部 worker 节点（save/load_cluster_all 用）
    size_t current_endpoint_idx;  // 当前已连接的查询端点索引，(size_t)-1 表示未连接任何端点
    size_t sticky_endpoint_idx;   // 绑定的 endpoint 索引，避免每条查询重连

    // 通信中间层的配置及实例，由 TransportFactory 根据 comm_mode 创建
    TransportConfig transport_config;
    std::unique_ptr<ITransport> transport;

    // SDK 使用虚拟 node_id 来管理通信连接，避免与集群真实节点 ID 冲突
    static constexpr int SDK_HOST_NODE_ID = 99999;
    int next_virtual_node_id = SDK_HOST_NODE_ID;

    int get_virtual_node_id() {
        return next_virtual_node_id++;
    }

public:
    bool init_routes()
    {
        try {
            // 通过临时 TCP 连接获取集群路由信息，确定各节点的地址和分区配置
            int virtual_id = get_virtual_node_id();
            transport->register_node(virtual_id, host_ip, host_port);
            int info_sock = transport->acquire_connection(virtual_id);
            if (info_sock < 0) {
                return false;
            }

            // 使用 RAII 模式确保临时连接在作用域结束时自动释放
            struct SockGuard {
                ITransport* t;
                int node_id;
                int fd;
                ~SockGuard() {
                    if (fd >= 0) {
                        t->release_connection(node_id, fd, false);
                    }
                }
            } guard{transport.get(), virtual_id, info_sock};

            Buffer buf;
            buf.write_val((int)MSG_GET_INFORMATION);

            uint64_t msg_size = buf.data.size();
            transport->send_raw(info_sock, &msg_size, sizeof(uint64_t));
            transport->send_raw(info_sock, buf.data.data(), msg_size);

            uint64_t resp_size = 0;
            transport->recv_raw(info_sock, &resp_size, sizeof(uint64_t));

            std::vector<char> resp_data(resp_size);
            transport->recv_raw(info_sock, resp_data.data(), resp_size);

            Buffer resp_buf;
            resp_buf.data = std::move(resp_data);
            std::string info_str = resp_buf.read_string();
            if (info_str.empty()) {
                return false;
            }

            auto info = json::parse(info_str, nullptr, false);
            if (info.is_discarded() || !info.contains("cluster") || info["cluster"].is_null()) {
                return false;
            }

            const auto &cluster = info["cluster"];
            if (!cluster.contains("nodes") || !cluster["nodes"].is_array()) {
                return false;
            }

            std::unordered_map<int, std::pair<std::string, int>> node_addr;
            for (const auto &node : cluster["nodes"]) {
                int node_id = node.value("node_id", -1);
                std::string ip = node.value("ip_addr", "");
                int port = node.value("port", -1);
                if (node_id >= 0 && !ip.empty() && port > 0) {
                    if (port == host_port) {
                        ip = host_ip;
                    }
                    node_addr[node_id] = {ip, port};
                }
            }
            if (node_addr.empty()) {
                return false;
            }

            const bool is_shard = cluster.value("is_shard", false);
            std::vector<std::string> ips;
            std::vector<int> ports;

            if (is_shard) {
                // Shard 模式：轮询所有节点，每个节点都能独立处理查询
                for (const auto &kv : node_addr) {
                    ips.push_back(kv.second.first);
                    ports.push_back(kv.second.second);
                }
            } else {
                // Forward 模式（layer / 非 layer）：轮询持有最高层入口点的节点
                // 服务端在 MSG_GET_INFORMATION 中返回 entry_node_ids
                if (cluster.contains("entry_node_ids") && cluster["entry_node_ids"].is_array()
                    && !cluster["entry_node_ids"].empty()) {
                    for (const auto &nid_json : cluster["entry_node_ids"]) {
                        int nid = nid_json.get<int>();
                        if (node_addr.count(nid)) {
                            ips.push_back(node_addr[nid].first);
                            ports.push_back(node_addr[nid].second);
                        }
                    }
                }
                // 回退：如果没有 entry_node_ids 信息，使用所有节点
                if (ips.empty()) {
                    for (const auto &kv : node_addr) {
                        ips.push_back(kv.second.first);
                        ports.push_back(kv.second.second);
                    }
                }
            }

            if (ips.empty()) {
                return false;
            }

            set_query_endpoints(ips, ports);

            // 同时保存全部 worker 节点列表（sort by node_id 保证顺序）
            all_work_endpoints.clear();
            for (const auto &kv : node_addr) {
                all_work_endpoints.push_back({kv.second.first, kv.second.second});
            }
            std::sort(all_work_endpoints.begin(), all_work_endpoints.end(),
                [](const QueryEndpoint &a, const QueryEndpoint &b) { return a.port < b.port; });

            return true;
        } catch (...) {
            return false;
        }
    }

    ClusterSDKClient(const std::string &ip = "127.0.0.1", int port = 19000, const std::string &comm_mode = "tcp")
        : host_ip(ip), host_port(port), sockfd(-1), connected(false), current_endpoint_idx((size_t)-1), sticky_endpoint_idx((size_t)-1)
    {
        // SDK 服务端（handle_sdk_client）使用裸 TCP 收发，不走 transport 抽象层，
        // 因此 SDK 客户端必须始终使用 TCP transport，无论集群节点间通信模式是什么。
        transport_config.comm_mode = "tcp";
        transport = TransportFactory::create(transport_config);
        // 注册默认 host 节点，使得首次 connect() 可直接使用 transport 连接池
        transport->register_node(SDK_HOST_NODE_ID, host_ip, host_port);
    }

    ~ClusterSDKClient() { disconnect(); }

    bool connect(const std::string &ip = "", int port = -1)
    {
        if (connected) return true;

        const std::string &target_ip = ip.empty() ? host_ip : ip;
        int target_port = (port >= 0) ? port : host_port;

        // 通过 transport 层建立到目标地址的网络连接
        int virtual_id = get_virtual_node_id();
        transport->register_node(virtual_id, target_ip, target_port);
        sockfd = transport->acquire_connection(virtual_id);
        if (sockfd < 0) return false;

        connected = true;
        return true;
    }

    void disconnect()
    {
        if (connected && sockfd >= 0) {
            transport->close_connection(sockfd);
            sockfd = -1;
            connected = false;
            current_endpoint_idx = (size_t)-1;
        }
    }

    bool is_connected() const { return connected; }

    void set_query_endpoints(const std::vector<std::string> &ips, const std::vector<int> &ports)
    {
        if (ips.size() != ports.size()) {
            throw std::runtime_error("set_query_endpoints: ips/ports size mismatch");
        }

        query_endpoints.clear();
        query_endpoints.reserve(ips.size());
        for (size_t i = 0; i < ips.size(); ++i) {
            query_endpoints.push_back(QueryEndpoint{ips[i], ports[i]});
        }

        // 每个 SDK 客户端绑定一个 endpoint（基于线程 ID 哈希），避免每条查询重连。
        // 旧方案：每条查询随机选 endpoint，5 个节点有 80% 概率换节点 → disconnect + connect，
        //   导致大量 TCP 握手开销、远端 handler 线程频繁创建销毁、TIME_WAIT socket 堆积。
        // 新方案：init_routes() 后绑定一个 endpoint，后续所有查询复用同一连接。
        //   多个 SDK 客户端通过线程 ID 哈希自然分布到不同节点，实现负载均衡。
        if (!query_endpoints.empty()) {
            // 使用 PID 而非 thread_id：fork() 后所有子进程的 pthread_self() 相同，
            // 会导致全部客户端打到同一个节点。PID 在进程间唯一，分布更均匀。
            // 注意：Forward 模式下 entry_node_ids 通常只有少数节点，分布效果有限。
            size_t hash = std::hash<pid_t>{}(getpid());
            sticky_endpoint_idx = hash % query_endpoints.size();
        }
    }

    void add_vector(int vid, const std::vector<T> &vec)
    {
        if (!connected && !connect()) {
            throw std::runtime_error("Failed to connect to cluster");
        }

        try {
            Buffer buf;
            buf.write_val((int)MSG_ADD_VECTOR);
            buf.write_val(vid);
            buf.write_vec(vec);

            uint64_t msg_size = buf.data.size();
            transport->send_raw(sockfd, &msg_size, sizeof(uint64_t));
            transport->send_raw(sockfd, buf.data.data(), msg_size);

            char ack = 0;
            transport->recv_raw(sockfd, &ack, 1);
            if (ack != 1) {
                throw std::runtime_error("Failed to add vector");
            }
        } catch (...) {
            disconnect();
            throw;
        }
        // 持久连接模式：不主动断开，后续的 add_vector / query 调用可复用同一连接
    }

    // 确保连接到绑定的 endpoint，失败时带指数退避重试
    void ensure_connected()
    {
        size_t target_idx = sticky_endpoint_idx;
        std::string target_ip = host_ip;
        int target_port = host_port;

        if (target_idx != (size_t)-1 && target_idx < query_endpoints.size()) {
            const QueryEndpoint &ep = query_endpoints[target_idx];
            target_ip = ep.ip;
            target_port = ep.port;
        }

        if (connected && current_endpoint_idx != target_idx) {
            disconnect();
        }

        if (!connected) {
            static constexpr int MAX_RETRIES = 5;
            static constexpr int BASE_DELAY_MS = 100;
            for (int attempt = 0; attempt < MAX_RETRIES; ++attempt) {
                if (connect(target_ip, target_port)) break;
                if (attempt + 1 < MAX_RETRIES) {
                    std::this_thread::sleep_for(
                        std::chrono::milliseconds(BASE_DELAY_MS << attempt));
                }
            }
            if (!connected) {
                throw std::runtime_error("Failed to connect to cluster");
            }
        }
        current_endpoint_idx = target_idx;
    }

    // 发送查询，不等待结果（流水线模式）
    void query_send(int query_id, const std::vector<T> &query_vector, Payload<DistT> &payload, int k = 10)
    {
        ensure_connected();

        try {
            Buffer buf;
            buf.write_val((int)MSG_QUERY);
            buf.write_val(query_id);
            buf.write_val(k);
            buf.write_vec(query_vector);
            // 客户端无本地 HNSWGraph，传 nullptr（堆本就为空，serialize 跳过转换）
            serialize_payload(buf, payload, (const HNSWGraph<T>*)nullptr);

            uint64_t msg_size = buf.data.size();
            double send_start_ms = steady_now_ms();
            transport->send_raw(sockfd, &msg_size, sizeof(uint64_t));
            transport->send_raw(sockfd, buf.data.data(), msg_size);
            payload.send_ms += steady_now_ms() - send_start_ms;
        } catch (...) {
            disconnect();
            throw;
        }
    }

    // 接收一个已完成的查询结果（阻塞直到有结果）
    // 返回 (query_id, results, payload)
    std::tuple<int, std::vector<int>, Payload<DistT>> query_recv()
    {
        try {
            uint64_t resp_size = 0;
            struct pollfd pfd{sockfd, POLLIN, 0};
            int pret;
            do {
                pret = ::poll(&pfd, 1, -1);
            } while (pret < 0 && errno == EINTR);
            if (pret <= 0 || !(pfd.revents & POLLIN)) {
                throw std::runtime_error("query_recv: socket not readable");
            }
            double recv_start_ms = steady_now_ms();
            transport->recv_raw(sockfd, &resp_size, sizeof(uint64_t));

            std::vector<char> resp_data(resp_size);
            transport->recv_raw(sockfd, resp_data.data(), resp_size);
            double recv_ms = steady_now_ms() - recv_start_ms;

            Buffer resp_buf;
            resp_buf.data = std::move(resp_data);

            int query_id = resp_buf.read_val<int>();
            int result_cnt = resp_buf.read_val<int>();
            std::vector<int> results;
            for (int i = 0; i < result_cnt; i++) {
                resp_buf.read_val<DistT>(); // dist (服务端按 DistT 写入，uint8 存储下为 int32)
                results.push_back(resp_buf.read_val<int>());
            }
            Payload<DistT> payload;
            // 客户端收回的堆项被置为 -(vid+1) 编码；客户端不再搜索，仅保留统计信息
            deserialize_payload(resp_buf, payload, (const HNSWGraph<T>*)nullptr);
            payload.recv_ms += recv_ms;

            return {query_id, std::move(results), std::move(payload)};
        } catch (...) {
            disconnect();
            throw;
        }
    }

    // 便捷批量接口：发送一批查询，全部接收后返回
    std::vector<std::tuple<int, std::vector<int>, Payload<DistT>>>
    query_batch(const std::vector<int> &query_ids,
                const std::vector<std::vector<T>> &query_vectors,
                std::vector<Payload<DistT>> &payloads, int k = 10)
    {
        if (query_ids.size() != query_vectors.size() || query_ids.size() != payloads.size())
            throw std::runtime_error("query_batch: size mismatch");

        for (size_t i = 0; i < query_ids.size(); i++)
            query_send(query_ids[i], query_vectors[i], payloads[i], k);

        std::vector<std::tuple<int, std::vector<int>, Payload<DistT>>> results;
        results.reserve(query_ids.size());
        for (size_t i = 0; i < query_ids.size(); i++)
            results.push_back(query_recv());

        return results;
    }

    std::string get_information()
    {
        if (!connected && !connect()) {
            throw std::runtime_error("Failed to connect to cluster");
        }

        try {
            Buffer buf;
            buf.write_val((int)MSG_GET_INFORMATION);

            uint64_t msg_size = buf.data.size();
            transport->send_raw(sockfd, &msg_size, sizeof(uint64_t));
            transport->send_raw(sockfd, buf.data.data(), msg_size);

            uint64_t resp_size = 0;
            transport->recv_raw(sockfd, &resp_size, sizeof(uint64_t));

            std::vector<char> resp_data(resp_size);
            transport->recv_raw(sockfd, resp_data.data(), resp_size);

            Buffer resp_buf;
            resp_buf.data = std::move(resp_data);
            std::string info = resp_buf.read_string();

            disconnect();
            return info;
        } catch (...) {
            disconnect();
            throw;
        }
    }

    void partition(const std::string &mode, float redundancy = 1.0f, const std::unordered_map<int, std::vector<int>> &layer_config = {}, const std::string &layer_partition_mode = "round_robin", const std::vector<bool> &layer_replicate_flags = {}, int num_threads = 0, const std::string &partition_distribution = "network-send", const std::string &partition_base = "")
    {
        if (!connected && !connect()) {
            throw std::runtime_error("Failed to connect to cluster");
        }

        try {
            Buffer buf;
            buf.write_val((int)MSG_PARTITION);
            buf.write_string(mode);
            buf.write_val(redundancy);
            serialize_layer_config(buf, layer_config);
            buf.write_string(layer_partition_mode);
            serialize_layer_replicate_flags(buf, layer_replicate_flags);
            buf.write_val(num_threads);
            buf.write_string(partition_distribution);
            buf.write_string(partition_base);

            uint64_t msg_size = buf.data.size();
            transport->send_raw(sockfd, &msg_size, sizeof(uint64_t));
            transport->send_raw(sockfd, buf.data.data(), msg_size);

            char ack = 0;
            transport->recv_raw(sockfd, &ack, 1);
            if (ack != 1) {
                throw std::runtime_error("Failed to partition");
            }
        } catch (...) {
            disconnect();
            throw;
        }
        disconnect();
    }

    void load_cluster(const std::string &filepath)
    {
        if (!connected && !connect()) {
            throw std::runtime_error("Failed to connect to cluster");
        }

        try {
            Buffer buf;
            buf.write_val((int)MSG_LOAD_CLUSTER);
            buf.write_string(filepath);

            uint64_t msg_size = buf.data.size();
            transport->send_raw(sockfd, &msg_size, sizeof(uint64_t));
            transport->send_raw(sockfd, buf.data.data(), msg_size);

            char ack = 0;
            transport->recv_raw(sockfd, &ack, 1);
            if (ack != 1) {
                throw std::runtime_error("Failed to load cluster");
            }
        } catch (...) {
            disconnect();
            throw;
        }
        disconnect();
    }

    void save_cluster(const std::string &filepath)
    {
        if (!connected && !connect()) {
            throw std::runtime_error("Failed to connect to cluster");
        }

        try {
            Buffer buf;
            buf.write_val((int)MSG_SAVE_CLUSTER);
            buf.write_string(filepath);

            uint64_t msg_size = buf.data.size();
            transport->send_raw(sockfd, &msg_size, sizeof(uint64_t));
            transport->send_raw(sockfd, buf.data.data(), msg_size);

            char ack = 0;
            transport->recv_raw(sockfd, &ack, 1);
            if (ack != 1) {
                throw std::runtime_error("Failed to save cluster");
            }
        } catch (...) {
            disconnect();
            throw;
        }
        disconnect();
    }

    // 并行批量 save
    void save_cluster_all(const std::string &basepath)
    {
        std::vector<std::future<void>> futures;
        futures.reserve(all_work_endpoints.size());
        for (size_t i = 0; i < all_work_endpoints.size(); i++)
        {
            futures.push_back(std::async(std::launch::async, [this, i, &basepath]() {
                std::string node_path = basepath + "-" + std::to_string(i) + ".bin";
                const auto &endpoint = all_work_endpoints[i];
                ClusterSDKClient node_client(endpoint.ip, endpoint.port, transport_config.comm_mode);
                if (!node_client.connect())
                    throw std::runtime_error("Failed to connect to node " + std::to_string(i) + " for save");
                try { node_client.save_cluster(node_path); }
                catch (...) { node_client.disconnect(); throw; }
                node_client.disconnect();
            }));
        }
        for (auto &f : futures) f.get(); // wait all + propagate exceptions
    }

    // 并行批量 load
    void load_cluster_all(const std::string &basepath)
    {
        std::vector<std::future<void>> futures;
        futures.reserve(all_work_endpoints.size());
        for (size_t i = 0; i < all_work_endpoints.size(); i++)
        {
            futures.push_back(std::async(std::launch::async, [this, i, &basepath]() {
                std::string node_path = basepath + "-" + std::to_string(i) + ".bin";
                const auto &endpoint = all_work_endpoints[i];
                ClusterSDKClient node_client(endpoint.ip, endpoint.port, transport_config.comm_mode);
                if (!node_client.connect())
                    throw std::runtime_error("Failed to connect to node " + std::to_string(i) + " for load");
                try { node_client.load_cluster(node_path); }
                catch (...) { node_client.disconnect(); throw; }
                node_client.disconnect();
            }));
        }
        for (auto &f : futures) f.get();
    }

    void save_hnsw(const std::string &filepath)
    {
        if (!connected && !connect()) {
            throw std::runtime_error("Failed to connect to cluster");
        }

        try {
            Buffer buf;
            buf.write_val((int)MSG_SAVE_HNSW);
            buf.write_string(filepath);

            uint64_t msg_size = buf.data.size();
            transport->send_raw(sockfd, &msg_size, sizeof(uint64_t));
            transport->send_raw(sockfd, buf.data.data(), msg_size);

            char ack = 0;
            transport->recv_raw(sockfd, &ack, 1);
            if (ack != 1) {
                throw std::runtime_error("Failed to save hnsw");
            }
        } catch (...) {
            disconnect();
            throw;
        }
        disconnect();
    }

    void load_hnsw(const std::string &filepath)
    {
        if (!connected && !connect()) {
            throw std::runtime_error("Failed to connect to cluster");
        }

        try {
            Buffer buf;
            buf.write_val((int)MSG_LOAD_HNSW);
            buf.write_string(filepath);

            uint64_t msg_size = buf.data.size();
            transport->send_raw(sockfd, &msg_size, sizeof(uint64_t));
            transport->send_raw(sockfd, buf.data.data(), msg_size);

            char ack = 0;
            transport->recv_raw(sockfd, &ack, 1);
            if (ack != 1) {
                throw std::runtime_error("Failed to load hnsw");
            }
        } catch (...) {
            disconnect();
            throw;
        }
        disconnect();
    }

    void release_full_hnsw()
    {
        if (!connected && !connect()) {
            throw std::runtime_error("Failed to connect to cluster");
        }

        try {
            Buffer buf;
            buf.write_val((int)MSG_RELEASE_FULL_HNSW);

            uint64_t msg_size = buf.data.size();
            transport->send_raw(sockfd, &msg_size, sizeof(uint64_t));
            transport->send_raw(sockfd, buf.data.data(), msg_size);

            char ack = 0;
            transport->recv_raw(sockfd, &ack, 1);
            if (ack != 1) {
                throw std::runtime_error("Failed to release full hnsw");
            }
        } catch (...) {
            disconnect();
            throw;
        }
        disconnect();
    }

    void release_full_hnsw_all()
    {
        std::vector<std::future<void>> futures;
        futures.reserve(all_work_endpoints.size());
        for (size_t i = 0; i < all_work_endpoints.size(); i++)
        {
            futures.push_back(std::async(std::launch::async, [this, i]() {
                const auto &endpoint = all_work_endpoints[i];
                ClusterSDKClient node_client(endpoint.ip, endpoint.port, transport_config.comm_mode);
                if (!node_client.connect())
                    throw std::runtime_error("Failed to connect to node " + std::to_string(i) + " for full graph release");
                try { node_client.release_full_hnsw(); }
                catch (...) { node_client.disconnect(); throw; }
                node_client.disconnect();
            }));
        }
        for (auto &f : futures) f.get();
    }

    // 专用参数设置接口：运行时更新已加载图的可调参数。
    // 约定：ef_search/ef_construction 为负值表示不更新；limited_level 为 -2 表示不更新，-1 表示不限制。
    void set_hnsw_params(int ef_search = -1, int ef_construction = -1, int limited_level = -2)
    {
        if (!connected && !connect()) {
            throw std::runtime_error("Failed to connect to cluster");
        }

        try {
            Buffer buf;
            buf.write_val((int)MSG_SET_HNSW_PARAMS);
            buf.write_val(ef_search);
            buf.write_val(ef_construction);
            buf.write_val(limited_level);

            uint64_t msg_size = buf.data.size();
            transport->send_raw(sockfd, &msg_size, sizeof(uint64_t));
            transport->send_raw(sockfd, buf.data.data(), msg_size);

            char ack = 0;
            transport->recv_raw(sockfd, &ack, 1);
            if (ack != 1) {
                throw std::runtime_error("Failed to set hnsw params");
            }
        } catch (...) {
            disconnect();
            throw;
        }
        disconnect();
    }
};
