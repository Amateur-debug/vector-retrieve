
#pragma once

#include "transport.h"

#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <unistd.h>
#include <poll.h>
#include <cstring>
#include <deque>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <string>
#include <stdexcept>

#include "log.h"

// ==================== TCP 连接池 ====================
// 与 cluster.h 中的 TcpConnectionPool 相同实现，迁移到此处统一管理
class TcpConnectionPool {
private:
    struct NodeConnections {
        std::deque<int> idle_sockets;
        std::mutex mutex;
        std::string ip;
        int port = 0;
        int active_count = 0;
    };
    
    std::unordered_map<int, NodeConnections> node_pools;
    std::mutex pools_mutex;
    
    int create_connection(const std::string& ip, int port) {
        int sock = socket(AF_INET, SOCK_STREAM, 0);
        if (sock < 0) return -1;
        
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
    
    bool is_socket_valid(int sock) {
        struct pollfd pfd;
        pfd.fd = sock;
        pfd.events = POLLIN | POLLOUT;
        pfd.revents = 0;
        
        int ret = poll(&pfd, 1, 0);
        if (ret <= 0) return true;
        
        if (pfd.revents & (POLLERR | POLLHUP | POLLNVAL)) {
            return false;
        }
        return true;
    }
    
public:
    void register_node(int node_id, const std::string& ip, int port) {
        std::lock_guard<std::mutex> lock(pools_mutex);
        auto result = node_pools.try_emplace(node_id);
        NodeConnections& nc = result.first->second;
        nc.ip = ip;
        nc.port = port;
    }
    
    int acquire(int node_id) {
        std::string ip;
        int port = 0;
        
        {
            std::lock_guard<std::mutex> lock(pools_mutex);
            auto it = node_pools.find(node_id);
            if (it == node_pools.end()) {
                return -1;
            }
            
            NodeConnections& nc = it->second;
            std::lock_guard<std::mutex> nc_lock(nc.mutex);
            
            while (!nc.idle_sockets.empty()) {
                int sock = nc.idle_sockets.front();
                nc.idle_sockets.pop_front();
                
                if (is_socket_valid(sock)) {
                    nc.active_count++;
                    int idle = (int)nc.idle_sockets.size();
                    int active = nc.active_count;
                    LOG_DEBUG_STREAM("[ConnectionPool] Acquired (reused) connection for node " << node_id 
                        << ", socket=" << sock 
                        << ", idle=" << idle
                        << ", active=" << active
                        << ", total=" << (idle + active));
                    return sock;
                }
                close(sock);
            }
            
            ip = nc.ip;
            port = nc.port;
        }
        
        if (!ip.empty() && port > 0) {
            int new_sock = create_connection(ip, port);
            if (new_sock >= 0) {
                std::lock_guard<std::mutex> lock(pools_mutex);
                auto it = node_pools.find(node_id);
                if (it != node_pools.end()) {
                    std::lock_guard<std::mutex> nc_lock(it->second.mutex);
                    it->second.active_count++;
                    int idle = (int)it->second.idle_sockets.size();
                    int active = it->second.active_count;
                    LOG_DEBUG_STREAM("[ConnectionPool] Acquired (created) connection for node " << node_id 
                        << ", socket=" << new_sock 
                        << ", idle=" << idle
                        << ", active=" << active
                        << ", total=" << (idle + active));
                }
            }
            return new_sock;
        }
        return -1;
    }
    
    void release(int node_id, int sock, bool valid = true) {
        if (sock < 0) return;
        
        std::lock_guard<std::mutex> lock(pools_mutex);
        auto it = node_pools.find(node_id);
        if (it == node_pools.end()) {
            close(sock);
            return;
        }
        
        NodeConnections& nc = it->second;
        std::lock_guard<std::mutex> nc_lock(nc.mutex);
        
        if (!valid || !is_socket_valid(sock)) {
            close(sock);
            if (nc.active_count > 0) nc.active_count--;
            LOG_DEBUG_STREAM("[ConnectionPool] Closed invalid connection for node " << node_id << ", socket=" << sock);
            return;
        }
        
        nc.idle_sockets.push_back(sock);
        if (nc.active_count > 0) nc.active_count--;
        
        int idle = (int)nc.idle_sockets.size();
        int active = nc.active_count;
        LOG_DEBUG_STREAM("[ConnectionPool] Released connection for node " << node_id 
            << ", socket=" << sock 
            << ", idle=" << idle
            << ", active=" << active
            << ", total=" << (idle + active));
    }
    
    void close_node(int node_id) {
        std::lock_guard<std::mutex> lock(pools_mutex);
        auto it = node_pools.find(node_id);
        if (it == node_pools.end()) return;
        
        NodeConnections& nc = it->second;
        std::lock_guard<std::mutex> nc_lock(nc.mutex);
        size_t closed_count = nc.idle_sockets.size();
        for (int sock : nc.idle_sockets) {
            close(sock);
        }
        nc.idle_sockets.clear();
        nc.active_count = 0;
        
        LOG_DEBUG_STREAM("[ConnectionPool] Closed all connections for node " << node_id 
            << ", closed_idle=" << closed_count);
    }
    
    std::string get_stats() {
        std::lock_guard<std::mutex> lock(pools_mutex);
        std::string stats;
        for (const auto& kv : node_pools) {
            stats += "Node " + std::to_string(kv.first) + 
                     ": " + std::to_string(kv.second.idle_sockets.size()) + " idle\n";
        }
        return stats;
    }
    
    void clear() {
        std::lock_guard<std::mutex> lock(pools_mutex);
        size_t total_closed = 0;
        for (auto& kv : node_pools) {
            std::lock_guard<std::mutex> nc_lock(kv.second.mutex);
            total_closed += kv.second.idle_sockets.size();
            for (int sock : kv.second.idle_sockets) {
                close(sock);
            }
            kv.second.idle_sockets.clear();
            kv.second.active_count = 0;
        }
        node_pools.clear();
        LOG_DEBUG_STREAM("[ConnectionPool] Cleared all connections, total_closed=" << total_closed);
    }
};


// ==================== TcpTransport 实现 ====================
class TcpTransport : public ITransport {
private:
    TcpConnectionPool pool_;
    int server_fd_ = -1;

    // 底层发送/接收辅助函数
    static void send_all_impl(int sock, const void* data, size_t size) {
        const char* ptr = static_cast<const char*>(data);
        size_t total_sent = 0;
        while (total_sent < size) {
            ssize_t sent = send(sock, ptr + total_sent, size - total_sent, 0);
            if (sent <= 0) {
                throw std::runtime_error("send failed");
            }
            total_sent += sent;
        }
    }

    static void recv_all_impl(int sock, void* data, size_t size) {
        char* ptr = static_cast<char*>(data);
        size_t total_read = 0;
        while (total_read < size) {
            ssize_t r = recv(sock, ptr + total_read, size - total_read, 0);
            if (r <= 0) {
                throw std::runtime_error("recv failed");
            }
            total_read += r;
        }
    }

public:
    TcpTransport() = default;

    ~TcpTransport() override {
        stop_server();
        pool_.clear();
    }

    // --- 服务端生命周期 ---

    int start_server(int port) override {
        server_fd_ = socket(AF_INET, SOCK_STREAM, 0);
        if (server_fd_ < 0) {
            return -1;
        }

        int opt = 1;
        setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        struct sockaddr_in addr;
        memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = INADDR_ANY;
        addr.sin_port = htons(port);

        if (bind(server_fd_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            perror("bind");
            close(server_fd_);
            server_fd_ = -1;
            return -1;
        }

        if (listen(server_fd_, 1024) < 0) {
            perror("listen");
            close(server_fd_);
            server_fd_ = -1;
            return -1;
        }

        return server_fd_;
    }

    void stop_server() override {
        if (server_fd_ != -1) {
            shutdown(server_fd_, SHUT_RDWR);
            close(server_fd_);
            server_fd_ = -1;
        }
    }

    int accept_client() override {
        if (server_fd_ < 0) return -1;

        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(server_fd_, (struct sockaddr*)&client_addr, &client_len);
        return client_fd;
    }

    // --- 节点注册与连接管理 ---

    void register_node(int node_id, const std::string& ip, int port) override {
        pool_.register_node(node_id, ip, port);
    }

    int acquire_connection(int node_id) override {
        return pool_.acquire(node_id);
    }

    void release_connection(int node_id, int conn_id, bool valid) override {
        pool_.release(node_id, conn_id, valid);
    }

    void close_connection(int conn_id) override {
        if (conn_id < 0) return;
        // 硬关闭：设置 SO_LINGER 为 0，发送 RST
        struct linger so_linger;
        so_linger.l_onoff = 1;
        so_linger.l_linger = 0;
        setsockopt(conn_id, SOL_SOCKET, SO_LINGER, &so_linger, sizeof(so_linger));
        close(conn_id);
    }

    // --- 数据传输 ---

    void send_message(int conn_id, const void* data, size_t size) override {
        uint64_t msg_size = size;
        send_all_impl(conn_id, &msg_size, sizeof(uint64_t));
        send_all_impl(conn_id, data, size);
    }

    size_t recv_message(int conn_id, std::vector<char>& buffer) override {
        uint64_t msg_size = 0;
        recv_all_impl(conn_id, &msg_size, sizeof(uint64_t));
        buffer.resize(msg_size);
        recv_all_impl(conn_id, buffer.data(), msg_size);
        return msg_size;
    }

    void send_raw(int conn_id, const void* data, size_t size) override {
        send_all_impl(conn_id, data, size);
    }

    void recv_raw(int conn_id, void* data, size_t size) override {
        recv_all_impl(conn_id, data, size);
    }

    // --- 资源管理 ---

    void close_node(int node_id) override {
        pool_.close_node(node_id);
    }

    void clear() override {
        pool_.clear();
    }

    std::string get_stats() override {
        return pool_.get_stats();
    }

    // --- 获取服务端 fd（供兼容层使用）---
    int get_server_fd() const { return server_fd_; }
};


// ==================== 条件引入 RDMA Transport ====================
#ifdef HAVE_RDMA
#include "rdma_transport.h"
#endif

// ==================== TransportFactory 实现 ====================
inline std::unique_ptr<ITransport> TransportFactory::create(const TransportConfig& config) {
    if (config.comm_mode == "tcp") {
        return std::make_unique<TcpTransport>();
    }
#ifdef HAVE_RDMA
    else if (config.comm_mode == "rdma") {
        return std::make_unique<RdmaTransport>(config);
    }
#endif
    else if (config.comm_mode == "rdma") {
        throw std::runtime_error("RDMA transport is not available: the project was compiled without RDMA support (HAVE_RDMA not defined). "
                                 "Please install libibverbs and librdmacm, then rebuild.");
    }
    throw std::runtime_error("Unsupported transport mode: " + config.comm_mode);
}

