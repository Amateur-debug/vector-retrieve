#pragma once

#include "cluster.h"

// SDK Client: 通过通信中间层与集群交互的客户端
template <typename T>
class ClusterSDKClient
{
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
    size_t current_endpoint_idx;  // 当前连接的端点索引，-1 表示无效

    // 通信中间层配置和实例
    TransportConfig transport_config;
    std::unique_ptr<ITransport> transport;

    // SDK 使用虚拟 node_id 来管理连接
    static constexpr int SDK_HOST_NODE_ID = 99999;
    int next_virtual_node_id = SDK_HOST_NODE_ID;

    int get_virtual_node_id() {
        return next_virtual_node_id++;
    }

public:
    bool init_routes()
    {
        try {
            // 使用 transport 进行路由初始化通信
            int virtual_id = get_virtual_node_id();
            transport->register_node(virtual_id, host_ip, host_port);
            int info_sock = transport->acquire_connection(virtual_id);
            if (info_sock < 0) {
                return false;
            }

            // 使用 RAII 确保资源释放
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

            const std::string partition_mode = cluster.value("partition_mode", "");
            std::vector<std::string> ips;
            std::vector<int> ports;

            if (partition_mode == "layer" && cluster.contains("layer_config")) {
                const auto &layer_config = cluster["layer_config"];
                std::vector<int> configured_levels;
                std::unordered_map<int, std::vector<int>> parsed_layer_config;

                if (layer_config.is_object()) {
                    configured_levels.reserve(layer_config.size());
                    for (auto it = layer_config.begin(); it != layer_config.end(); ++it) {
                        try {
                            int level = std::stoi(it.key());
                            if (level >= 0 && it.value().is_array()) {
                                parsed_layer_config[level] = it.value().get<std::vector<int>>();
                                configured_levels.push_back(level);
                            }
                        } catch (...) {
                            continue;
                        }
                    }
                } else if (layer_config.is_array()) {
                    configured_levels.reserve(layer_config.size());
                    for (const auto &item : layer_config) {
                        if (!item.is_array() || item.size() != 2) {
                            continue;
                        }
                        int level = -1;
                        try {
                            level = item[0].get<int>();
                        } catch (...) {
                            continue;
                        }
                        if (level < 0 || !item[1].is_array()) {
                            continue;
                        }
                        try {
                            parsed_layer_config[level] = item[1].get<std::vector<int>>();
                            configured_levels.push_back(level);
                        } catch (...) {
                            continue;
                        }
                    }
                }

                if (!configured_levels.empty()) {
                    std::sort(configured_levels.begin(), configured_levels.end(), std::greater<int>());
                    std::vector<int> entry_nodes;
                    int selected_entry_level = -1;
                    
                    for (int level : configured_levels) {
                        std::vector<int> level_nodes;
                        const auto it = parsed_layer_config.find(level);
                        if (it == parsed_layer_config.end()) {
                            continue;
                        }
                        for (int nid : it->second) {
                            if (node_addr.count(nid)) {
                                level_nodes.push_back(nid);
                            }
                        }
                        if (!level_nodes.empty()) {
                            selected_entry_level = level;
                            entry_nodes = std::move(level_nodes);
                            break;
                        }
                    }

                    if (selected_entry_level >= 0 && !entry_nodes.empty()) {
                        bool entry_replicated = false;
                        if (cluster.contains("layer_replicate_flags") && cluster["layer_replicate_flags"].is_array()) {
                            const auto &flags = cluster["layer_replicate_flags"];
                            if (selected_entry_level < (int)flags.size()) {
                                entry_replicated = flags[selected_entry_level].get<bool>();
                            }
                        }

                        if (entry_replicated) {
                            for (int nid : entry_nodes) {
                                ips.push_back(node_addr[nid].first);
                                ports.push_back(node_addr[nid].second);
                            }
                        } else {
                            int nid = entry_nodes.front();
                            ips.push_back(node_addr[nid].first);
                            ports.push_back(node_addr[nid].second);
                        }
                    }
                }
            } else {
                for (const auto &kv : node_addr) {
                    ips.push_back(kv.second.first);
                    ports.push_back(kv.second.second);
                }
            }

            if (ips.empty()) {
                return false;
            }

            set_query_endpoints(ips, ports);
            return true;
        } catch (...) {
            return false;
        }
    }

    ClusterSDKClient(const std::string &ip = "127.0.0.1", int port = 19000, const std::string &comm_mode = "tcp")
        : host_ip(ip), host_port(port), sockfd(-1), connected(false), current_endpoint_idx((size_t)-1)
    {
        transport_config.comm_mode = comm_mode;
        transport = TransportFactory::create(transport_config);
        // 注册默认 host 节点
        transport->register_node(SDK_HOST_NODE_ID, host_ip, host_port);
    }

    ~ClusterSDKClient() { disconnect(); }

    bool connect(const std::string &ip = "", int port = -1)
    {
        if (connected) return true;

        const std::string &target_ip = ip.empty() ? host_ip : ip;
        int target_port = (port >= 0) ? port : host_port;

        // 通过 transport 建立连接
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
        // 持久连接：不断开，下次 add_vector 复用同一连接
    }

    std::vector<int> query(int query_id, const std::vector<T> &query_vector, Payload<T> &payload, int k = 10)
    {
        size_t target_idx = (size_t)-1;
        std::string target_ip = host_ip;
        int target_port = host_port;
        
        if (!query_endpoints.empty()) {
            static thread_local std::mt19937 gen(std::random_device{}());
            std::uniform_int_distribution<size_t> dist(0, query_endpoints.size() - 1);
            target_idx = dist(gen);
            const QueryEndpoint &ep = query_endpoints[target_idx];
            target_ip = ep.ip;
            target_port = ep.port;
        }

        if (connected && current_endpoint_idx != target_idx) {
            disconnect();
        }

        if (!connected && !connect(target_ip, target_port)) {
            throw std::runtime_error("Failed to connect to cluster");
        }
        current_endpoint_idx = target_idx;

        try {
            Buffer buf;
            buf.write_val((int)MSG_QUERY);
            buf.write_val(query_id);
            buf.write_val(k);
            buf.write_vec(query_vector);
            serialize_payload(buf, payload);

            uint64_t msg_size = buf.data.size();
            transport->send_raw(sockfd, &msg_size, sizeof(uint64_t));
            transport->send_raw(sockfd, buf.data.data(), msg_size);

            uint64_t resp_size = 0;
            transport->recv_raw(sockfd, &resp_size, sizeof(uint64_t));

            std::vector<char> resp_data(resp_size);
            transport->recv_raw(sockfd, resp_data.data(), resp_size);

            Buffer resp_buf;
            resp_buf.data = std::move(resp_data);

            int result_cnt = resp_buf.read_val<int>();
            std::vector<int> results;
            for (int i = 0; i < result_cnt; i++) {
                resp_buf.read_val<T>(); // dist
                results.push_back(resp_buf.read_val<int>());
            }
            deserialize_payload(resp_buf, payload);

            return results;
        } catch (...) {
            disconnect();
            throw;
        }
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

    void partition(const std::string &mode, int redundancy = 1, const std::unordered_map<int, std::vector<int>> &layer_config = {}, const std::string &layer_partition_mode = "round_robin", const std::vector<bool> &layer_replicate_flags = {})
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
};
