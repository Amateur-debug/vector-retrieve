#pragma once

#include "cluster.h"

// SDK Client: TCP client for querying cluster via network
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
    int timeout_sec;
    std::vector<QueryEndpoint> query_endpoints;

public:
    bool init_routes()
    {
        try {
            int info_sock = socket(AF_INET, SOCK_STREAM, 0);
            if (info_sock < 0) {
                return false;
            }
            struct InfoSockGuard {
                int fd;
                ~InfoSockGuard() {
                    if (fd >= 0) {
                        shutdown(fd, SHUT_RDWR);
                        close(fd);
                    }
                }
            } info_guard{info_sock};

            struct sockaddr_in serv_addr;
            memset(&serv_addr, 0, sizeof(serv_addr));
            serv_addr.sin_family = AF_INET;
            serv_addr.sin_port = htons(host_port);
            if (inet_pton(AF_INET, host_ip.c_str(), &serv_addr.sin_addr) <= 0) {
                return false;
            }
            if (::connect(info_sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
                return false;
            }

            Buffer buf;
            buf.write_val((int)MSG_GET_INFORMATION);

            uint64_t msg_size = buf.data.size();
            send_all(info_sock, &msg_size, sizeof(uint64_t));
            send_all(info_sock, buf.data.data(), msg_size);

            uint64_t resp_size = 0;
            recv_all(info_sock, &resp_size, sizeof(uint64_t));

            std::vector<char> resp_data(resp_size);
            recv_all(info_sock, resp_data.data(), resp_size);

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
                    node_addr[node_id] = {ip, port};
                }
            }
            if (node_addr.empty()) {
                return false;
            }

            std::vector<int> entry_nodes;
            bool entry_replicated = false;
            int selected_entry_level = -1;
            const std::string partition_mode = cluster.value("partition_mode", "");
            if (partition_mode != "layer" || !cluster.contains("layer_config")) {
                return false;
            }

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
            } else {
                return false;
            }

            if (configured_levels.empty()) {
                return false;
            }

            std::sort(configured_levels.begin(), configured_levels.end(), std::greater<int>());
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

            if (selected_entry_level < 0 || entry_nodes.empty()) {
                return false;
            }

            if (cluster.contains("layer_replicate_flags") && cluster["layer_replicate_flags"].is_array()) {
                const auto &flags = cluster["layer_replicate_flags"];
                if (selected_entry_level < (int)flags.size()) {
                    entry_replicated = flags[selected_entry_level].get<bool>();
                }
            }

            std::vector<std::string> ips;
            std::vector<int> ports;
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

            set_query_endpoints(ips, ports);
            return true;
        } catch (...) {
            // 自动路由初始化失败时，保留默认 host_ip/host_port 作为回退路径。
            return false;
        }
    }

    ClusterSDKClient(const std::string &ip = "127.0.0.1", int port = 19000, int timeout = 5)
        : host_ip(ip), host_port(port), sockfd(-1), connected(false), timeout_sec(timeout) {}

    ~ClusterSDKClient() { disconnect(); }

    bool connect()
    {
        if (connected) return true;

        sockfd = socket(AF_INET, SOCK_STREAM, 0);
        if (sockfd < 0) return false;

        struct sockaddr_in serv_addr;
        memset(&serv_addr, 0, sizeof(serv_addr));
        serv_addr.sin_family = AF_INET;
        serv_addr.sin_port = htons(host_port);
        if (inet_pton(AF_INET, host_ip.c_str(), &serv_addr.sin_addr) <= 0) {
            close(sockfd); sockfd = -1; return false;
        }

        if (::connect(sockfd, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
            close(sockfd); sockfd = -1; return false;
        }

        connected = true;
        return true;
    }

    void disconnect()
    {
        if (connected && sockfd >= 0) {
            shutdown(sockfd, SHUT_RDWR);
            close(sockfd);
            sockfd = -1;
            connected = false;
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
            send_all(sockfd, &msg_size, sizeof(uint64_t));
            send_all(sockfd, buf.data.data(), msg_size);

            // Receive ACK
            char ack = 0;
            recv_all(sockfd, &ack, 1);
            if (ack != 1) {
                throw std::runtime_error("Failed to add vector");
            }
        } catch (...) {
            disconnect();
            throw;
        }
        disconnect();
    }

    std::vector<int> query(int query_id, const std::vector<T> &query_vector, Payload<T> &payload, int k = 10)
    {
        if (!query_endpoints.empty()) {
            static thread_local std::mt19937 gen(std::random_device{}());
            std::uniform_int_distribution<size_t> dist(0, query_endpoints.size() - 1);
            const QueryEndpoint &ep = query_endpoints[dist(gen)];
            host_ip = ep.ip;
            host_port = ep.port;
        }

        if (!connected && !connect()) {
            throw std::runtime_error("Failed to connect to cluster");
        }

        try {
            Buffer buf;
            buf.write_val((int)MSG_QUERY);
            buf.write_val(query_id);
            buf.write_val(k);
            buf.write_vec(query_vector);
            serialize_payload(buf, payload);

            uint64_t msg_size = buf.data.size();
            send_all(sockfd, &msg_size, sizeof(uint64_t));
            send_all(sockfd, buf.data.data(), msg_size);

            uint64_t resp_size = 0;
            recv_all(sockfd, &resp_size, sizeof(uint64_t));

            std::vector<char> resp_data(resp_size);
            recv_all(sockfd, resp_data.data(), resp_size);

            Buffer resp_buf;
            resp_buf.data = std::move(resp_data);

            int result_cnt = resp_buf.read_val<int>();
            std::vector<int> results;
            for (int i = 0; i < result_cnt; i++) {
                resp_buf.read_val<T>(); // dist
                results.push_back(resp_buf.read_val<int>());
            }
            deserialize_payload(resp_buf, payload);

            disconnect();
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
            send_all(sockfd, &msg_size, sizeof(uint64_t));
            send_all(sockfd, buf.data.data(), msg_size);

            uint64_t resp_size = 0;
            recv_all(sockfd, &resp_size, sizeof(uint64_t));

            std::vector<char> resp_data(resp_size);
            recv_all(sockfd, resp_data.data(), resp_size);

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
            send_all(sockfd, &msg_size, sizeof(uint64_t));
            send_all(sockfd, buf.data.data(), msg_size);

            // Receive ACK
            char ack = 0;
            recv_all(sockfd, &ack, 1);
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
            send_all(sockfd, &msg_size, sizeof(uint64_t));
            send_all(sockfd, buf.data.data(), msg_size);

            // Receive ACK
            char ack = 0;
            recv_all(sockfd, &ack, 1);
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
            send_all(sockfd, &msg_size, sizeof(uint64_t));
            send_all(sockfd, buf.data.data(), msg_size);

            // Receive ACK
            char ack = 0;
            recv_all(sockfd, &ack, 1);
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
            send_all(sockfd, &msg_size, sizeof(uint64_t));
            send_all(sockfd, buf.data.data(), msg_size);

            char ack = 0;
            recv_all(sockfd, &ack, 1);
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
            send_all(sockfd, &msg_size, sizeof(uint64_t));
            send_all(sockfd, buf.data.data(), msg_size);

            // Receive ACK
            char ack = 0;
            recv_all(sockfd, &ack, 1);
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

