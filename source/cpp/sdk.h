#pragma once

#include "cluster.h"

// SDK Client: TCP client for querying cluster via network
template <typename T>
class ClusterSDKClient
{
private:
    std::string host_ip;
    int host_port;
    int sockfd;
    bool connected;
    int timeout_sec;

public:
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

    void partition(const std::string &mode, int redundancy = 1, const std::unordered_map<int, std::vector<int>> &layer_config = {}, const std::string &layer_partition_mode = "round_robin")
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

