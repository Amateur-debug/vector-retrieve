
#pragma once

#include <string>
#include <memory>
#include <stdexcept>

// 前向声明，避免循环依赖
class Buffer;

// ==================== 通信配置 ====================
struct TransportConfig {
    std::string comm_mode = "tcp";   // 通信模式: "tcp" 或 "rdma"
    std::string ib_dev = "";         // RDMA: IB 设备名（空字符串表示自动检测）
    int ib_port = 1;                 // RDMA: IB 端口号
    int gid_index = -1;              // RDMA: GID 索引（-1 表示自动检测）
};

// ==================== 通信中间层抽象基类 ====================
// 统一的通信接口，屏蔽 TCP 和 RDMA 的底层差异
class ITransport {
public:
    virtual ~ITransport() = default;

    // --- 服务端生命周期 ---

    // 启动通信服务端监听，返回监听 socket/fd（失败返回 -1）
    virtual int start_server(int port) = 0;

    // 停止通信服务端
    virtual void stop_server() = 0;

    // 接受一个客户端连接，返回连接标识（fd/conn_id），失败返回 -1
    // 该方法会阻塞直到有新连接到来或服务端被停止
    virtual int accept_client() = 0;

    // --- 节点注册与连接管理 ---

    // 注册远程节点信息（用于连接池管理）
    virtual void register_node(int node_id, const std::string& ip, int port) = 0;

    // 获取到目标节点的连接，返回连接标识（fd/conn_id），失败返回 -1
    virtual int acquire_connection(int node_id) = 0;

    // 释放连接回池（valid=false 表示连接已不可用，应关闭）
    virtual void release_connection(int node_id, int conn_id, bool valid = true) = 0;

    // 关闭指定连接（硬关闭，发送 RST）
    virtual void close_connection(int conn_id) = 0;

    // --- 数据传输 ---

    // 在指定连接上发送消息（先发 uint64_t 长度头，再发数据体）
    virtual void send_message(int conn_id, const void* data, size_t size) = 0;

    // 在指定连接上接收消息到 buffer（先收 uint64_t 长度头，再收数据体）
    // 返回接收到的消息大小，失败抛出异常
    virtual size_t recv_message(int conn_id, std::vector<char>& buffer) = 0;

    // 在指定连接上发送原始字节流（不带长度头）
    virtual void send_raw(int conn_id, const void* data, size_t size) = 0;

    // 在指定连接上接收指定大小的原始字节流（不带长度头）
    virtual void recv_raw(int conn_id, void* data, size_t size) = 0;

    // --- 资源管理 ---

    // 关闭指定节点的所有连接
    virtual void close_node(int node_id) = 0;

    // 清空所有连接
    virtual void clear() = 0;

    // 获取统计信息（调试用）
    virtual std::string get_stats() = 0;
};

// ==================== 工厂类 ====================
// 前向声明具体实现
class TcpTransport;

class TransportFactory {
public:
    // 根据配置创建对应的 Transport 实例
    static std::unique_ptr<ITransport> create(const TransportConfig& config);
};

