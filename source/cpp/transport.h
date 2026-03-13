
#pragma once

#include <string>
#include <memory>
#include <stdexcept>

// 前向声明 Buffer 类，避免与 cluster.h 产生循环引用依赖
class Buffer;

// ==================== 通信配置结构体 ====================
struct TransportConfig {
    std::string comm_mode = "tcp";   // 通信模式："tcp" 或 "rdma"，决定底层传输实现
    std::string ib_dev = "";         // RDMA 模式：InfiniBand 设备名，空字符串表示自动检测第一个可用设备
    int ib_port = 1;                 // RDMA 模式：IB 端口编号，默认使用端口 1
    int gid_index = -1;              // RDMA 模式：GID 索引（-1 表示自动检测，RoCE 默认使用 0）
};

// ==================== 通信中间层抽象基类 ====================
// 提供统一的网络通信接口，屏蔽 TCP 和 RDMA 之间的底层差异。
// 上层代码（如 Cluster、SDK）通过此接口进行数据传输，无需关心具体传输协议。
class ITransport {
public:
    virtual ~ITransport() = default;

    // --- 服务端生命周期管理 ---

    // 启动服务端监听，绑定指定端口。返回监听套接字描述符（失败返回 -1）。
    virtual int start_server(int port) = 0;

    // 停止服务端监听，关闭监听套接字并释放相关资源。
    virtual void stop_server() = 0;

    // 接受一个客户端连接，返回连接标识符（fd 或 conn_id）。
    // 该方法为阻塞调用，会一直等待直到新连接到达或服务端被停止。失败返回 -1。
    virtual int accept_client() = 0;

    // --- 节点注册与连接池管理 ---

    // 注册远程节点信息（IP、端口），用于连接池初始化和地址解析。
    virtual void register_node(int node_id, const std::string& ip, int port) = 0;

    // 从连接池中获取一个到目标节点的可用连接，返回连接标识符。
    // 优先复用空闲连接，无空闲时创建新连接。失败返回 -1。
    virtual int acquire_connection(int node_id) = 0;

    // 将连接归还连接池。valid=false 表示连接已不可用，应直接关闭而非回收复用。
    virtual void release_connection(int node_id, int conn_id, bool valid = true) = 0;

    // 强制关闭指定连接（硬关闭）。TCP 模式下设置 SO_LINGER=0 发送 RST。
    virtual void close_connection(int conn_id) = 0;

    // --- 数据传输接口 ---

    // 在指定连接上发送带长度头的消息。先发送 uint64_t 的消息长度，再发送消息体。
    virtual void send_message(int conn_id, const void* data, size_t size) = 0;

    // 在指定连接上接收带长度头的消息。先接收 uint64_t 的长度头，再接收相应大小的消息体。
    // 返回实际接收的消息大小，失败则抛出异常。
    virtual size_t recv_message(int conn_id, std::vector<char>& buffer) = 0;

    // 在指定连接上发送原始字节流（不带长度头），由调用方自行管理消息边界。
    virtual void send_raw(int conn_id, const void* data, size_t size) = 0;

    // 在指定连接上接收指定大小的原始字节流（不带长度头）。
    virtual void recv_raw(int conn_id, void* data, size_t size) = 0;

    // --- 资源管理 ---

    // 关闭指定节点的所有连接（包括空闲和活跃连接）。
    virtual void close_node(int node_id) = 0;

    // 清空所有节点的所有连接，释放全部网络资源。
    virtual void clear() = 0;

    // 获取连接池的统计信息字符串（供调试和监控使用）。
    virtual std::string get_stats() = 0;
};

// ==================== 传输层工厂类 ====================
// 根据 TransportConfig 配置创建对应的传输层实例（TCP 或 RDMA）。
class TcpTransport;

class TransportFactory {
public:
    // 根据配置中的 comm_mode 创建对应的 ITransport 实现实例。
    // comm_mode="tcp" 返回 TcpTransport，comm_mode="rdma" 返回 RdmaTransport。
    static std::unique_ptr<ITransport> create(const TransportConfig& config);
};

