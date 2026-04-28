import os
import time
import argparse
import sys
import json
import signal

sys.setrecursionlimit(5000)

# Set unbuffered output to avoid BrokenPipeError
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
sys.stderr.reconfigure(line_buffering=True) if hasattr(sys.stderr, 'reconfigure') else None

project_root = os.path.abspath(os.path.dirname(__file__))
build_dir = os.path.join(project_root, 'build')

# 从 config.json 读取部署参数（与 main.py 共用同一文件）
_CONFIG_PATH = os.path.join(project_root, "config.json")
with open(_CONFIG_PATH, "r") as _f:
    _CLUSTER_CONFIG = json.load(_f)
# IP -> port 映射（任一参与过集群的节点都必须在 config.json 的 nodes 列表里）
_IP_TO_PORT = {nd["ip"]: int(nd["port"]) for nd in _CLUSTER_CONFIG["work_nodes"]}

if os.path.isdir(build_dir):
    os.environ['PYTHONPATH'] = f"{build_dir}:{os.environ.get('PYTHONPATH','')}"
    if build_dir not in sys.path:
        sys.path.insert(0, build_dir)

_DTYPE_CLASS_MAP = {
    "float":  ("PayloadFloat",  "HNSWGraphFloat",  "ClusterFloat"),
    "double": ("PayloadDouble", "HNSWGraphDouble", "ClusterDouble"),
    "int32":  ("PayloadInt32",  "HNSWGraphInt32",  "ClusterInt32"),
    "uint32": ("PayloadUInt32", "HNSWGraphUInt32", "ClusterUInt32"),
    # uint8 存储 + int32 距离：复用 PayloadInt32
    "uint8":  ("PayloadInt32",  "HNSWGraphUInt8",  "ClusterUInt8"),
}

# 提前解析 --dtype 参数
_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument("--dtype", type=str, default="float")
_pre_args, _ = _pre_parser.parse_known_args()

try:
    import vector_retrieve as _vr
    _payload_cls, _hnsw_cls, _cluster_cls = _DTYPE_CLASS_MAP[_pre_args.dtype]
    Payload = getattr(_vr, _payload_cls)
    HNSWGraph = getattr(_vr, _hnsw_cls)
    Cluster = getattr(_vr, _cluster_cls)
except (ImportError, KeyError) as e:
    print(f"Error: Could not import vector_retrieve module (dtype={_pre_args.dtype}): {e}")
    sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed HNSW Node Start")

    parser.add_argument("--node_id", type=int, required=True, help="ID of the current node")
    parser.add_argument("--node_ips", type=str, required=True, help="Comma separated list of IPs")
    parser.add_argument("--dtype", type=str, default="float", choices=list(_DTYPE_CLASS_MAP.keys()),
                        help="Vector element type")
    
    # HNSW params (needed for initialization)
    parser.add_argument("--m", type=int, default=8)
    parser.add_argument("--ef_construction", type=int, default=16)
    parser.add_argument("--ef_search", type=int, default=8)
    parser.add_argument("--limited_level", type=int, default=5)

    parser.add_argument("--distributed", default=False, action='store_true', help="Enable distributed mode")
    parser.add_argument("--shard", default=False, action='store_true', help="Enable shard mode (independent graphs per node)")
    parser.add_argument("--global-pull", dest="is_global_pull", default=False, action='store_true', help="Enable global-pull mode (full graph skeleton per node, remote vector fetch on demand)")
    # 通信模式和 RDMA 参数
    parser.add_argument("--comm_mode", type=str, default="tcp", choices=["tcp", "rdma"], help="Communication mode: tcp or rdma")
    parser.add_argument("--ib_dev", type=str, default="", help="RDMA IB device name (auto-detect if empty)")
    parser.add_argument("--ib_port", type=int, default=1, help="RDMA IB port number")
    parser.add_argument("--gid_index", type=int, default=-1, help="RDMA GID index (-1 for auto-detect)")
    parser.add_argument("--max_track_num", type=int, default=10, help="Sliding window size for backtrack detection (0 disables)")
    parser.add_argument("--num_threads", type=int, default=32, help="Message thread pool size (should match query threads)")


    args = parser.parse_args()

    node_id = args.node_id
    node_ips = args.node_ips.split(',')
    node_num = len(node_ips)
    
    # Safe print function to handle BrokenPipeError
    def safe_print(*args, **kwargs):
        try:
            print(*args, **kwargs)
            sys.stdout.flush()
        except BrokenPipeError:
            # Redirect stdout to devnull to suppress further errors
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, sys.stdout.fileno())
            sys.exit(1)
    
    safe_print(f"[Node {node_id}] Starting worker node on {node_ips[node_id]}...")

    # Initialize dummy HNSW (will be replaced/populated by graph partition)
    # Dim is hardcoded to 128 for sift/siftsmall, but ideally should be passed or generic.
    # However, the C++ side deserialize will handle the actual data.
    hnsw = HNSWGraph(dim=128, m=args.m, ef_construction=args.ef_construction, 
                     ef_search=args.ef_search, limited_level=args.limited_level)

    cluster = Cluster(hnsw, node_num, comm_mode=args.comm_mode, ib_dev=args.ib_dev, ib_port=args.ib_port, gid_index=args.gid_index, max_track_num=args.max_track_num)
    cluster.is_distributed = args.distributed
    cluster.is_shard = args.shard
    cluster.is_global_pull = args.is_global_pull

    # Configure all nodes
    # 关键逻辑：每个机器上的 Cluster 对象都必须包含所有节点的信息
    # 但只有当前机器对应的 node_id 被标记为本地节点 (is_remote=False)
    # 其他节点都被标记为远程节点 (is_remote=True)，用于跨节点通信
    for i, ip in enumerate(node_ips):
        is_remote = (i != node_id)
        if ip not in _IP_TO_PORT:
            raise ValueError(f"[Node {node_id}] IP {ip} 未在 config.json 的 nodes 中登记")
        port = _IP_TO_PORT[ip]
        cluster.set_node_info(i, ip, port, is_remote)
        print(f"[Node {node_id}] Configured Node {i}: IP={ip}, Port={port}, Remote={is_remote}")

    print(f"[Node {node_id}] Starting server...")
    cluster.start_node_server(node_id, msg_pool_size=args.num_threads)
    
    print(f"[Node {node_id}] Server running. Waiting for tasks...")
    # 使用 Python 循环保持主线程存活，这样可以响应 Ctrl+C 信号
    # 并且即使 C++ 线程启动失败（如端口占用），脚本也不会立即退出，便于查看日志
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print(f"\n[Node {node_id}] Received interrupt signal, shutting down...")
    finally:
        print(f"[Node {node_id}] Cleaning up resources...")
        # Python 退出时会自动调用 C++ 对象的析构函数
        # 析构函数会设置 stop_server=true, 关闭 sockfd, 并 join server_thread
        del cluster
        del hnsw
        print(f"[Node {node_id}] Shutdown complete.")
