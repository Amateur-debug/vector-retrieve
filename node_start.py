import os
import time
import argparse
import sys
import signal

sys.setrecursionlimit(5000)

# Set unbuffered output to avoid BrokenPipeError
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
sys.stderr.reconfigure(line_buffering=True) if hasattr(sys.stderr, 'reconfigure') else None

project_root = os.path.abspath(os.path.dirname(__file__))
build_dir = os.path.join(project_root, 'build')

if os.path.isdir(build_dir):
    os.environ['PYTHONPATH'] = f"{build_dir}:{os.environ.get('PYTHONPATH','')}"
    if build_dir not in sys.path:
        sys.path.insert(0, build_dir)

try:
    from vector_retrieve import PayloadFloat as Payload
    from vector_retrieve import HNSWGraphFloat as HNSWGraph
    from vector_retrieve import ClusterFloat as Cluster
except ImportError:
    print("Error: Could not import vector_retrieve module. Please ensure the C++ project is built.")
    sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed HNSW Node Start")
    
    parser.add_argument("--node_id", type=int, required=True, help="ID of the current node")
    parser.add_argument("--node_ips", type=str, required=True, help="Comma separated list of IPs")
    
    # HNSW params (needed for initialization)
    parser.add_argument("--m", type=int, default=8)
    parser.add_argument("--ef_construction", type=int, default=16)
    parser.add_argument("--ef_search", type=int, default=8)
    parser.add_argument("--limited_level", type=int, default=5)

    parser.add_argument("--distributed", default=False, action='store_true', help="Enable distributed mode")
    parser.add_argument("--reduce", default=False, action='store_true', help="Enable reduce mode (independent graphs per node)")


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

    cluster = Cluster(hnsw, node_num)
    cluster.is_distributed = args.distributed
    cluster.is_reduce = args.reduce

    # Configure all nodes
    # 关键逻辑：每个机器上的 Cluster 对象都必须包含所有节点的信息
    # 但只有当前机器对应的 node_id 被标记为本地节点 (is_remote=False)
    # 其他节点都被标记为远程节点 (is_remote=True)，用于跨节点通信
    for i, ip in enumerate(node_ips):
        is_remote = (i != node_id)
        port = 19000 + i
        cluster.set_node_info(i, ip, port, is_remote)
        print(f"[Node {node_id}] Configured Node {i}: IP={ip}, Port={port}, Remote={is_remote}")

    print(f"[Node {node_id}] Starting server...")
    cluster.start_node_server(node_id)
    
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
