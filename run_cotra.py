import os
import sys
import time
import argparse
import subprocess
import socket
import json

project_root = os.path.abspath(os.path.dirname(__file__))
cotra_dir = os.path.join(project_root, "CoTra")
cotra_build_dir = os.path.join(cotra_dir, "build")
cotra_scripts_dir = os.path.join(cotra_dir, "scripts")


# ──────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────

def get_local_ip():
    """获取本机 IP（取第一个非 127.x 地址）"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def write_ip_conf(node_ips, memcache_port, conf_path):
    """
    生成 CoTra IP 配置文件。
    格式：
        <leader_ip_or_hostname>
        <memcached_port>
        <ip>=<machine_id>
        ...
    """
    leader_ip = node_ips[0]
    lines = [leader_ip, str(memcache_port)]
    for idx, ip in enumerate(node_ips):
        lines.append(f"{ip}={idx}")
    with open(conf_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"IP 配置文件已写入: {conf_path}")


def restart_memcache(leader_ip, memcache_port, local_ip):
    """在 Leader 节点上重启 memcached 并初始化计数器"""
    if local_ip != leader_ip:
        return
    print(f"[Leader] 重启 memcached @ {leader_ip}:{memcache_port} ...")
    # 杀掉旧进程
    subprocess.run(
        "if [ -f /tmp/memcached.pid ]; then kill $(cat /tmp/memcached.pid) 2>/dev/null; rm -f /tmp/memcached.pid; fi",
        shell=True, check=False
    )
    time.sleep(0.5)
    # 启动 memcached
    subprocess.run(
        f"memcached -u root -l {leader_ip} -p {memcache_port} -c 10000 -d -P /tmp/memcached.pid",
        shell=True, check=True
    )
    time.sleep(1)
    # 初始化 serverNum / clientNum
    subprocess.run(
        f'echo -e "set serverNum 0 0 1\\r\\n0\\r\\nquit\\r" | nc {leader_ip} {memcache_port}',
        shell=True, check=False
    )
    subprocess.run(
        f'echo -e "set clientNum 0 0 1\\r\\n0\\r\\nquit\\r" | nc {leader_ip} {memcache_port}',
        shell=True, check=False
    )
    print("[Leader] memcached 初始化完成")


# ──────────────────────────────────────────────
# 核心 Runner
# ──────────────────────────────────────────────

class CoTraRunner:
    def __init__(self, args):
        self.args = args

        # 节点配置
        self.user_name = r"CONNECT\xchen740"
        self.default_ips = "192.168.200.204,192.168.200.205,192.168.200.206,192.168.200.207,192.168.200.20"
        self.node_ips = args.node_ips.split(",") if args.node_ips else self.default_ips.split(",")
        self.node_num = len(self.node_ips)
        self.host_ip = args.host_ip if args.host_ip else get_local_ip()
        self.leader_ip = self.node_ips[0]
        self.machine_id = self._detect_machine_id()

        # memcached
        self.memcache_port = args.memcache_port

        # 路径
        self.conf_path = os.path.join(cotra_scripts_dir, "ip_auto.conf")
        self.index_save_dir = args.index_save_dir
        self.index_prefix = os.path.join(self.index_save_dir, "merged_index")

        # 数据集参数（与 main.py 对齐）
        self.data_name = args.data_name
        self.data_path = args.data_path
        self.query_path = args.query_path
        self.groundtruth_path = args.groundtruth_path
        self.data_type = args.data_type
        self.dist_fn = args.dist_fn
        self.million = args.million
        self.k = args.k
        self.m = args.m
        self.ef_construction = args.ef_construction
        self.ef_search = args.ef_search
        self.limited_level = args.limited_level
        self.partition_mode = args.partition_mode
        self.distributed = args.distributed
        self.reduce = args.reduce
        self.layer = args.layer
        self.redundancy = args.redundancy

        # CoTra 特有索引参数
        self.graph_type = args.graph_type
        self.graph_path = args.graph_path
        self.cluster_path = args.cluster_path
        self.R = args.R
        self.L = args.L
        self.B = args.B
        self.M = args.M
        self.num_construction_threads = args.num_construction_threads
        self.num_query_threads = args.num_query_threads
        self.num_cpu = args.num_cpu
        self.save = args.save

    def _detect_machine_id(self):
        """根据本机 IP 自动检测 machine_id"""
        for idx, ip in enumerate(self.node_ips):
            if ip == self.host_ip:
                return idx
        print(f"[警告] 本机 IP {self.host_ip} 不在节点列表中，默认 machine_id=0")
        return 0

    def setup(self):
        """生成 IP 配置文件，并在 Leader 上重启 memcached"""
        os.makedirs(self.index_save_dir, exist_ok=True)
        write_ip_conf(self.node_ips, self.memcache_port, self.conf_path)
        restart_memcache(self.leader_ip, self.memcache_port, self.host_ip)

    def _base_cmd(self, binary):
        """返回可执行文件的绝对路径"""
        path = os.path.join(cotra_build_dir, "tests", binary)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"找不到可执行文件: {path}\n请先在 CoTra/build/ 目录下执行 cmake .. && make -j"
            )
        return path

    def build_index(self):
        """调用 scala_index 构建分布式索引"""
        print(f"\n{'='*50}")
        print(f"[machine_id={self.machine_id}] 开始构建索引 ...")
        print(f"{'='*50}")

        cmd = [
            self._base_cmd("scala_index"),
            "--config_file", self.conf_path,
            "--graph_type", self.graph_type,
            "--data_type", self.data_type,
            "--dist_fn", self.dist_fn,
            "--data_path", self.data_path,
            "--index_path_prefix", self.index_prefix,
            "-R", str(self.R),
            "-L", str(self.L),
            "-B", str(self.B),
            "-M", str(self.M),
            "-T", str(self.num_construction_threads),
            "-s", str(self.million),
            "-t", str(self.num_construction_threads),
        ]

        # scalagraph_v3 专用标志
        if self.graph_type == "scalagraph_v3":
            cmd += ["--scala_v3", "--scalagraph_v2"]

        print("CMD:", " ".join(cmd))
        start = time.time()
        ret = subprocess.run(cmd, cwd=cotra_build_dir)
        elapsed = time.time() - start

        if ret.returncode != 0:
            print(f"[错误] scala_index 退出码: {ret.returncode}")
        else:
            print(f"[完成] 索引构建耗时: {elapsed:.2f}s")

    def run_anns(self):
        """调用 scala_anns 执行分布式近似最近邻搜索"""
        print(f"\n{'='*50}")
        print(f"[machine_id={self.machine_id}] 开始执行 ANNS 搜索 ...")
        print(f"{'='*50}")

        app_type = "scala_v3" if self.graph_type == "scalagraph_v3" else "b2"

        cmd = [
            self._base_cmd("scala_anns"),
            "--config_file", self.conf_path,
            "--app_type", app_type,
            "--graph_type", self.graph_type,
            "--data_type", self.data_type,
            "--dist_fn", self.dist_fn,
            "--data_path", self.data_path,
            "--query_path", self.query_path,
            "--gt_path", self.groundtruth_path,
            "--index_path_prefix", self.index_prefix,
            "-R", str(self.R),
            "-L", str(self.L),
            "-B", str(self.B),
            "-M", str(self.M),
            "-T", str(self.num_construction_threads),
            "-s", str(self.million),
            "-t", str(self.num_construction_threads),
        ]

        if self.graph_type == "scalagraph_v3":
            cmd += ["--scala_v3"]

        print("CMD:", " ".join(cmd))
        start = time.time()
        ret = subprocess.run(cmd, cwd=cotra_build_dir)
        elapsed = time.time() - start

        if ret.returncode != 0:
            print(f"[错误] scala_anns 退出码: {ret.returncode}")
        else:
            print(f"[完成] ANNS 搜索耗时: {elapsed:.2f}s")

    def save_results(self):
        """保存运行参数到 JSON"""
        if not self.save:
            return
        results_dir = os.path.join(project_root, "results", "cotra")
        os.makedirs(results_dir, exist_ok=True)

        stats = {
            "data_name": self.data_name,
            "data_path": self.data_path,
            "query_path": self.query_path,
            "groundtruth_path": self.groundtruth_path,
            "data_type": self.data_type,
            "dist_fn": self.dist_fn,
            "million": self.million,
            "k": self.k,
            "m": self.m,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
            "limited_level": self.limited_level,
            "graph_type": self.graph_type,
            "R": self.R,
            "L": self.L,
            "B": self.B,
            "M": self.M,
            "node_num": self.node_num,
            "node_ips": self.node_ips,
            "machine_id": self.machine_id,
            "num_construction_threads": self.num_construction_threads,
            "num_query_threads": self.num_query_threads,
            "num_cpu": self.num_cpu,
            "partition_mode": self.partition_mode,
            "redundancy": self.redundancy,
            "distributed": self.distributed,
            "reduce": self.reduce,
            "layer": self.layer,
            "index_save_dir": self.index_save_dir,
        }

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        results_path = os.path.join(
            results_dir,
            f"cotra-{self.data_name}-{self.graph_type}-R{self.R}-L{self.L}"
            f"-node{self.node_num}-{timestamp}.json"
        )

        with open(results_path, "w") as f:
            json.dump(stats, f, indent=4)
        print(f"\n{'='*50}")
        print(f"Results: {os.path.abspath(results_path)}")
        print(f"{'='*50}\n")


# ──────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CoTra 分布式向量检索 Python 入口")

    # 节点配置
    parser.add_argument("--node_ips", type=str,
                        default="192.168.200.204,192.168.200.205,192.168.200.206,192.168.200.207,192.168.200.20",
                        help="所有节点 IP，逗号分隔，第一个为 Leader（machine_id=0）")
    parser.add_argument("--host_ip", type=str, default="",
                        help="本机 IP（留空自动检测）")
    parser.add_argument("--memcache_port", type=int, default=2378, help="memcached 端口")

    # 运行模式
    parser.add_argument("--mode", type=str, default="index",
                        choices=["index", "anns", "all"],
                        help="运行模式: index=仅构建索引, anns=仅搜索, all=先构建再搜索")

    # 数据集参数（与 main.py 对齐）
    parser.add_argument("--data_name", type=str, default="sift", help="Name of the dataset")
    parser.add_argument("--data_path", type=str, required=True, help="Path to base vectors file (.fbin/.u8bin)")
    parser.add_argument("--query_path", type=str, default="", help="Path to query vectors file (required for anns)")
    parser.add_argument("--groundtruth_path", type=str, default="", help="Path to groundtruth file (required for anns)")
    parser.add_argument("--graph_path", type=str, default=None, help="Path to save/load the graph")
    parser.add_argument("--cluster_path", type=str, default=None, help="Path to save/load the cluster")
    parser.add_argument("--k", type=int, default=10, help="Number of nearest neighbors to retrieve")
    parser.add_argument("--m", type=int, default=16, help="Max number of connections per element")
    parser.add_argument("--ef_construction", type=int, default=200, help="Size of the dynamic candidate list for construction")
    parser.add_argument("--ef_search", type=int, default=16, help="Size of the dynamic candidate list for search")
    parser.add_argument("--limited_level", type=int, default=5, help="Level limit for HNSW")
    parser.add_argument("--node_num", type=int, default=5, help="Number of cluster nodes")
    parser.add_argument("--partition_mode", type=str, default="k-means", help="Data partition mode across nodes")
    parser.add_argument("--layer", action='store_true', default=False, help="Enable layer-based partitioning")
    parser.add_argument("--redundancy", type=int, default=2, help="Redundancy level for data replication across nodes")
    parser.add_argument("--distributed", default=False, action='store_true', help="Enable distributed mode")
    parser.add_argument("--reduce", default=False, action='store_true', help="Enable reduce mode (independent graphs per node)")
    parser.add_argument("--data_type", type=str, default="float", choices=["float", "uint8", "int8"], help="向量数据类型")
    parser.add_argument("--dist_fn", type=str, default="l2", choices=["l2", "mips"], help="距离函数")
    parser.add_argument("--million", type=int, default=10, help="数据集大小（百万）")

    # CoTra 特有索引参数
    parser.add_argument("--graph_type", type=str, default="scalagraph_v3",
                        choices=["scalagraph_v3", "shared_nothing", "Kshared_nothing", "vamana"],
                        help="图索引类型")
    parser.add_argument("--index_save_dir", type=str, default="./index_output", help="索引保存目录")
    parser.add_argument("-R", type=int, default=48, help="图最大度数")
    parser.add_argument("-L", type=int, default=500, help="构建/搜索队列大小（ef）")
    parser.add_argument("-B", type=float, default=100, help="内存预算（GB）")
    parser.add_argument("-M", type=float, default=120, help="内存上限（GB）")
    parser.add_argument("--num_construction_threads", type=int, default=32,
                        help="Number of threads for parallel graph construction")
    parser.add_argument("--num_query_threads", type=int, default=32,
                        help="Number of threads for querying")
    parser.add_argument("--num_cpu", type=int, default=8,
                        help="Limit CPU cores per node process (0=unlimited)")
    parser.add_argument("--save", action='store_true', default=False,
                        help="Save the built graph to disk after construction")

    args = parser.parse_args()

    runner = CoTraRunner(args)
    runner.setup()

    if args.mode in ("index", "all"):
        runner.build_index()

    if args.mode in ("anns", "all"):
        if not args.query_path or not args.groundtruth_path:
            parser.error("--mode anns/all 时必须提供 --query_path 和 --groundtruth_path")
        runner.run_anns()

    runner.save_results()
