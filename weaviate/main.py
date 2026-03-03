import argparse
import numpy as np
import time
import weaviate
import subprocess
from weaviate.classes.config import Configure, Property, DataType, VectorDistances, Reconfigure
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing  # [新增] 引入 multiprocessing
from typing import List, Dict
from monitor import WeaviateMonitor
import yaml
from tqdm import tqdm

# --- 配置常量 ---
ALL_NODES = [
    {"ip": "192.168.200.204", "name": "weaviate_node0"},
    {"ip": "192.168.200.205", "name": "weaviate_node1"},
    {"ip": "192.168.200.206", "name": "weaviate_node2"},
    {"ip": "192.168.200.207", "name": "weaviate_node3"},
    {"ip": "192.168.200.208", "name": "weaviate_node4"},
    # {"ip": "192.168.200.209", "name": "weaviate_node5"},
]

RAFT_PORT = 19600
GOSSIP_PORT = 19601
DATA_PORT = 19602
HTTP_PORT = 19603
GRPC_PORT = 19604

# --- 独立的 Worker 函数 (必须定义在顶层以便 pickle) ---

def query_worker(node_ips, data_name, queries, global_indices, k, ground_truth):
    """
    单个进程的查询任务：建立到所有节点的连接，每次查询随机选择节点
    """
    latencies = []
    recalls = []  # 每个查询的召回率

    # 建立到所有节点的连接池
    clients = {}
    collections = {}
    try:
        for ip in node_ips:
            client = weaviate.connect_to_custom(
                http_host=ip, http_port=HTTP_PORT, http_secure=False,
                grpc_host=ip, grpc_port=GRPC_PORT, grpc_secure=False,
            )
            clients[ip] = client
            collections[ip] = client.collections.get(data_name) 
    except Exception as e:
        print(f"  [Fatal] Failed to connect to nodes: {e}")
        return [], []

    try:
        for i, vec in enumerate(queries):
            # 每次查询随机选择一个节点
            target_ip = np.random.choice(node_ips)
            collection = collections[target_ip]

            start_time = time.time()
            try:
                response = collection.query.near_vector(
                    near_vector=vec.tolist(), limit=k, return_properties=["original_index"]
                )
                ids = [obj.properties['original_index'] for obj in response.objects]
                latencies.append(time.time() - start_time)
                
                # 计算该查询的召回率
                gt_ids = set(ground_truth[global_indices[i]][:k])
                recall = len(gt_ids.intersection(set(ids))) / k
                recalls.append(recall)
            except Exception as e:
                latencies.append(0.0)
                recalls.append(0.0)
    finally:
        for client in clients.values():
            try:
                client.close()
            except:
                pass

    return latencies, recalls

class ClusterController:
    """负责集群节点的部署、资源限制和信息管理"""
    def __init__(self, all_nodes: List[Dict], user: str = "CONNECT\\xchen740"):
        self.all_nodes = all_nodes
        self.user = user
        self.weaviate_dir = "/home/CONNECT/xchen740/workspace/vector-retrieve/weaviate"
        self.weaviate_data_dir = f"/local-ssd/{self.user}/weaviate"
        self.ssh_processes = []  # 保持 SSH 连接的进程列表
        self.ssh_control_paths = {}  # 每个节点的 SSH control socket 路径

    def _run_ssh_cmd(self, ip: str, cmd: str, check: bool = True, capture_output: bool = True) -> subprocess.CompletedProcess:
        """通过持久 SSH 连接执行命令"""
        control_path = self.ssh_control_paths.get(ip)
        ssh_opts = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
        if control_path:
            ssh_opts.extend(["-o", f"ControlPath={control_path}"])
        
        full_cmd = ["ssh"] + ssh_opts + [f"{self.user}@{ip}", cmd]
        
        if capture_output:
            return subprocess.run(full_cmd, capture_output=True, check=check)
        else:
            return subprocess.run(full_cmd, check=check)

    def _run_ssh_cmd_output(self, ip: str, cmd: str) -> str:
        """通过持久 SSH 连接执行命令并返回输出"""
        control_path = self.ssh_control_paths.get(ip)
        ssh_opts = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
        if control_path:
            ssh_opts.extend(["-o", f"ControlPath={control_path}"])
        
        full_cmd = ["ssh"] + ssh_opts + [f"{self.user}@{ip}", cmd]
        return subprocess.check_output(full_cmd).decode().strip()

    def setup_ssh_connections(self, nodes: List[Dict]):
        """为每个节点创建持久的 SSH master 连接"""
        import tempfile
        import os
        
        print(f"\n=== 建立 SSH 持久连接 ({len(nodes)} 个节点) ===")
        
        # 创建临时目录存放 control sockets
        temp_dir = tempfile.mkdtemp(prefix="ssh_ctrl_")
        
        for node in nodes:
            ip = node["ip"]
            name = node["name"]
            control_path = os.path.join(temp_dir, f"{name}.sock")
            self.ssh_control_paths[ip] = control_path
            
            # 启动 SSH master 连接 (后台运行，保持连接)
            # -M: master mode, -S: control socket, -f: background
            master_cmd = [
                "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                "-o", "ControlMaster=auto", "-o", f"ControlPath={control_path}",
                "-o", "ControlPersist=600",  # 连接保持 10 分钟
                "-n",  # 不读取 stdin
                f"{self.user}@{ip}",
                "echo 'SSH connection established' && sleep infinity"
            ]
            
            print(f"  正在连接 {name} ({ip})...")
            proc = subprocess.Popen(
                master_cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            self.ssh_processes.append(proc)
            
            # 等待连接建立
            time.sleep(0.5)
            
            # 验证连接是否成功
            check_cmd = [
                "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
                "-o", f"ControlPath={control_path}",
                f"{self.user}@{ip}", "echo ok"
            ]
            try:
                result = subprocess.run(check_cmd, capture_output=True, timeout=5)
                if result.returncode == 0:
                    print(f"  [OK] {name} SSH 连接已建立")
                else:
                    print(f"  [WARN] {name} SSH 连接验证失败")
            except subprocess.TimeoutExpired:
                print(f"  [WARN] {name} SSH 连接超时")
        
        print(f"=== SSH 连接建立完成 ===\n")

    def cleanup_ssh_connections(self):
        """关闭所有 SSH 持久连接"""
        print("\n=== 清理 SSH 连接 ===")
        
        # 通过 control socket 关闭连接
        for ip, control_path in self.ssh_control_paths.items():
            exit_cmd = ["ssh", "-o", f"ControlPath={control_path}", "-O", "exit", f"{self.user}@{ip}"]
            try:
                subprocess.run(exit_cmd, capture_output=True, timeout=5)
                print(f"  [OK] 已关闭 {ip} 的 SSH 连接")
            except Exception as e:
                print(f"  [WARN] 关闭 {ip} SSH 连接失败: {e}")
        
        # 终止所有 master 进程
        for proc in self.ssh_processes:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
        
        self.ssh_processes = []
        self.ssh_control_paths = {}
        print("=== SSH 连接清理完成 ===\n")

    def _generate_compose_file(self, active_nodes: List[Dict]):
        """
        为每个节点生成独立的 docker-compose-{i}.yml
        关键点：动态计算 RAFT_BOOTSTRAP_EXPECT 和 CLUSTER_JOIN
        """
        print(f"正在生成 {len(active_nodes)} 个节点的 docker-compose 配置...")
        
        # 1. 准备基础配置
        cluster_join_list = [f"{n['ip']}:{GOSSIP_PORT}" for n in active_nodes]
        cluster_join_str = ",".join(cluster_join_list)
        
        raft_join_list = [n['name'].replace('weaviate_', '') for n in active_nodes] # e.g., node0, node1
        raft_join_str = ",".join(raft_join_list)
        
        bootstrap_expect = len(active_nodes)

        # 2. 为每个节点生成独立的配置文件
        for i, node in enumerate(active_nodes):
            name = node["name"] # weaviate_node0
            short_name = name.replace("weaviate_", "") # node0
            
            services = {
                name: {
                    "network_mode": "host",
                    "command": [
                        "--host", "0.0.0.0",
                        "--port", str(HTTP_PORT),
                        "--scheme", "http"
                    ],
                    "image": "cr.weaviate.io/semitechnologies/weaviate:1.34.2",
                    "volumes": [f"{name}_data:/var/lib/weaviate"],
                    "restart": "on-failure",
                    "environment": {
                        "QUERY_DEFAULTS_LIMIT": 25,
                        "AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED": "true",
                        "PROMETHEUS_MONITORING_ENABLED": "true",
                        "PERSISTENCE_DATA_PATH": "/var/lib/weaviate",
                        "GRPC_PORT": str(GRPC_PORT),
                        "ENABLE_MODULES": "",
                        "CLUSTER_HOSTNAME": short_name,
                        "CLUSTER_GOSSIP_BIND_PORT": str(GOSSIP_PORT),
                        "CLUSTER_DATA_BIND_PORT": str(DATA_PORT),
                        "RAFT_PORT": str(RAFT_PORT),
                        # 关键动态参数
                        "CLUSTER_JOIN": cluster_join_str,
                        "RAFT_JOIN": raft_join_str,
                        "RAFT_BOOTSTRAP_EXPECT": bootstrap_expect
                    }
                }
            }
            
            volumes = {
                f"{name}_data": {
                    "name": f"{name}_data",
                    "driver": "local",
                    "driver_opts": {
                        "type": "none",
                        "o": "bind",
                        "device": f"{self.weaviate_data_dir}/{name}_data"
                    }
                }
            }

            compose_config = {
                "services": services,
                "volumes": volumes
            }

            # 3. 写入独立文件
            output_path = f"{self.weaviate_dir}/docker-compose-{i}.yml"
            with open(output_path, 'w') as f:
                yaml.dump(compose_config, f, default_flow_style=False)
            print(f"  [OK] 已生成 docker-compose-{i}.yml (节点: {name})")
        
        print(f"  [OK] 共生成 {len(active_nodes)} 个配置文件 (Expect: {bootstrap_expect})")

    def deploy(self, nodes: List[Dict]):
        print(f"\n=== 开始部署 {len(nodes)} 节点 Weaviate 集群 ===")
        
        # 0. 建立 SSH 持久连接
        self.setup_ssh_connections(nodes)
        
        # 1. 清理旧环境 (停止所有节点 + 删除数据)
        # 关键修改：
        # 1. 使用 podman rm -f 直接删除容器，不依赖 docker-compose.yml (防止因文件更新导致无法停止旧服务)
        # 2. 执行 rm -rf 删除数据卷，防止旧 Raft 状态导致 "leader not found" 错误
        print(">>> [Step 1/2] 正在清理旧环境 (停止所有节点 & 删除数据)...")
        for node in self.all_nodes:
            ip, name = node["ip"], node["name"]
            
            clean_cmd = (
                f"podman ps -a --filter name={name} --format '{{{{.ID}}}}' | xargs -r podman rm -f && "
                f"podman volume rm -f 'weaviate_{name}_data' >/dev/null 2>&1 || true && "
                f"podman volume rm -f '{name}_data' >/dev/null 2>&1 || true && "
                f"rm -rf '{self.weaviate_data_dir}/{name}_data'"
            )
            try:
                self._run_ssh_cmd(ip, clean_cmd, check=False)
                print(f"  [OK] 节点 {name} 已清理 (容器 & 数据)")
            except Exception as e:
                print(f"  [WARN] 清理节点 {name} 失败: {e}")

        # 2. 动态生成配置文件 (此时生成，确保启动时使用的是最新配置)
        self._generate_compose_file(nodes)

        # 3. 启动新容器
        print("\n>>> [Step 2/2] 正在启动新环境...")
        for i, node in enumerate(nodes):
            ip, name = node["ip"], node["name"]
            print(f"正在启动节点 {name} ({ip})...")
            start_cmd = (
                f"mkdir -p '{self.weaviate_data_dir}/{name}_data' && "
                f"cd '{self.weaviate_dir}' && "
                f"systemctl restart --user podman.socket && "
                f"podman compose -f docker-compose-{i}.yml up -d"
            )
            try:
                self._run_ssh_cmd(ip, start_cmd, check=True)
                print(f"  [OK] 节点 {name} 启动完成")
            except subprocess.CalledProcessError as e:
                print(f"  [FAIL] 节点 {name} 启动失败: {e}")
                raise e
        print(f"=== 集群部署完成! ===\n")

    def update_cpu_limits(self, nodes: List[Dict], cpu_limit: float):
        if cpu_limit > 0:
            print(f"\n=== 正在应用 CPU 限制: {cpu_limit} 核/节点 ===")
            core_count = int(max(1, round(cpu_limit)))
            cpu_list = f"0-{core_count - 1}"
        else:
            print(f"\n=== 正在取消 CPU 限制 (恢复使用所有核心) ===")
            cpu_list = None

        for node in nodes:
            ip, name = node["ip"], node["name"]
            try:
                # 1. 获取容器 ID (podman compose 创建的容器名可能是 weaviate-{name}-1)
                find_id_cmd = f"podman ps --format '{{{{.Names}}}} {{{{.ID}}}}' | grep '{name}' | head -n 1 | cut -d' ' -f2"
                container_id = self._run_ssh_cmd_output(ip, find_id_cmd)
                
                if not container_id:
                    status_cmd = f"podman ps -a --format '{{{{.Names}}}} {{{{.Status}}}}' | grep '{name}' | head -n 1"
                    status_line = self._run_ssh_cmd_output(ip, status_cmd)
                    if status_line:
                        print(f"  [WARN] 节点 {name} ({ip}) 容器未运行: {status_line}")
                    else:
                        print(f"  [WARN] 节点 {name} ({ip}) 未找到容器")
                    continue

                # 确定目标 CPU 数
                if cpu_limit == 0:
                    nproc = int(self._run_ssh_cmd_output(ip, "nproc"))
                    target_cpus = nproc
                    cpu_list = f"0-{nproc - 1}"
                else:
                    target_cpus = cpu_limit

                # 2. 尝试 Podman Update (Cgroup)
                try:
                    update_cmd = f"podman update --cpus={target_cpus} {container_id}"
                    self._run_ssh_cmd(ip, update_cmd, check=True)
                    action_str = f"限制 CPU: {target_cpus}" if cpu_limit > 0 else "重置 Cgroup 配额"
                    print(f"  [OK] 节点 {name} ({ip}) 已通过 Cgroup {action_str}")
                except subprocess.CalledProcessError:
                    # 3. 回退 Taskset
                    if cpu_limit > 0:
                        get_pid_cmd = f"podman inspect -f '{{{{.State.Pid}}}}' {container_id}"
                        pid = self._run_ssh_cmd_output(ip, get_pid_cmd)
                        if pid and pid != "0":
                            taskset_cmd = f"taskset -acp {cpu_list} {pid}"
                            self._run_ssh_cmd(ip, taskset_cmd, check=True)
                            print(f"  [OK] 节点 {name} ({ip}) 已通过 taskset -a 绑定核心: {cpu_list} (PID: {pid})")
                        else:
                            print(f"  [FAIL] 无法获取容器 PID")
            except Exception as e:
                print(f"  [WARN] 节点 {name} ({ip}) CPU 限制操作失败: {e}")


class DatasetLoader:
    """负责读取向量数据集"""
    @staticmethod
    def read_fvecs(filename, c_contiguous=True):
        fv = np.fromfile(filename, dtype=np.float32)
        dim = fv.view(np.int32)[0]
        fv = fv.reshape(-1, 1 + dim)
        vectors = fv[:, 1:]
        return np.ascontiguousarray(vectors) if c_contiguous else vectors

    @staticmethod
    def read_ivecs(filename, c_contiguous=True):
        iv = np.fromfile(filename, dtype=np.int32)
        dim = iv[0]
        iv = iv.reshape(-1, 1 + dim)
        vectors = iv[:, 1:]
        return np.ascontiguousarray(vectors) if c_contiguous else vectors


class WeaviateIndexer:
    """负责索引构建和数据加载"""
    def __init__(self, collection_name: str):
        self.collection_name = collection_name

    def build(self, node_ip: str, data_path: str, m: int):
        print(f"\n=== 开始构建索引 (数据加载) ===")
        print(f"目标节点: {node_ip}, 数据: {data_path}")
        
        client = weaviate.connect_to_custom(
            http_host=node_ip, http_port=HTTP_PORT, http_secure=False,
            grpc_host=node_ip, grpc_port=GRPC_PORT, grpc_secure=False,
        )
        
        try:
            vectors = DatasetLoader.read_fvecs(data_path)
            print(f"加载向量数据: {vectors.shape}")
            
            if client.collections.exists(self.collection_name):
                print(f"删除旧集合: {self.collection_name}")
                client.collections.delete(self.collection_name)
                
            print(f"创建新集合: {self.collection_name} (HNSW: m={m}, efConstruction=200, RF=1, Sharding=5)")
            client.collections.create(
                name=self.collection_name,
                vector_config=Configure.Vectors.self_provided(
                    vector_index_config=Configure.VectorIndex.hnsw(
                        distance_metric=VectorDistances.L2_SQUARED,
                        max_connections=m,
                        ef_construction=200,
                    ),
                ),
                properties=[
                    Property(name="original_index", data_type=DataType.INT),
                ],
                replication_config=Configure.replication(
                    factor=1
                ),
                sharding_config=Configure.sharding(
                    desired_count=5
                ),
            )
            
            collection = client.collections.get(self.collection_name)
            print("开始批量上传...")
            start_time = time.time()
            with collection.batch.dynamic() as batch:
                for i, vec in tqdm(enumerate(vectors), total=len(vectors), desc="Building", unit="vec"):
                    batch.add_object(properties={"original_index": i}, vector=vec)
            
            print(f"数据加载完成! 耗时: {time.time() - start_time:.2f}s")
            print("索引构建阶段完成")
            
            # 显示集合配置和分片信息
            print("\n=== 集合配置信息 ===")
            try:
                config = client.collections.get(self.collection_name).config.get()
                rf = config.replication_config.factor if config.replication_config else 'N/A'
                print(f"  复制因子 (RF): {rf}")
                if hasattr(config, 'sharding_config') and config.sharding_config:
                    print(f"  分片数: {config.sharding_config.desired_count}")
                
                total_count = client.collections.get(self.collection_name).aggregate.over_all(total_count=True).total_count
                print(f"  集合总向量数: {total_count}")
                
                # 获取每个分片的信息
                print("\n=== 各分片向量数 ===")
                try:
                    shards_response = collection.shards()
                    if shards_response:
                        for shard in shards_response:
                            print(f"  Shard {shard.name}: {shard.object_count} vectors (state: {shard.state})")
                    else:
                        print("  无法获取分片信息")
                except Exception as shard_e:
                    print(f"  获取分片信息失败: {shard_e}")
            except Exception as e:
                print(f"  获取配置失败: {e}")
        finally:
            client.close()


class QueryBenchmark:
    """核心压测类：管理连接、执行查询、统计结果"""
    def __init__(self, args, active_nodes: List[Dict], query_vectors: np.ndarray, ground_truth: np.ndarray):
        self.args = args
        self.active_nodes = active_nodes
        self.query_vectors = query_vectors
        self.ground_truth = ground_truth
        # 监控器
        self.node_map = {n["name"]: n["ip"] for n in active_nodes}
        self.monitor = WeaviateMonitor(self.node_map)

    def update_config(self):
        """更新集群配置 (只需连接一个节点即可)"""
        if self.args.ef_search > 0:
            print(f"正在更新 HNSW ef 参数为 {self.args.ef_search} ...")
            ip = self.active_nodes[0]["ip"]
            client = None
            try:
                client = weaviate.connect_to_custom(
                    http_host=ip, http_port=HTTP_PORT, http_secure=False,
                    grpc_host=ip, grpc_port=GRPC_PORT, grpc_secure=False,
                )
                client.collections.get(self.args.data_name).config.update(
                    vector_index_config=Reconfigure.VectorIndex.hnsw(ef=self.args.ef_search)
                )
                print(f"  [OK] ef_search 参数已更新")
            except Exception as e:
                print(f"  [ERROR] 更新 ef_search 参数失败: {e}")
            finally:
                if client:
                    client.close()

    def run(self):
        # 1. 配置更新与同步
        self.update_config()
        print("\n[同步等待] 已关闭固定等待，立即启动压测")

        # 2. 初始指标
        print("正在抓取集群初始指标...")
        start_stats = self.monitor.queries_durations_ms_count

        print("\n=== 开始并发查询 ===")
        global_start_time = time.time()

        total_queries = len(self.query_vectors)
        num_threads = self.args.num_query_threads
        node_ips = [n["ip"] for n in self.active_nodes]

        # 打乱查询顺序
        indices = list(range(total_queries))
        np.random.shuffle(indices)
        shuffled_vectors = self.query_vectors[indices]

        # 每个进程都处理所有查询向量
        all_tasks = []
        for t in range(num_threads):
            all_tasks.append((
                node_ips,
                self.args.data_name,
                shuffled_vectors,
                indices,
                self.args.k,
                self.ground_truth,  # 传递 ground_truth
            ))

        print(f"启动 {len(all_tasks)} 个进程，每进程处理 {total_queries} 个查询...")

        # 使用 spawn 上下文
        ctx = multiprocessing.get_context('spawn')
        with ProcessPoolExecutor(max_workers=len(all_tasks), mp_context=ctx) as executor:
            futures = [executor.submit(query_worker, *task) for task in all_tasks]

            all_latencies = []
            all_recalls = []
            total_client_queries = total_queries * num_threads
            query_start_time = time.time()
            
            with tqdm(total=total_client_queries, desc="Querying", unit="qry") as pbar:
                last_count = 0
                for future in as_completed(futures):
                    lats, recs = future.result()
                    all_latencies.extend(lats)
                    all_recalls.extend(recs)
                    # 更新进度
                    current_count = len(all_latencies)
                    pbar.update(current_count - last_count)
                    last_count = current_count
                    # 更新 QPS
                    elapsed = time.time() - query_start_time
                    if elapsed > 0:
                        current_qps = current_count / elapsed
                        pbar.set_postfix(QPS=f"{current_qps:.1f}")

        global_end_time = time.time()
        total_duration = global_end_time - global_start_time

        # 3. 统计结果
        self.report(all_latencies, all_recalls, total_duration, start_stats)

    def report(self, latencies, recalls, duration, start_stats):
        # 召回率直接取平均
        avg_recall = np.mean(recalls) if recalls else 0.0
        avg_latency = np.mean(latencies) * 1000
        p99_latency = np.percentile(latencies, 99) * 1000

        # 抓取结束指标
        print("\n\n正在抓取集群结束指标...")
        end_stats = self.monitor.queries_durations_ms_count
        end_concurrent = self.monitor.concurrent_queries_count
        end_goroutines = self.monitor.concurrent_goroutines
        
        print("\n=== 集群各节点负载统计 ===")
        cluster_processed_count = 0
        for node in self.active_nodes:
            name, ip = node["name"], node["ip"]
            s = start_stats.get(name, -1)
            e = end_stats.get(name, -1)
            c = end_concurrent.get(name, -1)
            g = end_goroutines.get(name, -1)
            
            if s >= 0 and e >= 0:
                diff = int(e - s)
                cluster_processed_count += diff
                node_qps = diff / duration
                print(f"  Node {ip}: {diff} queries (QPS: {node_qps:.2f}, Concurrent: {int(c)}, Goroutines: {int(g)})")
            else:
                print(f"  Node {ip}: [指标获取失败]")
        
        global_qps = cluster_processed_count / duration
        
        # 理论 QPS (客户端发送的查询总数)
        total_client_queries = len(self.query_vectors) * self.args.num_query_threads
        client_qps = total_client_queries / duration

        print(f"--------------------------------")
        print(f"=== 测试结果 ===")
        print(f"总耗时: {duration:.6f}s")
        print(f"Recall@{self.args.k}: {avg_recall:.4f}")
        print(f"平均延迟: {avg_latency:.6f} ms")
        print(f"P99 延迟: {p99_latency:.6f} ms")
        print(f"--------------------------------")
        print(f"客户端发送查询数: {total_client_queries}")
        print(f"服务器处理查询数: {cluster_processed_count}")
        print(f"客户端 QPS: {client_qps:.2f}")
        print(f"服务器 QPS: {global_qps:.2f}")


def main(args):
    active_nodes = ALL_NODES[:args.node_num]
    active_ips = [n["ip"] for n in active_nodes]

    print(f"=== 节点配置 ===")
    print(f"使用节点数: {args.node_num}")
    print(f"deploy={args.deploy}, build={args.build}")  # 调试输出
    for n in active_nodes:
        print(f"  - {n['name']}: {n['ip']}")

    controller = ClusterController(ALL_NODES)

    try:
        # 1. 如果指定了部署标志，先执行部署
        if args.deploy:
            controller.deploy(active_nodes)
        elif args.num_cpu != 0:
            # 如果没有部署但需要设置 CPU 限制，先建立 SSH 连接
            controller.setup_ssh_connections(active_nodes)

        # 1.5 应用或取消 CPU 限制 (只要参数存在就执行，支持 0 取消)
        if args.num_cpu != 0:
            controller.update_cpu_limits(active_nodes, args.num_cpu)

        # 2. 如果指定了建图标志，执行数据加载
        if args.build:
            indexer = WeaviateIndexer(args.data_name)
            indexer.build(active_ips[0], args.data_path, args.m)

        # 3. 加载查询数据
        print(f"正在加载查询向量: {args.query_path}")
        all_queries = DatasetLoader.read_fvecs(args.query_path)
        print(f"正在加载 Ground Truth: {args.groundtruth_path}")
        all_ground_truth = DatasetLoader.read_ivecs(args.groundtruth_path)

        query_vectors = all_queries
        ground_truth = all_ground_truth
        total_queries = len(query_vectors)

        print(f"模式: 节点绑定并发 (多进程模式)")
        print(f"并发结构: {args.node_num} 节点 x {args.num_query_threads} 查询线程/节点")
        print(f"总查询数: {total_queries}")

        # 4. 执行压测
        benchmark = QueryBenchmark(args, active_nodes, query_vectors, ground_truth)
        benchmark.run()
    finally:
        # 清理 SSH 连接
        controller.cleanup_ssh_connections()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Weaviate Concurrent Query Benchmark")

    # 数据集配置
    parser.add_argument("--data_name", type=str, default="sift", help="Collection name")
    parser.add_argument("--data_path", type=str, default="/home/CONNECT/xchen740/workspace/vector-retrieve/dataset/sift/sift_base.fvecs")
    parser.add_argument("--query_path", type=str, default="/home/CONNECT/xchen740/workspace/vector-retrieve/dataset/sift/sift_query.fvecs")
    parser.add_argument("--groundtruth_path", type=str, default="/home/CONNECT/xchen740/workspace/vector-retrieve/dataset/sift/sift_groundtruth.ivecs")

    # 节点与部署配置
    parser.add_argument("--node_num", type=int, default=1, help="Number of nodes to use (and deploy)")
    parser.add_argument("--deploy", action="store_true", default=False, help="Deploy/Restart cluster before running benchmark")
    parser.add_argument("--num_cpu", type=float, default=8, help="Number of CPU cores per node (e.g. 4.0). 0 means no limit.")
    parser.add_argument("--build", action="store_true", default=False, help="Build index (load data) before querying")

    # HNSW 参数 (用于建图)
    parser.add_argument("--m", type=int, default=16, help="HNSW max connections")

    # HNSW 参数 (用于查询)
    parser.add_argument("--ef_search", type=int, default=16, help="HNSW ef_search parameter. -1 to use default.")

    # 查询参数
    parser.add_argument("--k", type=int, default=10, help="Top K")
    parser.add_argument("--num_query_threads", type=int, default=32, help="Total number of query processes (each process executes all queries)")

    args = parser.parse_args()
    main(args)