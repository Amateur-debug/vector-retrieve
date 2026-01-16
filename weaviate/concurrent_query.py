import argparse
import numpy as np
import time
import math
import weaviate
import subprocess
import weaviate.classes.config as wc
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing  # [新增] 引入 multiprocessing
from typing import List, Tuple, Dict, Optional
from monitor import WeaviateMonitor
import yaml

# --- 配置常量 ---
ALL_NODES = [
    {"ip": "192.168.200.204", "name": "weaviate_node0"},
    {"ip": "192.168.200.205", "name": "weaviate_node1"},
    {"ip": "192.168.200.206", "name": "weaviate_node2"},
    {"ip": "192.168.200.207", "name": "weaviate_node3"},
    {"ip": "192.168.200.208", "name": "weaviate_node4"},
    {"ip": "192.168.200.209", "name": "weaviate_node5"},
]

# --- 独立的 Worker 函数 (必须定义在顶层以便 pickle) ---

# [修改] 接收 client 对象，不再自己建立连接
def _process_query_task(client, data_name, queries, global_indices, k, loop_count, user_id):
    """
    单个用户的查询任务 (复用连接)
    """
    results = []
    latencies = []
    
    try:
        # 连接已建立，直接获取集合
        collection = client.collections.get(data_name)
        
        for loop in range(loop_count):
            for i, vec in enumerate(queries):
                start_time = time.time()
                try:
                    response = collection.query.near_vector(
                        near_vector=vec.tolist(), limit=k, return_properties=["original_index"]
                    )
                    if loop == loop_count - 1:
                        ids = [obj.properties['original_index'] for obj in response.objects]
                        results.append((global_indices[i], ids))
                    
                    latencies.append(time.time() - start_time)
                except Exception as e:
                    # print(f"Query error: {e}")
                    latencies.append(0.0)
                    if loop == loop_count - 1:
                        results.append((global_indices[i], []))
        
    except Exception as e:
        print(f"  [Fatal] Node User-{user_id} query failed: {e}")
        return [], []

    return results, latencies

def node_process_worker(node_ip, data_name, queries, global_indices, k, loop, users_in_this_process, start_user_id):
    """
    进程 Worker：负责处理分配给该进程的一批用户
    """
    results = []
    latencies = []
    
    # [新增] 在进程层面建立长连接 (模拟后端服务的连接池)
    try:
        client = weaviate.connect_to_custom(
            http_host=node_ip, http_port=8080, http_secure=False,
            grpc_host=node_ip, grpc_port=50051, grpc_secure=False,
        )
    except Exception as e:
        print(f"  [Fatal] Process connection to {node_ip} failed: {e}")
        return [], []

    try:
        # 计算切片逻辑保持不变
        if users_in_this_process > 0:
            chunk_size = math.ceil(len(queries) / users_in_this_process)
        else:
            chunk_size = 0
        
        for i in range(users_in_this_process):
            user_id = start_user_id + i
            start = i * chunk_size
            end = min(start + chunk_size, len(queries))
            
            if start >= len(queries): 
                break 
                
            user_queries = queries[start:end]
            user_indices = global_indices[start:end]

            # [修改] 将 client 传入任务
            res, lats = _process_query_task(
                client, data_name, user_queries, user_indices, k, loop, user_id
            )
            results.extend(res)
            latencies.extend(lats)
            
    finally:
        # [新增] 所有用户处理完后，才关闭连接
        client.close()
            
    return results, latencies

class ClusterController:
    """负责集群节点的部署、资源限制和信息管理"""
    def __init__(self, all_nodes: List[Dict], user: str = "CONNECT\\xchen740"):
        self.all_nodes = all_nodes
        self.user = user
        self.weaviate_dir = "/home/CONNECT/xchen740/workspace/vector-retrieve/weaviate"

    def get_active_nodes(self, num: int) -> List[Dict]:
        if num > len(self.all_nodes):
            raise ValueError(f"请求节点数 {num} 超过可用节点数 {len(self.all_nodes)}")
        return self.all_nodes[:num]

    def _generate_compose_file(self, active_nodes: List[Dict]):
        """
        根据当前激活的节点，动态生成 docker-compose.yml
        关键点：动态计算 RAFT_BOOTSTRAP_EXPECT 和 CLUSTER_JOIN
        """
        print(f"正在生成 {len(active_nodes)} 节点的 docker-compose.yml ...")
        
        # 1. 准备基础配置
        cluster_join_list = [f"{n['ip']}:7100" for n in active_nodes]
        cluster_join_str = ",".join(cluster_join_list)
        
        raft_join_list = [n['name'].replace('weaviate_', '') for n in active_nodes] # e.g., node0, node1
        raft_join_str = ",".join(raft_join_list)
        
        bootstrap_expect = len(active_nodes)

        # 2. 读取原始模板 (假设你有一个 docker-compose.template.yml 或者直接硬编码结构)
        # 这里为了简单，我们直接构建字典结构
        services = {}
        volumes = {}
        
        for node in active_nodes:
            name = node["name"] # weaviate_node0
            short_name = name.replace("weaviate_", "") # node0
            ip = node["ip"]
            
            services[name] = {
                "network_mode": "host",
                "command": [
                    "--host", "0.0.0.0",
                    "--port", "8080",
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
                    "ENABLE_MODULES": "",
                    "CLUSTER_HOSTNAME": short_name,
                    "CLUSTER_GOSSIP_BIND_PORT": "7100",
                    "CLUSTER_DATA_BIND_PORT": "7101",
                    # 关键动态参数
                    "CLUSTER_JOIN": cluster_join_str,
                    "RAFT_JOIN": raft_join_str,
                    "RAFT_BOOTSTRAP_EXPECT": bootstrap_expect
                }
            }
            
            volumes[f"{name}_data"] = {
                "driver": "local",
                "driver_opts": {
                    "type": "none",
                    "o": "bind",
                    "device": f"{self.weaviate_dir}/volumes/{name}_data"
                }
            }

        compose_config = {
            "services": services,
            "volumes": volumes
        }

        # 3. 写入文件
        output_path = f"{self.weaviate_dir}/docker-compose.yml"
        with open(output_path, 'w') as f:
            yaml.dump(compose_config, f, default_flow_style=False)
        
        print(f"  [OK] 已更新 docker-compose.yml (Expect: {bootstrap_expect})")

    def deploy(self, nodes: List[Dict]):
        print(f"\n=== 开始部署 {len(nodes)} 节点 Weaviate 集群 ===")
        
        # 1. 清理旧环境 (停止所有节点 + 删除数据)
        # 关键修改：
        # 1. 使用 podman rm -f 直接删除容器，不依赖 docker-compose.yml (防止因文件更新导致无法停止旧服务)
        # 2. 执行 rm -rf 删除数据卷，防止旧 Raft 状态导致 "leader not found" 错误
        print(">>> [Step 1/2] 正在清理旧环境 (停止所有节点 & 删除数据)...")
        for node in self.all_nodes:
            ip, name = node["ip"], node["name"]
            
            clean_cmd = (
                f"ssh '{self.user}@{ip}' "
                f"\"podman ps -a --filter name={name} --format '{{{{.ID}}}}' | xargs -r podman rm -f && "
                f"rm -rf '{self.weaviate_dir}/volumes/{name}_data'\""
            )
            try:
                subprocess.run(clean_cmd, shell=True, check=False)
                print(f"  [OK] 节点 {name} 已清理 (容器 & 数据)")
            except Exception as e:
                print(f"  [WARN] 清理节点 {name} 失败: {e}")

        # 2. 动态生成配置文件 (此时生成，确保启动时使用的是最新配置)
        self._generate_compose_file(nodes)

        # 3. 启动新容器
        print("\n>>> [Step 2/2] 正在启动新环境...")
        for node in nodes:
            ip, name = node["ip"], node["name"]
            print(f"正在启动节点 {name} ({ip})...")
            cmd = (
                f"ssh '{self.user}@{ip}' "
                f"\"mkdir -p '{self.weaviate_dir}/volumes/{name}_data' && "
                f"cd '{self.weaviate_dir}' && "
                f"systemctl restart --user podman.socket && "
                f"podman compose -f docker-compose.yml up -d '{name}'\""
            )
            try:
                subprocess.run(cmd, shell=True, check=True)
                print(f"  [OK] 节点 {name} 启动完成")
            except subprocess.CalledProcessError as e:
                print(f"  [FAIL] 节点 {name} 启动失败: {e}")
                raise e
        print(f"=== 集群部署完成! (等待初始化 15s) ===\n")
        time.sleep(15)

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
                # 1. 获取容器 ID
                find_id_cmd = f"ssh '{self.user}@{ip}' \"podman ps --filter name={name} --format '{{{{.ID}}}}' | head -n 1\""
                container_id = subprocess.check_output(find_id_cmd, shell=True).decode().strip()
                
                if not container_id:
                    print(f"  [WARN] 节点 {name} ({ip}) 未找到运行中的容器")
                    continue

                # 确定目标 CPU 数
                if cpu_limit == 0:
                    nproc_cmd = f"ssh '{self.user}@{ip}' \"nproc\""
                    nproc = int(subprocess.check_output(nproc_cmd, shell=True).decode().strip())
                    target_cpus = nproc
                    cpu_list = f"0-{nproc - 1}"
                else:
                    target_cpus = cpu_limit

                # 2. 尝试 Podman Update (Cgroup)
                try:
                    update_cmd = f"ssh '{self.user}@{ip}' \"podman update --cpus={target_cpus} {container_id}\""
                    subprocess.run(update_cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    action_str = f"限制 CPU: {target_cpus}" if cpu_limit > 0 else "重置 Cgroup 配额"
                    print(f"  [OK] 节点 {name} ({ip}) 已通过 Cgroup {action_str}")
                except subprocess.CalledProcessError:
                    # 3. 回退 Taskset
                    if cpu_limit > 0:
                        # print(f"  [INFO] 节点 {name} Cgroup 限制失败，切换到 taskset...")
                        get_pid_cmd = f"ssh '{self.user}@{ip}' \"podman inspect -f '{{{{.State.Pid}}}}' {container_id}\""
                        pid = subprocess.check_output(get_pid_cmd, shell=True).decode().strip()
                        if pid and pid != "0":
                            taskset_cmd = f"ssh '{self.user}@{ip}' \"taskset -acp {cpu_list} {pid}\""
                            subprocess.run(taskset_cmd, shell=True, check=True)
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

    def build(self, node_ip: str, data_path: str, m: int, ef_construction: int):
        print(f"\n=== 开始构建索引 (数据加载) ===")
        print(f"目标节点: {node_ip}, 数据: {data_path}")
        
        client = weaviate.connect_to_custom(
            http_host=node_ip, http_port=8080, http_secure=False,
            grpc_host=node_ip, grpc_port=50051, grpc_secure=False,
        )
        
        try:
            vectors = DatasetLoader.read_fvecs(data_path)
            print(f"加载向量数据: {vectors.shape}")
            
            if client.collections.exists(self.collection_name):
                print(f"删除旧集合: {self.collection_name}")
                client.collections.delete(self.collection_name)
                
            print(f"创建新集合: {self.collection_name} (HNSW: m={m}, ef_construction={ef_construction})")
            client.collections.create(
                name=self.collection_name,
                vectorizer_config=wc.Configure.Vectorizer.none(),
                vector_index_config=wc.Configure.VectorIndex.hnsw(
                    distance_metric=wc.VectorDistances.L2_SQUARED,
                    max_connections=m,
                    ef_construction=ef_construction,
                ),
                properties=[wc.Property(name="original_index", data_type=wc.DataType.INT)]
            )
            
            collection = client.collections.get(self.collection_name)
            print("开始批量上传...")
            start_time = time.time()
            with collection.batch.dynamic() as batch:
                for i, vec in enumerate(vectors):
                    batch.add_object(properties={"original_index": i}, vector=vec)
                    if (i + 1) % 1000 == 0:
                        print(f"  进度: {i + 1}/{len(vectors)}", end='\r')
            
            print(f"\n数据加载完成! 耗时: {time.time() - start_time:.2f}s")
            print("等待索引构建与同步 (5秒)...")
            time.sleep(5)
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
            try:
                client = weaviate.connect_to_custom(
                    http_host=ip, http_port=8080, http_secure=False,
                    grpc_host=ip, grpc_port=50051, grpc_secure=False,
                )
                client.collections.get(self.args.data_name).config.update(
                    vector_index_config=wc.Reconfigure.VectorIndex.hnsw(ef=self.args.ef_search)
                )
                print(f"  [OK] ef_search 参数已更新")
                client.close()
            except Exception as e:
                print(f"  [ERROR] 更新 ef_search 参数失败: {e}")

    def sync_start(self):
        now = time.time()
        next_minute = math.ceil(now / 60) * 60
        if next_minute - now < 10:
            next_minute += 60
        wait_seconds = next_minute - now
        start_time_str = time.strftime("%H:%M:%S", time.localtime(next_minute))
        
        print(f"\n[同步等待] 当前时间: {time.strftime('%H:%M:%S', time.localtime(now))}")
        print(f"[同步等待] 将在 {start_time_str} 启动压测 (等待 {wait_seconds:.1f} 秒)...")
        time.sleep(wait_seconds)

    def run(self):
        # 1. 配置更新与同步
        self.update_config()
        self.sync_start()

        # 2. 初始指标
        print("正在抓取集群初始指标...")
        start_stats = self.monitor.queries_durations_ms_count

        print("\n=== 开始并发查询 (纯多进程模式: 进程并行，用户串行) ===")
        global_start_time = time.time()
        
        all_results = []
        all_latencies = []
        num_nodes = len(self.active_nodes)
        total_queries = len(self.query_vectors)

        # [重构] 扁平化任务分配
        all_process_tasks = []
        
        for i in range(num_nodes):
            node_ip = self.active_nodes[i]["ip"]
            
            # 1. 切分给该节点的数据 (总查询向量分给节点)
            node_queries = self.query_vectors[i::num_nodes]
            node_indices = list(range(i, total_queries, num_nodes))
            
            # 2. 计算该节点需要的进程数
            # threads_per_node 在这里代表 "该节点总共要模拟的用户数"
            total_users_per_node = self.args.threads_per_node
            
            # 进程数
            n_procs = min(self.args.process_num, total_users_per_node)
            if n_procs < 1: n_procs = 1
            
            # 每个进程负责的用户数
            users_per_proc = math.ceil(total_users_per_node / n_procs)
            
            print(f"  -> [Node {node_ip}] 启动 {n_procs} 个进程 (每进程顺序处理 {users_per_proc} 用户)")

            for p in range(n_procs):
                start_user = p * users_per_proc
                # 计算该进程实际负责的用户数
                count = min(users_per_proc, total_users_per_node - start_user)
                if count <= 0: break
                
                # 计算该进程对应的数据切片
                # 我们需要把 node_queries 再切分给这 n_procs 个进程
                total_node_vectors = len(node_queries)
                vec_start = math.floor(p / n_procs * total_node_vectors)
                vec_end = math.floor((p + 1) / n_procs * total_node_vectors)
                
                proc_queries = node_queries[vec_start:vec_end]
                proc_indices = node_indices[vec_start:vec_end]
                
                if len(proc_queries) == 0 and count > 0:
                    # 向量太少，不够分，但还有用户。给空列表，让用户建立空连接（模拟负载）
                    proc_queries = np.array([])
                    proc_indices = []

                all_process_tasks.append((
                    node_ip,
                    self.args.data_name,
                    proc_queries,
                    proc_indices,
                    self.args.k,
                    self.args.loop,
                    count,      # users_in_this_process
                    start_user  # start_user_id
                ))

        print(f"启动全局进程池，总进程数: {len(all_process_tasks)} ...")
        
        # 使用 spawn 上下文
        ctx = multiprocessing.get_context('spawn')
        # 开启进程池，max_workers 设为任务总数，确保所有进程同时运行
        with ProcessPoolExecutor(max_workers=len(all_process_tasks), mp_context=ctx) as executor:
            futures = [executor.submit(node_process_worker, *args) for args in all_process_tasks]
            
            completed_count = 0
            for future in as_completed(futures):
                res, lats = future.result()
                all_results.extend(res)
                all_latencies.extend(lats)
                completed_count += 1
                print(f"  [Progress] {completed_count}/{len(all_process_tasks)} 进程任务完成。", end='\r')

        global_end_time = time.time()
        total_duration = global_end_time - global_start_time

        # 3. 统计结果
        self.report(all_results, all_latencies, total_duration, start_stats)

    def report(self, results, latencies, duration, start_stats):
        # 计算 Recall
        print("\n正在计算 Recall (基于最后一次循环)...")
        total_hits = 0
        for global_idx, retrieved_ids in results:
            gt_ids = set(self.ground_truth[global_idx][:self.args.k])
            total_hits += len(gt_ids.intersection(set(retrieved_ids)))
        
        recall = total_hits / (len(self.query_vectors) * self.args.k)
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

        print(f"--------------------------------")
        print(f"=== 测试结果 ===")
        print(f"总耗时: {duration:.6f}s")
        print(f"Recall@{self.args.k}: {recall:.2f}")
        print(f"平均延迟: {avg_latency:.6f} ms")
        print(f"P99 延迟: {p99_latency:.6f} ms")
        print(f"--------------------------------")
        print(f"集群总处理查询数: {cluster_processed_count}")
        print(f"全局 QPS:   {global_qps:.2f}")


def main(args):
    active_nodes = ALL_NODES[:args.node_num]
    active_ips = [n["ip"] for n in active_nodes]
    
    print(f"=== 节点配置 ===")
    print(f"使用节点数: {args.node_num}")
    for n in active_nodes:
        print(f"  - {n['name']}: {n['ip']}")

    controller = ClusterController(ALL_NODES)
    
    # 1. 如果指定了部署标志，先执行部署
    if args.deploy:
        controller.deploy(active_nodes)

    # 1.5 应用或取消 CPU 限制 (只要参数存在就执行，支持 0 取消)
    controller.update_cpu_limits(active_nodes, args.limit_cpu)

    # 2. 如果指定了建图标志，执行数据加载
    if args.build:
        indexer = WeaviateIndexer(args.data_name)
        indexer.build(active_ips[0], args.data_path, args.m, args.ef_construction)

    # 3. 加载查询数据
    print(f"正在加载查询向量: {args.query_path}")
    all_queries = DatasetLoader.read_fvecs(args.query_path)
    print(f"正在加载 Ground Truth: {args.groundtruth_path}")
    all_ground_truth = DatasetLoader.read_ivecs(args.groundtruth_path)

    query_vectors = all_queries[0]
    ground_truth = all_ground_truth
    total_queries = len(query_vectors)
    
    print(f"模式: 节点绑定并发 (多进程模式)")
    print(f"并发结构: {args.node_num} 进程 x {args.threads_per_node} 线程/进程")
    print(f"总查询数: {total_queries} (每轮) x {args.loop} 轮")
    
    # 4. 执行压测
    benchmark = QueryBenchmark(args, active_nodes, query_vectors, ground_truth)
    benchmark.run()

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
    parser.add_argument("--limit_cpu", type=float, default=0, help="Limit CPU cores per node (e.g. 4.0). 0 means no limit.")
    parser.add_argument("--build", action="store_true", default=False, help="Build index (load data) before querying")

    # HNSW 参数 (用于建图)
    parser.add_argument("--m", type=int, default=8, help="HNSW max connections")
    parser.add_argument("--ef_construction", type=int, default=16, help="HNSW ef construction")

    # HNSW 参数 (用于查询)
    parser.add_argument("--ef_search", type=int, default=16, help="HNSW ef_search parameter. -1 to use default.")

    # 查询参数
    parser.add_argument("--k", type=int, default=10, help="Top K")
    parser.add_argument("--loop", type=int, default=5, help="Number of query loops per query batch")
    parser.add_argument("--threads_per_node", type=int, default=100000, help="Number of threads (users) per node for concurrent query processing")
    parser.add_argument("--process_num", type=int, default=32, help="Number of processes per node client to drive the threads")

    args = parser.parse_args()
    main(args)