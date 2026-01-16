import argparse
import numpy as np
import time
import weaviate
import weaviate.classes.config as wc
from weaviate.classes.query import MetadataQuery
import requests

# 定义节点 IP 映射
nodes_ip = {
    "weaviate_node0": "192.168.200.204",
    "weaviate_node1": "192.168.200.205",
    "weaviate_node2": "192.168.200.206",
    "weaviate_node3": "192.168.200.207",
    "weaviate_node4": "192.168.200.208",
    "weaviate_node5": "192.168.200.209",
}

def read_fvecs(filename, c_contiguous=True):
    """读取 .fvecs 格式的文件"""
    # 将文件内容读入一个 (N, D+1) 的浮点数数组
    # 其中 D 是向量维度。第一列是维度，可以忽略。
    fv = np.fromfile(filename, dtype=np.float32)

    # 获取第一个向量的维度
    dim = fv.view(np.int32)[0]

    # 重塑数组
    fv = fv.reshape(-1, 1 + dim)

    # 去掉维度列
    vectors = fv[:, 1:]

    if c_contiguous:
        return np.ascontiguousarray(vectors)
    
    return vectors

def read_ivecs(filename, c_contiguous=True):
    """读取 .ivecs 格式的文件"""
    # 将文件内容读入一个 (N, D+1) 的整数数组
    iv = np.fromfile(filename, dtype=np.int32)

    # 获取第一个向量的维度
    dim = iv[0]

    # 重塑数组
    iv = iv.reshape(-1, 1 + dim)

    # 去掉维度列
    vectors = iv[:, 1:]

    if c_contiguous:
        return np.ascontiguousarray(vectors)
    
    return vectors

def get_batch_metrics():
    """从 Prometheus 端点获取批处理相关的指标"""
    metrics = {}
    for name, ip in nodes_ip.items():
        try:
            # 假设 Prometheus 指标暴露在 2112 端口
            url = f"http://{ip}:2112/metrics"
            resp = requests.get(url, timeout=0.5)
            if resp.status_code == 200:
                node_metrics = {
                    "batch_durations_ms": 0,
                    "batch_delete_durations_ms": 0,
                    "batch_size_bytes": 0,
                    "batch_size_objects": 0,
                    "batch_objects_processed_total": 0,
                    "batch_objects_processed_bytes": 0,
                }
                for line in resp.text.split('\n'):
                    # 解析 batch_durations_ms
                    if line.startswith("batch_durations_ms"):
                        parts = line.split(' ')
                        if len(parts) >= 2:
                            node_metrics["batch_durations_ms"] += float(parts[-1])
                    
                    # 解析 batch_delete_durations_ms
                    elif line.startswith("batch_delete_durations_ms"):
                        parts = line.split(' ')
                        if len(parts) >= 2:
                            node_metrics["batch_delete_durations_ms"] += float(parts[-1])
                    
                    # 解析 batch_size_bytes
                    elif line.startswith("batch_size_bytes"):
                        parts = line.split(' ')
                        if len(parts) >= 2:
                            node_metrics["batch_size_bytes"] += float(parts[-1])
                    
                    # 解析 batch_size_objects
                    elif line.startswith("batch_size_objects"):
                        parts = line.split(' ')
                        if len(parts) >= 2:
                            node_metrics["batch_size_objects"] += float(parts[-1])
                    
                    # 解析 batch_objects_processed_total
                    elif line.startswith("batch_objects_processed_total"):
                        parts = line.split(' ')
                        if len(parts) >= 2:
                            node_metrics["batch_objects_processed_total"] += float(parts[-1])
                    
                    # 解析 batch_objects_processed_bytes
                    elif line.startswith("batch_objects_processed_bytes"):
                        parts = line.split(' ')
                        if len(parts) >= 2:
                            node_metrics["batch_objects_processed_bytes"] += float(parts[-1])
                
                metrics[name] = node_metrics
        except Exception:
            metrics[name] = {
                "batch_durations_ms": -1,
                "batch_delete_durations_ms": -1,
                "batch_size_bytes": -1,
                "batch_size_objects": -1,
                "batch_objects_processed_total": -1,
                "batch_objects_processed_bytes": -1,
            }
    return metrics

def get_query_metrics():
    """从 Prometheus 端点获取查询相关的指标"""
    metrics = {}
    for name, ip in nodes_ip.items():
        try:
            # 假设 Prometheus 指标暴露在 2112 端口
            url = f"http://{ip}:2112/metrics"
            resp = requests.get(url, timeout=0.5)
            if resp.status_code == 200:
                node_metrics = {
                    "concurrent_queries_count": 0,
                    "queries_durations_ms_sum": 0,
                    "queries_durations_ms_count": 0,
                    "queries_filtered_vector_durations_ms": 0,
                    "concurrent_goroutines": 0,
                    "requests_total": 0,
                    "query_dimensions_total": 0,
                    "query_dimensions_combined_total": 0,
                }
                for line in resp.text.split('\n'):
                    # 解析 concurrent_queries_count
                    if line.startswith("concurrent_queries_count"):
                        parts = line.split(' ')
                        if len(parts) >= 2:
                            node_metrics["concurrent_queries_count"] += float(parts[-1])
                        print(line)
                    
                    # 解析 queries_durations_ms_sum
                    elif line.startswith("queries_durations_ms_sum"):
                        parts = line.split(' ')
                        if len(parts) >= 2:
                            node_metrics["queries_durations_ms_sum"] += float(parts[-1])
                    
                    # 解析 queries_durations_ms_count
                    elif line.startswith("queries_durations_ms_count"):
                        parts = line.split(' ')
                        if len(parts) >= 2:
                            node_metrics["queries_durations_ms_count"] += float(parts[-1])
                    
                    # 解析 queries_filtered_vector_durations_ms
                    elif line.startswith("queries_filtered_vector_durations_ms"):
                        parts = line.split(' ')
                        if len(parts) >= 2:
                            node_metrics["queries_filtered_vector_durations_ms"] += float(parts[-1])
                    
                    # 解析 concurrent_goroutines
                    elif line.startswith("concurrent_goroutines"):
                        parts = line.split(' ')
                        if len(parts) >= 2:
                            node_metrics["concurrent_goroutines"] += float(parts[-1])
                    
                    # 解析 requests_total
                    elif line.startswith("requests_total"):
                        parts = line.split(' ')
                        if len(parts) >= 2:
                            node_metrics["requests_total"] += float(parts[-1])
                    
                    # 解析 query_dimensions_total
                    elif line.startswith("query_dimensions_total"):
                        parts = line.split(' ')
                        if len(parts) >= 2:
                            node_metrics["query_dimensions_total"] += float(parts[-1])
                    
                    # 解析 query_dimensions_combined_total
                    elif line.startswith("query_dimensions_combined_total"):
                        parts = line.split(' ')
                        if len(parts) >= 2:
                            node_metrics["query_dimensions_combined_total"] += float(parts[-1])
                
                metrics[name] = node_metrics
        except Exception:
            metrics[name] = {
                "concurrent_queries_count": -1,
                "queries_durations_ms": -1,
                "queries_filtered_vector_durations_ms": -1,
                "concurrent_goroutines": -1,
                "requests_total": -1,
                "query_dimensions_total": -1,
                "query_dimensions_combined_total": -1,
            }
    return metrics

def main(args):
    # 连接到集群
    print(f"正在连接 Weaviate 集群 (Node: 192.168.200.204)...")
    client = weaviate.connect_to_custom(
        http_host="192.168.200.204",
        http_port=8080,
        http_secure=False,
        grpc_host="192.168.200.204",
        grpc_port=50051,
        grpc_secure=False,
    )

    collection_name = args.data_name

    try:
        # 1. 准备数据与集合
        if not args.skip_import:
            print(f"\n--- 阶段 1: 数据导入与索引构建 ---")
            
            # 如果集合存在，先删除以确保参数生效 (Benchmark 模式)
            if client.collections.exists(collection_name):
                print(f"集合 {collection_name} 已存在，正在删除重置以应用新参数...")
                client.collections.delete(collection_name)

            print(f"创建集合 {collection_name}")
            print(f"HNSW 参数: M={args.m}, ef_construction={args.ef_construction}, ef_search={args.ef_search}")
            
            client.collections.create(
                name=collection_name,
                vectorizer_config=wc.Configure.Vectorizer.none(),
                vector_index_config=wc.Configure.VectorIndex.hnsw(
                    distance_metric=wc.VectorDistances.L2_SQUARED,
                    max_connections=args.m,
                    ef_construction=args.ef_construction,
                    ef=args.ef_search
                ),
                sharding_config=wc.Configure.sharding(desired_count=6),
                replication_config=wc.Configure.replication(factor=1),
                properties=[
                    wc.Property(name="original_index", data_type=wc.DataType.INT),
                ]
            )

            # 读取数据
            print(f"读取基准向量: {args.data_path}")
            base_vectors = read_fvecs(args.data_path)
            print(f"数据规模: {base_vectors.shape}")

            # 导入数据
            collection = client.collections.get(collection_name)
            print("开始批量导入...")
            start_import = time.time()
            
            with collection.batch.dynamic() as batch:
                for i, vec in enumerate(base_vectors):
                    batch.add_object(
                        properties={"original_index": i},
                        vector=vec.tolist(),
                        uuid=weaviate.util.generate_uuid5(i)
                    )
                    if (i + 1) % 1000 == 0:
                        print(f"已导入 {i + 1} 条...", end='\r')
            
            import_time = time.time() - start_import
            print(f"\n导入完成! 耗时: {import_time:.2f}s")
        else:
            print(f"\n--- 阶段 1: 跳过数据导入 (使用现有集合) ---")
            if not client.collections.exists(collection_name):
                raise ValueError(f"集合 {collection_name} 不存在，无法跳过导入。请先运行一次不带 --skip_import 的命令。")
            collection = client.collections.get(collection_name)
            print(f"已加载集合 {collection_name}")

        # --- 验证总数 ---
        total_count = collection.aggregate.over_all(total_count=True).total_count
        print(f"✅ 集合实际总对象数: {total_count}")

        # --- 查看各节点分布 ---
        print(f"\n--- 节点数据分布情况 ---")
        try:
            # 获取集群中该集合的节点状态
            # 添加 output="verbose" 以获取分片详细统计
            nodes_info = client.cluster.nodes(collection=collection_name, output="verbose")
            # 按节点名称排序
            nodes_info.sort(key=lambda x: x.name)
            
            for node in nodes_info:
                # 统计该节点上该集合所有分片的对象数
                # node.shards 可能为 None，需做空值处理
                shards = node.shards or []
                node_count = sum(shard.object_count for shard in shards)
                print(f"节点: {node.name:<20} 向量数: {node_count}")
        except Exception as e:
            print(f"获取节点分布信息失败: {e}")
        
        # 2. 执行查询
        print(f"\n--- 阶段 2: 向量检索 (k={args.k}) ---")
        
        # --- 记录初始指标 ---
        print("正在抓取初始节点指标...")
        start_metrics = get_batch_metrics()

        # --- 记录初始查询指标 ---
        print("正在抓取初始查询指标...")
        start_query_metrics = get_query_metrics()

        # 执行查询
        query_vectors = read_fvecs(args.query_path)
        ground_truth = read_ivecs(args.groundtruth_path)
        print(f"查询向量规模: {query_vectors.shape}")

        hits = 0
        total_queries = len(query_vectors)
        start_query = time.time()

        for i, query_vec in enumerate(query_vectors):
            # 执行查询
            response = collection.query.near_vector(
                near_vector=query_vec.tolist(),
                limit=args.k,
                return_properties=["original_index"],
                return_metadata=MetadataQuery(distance=True)
            )

            # 计算 Recall
            gt_ids = set(ground_truth[i][:args.k])
            retrieved_ids = [obj.properties['original_index'] for obj in response.objects]
            hits += len(gt_ids.intersection(retrieved_ids))

            if (i + 1) % 10 == 0:
                print(f"正在查询 {i + 1}/{total_queries}...", end='\r')

        query_time = time.time() - start_query

        # --- 记录结束指标 ---
        print("\n正在抓取结束节点指标...")
        end_metrics = get_batch_metrics()

        # --- 记录结束查询指标 ---
        print("\n正在抓取结束查询指标...")
        end_query_metrics = get_query_metrics()

        # --- 打印查询指标变化 ---
        print("\n--- 各节点查询相关指标统计 ---")
        for name in nodes_ip.keys():
            start = start_query_metrics.get(name, {})
            end = end_query_metrics.get(name, {})
            
            if start and end:
                print(f"\n节点 {name} 指标变化：")
                for metric, start_value in start.items():
                    end_value = end.get(metric, -1)
                    if start_value != -1 and end_value != -1:
                        diff = end_value - start_value
                        print(f"  {metric}: {start_value} -> {end_value} (增量: {diff})")
                    else:
                        print(f"  {metric}: 无法获取")
            else:
                print(f"节点 {name}: 无法获取查询指标")

        print("\n--- 各节点批处理负载估算 (基于 Prometheus 指标) ---")
        for name in nodes_ip.keys():
            start = start_metrics.get(name, {})
            end = end_metrics.get(name, {})
            
            if start and end:
                print(f"\n节点 {name} 指标变化：")
                for metric, start_value in start.items():
                    end_value = end.get(metric, -1)
                    if start_value != -1 and end_value != -1:
                        diff = end_value - start_value
                        print(f"  {metric}: {start_value} -> {end_value} (增量: {diff})")
                    else:
                        print(f"  {metric}: 无法获取")
            else:
                print(f"节点 {name}: 无法获取指标")

        recall = hits / (total_queries * args.k)
        qps = total_queries / query_time

        print(f"\n\n=== 测试结果 ===")
        print(f"Dataset: {args.data_name}")
        print(f"Params: M={args.m}, ef_c={args.ef_construction}, ef_s={args.ef_search}")
        print(f"Recall@{args.k}: {recall:.4f}")
        print(f"QPS: {qps:.2f}")
        print(f"Total Query Time: {query_time:.2f}s")

    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HNSW Vector Retrieval Benchmark")
    parser.add_argument("--data_name", type=str, default="siftsmall", help="Name of the dataset")
    parser.add_argument("--data_path", type=str, default="/home/CONNECT/xchen740/workspace/vector-retrieve/dataset/siftsmall/siftsmall_base.fvecs", help="Path to base vectors file")
    parser.add_argument("--query_path", type=str, default="/home/CONNECT/xchen740/workspace/vector-retrieve/dataset/siftsmall/siftsmall_query.fvecs", help="Path to query vectors file")
    parser.add_argument("--groundtruth_path", type=str, default="/home/CONNECT/xchen740/workspace/vector-retrieve/dataset/siftsmall/siftsmall_groundtruth.ivecs", help="Path to groundtruth file")
    parser.add_argument("--k", type=int, default=10, help="Number of nearest neighbors to retrieve")
    parser.add_argument("--m", type=int, default=16, help="Max number of connections per element")
    parser.add_argument("--ef_construction", type=int, default=128, help="Size of the dynamic candidate list for construction")
    parser.add_argument("--ef_search", type=int, default=64, help="Size of the dynamic candidate list for search")
    parser.add_argument("--limited_level", type=int, default=5, help="Level limit for HNSW")
    parser.add_argument("--skip_import", action="store_true", help="Skip data import and use existing collection")
    
    args = parser.parse_args()
    main(args)
