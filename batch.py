import subprocess
import os
import itertools
import time
import concurrent.futures  # 引入并发库

SCRIPT_PATH = "main.py"

# 数据集根目录
DATASET_ROOT = "/home/CONNECT/xchen740/workspace/vector-retrieve/dataset"

# 1. 定义要运行的数据集
# 请根据实际情况修改路径和数量
DATASETS = [
    # {
    #     "name": "siftsmall",
    #     "base": "siftsmall/siftsmall_base.fvecs",
    #     "query": "siftsmall/siftsmall_query.fvecs",
    #     "groundtruth": "siftsmall/siftsmall_groundtruth.ivecs",
    # },
    {
        "name": "sift",
        "base": "sift/sift_base.fvecs",
        "query": "sift/sift_query.fvecs",
        "groundtruth": "sift/sift_groundtruth.ivecs",
    }
]

# 2. 定义参数网格
# 脚本将运行这些参数的所有笛卡尔积组合
PARAM_GRID = {
    "node_num": [1],   # 测试不同的节点数
    "partition_mode": ["k-means"],                   # 测试划分模式
    "redundancy": [3],                                     # 冗余级别
    "m": [24],                                   # HNSW 参数 M
    "ef_construction": [400],               # HNSW 建图参数
    "ef_search": [128],                              # HNSW 搜索参数
    "limited_level": [5],                                           # HNSW 参数 limited_level
    "k": [10],                                                       # Top-K
    "num_query_threads": [64],                                        # 查询线程数（传递给 main.py --num_query_threads）
    "reduce": [False],                                          # Reduce 模式开关（独立图每节点）
    "distributed": [True],                                           # 分布式模式开关
}

# 等待已生成 graph 最长秒数（0 表示无限等待）
GRAPH_WAIT_TIMEOUT = 0

def run_worker(cmd, task_id, total_tasks, description):
    """
    单个任务的工作函数
    """
    # 若命令带 --graph_path，则等待文件可用
    if "--graph_path" in cmd:
        idx = cmd.index("--graph_path")
        graph_path = cmd[idx + 1] if idx + 1 < len(cmd) else None
        if graph_path:
            waited = 0
            interval = 1
            print(f"[{task_id}/{total_tasks}] Waiting for graph file {graph_path} to exist...")
            while not os.path.exists(graph_path):
                time.sleep(interval)
                waited += interval
                if GRAPH_WAIT_TIMEOUT and waited >= GRAPH_WAIT_TIMEOUT:
                    print(f"[{task_id}/{total_tasks}] TIMEOUT waiting for {graph_path}. Skipping task.")
                    return
    print(f"[{task_id}/{total_tasks}] STARTING: {description}")
    start_time = time.time()
    try:
        # 这里的输出会直接打印到终端，多个任务并行时输出会交错。
        # 如果觉得乱，可以添加 stdout=subprocess.DEVNULL 来静默运行，或者重定向到文件。
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[{task_id}/{total_tasks}] ERROR: {e}")
    
    print(f"[{task_id}/{total_tasks}] FINISHED in {time.time() - start_time:.2f}s")

def run_batch():
    # 生成所有参数组合
    keys = PARAM_GRID.keys()
    values = PARAM_GRID.values()
    combinations = list(itertools.product(*values))
    
    # 准备任务列表
    tasks = []
    
    for dataset in DATASETS:
        # 构建完整路径
        data_name = dataset['name']
        data_path = os.path.join(DATASET_ROOT, dataset['base'])
        query_path = os.path.join(DATASET_ROOT, dataset['query'])
        groundtruth_path = os.path.join(DATASET_ROOT, dataset['groundtruth'])
        
        # 简单检查文件是否存在
        if not os.path.exists(data_path):
            print(f"Warning: Dataset file not found: {data_path}. Skipping...")
            continue

        # 用于追踪哪些 (m, ef_construction, ef_search, limited_level) 已经 build 过
        built_graph_keys = set()

        for combo in combinations:
            params = dict(zip(keys, combo))

            # 形成 graph 保存/加载路径（与 main.py 的命名一致）
            graph_filename = f"save/hnsw-{data_name}-m{params['m']}-efc{params['ef_construction']}-ll{params['limited_level']}.json"
            graph_path = os.path.join(".", graph_filename)

            key = (params['m'], params['ef_construction'], params['limited_level'])

            # 如果目标 graph 已存在于磁盘，直接视为已构建
            if os.path.exists(graph_path):
                built_graph_keys.add(key)

            # 构建命令基础
            cmd = [
                "python", SCRIPT_PATH,
                "--data_name", data_name,
                "--data_path", data_path,
                "--query_path", query_path,
                "--groundtruth_path", groundtruth_path,
                "--k", str(params['k']),
                "--m", str(params['m']),
                "--ef_construction", str(params['ef_construction']),
                "--ef_search", str(params['ef_search']),
                "--limited_level", str(params['limited_level']),
                "--num_query_threads", str(params['num_query_threads']),
                "--node_num", str(params['node_num']),
                "--partition_mode", params['partition_mode'],
                "--redundancy", str(params['redundancy']),
            ]

            if params.get('reduce'):
                cmd.append('--reduce')
            if params.get('distributed'):
                cmd.append('--distributed')

            description = f"Dataset: {dataset['name']} | Build graph | Params: {params}"

            # # 如果该组合已经建图过，后续任务都应该直接 load（传 --graph_path）
            # if params.get('reduce'):
            #     description = f"Dataset: {dataset['name']} | Build graph | Params: {params}"
            # else:
            #     if key in built_graph_keys:
            #         # 直接加载已经存在的 graph
            #         cmd += ["--graph_path", graph_path]
            #         description = f"Dataset: {dataset['name']} | Load graph {graph_path} | Params: {params}"
            #     else:
            #         # 第一次遇到此 m/efc 组合时，创建 build 任务（不传 --graph_path），并标记已构建
            #         built_graph_keys.add(key)
            #         description = f"Dataset: {dataset['name']} | Build graph (first time) | Params: {params}"
            
            tasks.append((cmd, description))

    total_experiments = len(tasks)
    print(f"=== Starting Batch Execution ===")
    print(f"Total experiments to run: {total_experiments}")

    # --- 并行执行配置 ---
    MAX_WORKERS = 1  # 设置并行数量
    print(f"Parallel Workers: {MAX_WORKERS}")

    if MAX_WORKERS > 1:
        print("Running tasks in parallel...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            for i, (cmd, desc) in enumerate(tasks):
                # 提交任务到线程池
                futures.append(executor.submit(run_worker, cmd, i + 1, total_experiments, desc))

            # 等待所有任务完成
            concurrent.futures.wait(futures)
    else:
        print("Running tasks sequentially...")
        for i, (cmd, desc) in enumerate(tasks):
            run_worker(cmd, i + 1, total_experiments, desc)

if __name__ == "__main__":
    run_batch()