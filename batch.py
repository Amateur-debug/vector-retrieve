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
    {
        "name": "siftsmall",
        "base": "siftsmall/siftsmall_base.fvecs",
        "query": "siftsmall/siftsmall_query.fvecs",
        "groundtruth": "siftsmall/siftsmall_groundtruth.ivecs",
    },
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
    "node_num": [2, 4, 8, 16],                      # 测试不同的节点数
    "partition_mode": ["round_robin", "k-means"],   # 测试划分模式
    "redundancy": [0, 1, 2],                        # 冗余级别
    "m": [8, 16, 32],                               # HNSW 参数 M
    "ef_construction": [64, 128],                   # HNSW 建图参数
    "ef_search": [16, 32, 64],                      # HNSW 搜索参数
    "k": [10]                                       # Top-K
}

def run_worker(cmd, task_id, total_tasks, description):
    """
    单个任务的工作函数
    """
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

        for combo in combinations:
            params = dict(zip(keys, combo))
            
            # 构建命令
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
                "--node_num", str(params['node_num']),
                "--partition_mode", params['partition_mode'],
                "--redundancy", str(params['redundancy']),
            ]
            
            description = f"Dataset: {dataset['name']} | Params: {params}"
            tasks.append((cmd, description))

    total_experiments = len(tasks)
    print(f"=== Starting Batch Execution ===")
    print(f"Total experiments to run: {total_experiments}")

    # --- 并行执行配置 ---
    MAX_WORKERS = 32  # 设置并行数量
    print(f"Parallel Workers: {MAX_WORKERS}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for i, (cmd, desc) in enumerate(tasks):
            # 提交任务到线程池
            futures.append(executor.submit(run_worker, cmd, i + 1, total_experiments, desc))
        
        # 等待所有任务完成
        concurrent.futures.wait(futures)

if __name__ == "__main__":
    run_batch()