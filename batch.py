import subprocess
import os
import itertools
import time
import concurrent.futures

SCRIPT_PATH = "main.py"
DATASET_ROOT = "/home/CONNECT/xchen740/workspace/vector-retrieve/dataset"

# ============================================================
# 数据集配置
# ============================================================
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

# ============================================================
# 参数网格 (将运行所有笛卡尔积组合)
# ============================================================
PARAM_GRID = {
    "node_num":           [1],
    "partition_mode":     ["k-means"],
    "redundancy":         [2, 1, 0],
    "m":                  [16],
    "ef_construction":    [200],
    "ef_search":          [16, 24, 32],
    "limited_level":      [5],
    "k":                  [10],
    "num_query_threads":  [8, 32],
    "reduce":             [False],
    "distributed":        [True],
    "layer":              [True, False],
    "comm_mode":          ["tcp"],  # 通信模式："tcp" 或 "rdma"
    "num_cpu":            [8],       # CPU 核心数
}

# 等待 graph 文件的最长秒数 (0 = 无限等待)
GRAPH_WAIT_TIMEOUT = 0
# 并行任务数 (1 = 顺序执行)
MAX_WORKERS = 1


def run_worker(cmd, task_id, total_tasks, description):
    """运行单个任务"""
    # 检查 graph 文件是否存在
    if "--graph_path" in cmd:
        idx = cmd.index("--graph_path")
        graph_path = cmd[idx + 1] if idx + 1 < len(cmd) else None
        if graph_path:
            waited = 0
            interval = 1
            print(f"[{task_id}/{total_tasks}] Waiting for graph file {graph_path}...")
            while not os.path.exists(graph_path):
                time.sleep(interval)
                waited += interval
                if GRAPH_WAIT_TIMEOUT and waited >= GRAPH_WAIT_TIMEOUT:
                    print(f"[{task_id}/{total_tasks}] TIMEOUT waiting for {graph_path}. Skipping.")
                    return

    print(f"[{task_id}/{total_tasks}] STARTING: {description}")
    start_time = time.time()

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[{task_id}/{total_tasks}] ERROR: {e}")

    print(f"[{task_id}/{total_tasks}] FINISHED in {time.time() - start_time:.2f}s")


def build_command(dataset, params):
    """构建命令行参数"""
    data_name = dataset['name']
    data_path = os.path.join(DATASET_ROOT, dataset['base'])
    query_path = os.path.join(DATASET_ROOT, dataset['query'])
    groundtruth_path = os.path.join(DATASET_ROOT, dataset['groundtruth'])

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
    if params.get('layer'):
        cmd.append('--layer')
    # 通信模式参数
    if params.get('comm_mode') and params['comm_mode'] != 'tcp':
        cmd += ['--comm_mode', str(params['comm_mode'])]
    # num_cpu 参数
    if params.get('num_cpu'):
        cmd += ['--num_cpu', str(params['num_cpu'])]

    return cmd


def get_graph_path(data_name, params):
    """生成图文件路径"""
    filename = f"hnsw-{data_name}-m{params['m']}-efc{params['ef_construction']}-ll{params['limited_level']}.json"
    return os.path.join(".", "save", filename)


def get_result_path(dataset_name, params):
    """生成结果文件路径"""
    filename = f"results-{dataset_name}-m{params['m']}-efc{params['ef_construction']}-efs{params['ef_search']}-ll{params['limited_level']}-node{params['node_num']}-{params['partition_mode']}-red{params['redundancy']}-threads{params['num_query_threads']}"
    if params.get('distributed'):
        filename += "-distributed"
    if params.get('reduce'):
        filename += "-reduce"
    if params.get('layer'):
        filename += "-layer"
    num_cpu = params.get('num_cpu', 8)
    filename += f"-cpus{num_cpu}.json"
    return os.path.join(".", "results", "cpp", filename)


def run_batch():
    """批量运行实验"""
    # 原始参数顺序
    keys = list(PARAM_GRID.keys())

    # 使用列表保持顺序与原始 PARAM_GRID 的 keys 一致
    param_grid_normal_items = [(k, PARAM_GRID[k]) for k in keys]
    param_grid_normal = dict(param_grid_normal_items)

    combinations_normal = list(itertools.product(*[v for _, v in param_grid_normal_items]))

    tasks = []
    built_graphs = set()  # 记录本次运行中已构建的图

    for dataset in DATASETS:
        data_path = os.path.join(DATASET_ROOT, dataset['base'])
        if not os.path.exists(data_path):
            print(f"Warning: Dataset file not found: {data_path}. Skipping...")
            continue

        # 处理任务
        for combo in combinations_normal:
            params = dict(zip(keys, combo))

            cmd = build_command(dataset, params)

            graph_path = get_graph_path(dataset['name'], params)
            graph_key = (dataset['name'], params['m'], params['ef_construction'], params['limited_level'])

            if os.path.exists(graph_path) or graph_key in built_graphs:
                # 图已存在或本次已构建，加载已有图
                cmd += ["--graph_path", graph_path]
                description = f"Dataset: {dataset['name']} | Load graph | Params: {params}"
            else:
                # 首次构建，保存图以便后续复用
                built_graphs.add(graph_key)
                cmd.append("--save")
                description = f"Dataset: {dataset['name']} | Build graph | Params: {params}"

            tasks.append((cmd, description, dataset, params))

    total_experiments = len(tasks)
    print("=" * 60)
    print(f"Total experiments: {total_experiments}")
    print(f"Parallel workers: {MAX_WORKERS}")
    print("=" * 60)

    # 第一次运行所有任务
    failed_tasks = []
    if MAX_WORKERS > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [
                executor.submit(run_worker, cmd, i + 1, total_experiments, desc)
                for i, (cmd, desc, _, _) in enumerate(tasks)
            ]
            concurrent.futures.wait(futures)
    else:
        for i, (cmd, desc, _, _) in enumerate(tasks):
            run_worker(cmd, i + 1, total_experiments, desc)

    # 检查哪些任务没有生成结果文件
    print("\n" + "=" * 60)
    print("Checking results...")
    print("=" * 60)

    missing_results = []
    for cmd, desc, dataset, params in tasks:
        result_path = get_result_path(dataset['name'], params)
        if not os.path.exists(result_path):
            print(f"MISSING: {result_path}")
            missing_results.append((cmd, desc))
        else:
            # 检查文件是否为空或损坏
            try:
                with open(result_path, 'r') as f:
                    content = f.read().strip()
                    if not content or content == '{}':
                        print(f"EMPTY: {result_path}")
                        missing_results.append((cmd, desc))
            except Exception as e:
                print(f"ERROR reading {result_path}: {e}")
                missing_results.append((cmd, desc))

    # 重新运行缺失结果的任务
    if missing_results:
        print(f"\n{'=' * 60}")
        print(f"Re-running {len(missing_results)} missing results...")
        print(f"{'=' * 60}")

        for i, (cmd, desc) in enumerate(missing_results):
            run_worker(cmd, i + 1, len(missing_results), f"[RETRY] {desc}")

        # 再次检查
        print(f"\n{'=' * 60}")
        print("Final check...")
        print(f"{'=' * 60}")

        still_missing = []
        for cmd, desc, dataset, params in tasks:
            result_path = get_result_path(dataset['name'], params)
            if not os.path.exists(result_path):
                still_missing.append(result_path)

        if still_missing:
            print(f"WARNING: {len(still_missing)} results still missing after retry:")
            for path in still_missing:
                print(f"  - {path}")
        else:
            print("All results generated successfully!")
    else:
        print("All results generated successfully!")


if __name__ == "__main__":
    run_batch()
