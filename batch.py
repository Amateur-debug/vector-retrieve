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
    "redundancy":         [2],
    "m":                  [16],
    "ef_construction":    [200],
    "ef_search":          [32],
    "limited_level":      [5],
    "k":                  [10],
    "num_query_threads":  [128],
    "reduce":             [False, True],
    "distributed":        [True],
    "layer":              [True, False],
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

    return cmd


def get_graph_path(data_name, params):
    """生成图文件路径"""
    filename = f"hnsw-{data_name}-m{params['m']}-efc{params['ef_construction']}-ll{params['limited_level']}.json"
    return os.path.join(".", "save", filename)


def run_batch():
    """批量运行实验"""
    # 原始参数顺序
    keys = list(PARAM_GRID.keys())

    # 为 reduce 模式生成参数组合（redundancy 只需一个值，因为 reduce 模式不分区）
    # 同时排除 layer（reduce 模式下 layer 无效）
    param_grid_reduce = {k: v for k, v in PARAM_GRID.items() if k not in ('redundancy', 'layer')}
    param_grid_reduce['redundancy'] = [0]  # reduce 模式只需一个值
    param_grid_reduce['reduce'] = [True]
    param_grid_reduce['layer'] = [False]  # reduce 模式下 layer 无效

    # 为非 reduce 模式生成参数组合
    param_grid_normal = {k: v for k, v in PARAM_GRID.items() if k != 'reduce'}
    param_grid_normal['reduce'] = [False]

    # 确保 key 顺序与原始 PARAM_GRID 一致
    param_grid_reduce = {k: param_grid_reduce.get(k, [False]) for k in keys}
    param_grid_normal = {k: param_grid_normal.get(k, [False]) for k in keys}

    combinations_reduce = list(itertools.product(*param_grid_reduce.values()))
    combinations_normal = list(itertools.product(*param_grid_normal.values()))

    tasks = []
    built_graphs = set()  # 记录本次运行中已构建的图

    for dataset in DATASETS:
        data_path = os.path.join(DATASET_ROOT, dataset['base'])
        if not os.path.exists(data_path):
            print(f"Warning: Dataset file not found: {data_path}. Skipping...")
            continue

        for combo in combinations_reduce + combinations_normal:
            params = dict(zip(keys, combo))

            cmd = build_command(dataset, params)

            graph_path = get_graph_path(dataset['name'], params)
            graph_key = (dataset['name'], params['m'], params['ef_construction'], params['limited_level'])

            # reduce 模式每个节点独立建图，不支持加载已有图
            if params.get('reduce'):
                description = f"Dataset: {dataset['name']} | Build (reduce) | Params: {params}"
            elif os.path.exists(graph_path) or graph_key in built_graphs:
                # 图已存在或本次已构建，加载已有图
                cmd += ["--graph_path", graph_path]
                description = f"Dataset: {dataset['name']} | Load graph | Params: {params}"
            else:
                # 首次构建，保存图以便后续复用
                built_graphs.add(graph_key)
                cmd.append("--save")
                description = f"Dataset: {dataset['name']} | Build graph | Params: {params}"

            tasks.append((cmd, description))

    total_experiments = len(tasks)
    print("=" * 60)
    print(f"Total experiments: {total_experiments}")
    print(f"Parallel workers: {MAX_WORKERS}")
    print("=" * 60)

    if MAX_WORKERS > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [
                executor.submit(run_worker, cmd, i + 1, total_experiments, desc)
                for i, (cmd, desc) in enumerate(tasks)
            ]
            concurrent.futures.wait(futures)
    else:
        for i, (cmd, desc) in enumerate(tasks):
            run_worker(cmd, i + 1, total_experiments, desc)


if __name__ == "__main__":
    run_batch()
