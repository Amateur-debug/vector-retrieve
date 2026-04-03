import subprocess
import os
import itertools
import time
import argparse
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
    # {
    #     "name": "sift",
    #     "base": "sift/sift_base.fvecs",
    #     "query": "sift/sift_query.fvecs",
    #     "groundtruth": "sift/sift_groundtruth.ivecs",
    # },
    {
        "name": "sift100m",
        "base": "sift100m/base.u8bin",
        "query": "sift100m/queries.u8bin",
        "groundtruth": "sift100m/gt_100.bin",
    }
]

# ============================================================
# 参数网格 (将运行所有笛卡尔积组合)
# ============================================================
PARAM_GRID = {
    "node_num":           [5],
    "partition_mode":     ["k-means"],
    "replication":        [2, 1, 0],
    "m":                  [10, 15, 20, 25, 30, 35, 40],
    "ef_construction":    [200],
    "ef_search":          [15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
    "limited_level":      [5],
    "k":                  [10],
    "num_query_threads":  [40],      # 客户端查询进程数
    "num_threads":        [8],       # 服务端每 node 搜索线程数
    "shard":              [False, True],
    "distributed":        [True],
    "layer":              [True, False],
    "comm_mode":          ["tcp"],  # 通信模式："tcp" 或 "rdma"
    "num_cpu":            [0],       # CPU 核心数
}

# 等待 graph 文件的最长秒数 (0 = 无限等待)
GRAPH_WAIT_TIMEOUT = 0
# 并行任务数 (1 = 顺序执行)
MAX_WORKERS = 1
# 每个实验重复运行次数（去掉最高最低取均值）
NUM_RUNS = 5


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


def build_command(dataset, params, skip_prepare=False, no_cleanup=False):
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
        "--num_threads", str(params['num_threads']),
        "--node_num", str(params['node_num']),
        "--partition_mode", params['partition_mode'],
        "--replication", str(params['replication']),
    ]

    if params.get('shard'):
        cmd.append('--shard')
    if params.get('distributed'):
        cmd.append('--distributed')
    if params.get('layer'):
        cmd.append('--layer')
    if params.get('comm_mode') and params['comm_mode'] != 'tcp':
        cmd += ['--comm_mode', str(params['comm_mode'])]
    if params.get('num_cpu'):
        cmd += ['--num_cpu', str(params['num_cpu'])]
    cmd += ['--num_runs', str(NUM_RUNS)]
    if skip_prepare:
        cmd.append('--skip_prepare')
    if no_cleanup:
        cmd.append('--no_cleanup')

    return cmd


def get_graph_path(data_name, params):
    """生成图文件路径"""
    filename = f"hnsw-{data_name}-m{params['m']}-efc{params['ef_construction']}-ll{params['limited_level']}.json"
    return os.path.join(".", "save", filename)


def get_result_path(dataset_name, params):
    """生成结果文件路径（与 main.py 的 save_results 对齐）"""
    distributed_suffix = "-distributed" if params.get('distributed') else ""
    cpus = params.get('num_cpu', 0)
    cpu_suffix = f"-cpus{cpus}" if params.get('distributed') and cpus != 0 else ""
    qt_suffix = f"-query_threads{params['num_query_threads']}"

    if params.get('shard'):
        filename = (f"results-{dataset_name}-m{params['m']}-efc{params['ef_construction']}-"
                    f"efs{params['ef_search']}-ll{params['limited_level']}-"
                    f"node{params['node_num']}-"
                    f"threads{params['num_threads']}{qt_suffix}{distributed_suffix}{cpu_suffix}-shard.json")
    else:
        layer_suffix = "-layer" if params.get('layer') else ""
        filename = (f"results-{dataset_name}-m{params['m']}-efc{params['ef_construction']}-"
                    f"efs{params['ef_search']}-ll{params['limited_level']}-"
                    f"node{params['node_num']}-{params['partition_mode']}-rep{params['replication']}-"
                    f"threads{params['num_threads']}{qt_suffix}{distributed_suffix}{cpu_suffix}{layer_suffix}.json")

    return os.path.join(".", "results", "cpp", filename)


def get_group_key(dataset_name, params):
    """生成分组键：相同 (dataset, m, efc, ll, node_num, partition_mode, replication, layer, ...) 的任务共享服务端"""
    return (
        dataset_name,
        params['m'],
        params['ef_construction'],
        params['limited_level'],
        params['node_num'],
        params['partition_mode'],
        params.get('replication', 0),
        params.get('layer', False),
        params.get('shard', False),
        params.get('distributed', False),
        params.get('comm_mode', 'tcp'),
        params.get('num_threads', 8),
        params.get('num_query_threads', 40),
        params.get('num_cpu', 0),
    )


def run_batch():
    """批量运行实验，相同分区配置的任务复用服务端，只变 ef_search"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip', action='store_true', help='跳过已有结果文件的任务')
    parser.add_argument('--offset', type=int, default=0, help='跳过前N个任务')
    batch_args = parser.parse_args()

    keys = list(PARAM_GRID.keys())
    combinations = list(itertools.product(*[PARAM_GRID[k] for k in keys]))

    # 按分组键聚合任务
    from collections import OrderedDict
    groups = OrderedDict()  # group_key -> [(dataset, params, efs), ...]
    built_graphs = set()
    skipped = 0

    for dataset in DATASETS:
        data_path = os.path.join(DATASET_ROOT, dataset['base'])
        if not os.path.exists(data_path):
            print(f"Warning: Dataset file not found: {data_path}. Skipping...")
            continue

        for combo in combinations:
            params = dict(zip(keys, combo))

            if batch_args.skip and os.path.exists(get_result_path(dataset['name'], params)):
                skipped += 1
                continue

            gk = get_group_key(dataset['name'], params)
            if gk not in groups:
                groups[gk] = []
            groups[gk].append((dataset, dict(params)))

    total_runs = sum(len(items) for items in groups.values())
    print("=" * 60)
    if batch_args.skip:
        print(f"Skipped (results exist): {skipped}")
    print(f"Total groups: {len(groups)} (covering {total_runs} ef_search runs)")
    print("=" * 60)

    task_idx = 0
    for gk, items in groups.items():
        # 第一个任务：正常启动服务端 + 建图 + partition，不清理
        # 后续任务：跳过 prepare，不清理（最后一个除外）
        group_started = False  # 当前组是否已有任务实际运行（用于处理 offset 跳过后的首任务）
        for i, (dataset, params) in enumerate(items):
            is_last = (i == len(items) - 1)
            task_idx += 1

            if task_idx <= batch_args.offset:
                print(f"[{task_idx}/{total_runs}] SKIPPED (offset)")
                continue

            is_first = not group_started  # 组内第一个实际运行的任务
            group_started = True

            cmd = build_command(dataset, params,
                                skip_prepare=not is_first,
                                no_cleanup=not is_last)

            # 组内首个实际运行的任务处理 graph 加载/构建
            if is_first:
                graph_path = get_graph_path(dataset['name'], params)
                graph_key = (dataset['name'], params['m'], params['ef_construction'], params['limited_level'])
                if os.path.exists(graph_path) or graph_key in built_graphs:
                    cmd += ["--graph_path", graph_path]
                else:
                    built_graphs.add(graph_key)
                    cmd.append("--save")

            efs = params['ef_search']
            layer_str = "layer" if params.get('layer') else "non-layer"
            mode_str = "shard" if params.get('shard') else f"rep={params.get('replication',0)}"
            flags = []
            if not is_first:
                flags.append("skip_prepare")
            if not is_last:
                flags.append("no_cleanup")
            flag_str = f" [{','.join(flags)}]" if flags else ""
            description = (f"m={params['m']} {mode_str} {layer_str} "
                           f"efs={efs}{flag_str}")

            run_worker(cmd, task_idx, total_runs, description)

    # 检查结果
    print("\n" + "=" * 60)
    print("Checking results...")
    print("=" * 60)

    all_missing = []
    for gk, items in groups.items():
        for dataset, params in items:
            result_path = get_result_path(dataset['name'], params)
            if not os.path.exists(result_path):
                print(f"MISSING: {result_path}")
                all_missing.append(result_path)

    if all_missing:
        print(f"WARNING: {len(all_missing)} results missing")
        for path in all_missing:
            print(f"  - {path}")
    else:
        print("All results generated successfully!")


if __name__ == "__main__":
    run_batch()
