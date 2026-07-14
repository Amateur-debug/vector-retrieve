import subprocess
import os
import itertools
import time
import argparse

SCRIPT_PATH = "main.py"
DATASET_ROOT = "/hpc2hdd/home/xchen740/workspace/vector-retrieve/dataset"

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
        "dtype": "uint8",
    },
    {
        "name": "deep100m",
        "base": "deep100m/base.fbin",
        "query": "deep100m/queries.fbin",
        "groundtruth": "deep100m/gt_100.bin",
        "dtype": "float",
    },
]

# ============================================================
# 参数网格 (将运行所有笛卡尔积组合)
# ============================================================
def _get_query_threads(node_num, num_threads):
    """query 线程数 = node_num × num_threads"""
    return node_num * num_threads


PARAM_GRID = {
    "node_num":           [8, 16],
    "partition_mode":     ["k-means", "fennel", "kmeans-fennel"],
    "replication":        [2, 1.5, 1, 0],
    "m":                  [20, 25, 30, 35, 40],
    "ef_construction":    [200],
    "ef_search":          [20, 25, 30, 35, 40, 45, 55, 60],
    "limited_level":      [5],
    "k":                  [10],
    "num_query_threads":  None,      # 由 node_num × num_threads 推导（8→64, 16→128）
    "num_threads":        [8],       # 服务端每 node 搜索线程数
    "shard":              [False],
    "global_pull":        [False],
    "distributed":        [True],
    "layer":              [True, False],
    "comm_mode":          ["tcp"],
    "num_cpu":            [0],
    "dtype":              [None],      # None = 使用 DATASETS 中的默认 dtype
}

# 仅跑接近 recall=0.80/0.85/0.90/0.95 的最高 QPS 配置（覆盖 node8/16 × layer T/F 的 union）
# 置空集合 set() 可关闭过滤，跑全部 (m, ef_search) 组合
ALLOWED_M_EFS = {
    (20, 35), (25, 40), (25, 55), (30, 25),
    (30, 45), (35, 30), (40, 20), (40, 60),
}

# uint8 模式下 data_name 追加的后缀，保证 graph/results 文件名与 float 不冲突
DTYPE_NAME_SUFFIX = {"uint8": "-u8"}


def effective_dtype(dataset, params):
    """解析实际 dtype：参数显式指定优先，否则使用数据集默认值。"""
    return params.get('dtype') or dataset.get('dtype', 'float')

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


def effective_data_name(dataset, params):
    """根据 dtype 给 data_name 加后缀，避免 float/uint8 结果互相覆盖。
    main.py 用 --data_name 决定 graph 与 results 文件名，所以后缀加在这里即可。"""
    return dataset['name'] + DTYPE_NAME_SUFFIX.get(effective_dtype(dataset, params), "")


def resolve_query_threads(params):
    """解析 num_query_threads：若为 None 则从 node_num * num_threads 推导"""
    qt = params.get('num_query_threads')
    if qt is not None:
        return qt
    return _get_query_threads(params['node_num'], params['num_threads'])


def resolve_query_nodes(node_num):
    """根据 worker 节点数推导 query 节点数：8→2, 16→4"""
    if node_num <= 8:
        return 2
    else:
        return 4


def build_command(dataset, params, skip_prepare=False, no_cleanup=False, use_slurm=False, save=False):
    """构建命令行参数"""
    data_name = effective_data_name(dataset, params)
    data_path = os.path.join(DATASET_ROOT, dataset['base'])
    query_path = os.path.join(DATASET_ROOT, dataset['query'])
    groundtruth_path = os.path.join(DATASET_ROOT, dataset['groundtruth'])

    cmd = [
        # -u: 关掉 stdout block-buffer，否则 main.py 进度日志会卡在 Python 缓冲区
        # 直到填满 4-8KB 才刷出，slurm 日志看起来像"卡住"
        "python", "-u", SCRIPT_PATH,
        "--data_name", data_name,
        "--data_path", data_path,
        "--query_path", query_path,
        "--groundtruth_path", groundtruth_path,
        "--k", str(params['k']),
        "--m", str(params['m']),
        "--ef_construction", str(params['ef_construction']),
        "--ef_search", str(params['ef_search']),
        "--limited_level", str(params['limited_level']),
        "--num_query_threads", str(resolve_query_threads(params)),
        "--num_threads", str(params['num_threads']),
        "--node_num", str(params['node_num']),
        "--num_query_nodes", str(resolve_query_nodes(params['node_num'])),
        "--partition_mode", params['partition_mode'],
        "--replication", str(params['replication']),
    ]

    if params.get('shard'):
        cmd.append('--shard')
    if params.get('global_pull'):
        cmd.append('--global-pull')
    if params.get('distributed'):
        cmd.append('--distributed')
    if params.get('layer'):
        cmd.append('--layer')
    if params.get('comm_mode') and params['comm_mode'] != 'tcp':
        cmd += ['--comm_mode', str(params['comm_mode'])]
    if params.get('num_cpu'):
        cmd += ['--num_cpu', str(params['num_cpu'])]
    cmd += ['--dtype', effective_dtype(dataset, params)]
    cmd += ['--num_runs', str(params.get('num_runs', NUM_RUNS))]
    cmd += ['--num_construction_threads', str(params.get('num_construction_threads', 64))]
    cmd += ['--partition_distribution', params.get('partition_distribution', 'nfs-load')]
    if params.get('graph_path'):
        cmd += ['--graph_path', params['graph_path']]
    if save:
        cmd.append('--save')
    if skip_prepare:
        cmd.append('--skip_prepare')
    if no_cleanup:
        cmd.append('--no_cleanup')
    if use_slurm:
        cmd.append('--slurm')

    return cmd


def get_graph_path(dataset, params):
    """生成图文件路径（与 main.py save_graph 对齐：分布式模式保存为 .bin）"""
    data_name = effective_data_name(dataset, params)
    filename = f"hnsw-{data_name}-m{params['m']}-efc{params['ef_construction']}-ll{params['limited_level']}.bin"
    return os.path.join(".", "save", filename)


def get_partition_base(dataset, params):
    """生成分区图基础路径（与 main.py prepare() 中默认 cluster_base 对齐）。
    实际分区文件为 base-0.bin, base-1.bin, ..., base-(node_num-1).bin"""
    data_name = effective_data_name(dataset, params)
    layer_sfx = "-layer" if params.get('layer') else ""
    return os.path.join(
        ".", "save", data_name, f"node{params['node_num']}",
        f"m{params['m']}-efc{params['ef_construction']}-ll{params['limited_level']}-"
        f"{params['partition_mode']}-rep{params.get('replication', 0)}{layer_sfx}"
    )


def all_partitions_exist(dataset, params):
    """所有 node 的分区文件都存在则返回 True"""
    if params.get('shard'):
        return False  # shard 模式不保存统一分区
    base = get_partition_base(dataset, params)
    return all(os.path.exists(f"{base}-{i}.bin") for i in range(params['node_num']))


def get_result_path(dataset, params):
    """生成结果文件路径（与 main.py 的 save_results 对齐）"""
    data_name = effective_data_name(dataset, params)
    is_node1 = params.get('node_num') == 1
    distributed_suffix = "-distributed" if params.get('distributed') and not is_node1 else ""
    cpus = params.get('num_cpu', 0)
    cpu_suffix = f"-cpus{cpus}" if params.get('distributed') and cpus != 0 else ""
    qt_suffix = f"-query_threads{resolve_query_threads(params)}"

    if params.get('shard'):
        filename = (f"results-{data_name}-m{params['m']}-efc{params['ef_construction']}-"
                    f"efs{params['ef_search']}-ll{params['limited_level']}-"
                    f"node{params['node_num']}-"
                    f"threads{params['num_threads']}{qt_suffix}{distributed_suffix}{cpu_suffix}-shard.json")
    elif params.get('global_pull'):
        filename = (f"results-{data_name}-m{params['m']}-efc{params['ef_construction']}-"
                    f"efs{params['ef_search']}-ll{params['limited_level']}-"
                    f"node{params['node_num']}-{params['partition_mode']}-rep{params['replication']}-"
                    f"threads{params['num_threads']}{qt_suffix}{distributed_suffix}{cpu_suffix}-global_pull.json")
    else:
        if is_node1:
            filename = (f"results-{data_name}-m{params['m']}-efc{params['ef_construction']}-"
                        f"efs{params['ef_search']}-ll{params['limited_level']}-"
                        f"node{params['node_num']}-"
                        f"threads{params['num_threads']}{qt_suffix}{cpu_suffix}.json")
        else:
            layer_suffix = "-layer" if params.get('layer') else ""
            filename = (f"results-{data_name}-m{params['m']}-efc{params['ef_construction']}-"
                        f"efs{params['ef_search']}-ll{params['limited_level']}-"
                        f"node{params['node_num']}-{params['partition_mode']}-rep{params['replication']}-"
                        f"threads{params['num_threads']}{qt_suffix}{distributed_suffix}{cpu_suffix}{layer_suffix}.json")

    return os.path.join(".", "results", "cpp", filename)


def get_group_key(dataset, params):
    """生成分组键：相同 (dataset, m, efc, ll, node_num, partition_mode, replication, layer, dtype, ...) 的任务共享服务端。
    dtype 必须进入分组键：不同 dtype 的 HNSW 类不同，graph 不可复用。"""
    return (
        dataset['name'],
        params['m'],
        params['ef_construction'],
        params['limited_level'],
        params['node_num'],
        params['partition_mode'],
        params.get('replication', 0),
        params.get('layer', False),
        params.get('shard', False),
        params.get('global_pull', False),
        params.get('distributed', False),
        params.get('comm_mode', 'tcp'),
        params.get('num_threads', 8),
        resolve_query_threads(params),
        params.get('num_cpu', 0),
        effective_dtype(dataset, params),
    )


def run_batch():
    """批量运行实验，相同分区配置的任务复用服务端，只变 ef_search。
    在一个 Slurm allocation 内串行执行 PARAM_GRID 的笛卡尔积组合。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip', action='store_true', help='跳过已有结果文件的任务')
    parser.add_argument('--offset', type=int, default=0, help='跳过前N个任务')
    parser.add_argument('--slurm', action='store_true', help='传递 --slurm 给 main.py（用 srun 代替 ssh）')
    batch_args = parser.parse_args()

    # 过滤掉值为 None 的 key（其值由其他参数推导，不参与网格展开）
    grid = {k: v for k, v in PARAM_GRID.items() if v is not None}
    keys = list(grid.keys())
    combinations = list(itertools.product(*grid.values()))

    # 按分组键聚合任务
    from collections import OrderedDict
    groups = OrderedDict()  # group_key -> [(dataset, params), ...]
    built_graphs = set()
    skipped = 0

    for dataset in DATASETS:
        data_path = os.path.join(DATASET_ROOT, dataset['base'])
        if not os.path.exists(data_path):
            print(f"Warning: Dataset file not found: {data_path}. Skipping...")
            continue

        for combo in combinations:
            params = dict(zip(keys, combo))

            # shard 与 global_pull 互斥
            if params.get('shard') and params.get('global_pull'):
                continue
            # global_pull 模式下 replication 和 layer 无意义（图骨架完整，无需冗余/分层），仅保留 replication=0、layer=False 的组合
            if params.get('global_pull') and (params.get('replication', 0) != 0 or params.get('layer')):
                continue
            # shard 模式 result 文件名不含 partition_mode/rep/layer，仅保留 replication=0、layer=False 的组合避免重复
            if params.get('shard') and (params.get('replication', 0) != 0 or params.get('layer')):
                continue
            # 仅保留显式允许的 (m, ef_search) 组合
            if ALLOWED_M_EFS and (params['m'], params['ef_search']) not in ALLOWED_M_EFS:
                continue

            if batch_args.skip and os.path.exists(get_result_path(dataset, params)):
                skipped += 1
                continue

            gk = get_group_key(dataset, params)
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
                                no_cleanup=not is_last,
                                use_slurm=batch_args.slurm)

            # 组内首个实际运行的任务处理 graph 加载/构建
            if is_first:
                graph_path = get_graph_path(dataset, params)
                graph_key = (dataset['name'], effective_dtype(dataset, params),
                             params['m'], params['ef_construction'], params['limited_level'])
                if os.path.exists(graph_path) or graph_key in built_graphs:
                    cmd += ["--graph_path", graph_path]
                else:
                    built_graphs.add(graph_key)
                # 分区图缺失时统一加 --save：覆盖"图已建好但分区未保存"的场景
                # （main.py 检测到分区文件齐全会跳过 --save；齐全则无需重复保存）
                if not all_partitions_exist(dataset, params):
                    cmd.append("--save")

            efs = params['ef_search']
            layer_str = "layer" if params.get('layer') else "non-layer"
            if params.get('shard'):
                mode_str = "shard"
            elif params.get('global_pull'):
                mode_str = "global_pull"
            else:
                mode_str = f"rep={params.get('replication',0)}"
            flags = []
            if not is_first:
                flags.append("skip_prepare")
            if not is_last:
                flags.append("no_cleanup")
            flag_str = f" [{','.join(flags)}]" if flags else ""
            description = (f"dtype={effective_dtype(dataset, params)} m={params['m']} {mode_str} {layer_str} "
                           f"efs={efs}{flag_str}")

            run_worker(cmd, task_idx, total_runs, description)

    # 检查结果
    print("\n" + "=" * 60)
    print("Checking results...")
    print("=" * 60)

    all_missing = []
    for gk, items in groups.items():
        for dataset, params in items:
            result_path = get_result_path(dataset, params)
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
