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
    "m":                  [10, 20, 30, 40, 50, 60],
    "ef_construction":    [200],
    "ef_search":          [15, 20, 30, 40, 50],
    "k":                  [10],
    "num_query_threads":  [8, 32, 64],
    "num_cpu":            [8],
}

# 并行任务数 (1 = 顺序执行)
MAX_WORKERS = 1


def run_worker(cmd, task_id, total_tasks, description):
    """运行单个任务"""
    print(f"[{task_id}/{total_tasks}] STARTING: {description}")
    start_time = time.time()

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[{task_id}/{total_tasks}] ERROR: {e}")

    print(f"[{task_id}/{total_tasks}] FINISHED in {time.time() - start_time:.2f}s")


def build_command(dataset, params, deploy=False, build=False):
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
        "--node_num", str(params['node_num']),
        "--m", str(params['m']),
        "--ef_construction", str(params['ef_construction']),
        "--ef_search", str(params['ef_search']),
        "--k", str(params['k']),
        "--num_query_threads", str(params['num_query_threads']),
        "--num_cpu", str(params['num_cpu']),
    ]

    if deploy:
        cmd.append("--deploy")
    if build:
        cmd.append("--build")

    return cmd


def get_result_path(dataset_name, params):
    """生成结果文件路径（与 main.py 的 report() 保存路径对齐，不含时间戳）"""
    cpu_suffix = f"-cpus{params['num_cpu']}" if params.get('num_cpu', 0) != 0 else ""

    # 匹配 main.py 中的命名格式：results-{data_name}-m{m}-efc{efc}-efs{efs}-node{node}-threads{threads}[-cpus{cpu}]-{timestamp}.json
    # 这里使用前缀匹配（不含时间戳），用于检查是否已有结果
    prefix = (f"results-{dataset_name}-m{params['m']}-"
              f"efc{params['ef_construction']}-efs{params['ef_search']}-"
              f"node{params['node_num']}-"
              f"threads{params['num_query_threads']}{cpu_suffix}-")

    return prefix


def check_result_exists(dataset_name, params):
    """检查是否已存在匹配的结果文件（前缀匹配，因为文件名含时间戳）"""
    results_dir = "./results"
    if not os.path.exists(results_dir):
        return False

    prefix = get_result_path(dataset_name, params)
    for filename in os.listdir(results_dir):
        if filename.startswith(prefix) and filename.endswith(".json"):
            filepath = os.path.join(results_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    content = f.read().strip()
                    if content and content != '{}':
                        return True
            except Exception:
                pass
    return False


def find_result_file(dataset_name, params):
    """查找匹配的结果文件路径（用于最终检查）"""
    results_dir = "./results"
    if not os.path.exists(results_dir):
        return None

    prefix = get_result_path(dataset_name, params)
    for filename in os.listdir(results_dir):
        if filename.startswith(prefix) and filename.endswith(".json"):
            filepath = os.path.join(results_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    content = f.read().strip()
                    if content and content != '{}':
                        return filepath
            except Exception:
                pass
    return None


def run_batch():
    """批量运行实验"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Weaviate Batch Benchmark Runner")
    parser.add_argument('--skip', action='store_true', help='跳过已有结果文件的任务')
    batch_args = parser.parse_args()

    # 生成参数组合
    keys = list(PARAM_GRID.keys())
    combinations = list(itertools.product(*[PARAM_GRID[k] for k in keys]))

    tasks = []
    # Weaviate 无法 save/load 图文件，每次 m 或 efc 变化都需重新 deploy+build
    # 同一 (data_name, node_num, m, efc) 组合只需在首次任务时 deploy+build
    built_indices = set()
    skipped = 0

    # 先收集所有待运行的任务项（不含 deploy/build 标记）
    pending_tasks = []

    for dataset in DATASETS:
        data_path = os.path.join(DATASET_ROOT, dataset['base'])
        if not os.path.exists(data_path):
            print(f"Warning: Dataset file not found: {data_path}. Skipping...")
            continue

        for combo in combinations:
            params = dict(zip(keys, combo))

            # --skip 模式：跳过已有结果的任务
            if batch_args.skip:
                if check_result_exists(dataset['name'], params):
                    skipped += 1
                    continue

            pending_tasks.append((dataset, params))

    # 按 (data_name, node_num, m, efc) 排序，确保相同索引的任务连续执行，
    # 避免 Weaviate 反复 deploy+build（Weaviate 无法 save/load 图文件）
    pending_tasks.sort(key=lambda x: (
        x[0]['name'],
        x[1]['node_num'],
        x[1]['m'],
        x[1]['ef_construction'],
    ))

    # 为排序后的任务分配 deploy/build 标记
    for dataset, params in pending_tasks:
        index_key = (dataset['name'], params['node_num'], params['m'], params['ef_construction'])
        need_deploy = index_key not in built_indices
        need_build = index_key not in built_indices

        if need_deploy or need_build:
            built_indices.add(index_key)

        cmd = build_command(dataset, params, deploy=need_deploy, build=need_build)
        description = (f"Dataset: {dataset['name']} | "
                       f"node={params['node_num']} m={params['m']} efc={params['ef_construction']} "
                       f"efs={params['ef_search']} k={params['k']} "
                       f"threads={params['num_query_threads']} cpus={params['num_cpu']}"
                       f"{' | deploy+build' if need_deploy or need_build else ''}")

        tasks.append((cmd, description, dataset, params))

    total_experiments = len(tasks)
    print("=" * 60)
    if batch_args.skip:
        print(f"Skipped experiments (results exist): {skipped}")
    print(f"Total experiments to run: {total_experiments}")
    print(f"Parallel workers: {MAX_WORKERS}")
    print("=" * 60)

    # 运行所有任务
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
        result_file = find_result_file(dataset['name'], params)
        if not result_file:
            print(f"MISSING: {get_result_path(dataset['name'], params)}*.json")
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
            result_file = find_result_file(dataset['name'], params)
            if not result_file:
                still_missing.append(get_result_path(dataset['name'], params))

        if still_missing:
            print(f"WARNING: {len(still_missing)} results still missing after retry:")
            for path in still_missing:
                print(f"  - {path}*.json")
        else:
            print("All results generated successfully!")
    else:
        print("All results generated successfully!")


if __name__ == "__main__":
    run_batch()
