import subprocess
import itertools
import argparse
import concurrent.futures
import time

SCRIPT_PATH = "draw.py"

# ============================================================
# 画图任务配置
# 每个任务定义 chart + mode + 固定参数，grid 中的参数取笛卡尔积
# ============================================================
DRAW_TASKS = [
    # --- compare 模式：三路对比折线图，threads vs qps ---
    {
        "chart": "line",
        "mode": "compare",
        "grid": {
            "x":            ["threads"],
            "y":            ["qps"],
            "data_name":    ["sift"],
            "m":            [10],
            "ef_construction": [200],
            "limited_level": [5],
            "node_num":     [5],
            "redundancy":   [2],
        },
    },
    # --- compare 模式：三路对比柱状图，threads vs qps ---
    {
        "chart": "bar",
        "mode": "compare",
        "grid": {
            "x":            ["threads"],
            "y":            ["qps"],
            "data_name":    ["sift"],
            "m":            [10],
            "ef_construction": [200],
            "limited_level": [5],
            "node_num":     [5],
            "redundancy":   [2],
        },
    },
]

# 并行进程数
MAX_WORKERS = 4


def build_command(chart, mode, params):
    """根据参数构建 draw.py 命令"""
    cmd = ["python", SCRIPT_PATH, "--chart", chart, "--mode", mode]
    for key, val in params.items():
        cmd += [f"--{key}", str(val)]
    return cmd


def run_worker(cmd, task_id, total_tasks, description):
    """运行单个画图任务"""
    print(f"[{task_id}/{total_tasks}] STARTING: {description}")
    start_time = time.time()

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        for line in result.stdout.strip().split('\n'):
            if 'saved to' in line.lower():
                print(f"[{task_id}/{total_tasks}] {line}")
        if result.returncode != 0:
            stderr = result.stderr.strip()
            if stderr:
                print(f"[{task_id}/{total_tasks}] ERROR: {stderr}")
    except Exception as e:
        print(f"[{task_id}/{total_tasks}] ERROR: {e}")

    elapsed = time.time() - start_time
    print(f"[{task_id}/{total_tasks}] FINISHED in {elapsed:.2f}s")


def run_batch(args):
    """批量画图"""
    tasks = []
    for task_def in DRAW_TASKS:
        chart = task_def["chart"]
        mode = task_def["mode"]
        grid = task_def["grid"]

        keys = list(grid.keys())
        combinations = list(itertools.product(*[grid[k] for k in keys]))

        for combo in combinations:
            params = dict(zip(keys, combo))
            cmd = build_command(chart, mode, params)
            desc = f"{chart}/{mode} | " + " | ".join(
                f"{k}={v}" for k, v in params.items() if k in ("x", "y", "ef_search"))
            tasks.append((cmd, desc))

    print("=" * 60)
    print(f"Total plots to generate: {len(tasks)}")
    print(f"Parallel workers: {args.workers}")
    print("=" * 60)

    if args.dry_run:
        for i, (cmd, desc) in enumerate(tasks):
            print(f"[{i+1}/{len(tasks)}] {' '.join(cmd)}")
        return

    if args.workers > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(run_worker, cmd, i + 1, len(tasks), desc)
                for i, (cmd, desc) in enumerate(tasks)
            ]
            concurrent.futures.wait(futures)
    else:
        for i, (cmd, desc) in enumerate(tasks):
            run_worker(cmd, i + 1, len(tasks), desc)

    print("=" * 60)
    print("All plots generated!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch draw plots")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS,
                        help="Number of parallel workers")
    parser.add_argument("--dry_run", action='store_true',
                        help="Only print commands without running")
    args = parser.parse_args()

    run_batch(args)
