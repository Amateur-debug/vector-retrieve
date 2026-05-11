import itertools
import argparse
import os
import time
from types import SimpleNamespace

from draw import run_draw

# ============================================================
# 节点配置：每个数据集的 (node_num, query_threads) 列表
# ============================================================
DATASET_NODE_CONFIG = {
    "sift":        [(5, 40)],
    "sift100m":    [(4, 32), (5, 40), (8, 64)],
    "sift100m-u8": [(8, 64), (16, 128)],
}

# ============================================================
# 任务模板 A：每个 (dataset, node_num) 扩展一次
# node_num / num_query_threads 由 _make_task_list 动态注入
# ============================================================
PER_NODE_TASKS = [
    # --- compare: recall vs qps ---
    {
        "chart": "line", "mode": "compare",
        "grid": {"x": ["avg_recall@10"], "y": ["qps"],
                 "replication": [2], "threads": [8]},
    },
    {
        "chart": "bar", "mode": "compare",
        "grid": {"x": ["avg_recall@10"], "y": ["qps"],
                 "replication": [2], "threads": [8]},
    },
    # --- compare: recall vs avg_distance_cnt (shard + global)
    #     layer (k-means) 当作 'Global'，no-layer 排除；cotra/weaviate 不画 ---
    {
        "chart": "line", "mode": "compare",
        "grid": {"x": ["avg_recall@10"], "y": ["avg_distance_cnt"],
                 "replication": [2],
                 "global": [True], "cotra": [False], "weaviate": [False]},
    },
    {
        "chart": "bar", "mode": "compare",
        "grid": {"x": ["avg_recall@10"], "y": ["avg_distance_cnt"],
                 "replication": [2],
                 "global": [True], "cotra": [False], "weaviate": [False]},
    },
    # --- compare: recall vs avg_remote_transfer_cnt ---
    {
        "chart": "line", "mode": "compare",
        "grid": {"x": ["avg_recall@10"], "y": ["avg_remote_transfer_cnt"],
                 "replication": [2]},
    },
    {
        "chart": "bar", "mode": "compare",
        "grid": {"x": ["avg_recall@10"], "y": ["avg_remote_transfer_cnt"],
                 "replication": [2]},
    },
    # --- compare: recall vs communication count (Ours + CoTra only), layer ---
    {
        "chart": "bar", "mode": "compare",
        "grid": {"x": ["avg_recall@10"], "y": ["avg_remote_transfer_cnt"],
                 "replication": [0, 1, 2], "partition_mode": ["k-means"], "layer": [True],
                 "shard": [False], "no_layer": [False], "global": [False],
                 "weaviate": [False], "cotra": [True],
                 "label_overrides": [{"layer": "Ours"}],
                 "output_tag": ["ours"]},
    },
    # --- compare: recall vs communication count (Ours + CoTra only), no-layer ---
    {
        "chart": "bar", "mode": "compare",
        "grid": {"x": ["avg_recall@10"], "y": ["avg_remote_transfer_cnt"],
                 "replication": [0, 1, 2], "partition_mode": ["k-means"], "layer": [False],
                 "shard": [False], "no_layer": [True], "global": [False],
                 "weaviate": [False], "cotra": [True],
                 "label_overrides": [{"no-layer": "Ours"}],
                 "output_tag": ["ours"]},
    },
    # --- compare: ef_search vs qps ---
    {
        "chart": "line", "mode": "compare",
        "grid": {"x": ["ef_search"], "y": ["qps"],
                 "m": [10, 15, 20, 25, 30, 35, 40],
                 "ef_construction": [200], "limited_level": [5],
                 "replication": [2], "threads": [8]},
    },
    # --- compare: ef_search vs avg_remote_transfer_cnt ---
    {
        "chart": "line", "mode": "compare",
        "grid": {"x": ["ef_search"], "y": ["avg_remote_transfer_cnt"],
                 "m": [10, 15, 20, 25, 30, 35, 40],
                 "ef_construction": [200], "limited_level": [5],
                 "replication": [2], "threads": [8]},
    },
    # --- compare: threads vs qps ---
    {
        "chart": "line", "mode": "compare",
        "grid": {"x": ["threads"], "y": ["qps"],
                 "m": [10], "ef_construction": [200], "limited_level": [5],
                 "replication": [2]},
    },
    # --- compare: replication vs avg_remote_transfer_cnt ---
    {
        "chart": "bar", "mode": "compare",
        "grid": {"x": ["replication"], "y": ["avg_remote_transfer_cnt"],
                 "m": [10], "ef_construction": [200], "ef_search": [15],
                 "limited_level": [5], "threads": [8]},
    },
    # --- compare: replication vs qps，按 recall 取最近实验（每个 recall 一张图）---
    {
        "chart": "bar", "mode": "compare",
        "grid": {"x": ["replication"], "y": ["qps"],
                 "threads": [8],
                 "recall": [0.85, 0.9, 0.95]},
    },
    # --- compare: replication vs avg_remote_transfer_data_per_transfer ---
    {
        "chart": "bar", "mode": "compare",
        "grid": {"x": ["replication"], "y": ["avg_remote_transfer_data_per_transfer"],
                 "m": [10], "ef_construction": [200], "ef_search": [15],
                 "limited_level": [5]},
    },
    # --- compare: replication vs avg_remote_transfer_data_per_query ---
    {
        "chart": "bar", "mode": "compare",
        "grid": {"x": ["replication"], "y": ["avg_remote_transfer_data_per_query"],
                 "m": [10], "ef_construction": [200], "ef_search": [15],
                 "limited_level": [5]},
    },
    # --- group: recall vs avg_remote_transfer_cnt by replication ---
    # layer=True → 只扫 layer 文件；layer=False → 只扫 no-layer 文件
    {
        "chart": "line", "mode": "group",
        "grid": {"x": ["avg_recall@10"], "y": ["avg_remote_transfer_cnt"],
                 "group_by": ["replication"], "threads": [8],
                 "layer": [True, False]},
    },
]

# ============================================================
# 任务模板 B：每个 dataset 只扩展一次（跨节点数量对比）
# node_num 不过滤（由 group_by 收集所有节点）
# query_threads 不过滤（各节点使用自身线程数）
# ============================================================
NODE_COMPARE_TASKS = [
    # --- node compare: recall vs qps (line) ---
    {
        "chart": "line", "mode": "group",
        "grid": {"x": ["avg_recall@10"], "y": ["qps"],
                 "group_by": ["node_num"],
                 "replication": [2], "threads": [8], "layer": [True, False]},
    },
    # --- node compare: recall vs qps (bar) — 每个 recall 档位并排不同 node 的柱子 ---
    {
        "chart": "bar", "mode": "group",
        "grid": {"x": ["avg_recall@10"], "y": ["qps"],
                 "group_by": ["node_num"],
                 "replication": [2], "threads": [8], "layer": [True, False]},
    },
    # --- node compare: recall vs avg_distance_cnt (line) ---
    {
        "chart": "line", "mode": "group",
        "grid": {"x": ["avg_recall@10"], "y": ["avg_distance_cnt"],
                 "group_by": ["node_num"],
                 "replication": [2], "layer": [True, False]},
    },
    # --- node compare: recall vs avg_distance_cnt (bar) ---
    {
        "chart": "bar", "mode": "group",
        "grid": {"x": ["avg_recall@10"], "y": ["avg_distance_cnt"],
                 "group_by": ["node_num"],
                 "replication": [2], "layer": [True, False]},
    },
    # --- node compare: recall vs avg_remote_transfer_cnt (line) ---
    {
        "chart": "line", "mode": "group",
        "grid": {"x": ["avg_recall@10"], "y": ["avg_remote_transfer_cnt"],
                 "group_by": ["node_num"],
                 "replication": [2], "layer": [True, False]},
    },
    # --- node compare: recall vs avg_remote_transfer_cnt (bar) ---
    {
        "chart": "bar", "mode": "group",
        "grid": {"x": ["avg_recall@10"], "y": ["avg_remote_transfer_cnt"],
                 "group_by": ["node_num"],
                 "replication": [2], "layer": [True, False]},
    },
]


def _make_task_list(node_configs):
    """Expand PER_NODE_TASKS for each node config, then append NODE_COMPARE_TASKS."""
    tasks = []
    for node_num, query_threads in node_configs:
        for tmpl in PER_NODE_TASKS:
            task = {
                "chart": tmpl["chart"],
                "mode": tmpl["mode"],
                "grid": {**tmpl["grid"],
                         "node_num": [node_num],
                         "num_query_threads": [query_threads]},
            }
            tasks.append(task)
    tasks.extend(NODE_COMPARE_TASKS)
    return tasks


DRAW_TASKS_BY_DATASET = {
    name: _make_task_list(node_configs)
    for name, node_configs in DATASET_NODE_CONFIG.items()
}


# 默认参数（与 draw.py 的 argparse 默认值对齐）
DEFAULTS = {
    "json_path": "./results/cpp",
    "weaviate_path": "./weaviate/results/",
    "output_dir": "./plot",
    "chart": "line",
    "mode": "compare",
    "x": "ef_search",
    "y": "qps",
    "data_name": "sift",
    "m": None,
    "ef_construction": None,
    "ef_search": None,
    "limited_level": None,
    "node_num": 5,
    "partition_mode": "k-means",
    "layer": True,
    "no_layer": True,
    "replication": None,
    "threads": None,
    "num_query_threads": None,
    "shard": True,
    "global": False,
    "cotra": True,
    "weaviate": True,
    "distributed": False,
    "min_recall": 0.8,
    "recall": None,
    "group_by": None,
    "debug": False,
    "split": False,
    "vertical": False,
    "label_overrides": None,
    "output_tag": None,
}


def make_args(chart, mode, params, debug=False):
    """构建 SimpleNamespace 参数对象（直接传给 run_draw）"""
    d = dict(DEFAULTS)
    d["chart"] = chart
    d["mode"] = mode
    d["debug"] = debug
    if "partition_mode" in params:
        d["filter_partition_mode"] = params["partition_mode"]
    if "layer" in params:
        d["filter_layer"] = params["layer"]
    d.update(params)
    # Route output to ./plot/<data_name>/{pdf,png}
    if d.get("data_name"):
        d["output_dir"] = os.path.join(DEFAULTS["output_dir"], d["data_name"])
    return SimpleNamespace(**d)


def run_batch(args):
    """批量画图"""
    tasks = []
    selected = args.dataset if getattr(args, "dataset", None) else list(DRAW_TASKS_BY_DATASET.keys())
    for data_name in selected:
        task_list = DRAW_TASKS_BY_DATASET.get(data_name)
        if task_list is None:
            print(f"[WARN] no task list for dataset '{data_name}', skipping")
            continue
        for task_def in task_list:
            chart = task_def["chart"]
            mode = task_def["mode"]
            grid = task_def["grid"]

            keys = list(grid.keys())
            combinations = list(itertools.product(*[grid[k] for k in keys]))

            for combo in combinations:
                params = dict(zip(keys, combo))
                params["data_name"] = data_name
                draw_args = make_args(chart, mode, params, debug=args.debug)
                desc = f"[{data_name}] {chart}/{mode} | " + " | ".join(
                    f"{k}={v}" for k, v in params.items() if k in ("x", "y"))
                tasks.append((draw_args, desc))

    print("=" * 60)
    print(f"Total plots to generate: {len(tasks)}")
    print("=" * 60)

    for i, (draw_args, desc) in enumerate(tasks):
        task_id = i + 1
        print(f"[{task_id}/{len(tasks)}] {desc}")
        start_time = time.time()
        try:
            run_draw(draw_args)
        except Exception as e:
            print(f"[{task_id}/{len(tasks)}] ERROR: {e}")
        print(f"[{task_id}/{len(tasks)}] FINISHED in {time.time() - start_time:.2f}s")

    print("=" * 60)
    print("All plots generated!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch draw plots")
    parser.add_argument("--debug", action='store_true',
                        help="Pass --debug to each draw call")
    parser.add_argument("--dataset", nargs="+", default=None,
                        choices=list(DRAW_TASKS_BY_DATASET.keys()),
                        help="Only run for these data_names (default: all in DRAW_TASKS_BY_DATASET)")
    args = parser.parse_args()

    run_batch(args)
