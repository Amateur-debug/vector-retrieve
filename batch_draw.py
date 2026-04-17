import itertools
import argparse
import os
import time
from types import SimpleNamespace

from draw import run_draw

# ============================================================
# 数据集总控：所有任务都会对每个 data_name 分别运行
# ============================================================
DATA_NAMES = ["sift100m"]

# ============================================================
# 画图任务配置
# 每个任务定义 chart + mode + 固定参数，grid 中的参数取笛卡尔积
# ============================================================
DRAW_TASKS = [
    # ==================== 论文用图 ====================
    # --- [论文] compare：replication vs avg_remote_transfer_cnt 柱状图（Fig 5.2）---
    {
        "chart": "bar",
        "mode": "compare",
        "grid": {
            "x":            ["replication"],
            "y":            ["avg_remote_transfer_cnt"],
            "m":            [10],
            "ef_construction": [200],
            "ef_search":    [15],
            "limited_level": [5],
            "node_num":     [5],
            "threads":      [8],
            "num_query_threads": [40],
        },
    },
    # --- [论文] compare：efs vs qps 折线图（Fig 5.3）---
    {
        "chart": "line",
        "mode": "compare",
        "grid": {
            "x":            ["ef_search"],
            "y":            ["qps"],
            "m":            [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
            "ef_construction": [200],
            "limited_level": [5],
            "node_num":     [5],
            "replication":   [0, 1, 2],
            "threads":      [8, 16, 32, 64],
            "num_query_threads": [40],
        },
    },
    # --- [论文] compare：efs vs avg_remote_transfer_cnt 折线图（Fig 5.4）---
    {
        "chart": "line",
        "mode": "compare",
        "grid": {
            "x":            ["ef_search"],
            "y":            ["avg_remote_transfer_cnt"],
            "m":            [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
            "ef_construction": [200],
            "limited_level": [5],
            "node_num":     [5],
            "replication":   [0, 1, 2],
            "threads":      [8, 16, 32, 64],
            "num_query_threads": [40],
        },
    },
    # --- [论文] compare：recall vs avg_distance_cnt 折线图 ---
    {
        "chart": "line",
        "mode": "compare",
        "grid": {
            "x":            ["avg_recall@10"],
            "y":            ["avg_distance_cnt"],
            "node_num":     [5],
            "replication":   [0, 1, 2],
        },
    },
    # --- [论文] compare：recall vs avg_distance_cnt 柱状图 ---
    {
        "chart": "bar",
        "mode": "compare",
        "grid": {
            "x":            ["avg_recall@10"],
            "y":            ["avg_distance_cnt"],
            "node_num":     [5],
            "replication":   [0, 1, 2],
        },
    },
    # --- [论文] compare：recall vs avg_remote_transfer_cnt 折线图 ---
    {
        "chart": "line",
        "mode": "compare",
        "grid": {
            "x":            ["avg_recall@10"],
            "y":            ["avg_remote_transfer_cnt"],
            "node_num":     [5],
            "replication":   [0, 1, 2],
        },
    },
    # --- [论文] compare：recall vs avg_remote_transfer_cnt 柱状图（Fig 1.2）---
    {
        "chart": "bar",
        "mode": "compare",
        "grid": {
            "x":            ["avg_recall@10"],
            "y":            ["avg_remote_transfer_cnt"],
            "node_num":     [5],
            "replication":   [0, 1, 2],
        },
    },
    # --- [论文] compare：recall vs avg_distance_cnt 柱状图（shard + global）（Fig 1.1）---
    {
        "chart": "bar",
        "mode": "compare",
        "grid": {
            "x":            ["avg_recall@10"],
            "y":            ["avg_distance_cnt"],
            "node_num":     [5],
            "replication":   [0, 1, 2],
            "threads":      [8, 16, 32, 64],
            "num_query_threads": [40],
            "global":       [True],
            "cotra":        [False],
            "weaviate":     [False],
        },
    },
    # --- [论文] compare：recall vs qps 折线图 ---
    {
        "chart": "line",
        "mode": "compare",
        "grid": {
            "x":            ["avg_recall@10"],
            "y":            ["qps"],
            "node_num":     [5],
            "replication":   [0, 1, 2],
            "threads":      [8, 16, 32, 64],
            "num_query_threads": [40],
        },
    },
    # --- [论文] compare：recall vs qps 柱状图（Fig 5.1）---
    {
        "chart": "bar",
        "mode": "compare",
        "grid": {
            "x":            ["avg_recall@10"],
            "y":            ["qps"],
            "node_num":     [5],
            "replication":   [0, 1, 2],
            "threads":      [8, 16, 32, 64],
            "num_query_threads": [40],
        },
    },
    # --- [论文] compare：threads vs qps 折线图 ---
    {
        "chart": "line",
        "mode": "compare",
        "grid": {
            "x":            ["threads"],
            "y":            ["qps"],
            "m":            [10],
            "ef_construction": [200],
            "limited_level": [5],
            "node_num":     [5],
            "replication":   [2],
            "num_query_threads": [40],
        },
    },
    # ==================== 论文未用图（已注释）====================
    # # --- compare：三路对比柱状图，threads vs qps ---
    # {
    #     "chart": "bar",
    #     "mode": "compare",
    #     "grid": {
    #         "x":            ["threads"],
    #         "y":            ["qps"],
    #         "m":            [10],
    #         "ef_construction": [200],
    #         "limited_level": [5],
    #         "node_num":     [5],
    #         "replication":   [2],
    #         "num_query_threads": [40],
    #     },
    # },
    # # --- compare：固定 recall，threads vs qps 折线图 ---
    # {
    #     "chart": "line",
    #     "mode": "compare",
    #     "grid": {
    #         "x":            ["threads"],
    #         "y":            ["qps"],
    #         "node_num":     [5],
    #         "replication":   [2],
    #         "num_query_threads": [40],
    #         "recall":       [0.95, 0.9, 0.85],
    #     },
    # },
    # # --- compare：固定 recall，threads vs qps 柱状图 ---
    # {
    #     "chart": "bar",
    #     "mode": "compare",
    #     "grid": {
    #         "x":            ["threads"],
    #         "y":            ["qps"],
    #         "node_num":     [5],
    #         "replication":   [2],
    #         "num_query_threads": [40],
    #         "recall":       [0.95, 0.9, 0.85],
    #     },
    # },
    # # --- compare：replication vs replication_num 柱状图 ---
    # {
    #     "chart": "bar",
    #     "mode": "compare",
    #     "grid": {
    #         "x":            ["replication"],
    #         "y":            ["replication_num"],
    #         "m":            [10],
    #         "ef_construction": [200],
    #         "ef_search":    [15],
    #         "limited_level": [5],
    #         "node_num":     [5],
    #         "threads":      [8],
    #         "num_query_threads": [40],
    #     },
    # },
    # # --- compare：replication vs avg_remote_transfer_cnt 折线图 ---
    # {
    #     "chart": "line",
    #     "mode": "compare",
    #     "grid": {
    #         "x":            ["replication"],
    #         "y":            ["avg_remote_transfer_cnt"],
    #         "m":            [10],
    #         "ef_construction": [200],
    #         "ef_search":    [15],
    #         "limited_level": [5],
    #         "node_num":     [5],
    #         "threads":      [8],
    #         "num_query_threads": [40],
    #     },
    # },
    # # --- compare：replication vs avg_remote_transfer_data_per_transfer 折线图 ---
    # {
    #     "chart": "line",
    #     "mode": "compare",
    #     "grid": {
    #         "x":            ["replication"],
    #         "y":            ["avg_remote_transfer_data_per_transfer"],
    #         "m":            [10],
    #         "ef_construction": [200],
    #         "ef_search":    [15],
    #         "limited_level": [5],
    #         "node_num":     [5],
    #     },
    # },
    # # --- compare：replication vs avg_remote_transfer_data_per_transfer 柱状图 ---
    # {
    #     "chart": "bar",
    #     "mode": "compare",
    #     "grid": {
    #         "x":            ["replication"],
    #         "y":            ["avg_remote_transfer_data_per_transfer"],
    #         "m":            [10],
    #         "ef_construction": [200],
    #         "ef_search":    [15],
    #         "limited_level": [5],
    #         "node_num":     [5],
    #     },
    # },
    # # --- compare：replication vs avg_remote_transfer_data_per_query 折线图 ---
    # {
    #     "chart": "line",
    #     "mode": "compare",
    #     "grid": {
    #         "x":            ["replication"],
    #         "y":            ["avg_remote_transfer_data_per_query"],
    #         "m":            [10],
    #         "ef_construction": [200],
    #         "ef_search":    [15],
    #         "limited_level": [5],
    #         "node_num":     [5],
    #     },
    # },
    # # --- compare：replication vs avg_remote_transfer_data_per_query 柱状图 ---
    # {
    #     "chart": "bar",
    #     "mode": "compare",
    #     "grid": {
    #         "x":            ["replication"],
    #         "y":            ["avg_remote_transfer_data_per_query"],
    #         "m":            [10],
    #         "ef_construction": [200],
    #         "ef_search":    [15],
    #         "limited_level": [5],
    #         "node_num":     [5],
    #     },
    # },
    # # --- compare：avg_distance_cnt vs recall 折线图 ---
    # {
    #     "chart": "line",
    #     "mode": "compare",
    #     "grid": {
    #         "x":            ["avg_distance_cnt"],
    #         "y":            ["avg_recall@10"],
    #         "node_num":     [5],
    #         "replication":   [0, 1, 2],
    #         "threads":      [8],
    #         "num_query_threads": [40],
    #     },
    # },
    # # --- compare：avg_remote_transfer_cnt vs qps 折线图（split）---
    # {
    #     "chart": "line",
    #     "mode": "compare",
    #     "grid": {
    #         "x":            ["avg_remote_transfer_cnt"],
    #         "y":            ["qps"],
    #         "m":            [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
    #         "ef_construction": [200],
    #         "limited_level": [5],
    #         "node_num":     [5],
    #         "replication":   [0, 1, 2],
    #         "threads":      [8, 16, 32, 64],
    #         "num_query_threads": [40],
    #         "split":        [True],
    #     },
    # },
    # # --- compare：efs vs qps 柱状图 ---
    # {
    #     "chart": "bar",
    #     "mode": "compare",
    #     "grid": {
    #         "x":            ["ef_search"],
    #         "y":            ["qps"],
    #         "m":            [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
    #         "ef_construction": [200],
    #         "limited_level": [5],
    #         "node_num":     [5],
    #         "replication":   [0, 1, 2],
    #         "threads":      [8, 16, 32, 64],
    #         "num_query_threads": [40],
    #     },
    # },
    # # --- compare：efs vs recall 折线图 ---
    # {
    #     "chart": "line",
    #     "mode": "compare",
    #     "grid": {
    #         "x":            ["ef_search"],
    #         "y":            ["avg_recall@10"],
    #         "m":            [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
    #         "ef_construction": [200],
    #         "limited_level": [5],
    #         "node_num":     [5],
    #         "replication":   [0, 1, 2],
    #         "threads":      [8, 16, 32, 64],
    #         "num_query_threads": [40],
    #     },
    # },
    # # --- compare：efs vs recall 柱状图 ---
    # {
    #     "chart": "bar",
    #     "mode": "compare",
    #     "grid": {
    #         "x":            ["ef_search"],
    #         "y":            ["avg_recall@10"],
    #         "m":            [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
    #         "ef_construction": [200],
    #         "limited_level": [5],
    #         "node_num":     [5],
    #         "replication":   [0, 1, 2],
    #         "threads":      [8, 16, 32, 64],
    #         "num_query_threads": [40],
    #     },
    # },
    # # --- compare：efs vs avg_remote_transfer_data_per_transfer 折线图 ---
    # {
    #     "chart": "line",
    #     "mode": "compare",
    #     "grid": {
    #         "x":            ["ef_search"],
    #         "y":            ["avg_remote_transfer_data_per_transfer"],
    #         "m":            [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
    #         "ef_construction": [200],
    #         "limited_level": [5],
    #         "node_num":     [5],
    #         "replication":   [0, 1, 2],
    #     },
    # },
    # # --- compare：efs vs avg_remote_transfer_data_per_transfer 柱状图 ---
    # {
    #     "chart": "bar",
    #     "mode": "compare",
    #     "grid": {
    #         "x":            ["ef_search"],
    #         "y":            ["avg_remote_transfer_data_per_transfer"],
    #         "m":            [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
    #         "ef_construction": [200],
    #         "limited_level": [5],
    #         "node_num":     [5],
    #         "replication":   [0, 1, 2],
    #     },
    # },
    # # --- compare：efs vs avg_remote_transfer_data_per_query 折线图 ---
    # {
    #     "chart": "line",
    #     "mode": "compare",
    #     "grid": {
    #         "x":            ["ef_search"],
    #         "y":            ["avg_remote_transfer_data_per_query"],
    #         "m":            [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
    #         "ef_construction": [200],
    #         "limited_level": [5],
    #         "node_num":     [5],
    #         "replication":   [0, 1, 2],
    #     },
    # },
    # # --- compare：efs vs avg_remote_transfer_data_per_query 柱状图 ---
    # {
    #     "chart": "bar",
    #     "mode": "compare",
    #     "grid": {
    #         "x":            ["ef_search"],
    #         "y":            ["avg_remote_transfer_data_per_query"],
    #         "m":            [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
    #         "ef_construction": [200],
    #         "limited_level": [5],
    #         "node_num":     [5],
    #         "replication":   [0, 1, 2],
    #     },
    # },
    # # --- compare：recall vs avg_remote_transfer_data_per_transfer 折线图 ---
    # {
    #     "chart": "line",
    #     "mode": "compare",
    #     "grid": {
    #         "x":            ["avg_recall@10"],
    #         "y":            ["avg_remote_transfer_data_per_transfer"],
    #         "node_num":     [5],
    #         "replication":   [0, 1, 2],
    #     },
    # },
    # # --- compare：recall vs avg_remote_transfer_data_per_transfer 柱状图 ---
    # {
    #     "chart": "bar",
    #     "mode": "compare",
    #     "grid": {
    #         "x":            ["avg_recall@10"],
    #         "y":            ["avg_remote_transfer_data_per_transfer"],
    #         "node_num":     [5],
    #         "replication":   [0, 1, 2],
    #     },
    # },
    # # --- compare：recall vs avg_remote_transfer_data_per_query 折线图 ---
    # {
    #     "chart": "line",
    #     "mode": "compare",
    #     "grid": {
    #         "x":            ["avg_recall@10"],
    #         "y":            ["avg_remote_transfer_data_per_query"],
    #         "node_num":     [5],
    #         "replication":   [0, 1, 2],
    #     },
    # },
    # # --- compare：recall vs avg_remote_transfer_data_per_query 柱状图 ---
    # {
    #     "chart": "bar",
    #     "mode": "compare",
    #     "grid": {
    #         "x":            ["avg_recall@10"],
    #         "y":            ["avg_remote_transfer_data_per_query"],
    #         "node_num":     [5],
    #         "replication":   [0, 1, 2],
    #     },
    # },
    # # --- compare：recall vs avg_distance_cnt 折线图（shard + global）---
    # {
    #     "chart": "line",
    #     "mode": "compare",
    #     "grid": {
    #         "x":            ["avg_recall@10"],
    #         "y":            ["avg_distance_cnt"],
    #         "node_num":     [5],
    #         "replication":   [0, 1, 2],
    #         "threads":      [8, 16, 32, 64],
    #         "num_query_threads": [40],
    #         "global":       [True],
    #         "cotra":        [False],
    #         "weaviate":     [False],
    #     },
    # },
    # # --- group：x=ef_search, y=qps, 按 m 分组 ---
    # {
    #     "chart": "line",
    #     "mode": "group",
    #     "grid": {
    #         "x":            ["ef_search"],
    #         "y":            ["qps"],
    #         "group_by":     ["m"],
    #         "ef_construction": [200],
    #         "limited_level": [5],
    #         "node_num":     [5],
    #         "replication":   [0, 1, 2],
    #         "threads":      [8, 16, 32, 64],
    #         "num_query_threads": [40],
    #         "layer":        [True, False],
    #     },
    # },
    # # --- group：x=m, y=qps, 按 ef_search 分组 ---
    # {
    #     "chart": "line",
    #     "mode": "group",
    #     "grid": {
    #         "x":            ["m"],
    #         "y":            ["qps"],
    #         "group_by":     ["ef_search"],
    #         "ef_construction": [200],
    #         "limited_level": [5],
    #         "node_num":     [5],
    #         "replication":   [0, 1, 2],
    #         "threads":      [8, 16, 32, 64],
    #         "num_query_threads": [40],
    #         "layer":        [True, False],
    #     },
    # },
    # # --- group：recall vs avg_remote_transfer_cnt，按 replication 分组 ---
    # {
    #     "chart": "line",
    #     "mode": "group",
    #     "grid": {
    #         "x":            ["avg_recall@10"],
    #         "y":            ["avg_remote_transfer_cnt"],
    #         "group_by":     ["replication"],
    #         "node_num":     [5],
    #         "threads":      [8, 16, 32, 64],
    #         "num_query_threads": [40],
    #         "layer":        [True, False],
    #     },
    # },
    # # --- compare：efs vs avg_distance_cnt 折线图 ---
    # {
    #     "chart": "line",
    #     "mode": "compare",
    #     "grid": {
    #         "x":            ["ef_search"],
    #         "y":            ["avg_distance_cnt"],
    #         "m":            [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
    #         "ef_construction": [200],
    #         "limited_level": [5],
    #         "node_num":     [5],
    #         "replication":   [0, 1, 2],
    #         "threads":      [8, 16, 32, 64],
    #         "num_query_threads": [40],
    #     },
    # },
    # # --- compare：efs vs avg_distance_cnt 柱状图 ---
    # {
    #     "chart": "bar",
    #     "mode": "compare",
    #     "grid": {
    #         "x":            ["ef_search"],
    #         "y":            ["avg_distance_cnt"],
    #         "m":            [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
    #         "ef_construction": [200],
    #         "limited_level": [5],
    #         "node_num":     [5],
    #         "replication":   [0, 1, 2],
    #         "threads":      [8, 16, 32, 64],
    #         "num_query_threads": [40],
    #     },
    # },
]


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
}


def make_args(chart, mode, params, debug=False):
    """构建 SimpleNamespace 参数对象（直接传给 run_draw）"""
    d = dict(DEFAULTS)
    d["chart"] = chart
    d["mode"] = mode
    d["debug"] = debug
    d.update(params)
    # Route output to ./plot/<data_name>/{pdf,png}
    if d.get("data_name"):
        d["output_dir"] = os.path.join(DEFAULTS["output_dir"], d["data_name"])
    return SimpleNamespace(**d)


def run_batch(args):
    """批量画图"""
    tasks = []
    for data_name in DATA_NAMES:
        for task_def in DRAW_TASKS:
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
    args = parser.parse_args()

    run_batch(args)
