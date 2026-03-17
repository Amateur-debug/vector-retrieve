import os
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict
import json
import re
import numpy as np


# ============================================================
# 工具函数
# ============================================================

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def get_param_value(data, key):
    """在 JSON 结构中查找参数"""
    if key in data:
        return data[key]
    if 'hnsw_params' in data and key in data['hnsw_params']:
        return data['hnsw_params'][key]
    if 'cluster_params' in data and key in data['cluster_params']:
        return data['cluster_params'][key]
    return None


def parse_filename_params(filename):
    """从结果文件名中解析参数，返回参数字典"""
    params = {}

    match = re.search(r'results-([a-z]+?)-m', filename)
    if match:
        params['data_name'] = match.group(1)

    match = re.search(r'-m(\d+)-', filename)
    if match:
        params['m'] = int(match.group(1))

    match = re.search(r'-efc(\d+)-', filename)
    if match:
        params['ef_construction'] = int(match.group(1))

    match = re.search(r'-efs(\d+)-', filename)
    if match:
        params['ef_search'] = int(match.group(1))

    match = re.search(r'-ll(\d+)-', filename)
    if match:
        params['limited_level'] = int(match.group(1))

    match = re.search(r'-node(\d+)-', filename)
    if match:
        params['node_num'] = int(match.group(1))

    match = re.search(r'-node\d+-([a-z_-]+)-red', filename)
    if match:
        params['partition_mode'] = match.group(1)

    match = re.search(r'-red(\d+)', filename)
    if match:
        params['redundancy'] = int(match.group(1))

    match = re.search(r'-threads(\d+)', filename)
    if match:
        params['threads'] = int(match.group(1))

    params['reduce'] = 'reduce' in filename
    params['distributed'] = 'distributed' in filename
    params['layer'] = filename.rstrip('.json').endswith('-layer')

    return params


def find_matching_file(directory, target_params):
    """从目录中找到匹配参数的第一个 JSON 文件"""
    if not os.path.isdir(directory):
        return None

    for filename in sorted(os.listdir(directory)):
        if not filename.endswith('.json'):
            continue

        file_params = parse_filename_params(filename)

        match = True
        for key, value in target_params.items():
            if value is None:
                continue
            if file_params.get(key) != value:
                match = False
                break

        if match:
            return os.path.join(directory, filename)

    return None


def classify_mode(filepath, data=None):
    """根据文件名判断实验模式，并生成包含分区方式的标签。

    返回 (mode_key, label):
      - reduce:   ('reduce', 'Reduce')
      - no-layer: ('no-layer', 'No-Layer (k-means)')
      - layer:    ('layer', 'Layer (k-means)')
    """
    basename = os.path.basename(filepath)
    if 'reduce' in basename:
        return 'reduce', 'Reduce'

    if data:
        cp = data.get('cluster_params', {})
    else:
        cp = {}

    if basename.rstrip('.json').endswith('-layer'):
        layer_pm = cp.get('layer_partition_mode', 'unknown')
        return 'layer', f'Layer ({layer_pm})'
    else:
        pm = cp.get('partition_mode', 'unknown')
        return 'no-layer', f'No-Layer ({pm})'


def get_y_value(data, y_key, data_name):
    """根据 y_key 从 data 中提取 y 值（支持自定义派生指标）"""
    if y_key == "red_per_node":
        redundancy_num = get_param_value(data, "redundancy_num")
        node_num = get_param_value(data, "node_num")
        return redundancy_num / node_num if node_num else 0
    elif y_key == "data_per_node":
        total_data = {'siftsmall': 10000, 'sift': 1000000, 'gist': 1000000, 'deep': 1000000}.get(data_name, 0)
        total_data_redundant = total_data + (get_param_value(data, "redundancy_num") or 0)
        node_num = get_param_value(data, "node_num")
        return total_data_redundant / node_num if node_num else 0
    else:
        return get_param_value(data, y_key)


def make_output_filename(prefix, args, extra_keys=None):
    """生成输出文件名"""
    parts = [
        prefix,
        f"{args.x}-{args.y}-{args.data_name}",
        f"m{args.m}-efc{args.ef_construction}-ll{args.limited_level}",
    ]
    if args.node_num is not None:
        parts.append(f"node{args.node_num}")
    if args.redundancy is not None:
        parts.append(f"red{args.redundancy}")
    if args.threads is not None:
        parts.append(f"threads{args.threads}")
    if getattr(args, 'recall', None) is not None:
        parts.append(f"recall{args.recall}")
    if extra_keys:
        parts.extend(extra_keys)
    return "-".join(parts) + ".pdf"


def setup_legend(ax, n_labels):
    """图例统一放在图上方，水平排列"""
    ax.legend(fontsize=10, loc='lower center', bbox_to_anchor=(0.5, 1.02),
              ncol=n_labels, frameon=True, fancybox=True)


RECALL_BUCKET_SIZE = 0.05


def is_recall_x(x_key):
    """判断 x 轴是否为 recall 类指标"""
    return 'recall' in x_key.lower()


def snap_to_recall_bucket(val):
    """将 recall 值对齐到最近的 0.05 档位"""
    return round(round(val / RECALL_BUCKET_SIZE) * RECALL_BUCKET_SIZE, 2)


def bucket_recall_points(points_dict):
    """将 {raw_recall: y_val} 按 0.05 档位聚合。

    同一档位有多个点时，保留 recall 最接近档位中心的那个。
    返回 {bucketed_recall: y_val}。
    """
    # 先按 bucket 分组：bucket -> [(raw_x, y), ...]
    buckets = defaultdict(list)
    for raw_x, y in points_dict.items():
        bucket = snap_to_recall_bucket(raw_x)
        buckets[bucket].append((raw_x, y))

    # 每个 bucket 保留最接近中心的点
    result = {}
    for bucket, candidates in buckets.items():
        best = min(candidates, key=lambda c: abs(c[0] - bucket))
        result[bucket] = best[1]

    return result


# ============================================================
# 数据收集函数（按 mode 分类）
# ============================================================

def collect_compare(args):
    """compare 模式：收集三路对比数据 (reduce / no-layer / layer)。

    返回 (mode_points, mode_labels, mode_debug):
      mode_points: {mode_key: {x_val: y_val}}
      mode_labels: {mode_key: display_label}
      mode_debug:  {mode_key: {x_val: (recall, filename)}} or None
    """
    directory = args.json_path
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a directory.")
        return None, None

    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.json')]

    mode_points = defaultdict(dict)
    mode_labels = {}
    # --recall 模式：收集所有候选，稍后按 recall 筛选
    # candidates: {mode_key: {x_val: [(recall, y_val, filename), ...]}}
    mode_candidates = defaultdict(lambda: defaultdict(list))
    target_recall = getattr(args, 'recall', None)

    for f in files:
        try:
            data = load_json(f)
            file_params = parse_filename_params(os.path.basename(f))

            skip = False
            for param_name, arg_val in [
                ('data_name', args.data_name),
                ('m', args.m),
                ('ef_construction', args.ef_construction),
                ('limited_level', args.limited_level),
                ('node_num', args.node_num),
                ('threads', args.threads),
            ]:
                if arg_val is not None and file_params.get(param_name) != arg_val:
                    skip = True
                    break
            if skip:
                continue

            if not file_params.get('reduce') and args.redundancy is not None:
                if file_params.get('redundancy') != args.redundancy:
                    continue

            mode_key, label = classify_mode(f, data)
            mode_labels[mode_key] = label

            x_val = get_param_value(data, args.x)
            if x_val is None:
                x_val = file_params.get(args.x)
            y_val = get_y_value(data, args.y, args.data_name)

            if x_val is None or y_val is None:
                continue

            if target_recall is not None:
                # 收集候选，稍后按 recall 距离筛选
                recall_val = get_param_value(data, 'avg_recall@10')
                if recall_val is not None:
                    mode_candidates[mode_key][x_val].append(
                        (recall_val, y_val, os.path.basename(f)))
            else:
                mode_points[mode_key][x_val] = y_val
                print(f"[{label}] {os.path.basename(f)} | {args.x}={x_val} | {args.y}={y_val}")

        except Exception as e:
            print(f"Skipping {f}: {e}")

    # --recall 模式：对每个 (mode, x_val) 选 recall 最接近目标的
    mode_debug = None
    if target_recall is not None:
        mode_debug = defaultdict(dict)
        for mode_key, x_candidates in mode_candidates.items():
            for x_val, candidates in x_candidates.items():
                # 先按 recall 距离分组，取最近的一批，再从中选 y 最大的
                min_dist = min(abs(c[0] - target_recall) for c in candidates)
                nearest = [c for c in candidates if abs(c[0] - target_recall) - min_dist < 0.025]
                best = max(nearest, key=lambda c: c[1])
                mode_points[mode_key][x_val] = best[1]
                mode_debug[mode_key][x_val] = (best[0], best[2])  # (recall, filename)
                label = mode_labels.get(mode_key, mode_key)
                print(f"[{label}] {best[2]} | {args.x}={x_val} | {args.y}={best[1]} "
                      f"| recall={best[0]:.4f} (target={target_recall})")

    if not any(mode_points.values()):
        print("No valid data points found.")
        return None, None, None

    # recall x 轴时，按 0.05 档位聚合
    if is_recall_x(args.x):
        for mode_key in list(mode_points.keys()):
            mode_points[mode_key] = bucket_recall_points(mode_points[mode_key])

    return mode_points, mode_labels, mode_debug


def collect_detail(args):
    """detail 模式：收集单次实验的逐查询详细数据。

    返回 {series_label: [(x, y), ...]}，只有一个 series。
    """
    json_file = args.json_path

    if os.path.isdir(args.json_path):
        target_params = {
            'data_name': args.data_name,
            'm': args.m,
            'ef_construction': args.ef_construction,
            'ef_search': args.ef_search,
            'limited_level': args.limited_level,
            'node_num': args.node_num,
            'partition_mode': args.partition_mode,
            'redundancy': args.redundancy,
            'threads': args.threads,
            'reduce': args.reduce,
            'distributed': args.distributed,
        }

        json_file = find_matching_file(args.json_path, target_params)

        if json_file is None:
            print("Error: No matching JSON file found with the specified parameters.")
            print(f"Target parameters: {target_params}")
            return None, None

        print(f"Found matching file: {json_file}")

    data = load_json(json_file)
    raw_y_data = data.get(args.y)

    if not raw_y_data or not isinstance(raw_y_data, dict):
        print(f"Error: Key '{args.y}' not found or is not a dictionary in the JSON file.")
        print(f"Available keys: {list(data.keys())}")
        return None, None

    sorted_items = sorted(
        [(int(k), v) for k, v in raw_y_data.items() if k.isdigit()],
        key=lambda x: x[0]
    )

    if not sorted_items:
        print(f"Error: No valid numeric keys found in {args.y}")
        return None, None

    label = args.y
    return {label: sorted_items}, json_file


def collect_merge(args):
    """merge 模式：从多个文件中提取单条汇总曲线。

    返回 {series_label: [(x, y), ...]}，只有一个 series。
    """
    directory = args.json_path
    files = []
    if os.path.isdir(directory):
        files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.json')]
    else:
        files = [directory]

    points = {}
    candidates = defaultdict(list)  # x_val -> [(recall, y_val, filename), ...]
    target_recall = getattr(args, 'recall', None)

    for f in files:
        try:
            data = load_json(f)
            if (
                args.data_name is not None and get_param_value(data, "data_name") != args.data_name or
                args.m is not None and get_param_value(data, "m") != args.m or
                args.ef_construction is not None and get_param_value(data, "ef_construction") != args.ef_construction or
                args.ef_search is not None and get_param_value(data, "ef_search") != args.ef_search or
                args.limited_level is not None and get_param_value(data, "limited_level") != args.limited_level or
                args.node_num is not None and get_param_value(data, "node_num") != args.node_num or
                args.partition_mode is not None and get_param_value(data, "partition_mode") != args.partition_mode or
                args.redundancy is not None and get_param_value(data, "redundancy_level") != args.redundancy
            ):
                continue

            x_val = get_param_value(data, args.x)
            y_val = get_y_value(data, args.y, args.data_name)

            if x_val is None or y_val is None:
                continue

            if target_recall is not None:
                recall_val = get_param_value(data, 'avg_recall@10')
                if recall_val is not None:
                    candidates[x_val].append((recall_val, y_val, os.path.basename(f)))
            else:
                print(f"File: {f} | {args.x}: {x_val} | {args.y}: {y_val}")
                points[x_val] = y_val

        except Exception as e:
            print(f"Skipping {f}: {e}")

    if target_recall is not None:
        for x_val, cands in candidates.items():
            min_dist = min(abs(c[0] - target_recall) for c in cands)
            nearest = [c for c in cands if abs(c[0] - target_recall) - min_dist < 0.025]
            best = max(nearest, key=lambda c: c[1])
            points[x_val] = best[1]
            print(f"{best[2]} | {args.x}={x_val} | {args.y}={best[1]} "
                  f"| recall={best[0]:.4f} (target={target_recall})")

    if not points:
        print("No valid data points found.")
        return None

    # recall x 轴时，按 0.05 档位聚合
    if is_recall_x(args.x):
        points = bucket_recall_points(points)

    sorted_pts = sorted(points.items(), key=lambda p: p[0])
    label = args.y
    return {label: sorted_pts}


# ============================================================
# 绘图函数（按 chart 类型分类）
# ============================================================

MODE_ORDER = ['reduce', 'no-layer', 'layer']
MODE_STYLES = {
    'reduce':   {'marker': 'o', 'linestyle': '-',  'color': '#1f77b4'},
    'no-layer': {'marker': 's', 'linestyle': '--', 'color': '#ff7f0e'},
    'layer':    {'marker': '^', 'linestyle': '-.', 'color': '#2ca02c'},
}


def draw_line_compare(mode_points, mode_labels, args, mode_debug=None):
    """折线图 × compare 模式"""
    debug = getattr(args, 'debug', False) and mode_debug is not None
    n_x = max(len(pts) for pts in mode_points.values())
    fig_width = max(8, n_x * 1.5 + 3)
    fig_height = 7.5 if debug else 6
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    for mode_key in MODE_ORDER:
        points = mode_points.get(mode_key, {})
        if not points:
            continue
        sorted_pts = sorted(points.items(), key=lambda p: p[0])
        xs, ys = zip(*sorted_pts)
        style = MODE_STYLES[mode_key]
        ax.plot(xs, ys, marker=style['marker'], linestyle=style['linestyle'],
                color=style['color'], label=mode_labels[mode_key], linewidth=2, markersize=8)
        for x, y in sorted_pts:
            if debug and mode_debug.get(mode_key, {}).get(x):
                recall, fname = mode_debug[mode_key][x]
                ax.annotate(f"{y:.2f}\nrecall={recall:.4f}\n{fname}",
                            (x, y), fontsize=6, va='bottom', ha='left',
                            xytext=(5, 5), textcoords='offset points')
            else:
                ax.text(x, y, f" {y:.2f}", fontsize=8, va='bottom')

    ax.set_xlabel(args.x, fontsize=12)
    ax.set_ylabel(args.y, fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    n_labels = sum(1 for m in MODE_ORDER if m in mode_points)
    setup_legend(ax, n_labels)
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)

    output_path = os.path.join(args.output_dir, make_output_filename("line-compare", args))
    fig.savefig(output_path, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    plt.close(fig)


def draw_bar_compare(mode_points, mode_labels, args, mode_debug=None):
    """柱状图 × compare 模式"""
    debug = getattr(args, 'debug', False) and mode_debug is not None
    all_x_vals = sorted(set().union(*[pts.keys() for pts in mode_points.values()]))
    x_labels = [str(v) for v in all_x_vals]

    active_modes = [m for m in MODE_ORDER if mode_points.get(m)]

    n_groups = len(all_x_vals)
    n_bars = len(active_modes)
    bar_width = 0.8 / max(n_bars, 1)
    x_indices = np.arange(n_groups)

    fig_width = max(8, n_groups * n_bars * 0.6 + 3)
    fig_height = 7.5 if debug else 6
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    for i, mode_key in enumerate(active_modes):
        y_vals = [mode_points[mode_key].get(x, 0) for x in all_x_vals]
        offset = (i - (n_bars - 1) / 2) * bar_width
        bars = ax.bar(x_indices + offset, y_vals, bar_width,
                      label=mode_labels[mode_key], color=MODE_STYLES[mode_key]['color'], alpha=0.85)
        for j, (bar, val) in enumerate(zip(bars, y_vals)):
            if val != 0:
                x_key = all_x_vals[j]
                if debug and mode_debug.get(mode_key, {}).get(x_key):
                    recall, fname = mode_debug[mode_key][x_key]
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                            f'{val:.2f}\nr={recall:.4f}\n{fname}',
                            ha='center', va='bottom', fontsize=5.5)
                else:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                            f'{val:.2f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x_indices)
    ax.set_xticklabels(x_labels, fontsize=11)
    ax.set_xlabel(args.x, fontsize=12)
    ax.set_ylabel(args.y, fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5, axis='y')
    setup_legend(ax, n_bars)
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)

    output_path = os.path.join(args.output_dir, make_output_filename("bar-compare", args))
    fig.savefig(output_path, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    plt.close(fig)


def draw_scatter_compare(mode_points, mode_labels, args, mode_debug=None):
    """散点图 × compare 模式"""
    debug = getattr(args, 'debug', False) and mode_debug is not None
    n_x = max(len(pts) for pts in mode_points.values())
    fig_width = max(8, n_x * 1.5 + 3)
    fig_height = 7.5 if debug else 6
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    for mode_key in MODE_ORDER:
        points = mode_points.get(mode_key, {})
        if not points:
            continue
        sorted_pts = sorted(points.items(), key=lambda p: p[0])
        xs, ys = zip(*sorted_pts)
        ax.scatter(xs, ys, color=MODE_STYLES[mode_key]['color'],
                   label=mode_labels[mode_key], s=80, alpha=0.8,
                   marker=MODE_STYLES[mode_key]['marker'])
        for x, y in sorted_pts:
            if debug and mode_debug.get(mode_key, {}).get(x):
                recall, fname = mode_debug[mode_key][x]
                ax.annotate(f"{y:.2f}\nrecall={recall:.4f}\n{fname}",
                            (x, y), fontsize=6, va='bottom', ha='left',
                            xytext=(5, 5), textcoords='offset points')
            else:
                ax.text(x, y, f" {y:.2f}", fontsize=8, va='bottom')

    ax.set_xlabel(args.x, fontsize=12)
    ax.set_ylabel(args.y, fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    n_labels = sum(1 for m in MODE_ORDER if m in mode_points)
    setup_legend(ax, n_labels)
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)

    output_path = os.path.join(args.output_dir, make_output_filename("scatter-compare", args))
    fig.savefig(output_path, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    plt.close(fig)


def draw_line_single(series_data, args, prefix="line"):
    """折线图 × 单系列模式（detail / merge）"""
    n_points = max(len(pts) for pts in series_data.values())
    fig, ax = plt.subplots(figsize=(16, 6))

    all_xs = []
    for label, pts in series_data.items():
        xs, ys = zip(*pts)
        all_xs.extend(xs)
        ax.plot(xs, ys, linestyle='-', linewidth=0.5, label=label)

    if all_xs:
        ax.set_xlim(min(all_xs), max(all_xs))

    x_label = "Query Index" if args.mode == "detail" else args.x
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(args.y, fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    setup_legend(ax, len(series_data))
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)

    output_path = os.path.join(args.output_dir, make_output_filename(f"{prefix}-{args.mode}", args))
    fig.savefig(output_path, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    plt.close(fig)


def draw_scatter_single(series_data, args, prefix="scatter"):
    """散点图 × 单系列模式（detail / merge）"""
    n_points = max(len(pts) for pts in series_data.values())
    fig, ax = plt.subplots(figsize=(16, 6))

    all_xs = []
    for label, pts in series_data.items():
        xs, ys = zip(*pts)
        all_xs.extend(xs)
        s = 1 if n_points > 500 else 20
        alpha = 0.3 if n_points > 500 else 0.6
        ax.scatter(xs, ys, alpha=alpha, s=s, label=label)

    if all_xs:
        ax.set_xlim(min(all_xs), max(all_xs))

    x_label = "Query Index" if args.mode == "detail" else args.x
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(args.y, fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    setup_legend(ax, len(series_data))
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)

    output_path = os.path.join(args.output_dir, make_output_filename(f"{prefix}-{args.mode}", args))
    fig.savefig(output_path, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    plt.close(fig)


def draw_bar_single(series_data, args, prefix="bar"):
    """柱状图 × 单系列模式（detail / merge）"""
    n_points = max(len(pts) for pts in series_data.values())
    fig_width = max(6, min(20, n_points * 1.2 + 2))
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    for label, pts in series_data.items():
        xs, ys = zip(*pts)
        x_labels = [str(x) for x in xs]
        x_indices = np.arange(len(xs))
        bar_width = min(0.6, max(0.2, 6.0 / max(len(xs), 1)))
        bars = ax.bar(x_indices, ys, width=bar_width, color='#1f77b4', alpha=0.85, label=label)
        for bar, val in zip(bars, ys):
            if val != 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f'{val:.2f}', ha='center', va='bottom', fontsize=7)
        ax.set_xticks(x_indices)
        ax.set_xticklabels(x_labels, fontsize=9, rotation=45 if len(xs) > 20 else 0)

    x_label = "Query Index" if args.mode == "detail" else args.x
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(args.y, fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5, axis='y')
    setup_legend(ax, len(series_data))
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)

    output_path = os.path.join(args.output_dir, make_output_filename(f"{prefix}-{args.mode}", args))
    fig.savefig(output_path, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    plt.close(fig)


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Vector Retrieval Results")
    parser.add_argument("--json_path", type=str, default="./results/cpp",
                        help="Path to JSON file or directory")
    parser.add_argument("--output_dir", type=str, default="./plot",
                        help="Directory to save plot")

    # 两个独立维度
    parser.add_argument("--chart", type=str, default="line", choices=["line", "bar", "scatter"],
                        help="图表类型: line (折线图), bar (柱状图), scatter (散点图)")
    parser.add_argument("--mode", type=str, default="compare", choices=["compare", "detail", "merge"],
                        help="数据模式: compare (三路对比), detail (单实验逐查询), merge (单曲线汇总)")

    # 通用参数
    parser.add_argument("--x", type=str, default="ef_search", help="X-axis parameter")
    parser.add_argument("--y", type=str, default="qps", help="Y-axis metric")

    # 筛选参数
    parser.add_argument("--data_name", type=str, default="sift", help="Dataset name")
    parser.add_argument("--m", type=int, default=10, help="Max connections per element")
    parser.add_argument("--ef_construction", type=int, default=200, help="ef_construction")
    parser.add_argument("--ef_search", type=int, default=None, help="ef_search (for detail/merge filtering)")
    parser.add_argument("--limited_level", type=int, default=5, help="Level limit for HNSW")
    parser.add_argument("--node_num", type=int, default=5, help="Number of cluster nodes")
    parser.add_argument("--partition_mode", type=str, default="k-means", help="Partition mode (for detail/merge)")
    parser.add_argument("--redundancy", type=int, default=2, help="Redundancy level")
    parser.add_argument("--threads", type=int, default=None, help="Number of query threads")
    parser.add_argument("--reduce", action='store_true', help="Enable reduce mode (for detail/merge)")
    parser.add_argument("--distributed", action='store_true', help="Enable distributed mode (for detail/merge)")

    # recall 约束：指定目标 recall，对每个 x 值找 recall 最接近的实验结果
    parser.add_argument("--recall", type=float, default=None,
                        help="Target recall (e.g. 0.9), pick experiments closest to this recall per x value")
    parser.add_argument("--debug", action='store_true',
                        help="Debug mode: annotate recall and filename on chart (used with --recall)")

    # 兼容旧参数
    parser.add_argument("--merge", action='store_true', help="(deprecated) Use --mode merge instead")
    parser.add_argument("--plot_type", type=str, default=None,
                        help="(deprecated) Use --chart instead")

    args = parser.parse_args()

    if not os.path.exists(args.json_path):
        print(f"Error: {args.json_path} not found.")
        exit(1)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 兼容旧参数
    if args.merge:
        args.mode = "merge"
    if args.plot_type == "scatter":
        args.chart = "scatter"

    # 根据 mode 收集数据，根据 chart 绘图
    if args.mode == "compare":
        mode_points, mode_labels, mode_debug = collect_compare(args)
        if mode_points is None:
            exit(1)
        if args.chart == "line":
            draw_line_compare(mode_points, mode_labels, args, mode_debug)
        elif args.chart == "bar":
            draw_bar_compare(mode_points, mode_labels, args, mode_debug)
        elif args.chart == "scatter":
            draw_scatter_compare(mode_points, mode_labels, args, mode_debug)

    elif args.mode == "detail":
        series_data, json_file = collect_detail(args)
        if series_data is None:
            exit(1)
        if args.chart == "line":
            draw_line_single(series_data, args)
        elif args.chart == "bar":
            draw_bar_single(series_data, args)
        elif args.chart == "scatter":
            draw_scatter_single(series_data, args)

    elif args.mode == "merge":
        series_data = collect_merge(args)
        if series_data is None:
            exit(1)
        if args.chart == "line":
            draw_line_single(series_data, args)
        elif args.chart == "bar":
            draw_bar_single(series_data, args)
        elif args.chart == "scatter":
            draw_scatter_single(series_data, args)
