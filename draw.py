import os
import argparse
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import defaultdict
import json
import re
import numpy as np

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman']
matplotlib.rcParams['mathtext.fontset'] = 'stix'

# ============================================================
# 全局绘图配置（字体、尺寸、线宽等）
# ============================================================
PLOT_CONFIG = {
    # 字体大小
    'font_axis_label': 14,      # 坐标轴标签 (xlabel / ylabel)
    'font_tick_label': 12,      # 刻度标签
    'font_legend': 14,          # 图例
    'font_title': 16,           # 子图标题 (split 模式)

    # 图尺寸 (inch)
    'fig_height_ratio': 0.4,    # 图高度 = 宽度 × 此比例
    'fig_height_min': 2,        # 最小高度 (inch)
    'fig_width_single': 6,      # 单系列图 (detail/merge/group) 固定宽度
    'fig_height_debug_extra': 1, # debug 模式额外高度

    # calc_fig_width 参数
    'fig_width_per_item': 0.5,  # 每个数据项宽度
    'fig_width_min': 4,         # 最小宽度
    'fig_width_max': 10,        # 最大宽度
    'fig_width_base': 2.0,      # 基础宽度

    # 柱状图
    'bar_width_max': 0.25,      # 柱宽上限 (compare 模式)

    # 折线图
    'linewidth': 2,             # 线宽
    'markersize': 6,            # 标记点大小

    # 布局（固定边距，保证所有图绘图区域一致）
    'margin_left': 0.14,        # 左边距（给 y 轴标签留空间）
    'margin_right': 0.96,       # 右边距
    'margin_bottom': 0.15,      # 下边距
    'margin_top': 0.82,         # 上边距（给图例留空间）
    'y_nbins': 6,               # y 轴刻度数量
}


# ============================================================
# 工具函数
# ============================================================

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


_summary_cache = {}


def load_json_summary(path, max_bytes=4096):
    """快速加载 JSON 文件的汇总信息（跳过大数组字段）。
    只读取文件前 max_bytes 字节，用正则提取 key-value 对。
    适用于 compare/merge 模式，不需要逐查询数据。
    结果会缓存，同一文件只读一次。
    """
    if path in _summary_cache:
        return _summary_cache[path]

    with open(path, 'r') as f:
        head = f.read(max_bytes)

    result = {}
    # 提取嵌套和顶层的 key: number/string 对
    for m in re.finditer(r'"([^"]+)"\s*:\s*(-?\d+(?:\.\d+)?)\b', head):
        key, val = m.group(1), m.group(2)
        result[key] = float(val) if '.' in val else int(val)

    _summary_cache[path] = result
    return result


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

    match = re.search(r'results-([a-z][a-z0-9]*)-m', filename)
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

    match = re.search(r'-node\d+-([a-z_-]+)-rep', filename)
    if match:
        params['partition_mode'] = match.group(1)

    match = re.search(r'-rep(\d+)', filename)
    if match:
        params['replication'] = int(match.group(1))

    match = re.search(r'-threads(\d+)', filename)
    if match:
        params['threads'] = int(match.group(1))

    match = re.search(r'-query_threads(\d+)', filename)
    if match:
        params['query_threads'] = int(match.group(1))

    params['shard'] = 'shard' in filename
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


def classify_mode(filepath, data=None, is_weaviate=False, is_global=False):
    """根据文件名判断实验模式，并生成包含分区方式的标签。

    返回 (mode_key, label):
      - shard:    ('shard', 'Shard')
      - global:   ('global', 'Global')   — is_global=True 时 no-layer 数据归为 global
      - no-layer: ('no-layer', 'No-Layer (k-means)')
      - layer:    ('layer', 'Layer (k-means)')
      - weaviate: ('weaviate', 'Weaviate')
    """
    if is_weaviate:
        return 'weaviate', 'Weaviate'

    basename = os.path.basename(filepath)
    if 'shard' in basename:
        return 'shard', 'Shard'

    # 从文件名提取 partition_mode
    file_params = parse_filename_params(basename)
    pm = file_params.get('partition_mode', 'unknown')

    if basename.rstrip('.json').endswith('-layer'):
        return 'layer', f'Layer ({pm})'
    else:
        if is_global:
            return 'global', 'Global'
        return 'no-layer', f'No-Layer ({pm})'


def get_y_value(data, y_key, data_name):
    """根据 y_key 从 data 中提取 y 值（支持自定义派生指标）"""
    if y_key == "rep_per_node":
        replication_num = get_param_value(data, "replication_num")
        node_num = get_param_value(data, "node_num")
        return replication_num / node_num if node_num else 0
    elif y_key == "data_per_node":
        total_data = {'siftsmall': 10000, 'sift': 1000000, 'gist': 1000000, 'deep': 1000000}.get(data_name, 0)
        total_data_redundant = total_data + (get_param_value(data, "replication_num") or 0)
        node_num = get_param_value(data, "node_num")
        return total_data_redundant / node_num if node_num else 0
    else:
        return get_param_value(data, y_key)


def make_output_filename(prefix, args, extra_keys=None):
    """生成输出文件名"""
    def _label(s):
        return filename_label(s)

    parts = [
        prefix,
        f"{_label(args.x)}-{_label(args.y)}-{args.data_name}",
    ]
    if args.m is not None:
        parts.append(f"m{args.m}")
    if args.ef_construction is not None:
        parts.append(f"efc{args.ef_construction}")
    # efs: 没有固定 recall 且 y 轴是 qps 时才写入文件名
    if args.ef_search is not None and getattr(args, 'recall', None) is None and args.y == 'qps':
        parts.append(f"efs{args.ef_search}")
    if args.limited_level is not None:
        parts.append(f"ll{args.limited_level}")
    if args.node_num is not None:
        parts.append(f"node{args.node_num}")
    if getattr(args, 'partition_mode', None) is not None:
        parts.append(args.partition_mode)
    if getattr(args, 'global', False):
        parts.append("global")
    if args.replication is not None:
        parts.append(f"rep{args.replication}")
    _thread_independent = {"replication_num", "rep_per_node", "data_per_node",
                           "avg_remote_transfer_cnt", "avg_distance_cnt",
                           "avg_remote_transfer_data_per_transfer",
                           "avg_remote_transfer_data_per_query"}
    if args.threads is not None and args.y not in _thread_independent:
        parts.append(f"threads{args.threads}")
    if getattr(args, 'num_query_threads', None) is not None and args.y not in _thread_independent:
        parts.append(f"query_threads{args.num_query_threads}")
    if getattr(args, 'recall', None) is not None:
        parts.append(f"recall{args.recall}")
    if getattr(args, 'group_by', None) is not None:
        parts.append(f"by-{args.group_by}")
    if extra_keys:
        parts.extend(extra_keys)
    return "-".join(parts) + ".pdf"


DISPLAY_NAMES = {
    'avg_distance_cnt': 'Computation Count',
    'avg_remote_transfer_cnt': 'Transfer Count',
    'avg_remote_transfer_data_per_transfer': 'Data per Transfer (bit)',
    'avg_remote_transfer_data_per_query': 'Data per Query (bit)',
    'qps': 'QPS',
    'threads': 'Threads',
    'replication': 'Replication',
    'replication_num': 'Replication Num',
    'avg_recall@10': 'Recall@10',
}

# 文件名用短名，不带单位和特殊字符
FILENAME_NAMES = {
    'avg_distance_cnt': 'distance_cnt',
    'avg_remote_transfer_cnt': 'remote_transfer_cnt',
    'avg_remote_transfer_data_per_transfer': 'data_per_transfer',
    'avg_remote_transfer_data_per_query': 'data_per_query',
    'avg_recall@10': 'recall@10',
}


def display_label(s):
    """将内部指标名映射为显示名（用于图标签）"""
    if s in DISPLAY_NAMES:
        return DISPLAY_NAMES[s]
    return s


def filename_label(s):
    """将内部指标名映射为文件名用短名"""
    if s in FILENAME_NAMES:
        return FILENAME_NAMES[s]
    return s


def style_ax(ax, x_label, y_label, title=None, tick_axis='both', grid_axis='both'):
    """统一设置坐标轴样式"""
    C = PLOT_CONFIG
    ax.set_xlabel(x_label, fontsize=C['font_axis_label'])
    ax.set_ylabel(y_label, fontsize=C['font_axis_label'])
    if title:
        ax.set_title(title, fontsize=C['font_title'])
    ax.tick_params(axis=tick_axis, labelsize=C['font_tick_label'])
    ax.yaxis.set_major_locator(MaxNLocator(nbins=C['y_nbins']))
    ax.grid(True, linestyle='--', alpha=0.7, axis=grid_axis)


def setup_legend(ax, n_labels):
    """图例统一放在图上方，水平排列"""
    ax.legend(fontsize=PLOT_CONFIG['font_legend'], loc='lower center',
              bbox_to_anchor=(0.5, 1.02), ncol=n_labels, frameon=True, fancybox=True)


def calc_fig_width(n_items, per_item=None, min_width=None, max_width=None, base=None):
    """根据数据项数量计算图宽度，带上下界"""
    C = PLOT_CONFIG
    per_item = per_item if per_item is not None else C['fig_width_per_item']
    min_width = min_width if min_width is not None else C['fig_width_min']
    max_width = max_width if max_width is not None else C['fig_width_max']
    base = base if base is not None else C['fig_width_base']
    return max(min_width, min(max_width, n_items * per_item + base))


def make_fig(width, debug=False):
    """创建 figure，高度按宽度比例计算，不低于最小高度"""
    C = PLOT_CONFIG
    h = max(width * C['fig_height_ratio'], C['fig_height_min'])
    if debug:
        h += C['fig_height_debug_extra']
    return plt.subplots(figsize=(width, h))


def apply_layout(fig):
    """应用固定边距布局，保证所有图绘图区域尺寸一致"""
    C = PLOT_CONFIG
    fig.subplots_adjust(left=C['margin_left'], right=C['margin_right'],
                        bottom=C['margin_bottom'], top=C['margin_top'])


RECALL_BUCKET_SIZE = 0.05


def is_recall_x(x_key):
    """判断 x 轴是否为 recall 类指标"""
    return 'recall' in x_key.lower()


def snap_to_recall_bucket(val):
    """将 recall 值对齐到最近的 0.05 档位"""
    return round(round(val / RECALL_BUCKET_SIZE) * RECALL_BUCKET_SIZE, 2)


def bucket_recall_points(points_list, min_recall=None, max_recall=0.95):
    """将 [(raw_recall, y_val, filename), ...] 按 0.05 档位聚合。

    recall 在档位 ±0.025 范围内视为满足该档位要求。
    同一档位有多个点时，保留 y 值最大的那个（如 qps 取最高）。
    min_recall: 过滤掉低于此值的档位。
    max_recall: 过滤掉高于此值的档位（默认 0.99，排除 recall=1.0）。
    返回 ({bucketed_recall: y_val}, {bucketed_recall: (raw_recall, filename)})。
    """
    # 生成所有可能的档位
    all_buckets = set()
    for raw_x, _, _ in points_list:
        b = snap_to_recall_bucket(raw_x)
        if (min_recall is None or b >= min_recall) and b <= max_recall:
            all_buckets.add(b)

    # 每个点分配到 ±0.025 范围内的所有档位
    buckets = defaultdict(list)
    for bucket in all_buckets:
        for raw_x, y, fname in points_list:
            if abs(raw_x - bucket) <= RECALL_BUCKET_SIZE / 2.0:
                buckets[bucket].append((raw_x, y, fname))

    result = {}
    debug_info = {}
    for bucket, candidates in buckets.items():
        best = max(candidates, key=lambda c: c[1])
        result[bucket] = best[1]
        debug_info[bucket] = (best[0], best[2])

    return result, debug_info


# ============================================================
# 数据收集函数（按 mode 分类）
# ============================================================

def collect_compare(args):
    """compare 模式：收集多路对比数据 (shard / no-layer / layer / weaviate / cotra)。

    返回 (mode_points, mode_labels, mode_debug):
      mode_points: {mode_key: {x_val: y_val}}
      mode_labels: {mode_key: display_label}
      mode_debug:  {mode_key: {x_val: (recall, filename)}} or None
    """
    directory = args.json_path
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a directory.")
        return None, None

    # 自研系统结果文件: (filepath, is_weaviate)
    files = [(os.path.join(directory, f), False)
             for f in os.listdir(directory) if f.endswith('.json')]

    # Weaviate 结果文件（递归扫描子目录）
    weaviate_dir = args.weaviate_path
    if weaviate_dir and os.path.isdir(weaviate_dir):
        for root, _, fnames in os.walk(weaviate_dir):
            files.extend([(os.path.join(root, f), True)
                           for f in fnames if f.endswith('.json')])

    mode_points = defaultdict(dict)
    mode_labels = {}
    # recall x 轴模式：收集所有候选点 {mode_key: [(x_val, y_val, filename), ...]}
    mode_recall_candidates = defaultdict(list)
    # --recall 模式：收集所有候选，稍后按 recall 筛选
    # candidates: {mode_key: {x_val: [(recall, y_val, filename), ...]}}
    mode_candidates = defaultdict(lambda: defaultdict(list))
    target_recall = getattr(args, 'recall', None)
    recall_x = is_recall_x(args.x)

    for f, is_weaviate in files:
        try:
            file_params = parse_filename_params(os.path.basename(f))

            # 先用文件名筛选，避免加载大文件
            if is_weaviate:
                skip = False
                for param_name, arg_val in [
                    ('data_name', args.data_name),
                    ('m', args.m),
                    ('ef_construction', args.ef_construction),
                    ('ef_search', args.ef_search),
                    ('node_num', args.node_num),
                    ('threads', args.threads),
                    ('query_threads', args.num_query_threads),
                ]:
                    if arg_val is not None and file_params.get(param_name) != arg_val:
                        skip = True
                        break
            else:
                skip = False
                for param_name, arg_val in [
                    ('data_name', args.data_name),
                    ('m', args.m),
                    ('ef_construction', args.ef_construction),
                    ('ef_search', args.ef_search),
                    ('limited_level', args.limited_level),
                    ('node_num', args.node_num),
                    ('threads', args.threads),
                    ('query_threads', args.num_query_threads),
                ]:
                    if arg_val is not None and file_params.get(param_name) != arg_val:
                        skip = True
                        break
            if skip:
                continue

            if not is_weaviate and not file_params.get('shard') and args.replication is not None:
                if file_params.get('replication') != args.replication:
                    continue

            # --global 模式：排除 layer 文件，no-layer 数据归为 global
            is_global = getattr(args, 'global', False)
            if is_global and not is_weaviate and file_params.get('layer'):
                continue

            data = load_json_summary(f)
            mode_key, label = classify_mode(f, data, is_weaviate=is_weaviate,
                                            is_global=is_global)
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
            elif recall_x:
                # recall 作为 x 轴：收集所有候选，稍后按档位聚合取 y 最大
                mode_recall_candidates[mode_key].append(
                    (x_val, y_val, os.path.basename(f)))
            else:
                mode_points[mode_key][x_val] = y_val
                print(f"[{label}] {os.path.basename(f)} | {args.x}={x_val} | {args.y}={y_val}")

        except Exception as e:
            print(f"Skipping {f}: {e}")

    if args.y == 'qps' and recall_x:
        cotra_base_qps = 68900
        cotra_base_node_num = 8
        cotra_base_threads = 8
        node_scale = (args.node_num / cotra_base_node_num) if args.node_num is not None else 1.0
        thread_scale = (args.threads / cotra_base_threads) if args.threads is not None else 1.0
        cotra_qps = cotra_base_qps * node_scale * thread_scale
        mode_points['cotra'][0.95] = cotra_qps
        mode_labels['cotra'] = 'CoTra'
        print(f"[CoTra] hardcoded | {args.x}=0.95 | {args.y}={cotra_qps}")

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

    # recall x 轴时，按 0.05 档位聚合，同档位取 y 最大
    if recall_x and mode_recall_candidates:
        if mode_debug is None:
            mode_debug = defaultdict(dict)
        for mode_key, candidates in mode_recall_candidates.items():
            bucketed, debug_info = bucket_recall_points(candidates, min_recall=getattr(args, 'min_recall', None))
            mode_points[mode_key] = bucketed
            mode_debug[mode_key].update(debug_info)
            label = mode_labels.get(mode_key, mode_key)
            for bucket in sorted(bucketed.keys()):
                raw_recall, fname = debug_info[bucket]
                print(f"[{label}] {fname} | {args.x}={bucket} | {args.y}={bucketed[bucket]}"
                      f" | raw_recall={raw_recall:.4f}")

    if not any(mode_points.values()):
        print("No valid data points found.")
        return None, None, None

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
            'replication': args.replication,
            'threads': args.threads,
            'query_threads': args.num_query_threads,
            'shard': args.shard,
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
    recall_candidates = []  # [(x_val, y_val, filename), ...]
    candidates = defaultdict(list)  # x_val -> [(recall, y_val, filename), ...]
    target_recall = getattr(args, 'recall', None)
    recall_x = is_recall_x(args.x)

    for f in files:
        try:
            data = load_json_summary(f)
            if (
                args.data_name is not None and get_param_value(data, "data_name") != args.data_name or
                args.m is not None and get_param_value(data, "m") != args.m or
                args.ef_construction is not None and get_param_value(data, "ef_construction") != args.ef_construction or
                args.ef_search is not None and get_param_value(data, "ef_search") != args.ef_search or
                args.limited_level is not None and get_param_value(data, "limited_level") != args.limited_level or
                args.node_num is not None and get_param_value(data, "node_num") != args.node_num or
                args.partition_mode is not None and get_param_value(data, "partition_mode") != args.partition_mode or
                args.replication is not None and get_param_value(data, "replication_level") != args.replication or
                args.threads is not None and get_param_value(data, "num_threads") != args.threads or
                args.num_query_threads is not None and get_param_value(data, "num_query_threads") != args.num_query_threads
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
            elif recall_x:
                recall_candidates.append((x_val, y_val, os.path.basename(f)))
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

    if recall_x and recall_candidates:
        points, _ = bucket_recall_points(recall_candidates, min_recall=getattr(args, 'min_recall', None))

    if not points:
        print("No valid data points found.")
        return None

    sorted_pts = sorted(points.items(), key=lambda p: p[0])
    label = args.y
    return {label: sorted_pts}


def collect_group(args):
    """group 模式：按 group_by 参数分组，每个分组一条曲线。

    返回 {group_label: [(x, y), ...]}，多个 series。
    """
    group_by = getattr(args, 'group_by', None)
    if not group_by:
        print("Error: --group_by is required for group mode.")
        return None

    directory = args.json_path
    files = []
    if os.path.isdir(directory):
        files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.json')]
    else:
        files = [directory]

    # {group_val: {x_val: y_val}}
    group_points = defaultdict(dict)

    for f in files:
        try:
            file_params = parse_filename_params(os.path.basename(f))

            # 筛选：跳过不匹配的文件
            skip = False
            for param_name, arg_val in [
                ('data_name', args.data_name),
                ('m', args.m),
                ('ef_construction', args.ef_construction),
                ('ef_search', args.ef_search),
                ('limited_level', args.limited_level),
                ('node_num', args.node_num),
                ('threads', args.threads),
                ('query_threads', args.num_query_threads),
            ]:
                # group_by 参数不做筛选
                if param_name == group_by:
                    continue
                if arg_val is not None and file_params.get(param_name) != arg_val:
                    skip = True
                    break
            if skip:
                continue

            if args.replication is not None and not file_params.get('shard'):
                if file_params.get('replication') != args.replication:
                    continue

            # 按 partition_mode 筛选
            if getattr(args, 'partition_mode', None) is not None and group_by != 'partition_mode':
                if file_params.get('partition_mode') != args.partition_mode:
                    continue

            # 按 layer 筛选
            if getattr(args, 'layer', None) is not None:
                if file_params.get('layer') != args.layer:
                    continue

            # 只取非 shard、非 weaviate 的分布式结果
            if file_params.get('shard') or not file_params.get('distributed'):
                continue

            data = load_json_summary(f)

            x_val = get_param_value(data, args.x)
            if x_val is None:
                x_val = file_params.get(args.x)
            y_val = get_y_value(data, args.y, args.data_name)

            group_val = file_params.get(group_by)
            if group_val is None:
                group_val = get_param_value(data, group_by)

            if x_val is None or y_val is None or group_val is None:
                continue

            group_points[group_val][x_val] = y_val
            print(f"[{group_by}={group_val}] {os.path.basename(f)} | {args.x}={x_val} | {args.y}={y_val}")

        except Exception as e:
            print(f"Skipping {f}: {e}")

    if not group_points:
        print("No valid data points found.")
        return None

    # 转为 {label: [(x, y), ...]} 格式，按 group_val 排序
    series_data = {}
    for gval in sorted(group_points.keys()):
        label = f"{group_by}={gval}"
        series_data[label] = sorted(group_points[gval].items(), key=lambda p: p[0])

    return series_data


# ============================================================
# 绘图函数（按 chart 类型分类）
# ============================================================

def save_figure(fig, args, prefix, extra_keys=None):
    """Save figure as PDF to ./plot/pdf and PNG to ./plot/png."""
    pdf_dir = os.path.join(args.output_dir, 'pdf')
    png_dir = os.path.join(args.output_dir, 'png')
    os.makedirs(pdf_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)
    basename = make_output_filename(prefix, args, extra_keys)
    pdf_path = os.path.join(pdf_dir, basename)
    png_path = os.path.join(png_dir, basename[:-4] + '.png')
    fig.savefig(pdf_path, bbox_inches='tight')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {pdf_path}")
    print(f"Plot saved to {png_path}")
    plt.close(fig)

MODE_ORDER = ['shard', 'global', 'weaviate', 'cotra', 'no-layer', 'layer']
LINE_STYLES = {
    'shard':    {'marker': 'o', 'linestyle': '-',  'color': '#cae9ff'},
    'global':   {'marker': 's', 'linestyle': '--', 'color': '#1a759f'},
    'no-layer': {'marker': 's', 'linestyle': '--', 'color': '#1a759f'},
    'layer':    {'marker': '^', 'linestyle': '-.', 'color': '#5fa8d3'},
    'weaviate': {'marker': 'D', 'linestyle': ':',  'color': '#a2d2ff'},
    'cotra':    {'marker': 'X', 'linestyle': '-',  'color': '#74c0fc'},
}
BAR_STYLES = {
    'shard':    {'color': '#cae9ff', 'hatch': '\\\\'},
    'global':   {'color': '#1a759f', 'hatch': 'xx'},
    'no-layer': {'color': '#1a759f', 'hatch': 'xx'},
    'layer':    {'color': '#5fa8d3', 'hatch': '//'},
    'weaviate': {'color': '#a2d2ff', 'hatch': '--'},
    'cotra':    {'color': '#74c0fc', 'hatch': '||'},
}


def draw_line_compare(mode_points, mode_labels, args, mode_debug=None):
    """折线图 × compare 模式"""
    C = PLOT_CONFIG
    debug = getattr(args, 'debug', False) and mode_debug is not None
    n_x = max(len(pts) for pts in mode_points.values())
    fig, ax = make_fig(calc_fig_width(n_x), debug)

    for mode_key in MODE_ORDER:
        points = mode_points.get(mode_key, {})
        if not points or all(v == 0 for v in points.values()):
            continue
        sorted_pts = sorted(points.items(), key=lambda p: p[0])
        xs, ys = zip(*sorted_pts)
        style = LINE_STYLES[mode_key]
        ax.plot(xs, ys, marker=style['marker'], linestyle=style['linestyle'],
                color=style['color'], label=mode_labels[mode_key],
                linewidth=C['linewidth'], markersize=C['markersize'])

    style_ax(ax, display_label(args.x), display_label(args.y))
    n_labels = sum(1 for m in MODE_ORDER if mode_points.get(m) and any(v != 0 for v in mode_points[m].values()))
    setup_legend(ax, n_labels)
    apply_layout(fig)

    save_figure(fig, args, "line-compare")


def draw_bar_compare(mode_points, mode_labels, args, mode_debug=None):
    """柱状图 × compare 模式"""
    C = PLOT_CONFIG
    debug = getattr(args, 'debug', False) and mode_debug is not None
    all_x_vals = sorted(set().union(*[pts.keys() for pts in mode_points.values()]))
    x_labels = [str(v) for v in all_x_vals]

    active_modes = [m for m in MODE_ORDER if mode_points.get(m) and any(v != 0 for v in mode_points[m].values())]

    n_groups = len(all_x_vals)
    max_bars_per_group = len(active_modes)
    bar_width = min(0.8 / max(max_bars_per_group, 1), C['bar_width_max'])
    x_indices = np.arange(n_groups)

    fig, ax = make_fig(calc_fig_width(n_groups * max_bars_per_group), debug)

    legend_added = set()
    for g, x_val in enumerate(all_x_vals):
        present_modes = [m for m in active_modes if x_val in mode_points.get(m, {})]
        n_present = len(present_modes)
        for i, mode_key in enumerate(present_modes):
            offset = (i - (n_present - 1) / 2) * bar_width
            bstyle = BAR_STYLES[mode_key]
            val = mode_points[mode_key][x_val]
            label = mode_labels[mode_key] if mode_key not in legend_added else None
            legend_added.add(mode_key)
            ax.bar(x_indices[g] + offset, val, bar_width,
                   label=label,
                   color=bstyle['color'], hatch=bstyle['hatch'],
                   edgecolor='black', alpha=0.85)

    ax.set_xticks(x_indices)
    ax.set_xticklabels(x_labels, fontsize=C['font_tick_label'])
    style_ax(ax, display_label(args.x), display_label(args.y), tick_axis='y', grid_axis='y')
    setup_legend(ax, len(active_modes))
    apply_layout(fig)

    save_figure(fig, args, "bar-compare")


def draw_scatter_compare(mode_points, mode_labels, args, mode_debug=None):
    """散点图 × compare 模式"""
    C = PLOT_CONFIG
    debug = getattr(args, 'debug', False) and mode_debug is not None
    n_x = max(len(pts) for pts in mode_points.values())
    fig, ax = make_fig(calc_fig_width(n_x), debug)

    for mode_key in MODE_ORDER:
        points = mode_points.get(mode_key, {})
        if not points or all(v == 0 for v in points.values()):
            continue
        sorted_pts = sorted(points.items(), key=lambda p: p[0])
        xs, ys = zip(*sorted_pts)
        ax.scatter(xs, ys, color=LINE_STYLES[mode_key]['color'],
                   label=mode_labels[mode_key], s=80, alpha=0.8,
                   marker=LINE_STYLES[mode_key]['marker'])

    style_ax(ax, display_label(args.x), display_label(args.y))
    n_labels = sum(1 for m in MODE_ORDER if mode_points.get(m) and any(v != 0 for v in mode_points[m].values()))
    setup_legend(ax, n_labels)
    apply_layout(fig)

    save_figure(fig, args, "scatter-compare")


def draw_line_compare_split(mode_points, mode_labels, args, mode_debug=None):
    """折线图 × compare 模式 × split：每个 mode 一个子图"""
    C = PLOT_CONFIG
    vertical = getattr(args, 'vertical', False)
    active_modes = [m for m in MODE_ORDER if mode_points.get(m) and any(v != 0 for v in mode_points[m].values())]
    n = len(active_modes)
    if n == 0:
        return

    n_x = max(len(pts) for pts in mode_points.values())
    sub_w = calc_fig_width(n_x)
    sub_h = max(sub_w * C['fig_height_ratio'], C['fig_height_min'])

    if vertical:
        fig, axes = plt.subplots(n, 1, figsize=(sub_w, sub_h * n))
    else:
        fig, axes = plt.subplots(1, n, figsize=(sub_w * n, sub_h))
    if n == 1:
        axes = [axes]

    for ax, mode_key in zip(axes, active_modes):
        points = mode_points[mode_key]
        sorted_pts = sorted(points.items(), key=lambda p: p[0])
        xs, ys = zip(*sorted_pts)
        style = LINE_STYLES[mode_key]
        ax.plot(xs, ys, marker=style['marker'], linestyle=style['linestyle'],
                color=style['color'], label=mode_labels[mode_key],
                linewidth=C['linewidth'], markersize=C['markersize'])
        style_ax(ax, display_label(args.x), display_label(args.y),
                 title=mode_labels[mode_key])

    apply_layout(fig)
    save_figure(fig, args, "line-compare-split")


def draw_bar_compare_split(mode_points, mode_labels, args, mode_debug=None):
    """柱状图 × compare 模式 × split：每个 mode 一个子图"""
    C = PLOT_CONFIG
    vertical = getattr(args, 'vertical', False)
    active_modes = [m for m in MODE_ORDER if mode_points.get(m) and any(v != 0 for v in mode_points[m].values())]
    n = len(active_modes)
    if n == 0:
        return

    all_x_vals = sorted(set().union(*[pts.keys() for pts in mode_points.values()]))
    x_labels = [str(v) for v in all_x_vals]
    n_groups = len(all_x_vals)
    x_indices = np.arange(n_groups)
    bar_width = min(0.6, max(0.2, 6.0 / max(n_groups, 1)))

    sub_w = calc_fig_width(n_groups)
    sub_h = max(sub_w * C['fig_height_ratio'], C['fig_height_min'])

    if vertical:
        fig, axes = plt.subplots(n, 1, figsize=(sub_w, sub_h * n))
    else:
        fig, axes = plt.subplots(1, n, figsize=(sub_w * n, sub_h))
    if n == 1:
        axes = [axes]

    for ax, mode_key in zip(axes, active_modes):
        bstyle = BAR_STYLES[mode_key]
        points = mode_points[mode_key]
        vals = [points.get(xv, 0) for xv in all_x_vals]
        ax.bar(x_indices, vals, bar_width,
               color=bstyle['color'], hatch=bstyle['hatch'],
               edgecolor='black', alpha=0.85)
        ax.set_xticks(x_indices)
        ax.set_xticklabels(x_labels, fontsize=C['font_tick_label'])
        style_ax(ax, display_label(args.x), display_label(args.y),
                 title=mode_labels[mode_key], tick_axis='y', grid_axis='y')

    apply_layout(fig)
    save_figure(fig, args, "bar-compare-split")


def draw_scatter_compare_split(mode_points, mode_labels, args, mode_debug=None):
    """散点图 × compare 模式 × split：每个 mode 一个子图"""
    C = PLOT_CONFIG
    vertical = getattr(args, 'vertical', False)
    active_modes = [m for m in MODE_ORDER if mode_points.get(m) and any(v != 0 for v in mode_points[m].values())]
    n = len(active_modes)
    if n == 0:
        return

    n_x = max(len(pts) for pts in mode_points.values())
    sub_w = calc_fig_width(n_x)
    sub_h = max(sub_w * C['fig_height_ratio'], C['fig_height_min'])

    if vertical:
        fig, axes = plt.subplots(n, 1, figsize=(sub_w, sub_h * n))
    else:
        fig, axes = plt.subplots(1, n, figsize=(sub_w * n, sub_h))
    if n == 1:
        axes = [axes]

    for ax, mode_key in zip(axes, active_modes):
        points = mode_points[mode_key]
        sorted_pts = sorted(points.items(), key=lambda p: p[0])
        xs, ys = zip(*sorted_pts)
        ax.scatter(xs, ys, color=LINE_STYLES[mode_key]['color'],
                   s=80, alpha=0.8, marker=LINE_STYLES[mode_key]['marker'])
        style_ax(ax, display_label(args.x), display_label(args.y),
                 title=mode_labels[mode_key])

    apply_layout(fig)
    save_figure(fig, args, "scatter-compare-split")


_MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', 'h', '<', '>']


def draw_line_single(series_data, args, prefix="line"):
    """折线图 × 单系列模式（detail / merge）"""
    C = PLOT_CONFIG
    fig, ax = make_fig(C['fig_width_single'])

    all_xs = []
    multi = len(series_data) > 1
    for i, (label, pts) in enumerate(series_data.items()):
        xs, ys = zip(*pts)
        all_xs.extend(xs)
        marker = _MARKERS[i % len(_MARKERS)] if multi else None
        lw = C['linewidth'] if multi else 1.5
        ax.plot(xs, ys, linestyle='-', linewidth=lw, marker=marker,
                markersize=C['markersize'], label=label)

    if all_xs:
        ax.set_xlim(min(all_xs), max(all_xs))

    x_label = "Query Index" if args.mode == "detail" else display_label(args.x)
    style_ax(ax, x_label, display_label(args.y))
    setup_legend(ax, len(series_data))
    apply_layout(fig)

    save_figure(fig, args, prefix)


def draw_scatter_single(series_data, args, prefix="scatter"):
    """散点图 × 单系列模式（detail / merge）"""
    C = PLOT_CONFIG
    n_points = max(len(pts) for pts in series_data.values())
    fig, ax = make_fig(C['fig_width_single'])

    all_xs = []
    for label, pts in series_data.items():
        xs, ys = zip(*pts)
        all_xs.extend(xs)
        s = 1 if n_points > 500 else 20
        alpha = 0.3 if n_points > 500 else 0.6
        ax.scatter(xs, ys, alpha=alpha, s=s, label=label)

    if all_xs:
        ax.set_xlim(min(all_xs), max(all_xs))

    x_label = "Query Index" if args.mode == "detail" else display_label(args.x)
    style_ax(ax, x_label, display_label(args.y))
    setup_legend(ax, len(series_data))
    apply_layout(fig)

    save_figure(fig, args, prefix)


def draw_bar_single(series_data, args, prefix="bar"):
    """柱状图 × 单系列模式（detail / merge）"""
    C = PLOT_CONFIG
    fig_width = calc_fig_width(max(len(pts) for pts in series_data.values()))
    fig, ax = make_fig(fig_width)

    for label, pts in series_data.items():
        xs, ys = zip(*pts)
        x_labels = [str(x) for x in xs]
        x_indices = np.arange(len(xs))
        bar_width = min(0.6, max(0.2, 6.0 / max(len(xs), 1)))
        ax.bar(x_indices, ys, width=bar_width, color='#1f77b4', alpha=0.85, label=label)
        ax.set_xticks(x_indices)
        ax.set_xticklabels(x_labels, fontsize=C['font_tick_label'],
                           rotation=45 if len(xs) > 20 else 0)

    x_label = "Query Index" if args.mode == "detail" else display_label(args.x)
    style_ax(ax, x_label, display_label(args.y), tick_axis='y', grid_axis='y')
    setup_legend(ax, len(series_data))
    apply_layout(fig)

    save_figure(fig, args, prefix)


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Vector Retrieval Results")
    parser.add_argument("--json_path", type=str, default="./results/cpp")
    parser.add_argument("--weaviate_path", type=str, default="./weaviate/results/")
    parser.add_argument("--output_dir", type=str, default="./plot")

    # 两个独立维度
    parser.add_argument("--chart", type=str, default="line", choices=["line", "bar", "scatter"],
                        help="图表类型: line (折线图), bar (柱状图), scatter (散点图)")
    parser.add_argument("--mode", type=str, default="compare", choices=["compare", "detail", "merge", "group"],
                        help="数据模式: compare (三路对比), detail (单实验逐查询), merge (单曲线汇总), group (按参数分组多曲线)")

    # 通用参数
    parser.add_argument("--x", type=str, default="ef_search", help="X-axis parameter")
    parser.add_argument("--y", type=str, default="qps", help="Y-axis metric")

    # 筛选参数
    parser.add_argument("--data_name", type=str, default="sift", help="Dataset name")
    parser.add_argument("--m", type=int, default=None, help="Max connections per element")
    parser.add_argument("--ef_construction", type=int, default=None, help="ef_construction")
    parser.add_argument("--ef_search", type=int, default=None, help="ef_search (for detail/merge filtering)")
    parser.add_argument("--limited_level", type=int, default=None, help="Level limit for HNSW")
    parser.add_argument("--node_num", type=int, default=5, help="Number of cluster nodes")
    parser.add_argument("--partition_mode", type=str, default="k-means", help="Partition mode (for detail/merge)")
    parser.add_argument("--layer", action='store_true', default=None, help="Filter for layer partition mode")
    parser.add_argument("--no_layer", action='store_true', default=None, help="Filter for non-layer partition mode")
    parser.add_argument("--replication", type=int, default=None, help="Redundancy level")
    parser.add_argument("--threads", "--num_threads", type=int, default=None, dest="threads",
                        help="Number of server-side search threads per node (num_threads)")
    parser.add_argument("--num_query_threads", type=int, default=None,
                        help="Number of client query processes (num_query_threads)")
    parser.add_argument("--group_by", type=str, default=None, help="Parameter to group by (for group mode, e.g. 'm' or 'ef_search')")
    parser.add_argument("--shard", action='store_true', help="Enable shard mode")
    parser.add_argument("--global", action='store_true', dest="global",
                        help="Global mode: use no-layer data as 'Global', exclude layer")
    parser.add_argument("--cotra", action='store_true', help="Enable CoTra mode")
    parser.add_argument("--weaviate", action='store_true', help="Enable Weaviate mode")

    parser.add_argument("--distributed", action='store_true', help="Enable distributed mode (for detail/merge)")

    # recall 约束
    parser.add_argument("--min_recall", type=float, default=0.8,
                        help="Minimum recall bucket when x-axis is recall (default: 0.8)")
    parser.add_argument("--recall", type=float, default=None,
                        help="Target recall (e.g. 0.9), pick experiments closest to this recall per x value")
    parser.add_argument("--debug", action='store_true',
                        help="Debug mode: annotate recall and filename on chart (used with --recall)")
    parser.add_argument("--split", action='store_true',
                        help="Split compare chart into separate subplots per mode (horizontal by default)")
    parser.add_argument("--vertical", action='store_true',
                        help="When used with --split, arrange subplots vertically instead of horizontally")

    args = parser.parse_args()

    # 将 --layer / --no_layer 合并为 args.layer: True/False/None
    if args.layer:
        args.layer = True
    elif args.no_layer:
        args.layer = False
    else:
        args.layer = None

    if not os.path.exists(args.json_path):
        print(f"Error: {args.json_path} not found.")
        exit(1)

    run_draw(args)


def run_draw(args):
    """核心绘图入口，可被外部直接调用"""
    # 兼容 batch_draw：将 no_layer 合并到 layer
    if getattr(args, 'no_layer', False) and getattr(args, 'layer', None) is None:
        args.layer = False

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.mode == "compare":
        mode_points, mode_labels, mode_debug = collect_compare(args)
        if mode_points is None:
            return

        # 按模式开关过滤：只保留为 True 的模式；全部未设置时保留所有
        _mode_flags = {
            'shard':    getattr(args, 'shard', False),
            'layer':    getattr(args, 'layer', None) is True,
            'no-layer': getattr(args, 'no_layer', False),
            'global':   getattr(args, 'global', False),
            'cotra':    getattr(args, 'cotra', False),
            'weaviate': getattr(args, 'weaviate', False),
        }
        _enabled = {k for k, v in _mode_flags.items() if v}
        if _enabled:
            for mk in list(mode_points.keys()):
                if mk not in _enabled:
                    del mode_points[mk]
                    mode_labels.pop(mk, None)
                    if mode_debug:
                        mode_debug.pop(mk, None)

        if not any(mode_points.values()):
            print("No valid data points after mode filtering.")
            return

        split = getattr(args, 'split', False)
        if args.chart == "line":
            (draw_line_compare_split if split else draw_line_compare)(mode_points, mode_labels, args, mode_debug)
        elif args.chart == "bar":
            (draw_bar_compare_split if split else draw_bar_compare)(mode_points, mode_labels, args, mode_debug)
        elif args.chart == "scatter":
            (draw_scatter_compare_split if split else draw_scatter_compare)(mode_points, mode_labels, args, mode_debug)

    elif args.mode == "detail":
        series_data, json_file = collect_detail(args)
        if series_data is None:
            return
        if args.chart == "line":
            draw_line_single(series_data, args, prefix="line-detail")
        elif args.chart == "bar":
            draw_bar_single(series_data, args, prefix="bar-detail")
        elif args.chart == "scatter":
            draw_scatter_single(series_data, args, prefix="scatter-detail")

    elif args.mode == "merge":
        series_data = collect_merge(args)
        if series_data is None:
            return
        if args.chart == "line":
            draw_line_single(series_data, args, prefix="line-merge")
        elif args.chart == "bar":
            draw_bar_single(series_data, args, prefix="bar-merge")
        elif args.chart == "scatter":
            draw_scatter_single(series_data, args, prefix="scatter-merge")

    elif args.mode == "group":
        series_data = collect_group(args)
        if series_data is None:
            return
        if args.chart == "line":
            draw_line_single(series_data, args, prefix="line-group")
        elif args.chart == "bar":
            draw_bar_single(series_data, args, prefix="bar-group")
        elif args.chart == "scatter":
            draw_scatter_single(series_data, args, prefix="scatter-group")
