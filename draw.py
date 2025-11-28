import os
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Vector Retrieval Results")
    parser.add_argument("--json_path", type=str, default="./results/cpp", help="Path to the results JSON file or directory containing multiple JSON files")
    parser.add_argument("--merge", action='store_true', help="Whether to merge multiple JSON files")
    parser.add_argument("--output_dir", type=str, default="./plot", help="Directory to save plot")
    parser.add_argument("--x", type=str, default="node_num", help="X-axis parameter for plotting (Merge mode)")
    parser.add_argument("--y", type=str, default="avg_remote_transfer_cnt", help="Y-axis parameter for plotting")
    parser.add_argument("--data_name", type=str, default="siftsmall", help="Name of the dataset")
    parser.add_argument("--m", type=int, default=16, help="Max number of connections per element")
    parser.add_argument("--ef_construction", type=int, default=64, help="Size of the dynamic candidate list for construction")
    parser.add_argument("--ef_search", type=int, default=64, help="Size of the dynamic candidate list for search")
    parser.add_argument("--limited_level", type=int, default=5, help="Level limit for HNSW")
    parser.add_argument("--node_num", type=int, default=None, help="Number of cluster nodes")
    parser.add_argument("--partition_mode", type=str, default="k-means", help="Data partition mode across nodes")
    parser.add_argument("--redundancy", type=int, default=1, help="Redundancy level for data replication across nodes")

    args = parser.parse_args()

    if not os.path.exists(args.json_path):
        print(f"Error: File {args.json_path} not found.")
        exit(1)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    def load_json(path):
        with open(path, 'r') as f:
            return json.load(f)

    # 辅助函数：在 JSON 结构中查找参数
    def get_param_value(data, key):
        # 尝试直接在根节点找
        if key in data: return data[key]
        # 尝试在 hnsw_params 中找
        if 'hnsw_params' in data and key in data['hnsw_params']:
            return data['hnsw_params'][key]
        # 尝试在 cluster_params 中找
        if 'cluster_params' in data and key in data['cluster_params']:
            return data['cluster_params'][key] 
        return None

    plt.figure(figsize=(10, 6))
    
    # --- 1. Merge 模式: 比较不同参数下的汇总指标 ---
    if args.merge:
        files = []
        if os.path.isdir(args.json_path):
            files = [os.path.join(args.json_path, f) for f in os.listdir(args.json_path) if f.endswith('.json')]
        else:
            files = [args.json_path]

        points = {}
        for f in files:
            try:
                data = load_json(f)
                # 参数筛选
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

                # 获取 X 轴 (参数)
                x_val = get_param_value(data, args.x)
                # 获取 Y 轴 (汇总指标)
                if args.y == "red_per_node":
                    redundancy_num = get_param_value(data, "redundancy_num")
                    node_num = get_param_value(data, "node_num")
                    y_val = redundancy_num / node_num if node_num else 0
                elif args.y == "data_per_node":
                    if args.data_name == "siftsmall":
                        total_data = 10000
                    elif args.data_name == "sift":
                        total_data = 1000000
                    
                    total_data_redundant = total_data + get_param_value(data, "redundancy_num")
                    node_num = get_param_value(data, "node_num")
                    y_val = total_data_redundant / node_num if node_num else 0
                else:
                    y_val = get_param_value(data, args.y)
                    node_num = get_param_value(data, "node_num")

                print(f"File: {f} | {args.x}: {x_val} | {args.y}: {y_val}")

                points[x_val] = y_val

            except Exception as e:
                print(f"Skipping {f}: {e}")
    
        # 按 X 轴排序并绘图
        points_list = sorted(points.items(), key=lambda p: p[0])
        if points_list:
            xs, ys = zip(*points_list)
            plt.plot(xs, ys, marker='s', linestyle='--', color='b')
            plt.xlabel(f"{args.x}")
            plt.title(
                f"data_name={args.data_name}, m={args.m}, efc={args.ef_construction}, efs={args.ef_search}, ll={args.limited_level}\n"
                f"node_num={args.node_num}, partition_mode={args.partition_mode}, redundancy={args.redundancy}"
            )
            
            # 在点旁边标注数值
            for x, y in points_list:
                plt.text(x, y + 0.1, f"{y:.0f}", fontsize=9, ha='right', va='bottom')
        else:
            print("No valid data points found.")
    
    # --- 2. 非 Merge 模式: 查看单次实验中每次查询的详细数据 ---
    else:
        if os.path.isdir(args.json_path):
            print("Error: In non-merge mode, json_path must be a file.")
            exit(1)
            
        data = load_json(args.json_path)
        
        # 获取详细数据字典 (例如 recall, remote_transfer_cnt)
        raw_y_data = data.get(args.y)
        
        if raw_y_data and isinstance(raw_y_data, dict):
            # JSON 中的 key 是字符串 "0", "1"... 需要转成 int 排序
            # 过滤掉非数字 key (如果有的话)
            sorted_items = []
            for k, v in raw_y_data.items():
                if k.isdigit():
                    sorted_items.append((int(k), v))
            
            sorted_items.sort(key=lambda x: x[0])
            
            if sorted_items:
                x_vals = [item[0] for item in sorted_items]
                y_vals = [item[1] for item in sorted_items]
                
                plt.plot(x_vals, y_vals, marker='.', linestyle='-', linewidth=0.5, markersize=2, label=os.path.basename(args.json_path))
                plt.xlabel("Query Index (nth query)")
                plt.title(f"{args.y} per Query")
            else:
                print(f"Error: No valid numeric keys found in {args.y}")
        else:
            print(f"Error: Key '{args.y}' not found or is not a dictionary in the JSON file.")
            print(f"Available keys: {list(data.keys())}")

    plt.ylabel(args.y)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    if args.merge:
        output_filename = (
            f"{args.x}-{args.y}-{args.data_name}-"
            f"m{args.m}-efc{args.ef_construction}-efs{args.ef_search}-ll{args.limited_level}-"
            + (f"node{args.node_num}-" if args.node_num is not None else "")
            + f"{args.partition_mode}-red{args.redundancy}.pdf"
        )
    else:
        base = os.path.splitext(os.path.basename(args.json_path))[0]
        output_filename = f"{base}_{args.y}.pdf"

    output_path = os.path.join(args.output_dir, output_filename)
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
