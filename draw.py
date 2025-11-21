import os
import argparse
import matplotlib.pyplot as plt
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Vector Retrieval Results")
    parser.add_argument("--json_path", type=str, required=True, help="Path to the results JSON file or directory containing multiple JSON files")
    parser.add_argument("--merge", action='store_true', help="Whether to merge multiple JSON files")
    parser.add_argument("--output_dir", type=str, default="./plot", help="Directory to save plot")
    parser.add_argument("--x", type=str, default="ef_search", help="X-axis parameter for plotting (Merge mode)")
    parser.add_argument("--y", type=str, default="avg_remote_transfer_latency_query", help="Y-axis parameter for plotting")

    args = parser.parse_args()

    if not os.path.exists(args.json_path):
        print(f"Error: File {args.json_path} not found.")
        exit(1)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    def load_json(path):
        with open(path, 'r') as f:
            return json.load(f)

    # 辅助函数：在 JSON 结构中查找参数 (用于 Merge 模式的 X 轴)
    def get_param_value(data, key):
        # 1. 尝试直接在根节点找
        if key in data: return data[key]
        # 2. 尝试在 hnsw_params 中找
        if 'hnsw_params' in data and key in data['hnsw_params']:
            return data['hnsw_params'][key]
        # 3. 尝试在 cluster_params 中找
        if 'cluster_params' in data and key in data['cluster_params']:
            return data['cluster_params'][key]
        return None

    # 辅助函数：查找指标值 (用于 Merge 模式的 Y 轴)
    def get_metric_value(data, key):
        # 1. 精确匹配
        if key in data: return data[key]
        # 2. 模糊匹配 (例如用户输入 avg_recall，匹配 avg_recall@10)
        for k in data.keys():
            if k.startswith(key) and (k == key or k[len(key)] == '@'):
                return data[k]
        return None

    plt.figure(figsize=(10, 6))
    
    # --- 1. Merge 模式: 比较不同参数下的汇总指标 ---
    if args.merge:
        files = []
        if os.path.isdir(args.json_path):
            files = [os.path.join(args.json_path, f) for f in os.listdir(args.json_path) if f.endswith('.json')]
        else:
            files = [args.json_path]
        
        points = []
        for f in files:
            try:
                data = load_json(f)
                
                # 获取 X 轴 (参数)
                x_val = get_param_value(data, args.x)
                
                # 获取 Y 轴 (汇总指标)
                y_val = get_metric_value(data, args.y)
                
                if x_val is not None and y_val is not None:
                    points.append((x_val, y_val))
                else:
                    print(f"Warning: Could not find x='{args.x}' or y='{args.y}' in {os.path.basename(f)}")
            except Exception as e:
                print(f"Skipping {f}: {e}")
        
        # 按 X 轴排序并绘图
        points.sort(key=lambda p: p[0])
        if points:
            xs, ys = zip(*points)
            plt.plot(xs, ys, marker='s', linestyle='--', color='b', label='Merged Results')
            plt.xlabel(f"Parameter: {args.x}")
            plt.title(f"{args.y} vs {args.x}")
            
            # 在点旁边标注数值
            for x, y in points:
                plt.text(x, y, f"{y:.4f}", fontsize=9, ha='right')
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
    
    output_filename = f"{'merge' if args.merge else 'single'}_{args.y}_vs_{args.x if args.merge else 'query'}.pdf"
    output_path = os.path.join(args.output_dir, output_filename)
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
