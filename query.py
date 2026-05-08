#!/usr/bin/env python
"""
query.py — 分布式查询客户端 (multiprocessing 版)
每个 worker 是独立进程（独立 PID），SDK 内部按 PID 哈希分配 sticky endpoint，
天然把请求分散到所有 worker 入口；避免单进程多线程下 hash(pid) 相同导致的
所有线程打到同一个 worker 入口的瓶颈。

用法:
  python query.py --data_name sift --query_rank 0 --query_world_size 1 --num_query_threads 40
"""

import os
import sys
import time
import json
import argparse
import random
import multiprocessing as mp

project_root = os.path.abspath(os.path.dirname(__file__))
build_dir = os.path.join(project_root, 'build')
sys.path.insert(0, build_dir)

# ---------- 参数 ----------
parser = argparse.ArgumentParser(description="Distributed Query Client (multiprocessing)")
parser.add_argument("--data_name", type=str, default="sift", help="Dataset name")
parser.add_argument("--query_path", type=str, default=None)
parser.add_argument("--groundtruth_path", type=str, default=None)
parser.add_argument("--dtype", type=str, default="float", choices=["float","double","int32","uint32","uint8"])
parser.add_argument("--k", type=int, default=10)
parser.add_argument("--ef_search", type=int, default=15)
parser.add_argument("--num_query_threads", type=int, default=32, help="Number of worker processes per instance")
parser.add_argument("--num_runs", type=int, default=1)
parser.add_argument("--query_rank", type=int, default=0)
parser.add_argument("--query_world_size", type=int, default=1)
parser.add_argument("--comm_mode", type=str, default="tcp", choices=["tcp","rdma"])
parser.add_argument("--barrier_dir", type=str, default=None)
parser.add_argument("--barrier_timeout", type=int, default=600)
args = parser.parse_args()

# ---------- 数据集路径：复用 main.py 的 preset 和数据读取 ----------
from main import DATASET_PRESETS, read_vectors, read_groundtruth

if args.query_path is None:
    args.query_path = DATASET_PRESETS[args.data_name]["query_path"]
if args.groundtruth_path is None:
    args.groundtruth_path = DATASET_PRESETS[args.data_name]["groundtruth_path"]

# ---------- 读取 config.json ----------
_CONFIG_PATH = os.path.join(project_root, "config.json")
with open(_CONFIG_PATH, "r") as f:
    _CFG = json.load(f)

nodes = _CFG["work_nodes"]
# 按 rank 分散入口节点，避免所有 query rank 的连接集中打到 nodes[0]。
# init_routes() 之后 ensure_connected() 会进一步按 PID hash 打散到 entry_node_ids。
entry_node = nodes[args.query_rank % len(nodes)]
sdk_ip = entry_node["ip"]
sdk_port = int(entry_node["port"])

# ---------- Worker 进程：spawn 启动；每个子进程独立加载数据和 SDK ----------
# 用 spawn 而非 fork：fork 在父进程已加载 pybind11 / numpy 时容易因线程锁状态死锁。
# 每个子进程自行读取 queries/groundtruth（OS page cache 命中后很快），并独立 init SDK。
_QUERIES = None
_GROUNDTRUTH = None
_MY_INDICES = None
_NUM_QUERIES = None
_PAYLOAD_CLS = None
_SDK_CLS = None
_SDK_IP = None
_SDK_PORT = None
_COMM_MODE = None

def _pool_init(dtype, sdk_ip, sdk_port, comm_mode,
               query_path, groundtruth_path, query_rank, query_world_size):
    """每个 pool worker 进程启动时调用一次：加载数据，准备好 Payload 类和 SDK 构造参数。"""
    global _PAYLOAD_CLS, _QUERIES, _GROUNDTRUTH, _MY_INDICES, _NUM_QUERIES
    global _SDK_CLS, _SDK_IP, _SDK_PORT, _COMM_MODE

    _QUERIES = read_vectors(query_path)
    _GROUNDTRUTH = read_groundtruth(groundtruth_path)
    _NUM_QUERIES = _QUERIES.shape[0]

    all_indices = list(range(_NUM_QUERIES))
    random.seed(42)
    random.shuffle(all_indices)
    _MY_INDICES = [q for j, q in enumerate(all_indices) if (j % query_world_size) == query_rank]

    import vector_retrieve as _vr
    _DTYPE_CLASS_MAP = {
        "float":  ("PayloadFloat",  "ClusterSDKClientFloat"),
        "double": ("PayloadDouble", "ClusterSDKClientDouble"),
        "int32":  ("PayloadInt32",  "ClusterSDKClientInt32"),
        "uint32": ("PayloadUInt32", "ClusterSDKClientUInt32"),
        "uint8":  ("PayloadInt32",  "ClusterSDKClientUInt8"),
    }
    payload_cls_name, sdk_cls_name = _DTYPE_CLASS_MAP[dtype]
    _PAYLOAD_CLS = getattr(_vr, payload_cls_name)
    _SDK_CLS = getattr(_vr, sdk_cls_name)
    _SDK_IP = sdk_ip
    _SDK_PORT = sdk_port
    _COMM_MODE = comm_mode
    print(f"[pid={os.getpid()}] pool worker initialized: queries={_NUM_QUERIES}, my_indices={len(_MY_INDICES)}", flush=True)

def _warmup(_):
    """空函数：用于触发所有 pool worker 完成 _pool_init。"""
    return os.getpid()


def _do_round(args_tuple):
    """单个进程跑一轮全部查询。返回该进程本轮的 stats。
    所有进程通过 scheduled_start 时间戳同步起始。"""
    (run_idx, thread_id, scheduled_start, k, num_query_threads,
     query_rank, query_world_size) = args_tuple

    global _PAYLOAD_CLS, _QUERIES, _GROUNDTRUTH, _MY_INDICES, _NUM_QUERIES
    global _SDK_CLS, _SDK_IP, _SDK_PORT, _COMM_MODE

    # 每轮创建新的 SDK 客户端（与老代码 query_worker_process 完全一致）
    sdk_client = _SDK_CLS(ip=_SDK_IP, port=_SDK_PORT, comm_mode=_COMM_MODE)
    if not sdk_client.init_routes():
        if thread_id == 0:
            print(f"[pid={os.getpid()}] WARNING: init_routes failed", flush=True)

    my_pid = os.getpid()
    print(f"[pid={my_pid}] _do_round run={run_idx} tid={thread_id} start, "
          f"indices={len(_MY_INDICES)}", flush=True)

    try:
        # 同步等到 scheduled_start
        while True:
            delta = scheduled_start - time.time()
            if delta <= 0:
                break
            if delta > 0.05:
                time.sleep(delta - 0.05)
            # 最后 50ms busy-loop

        actual_start = time.time()
        drift_ms = (actual_start - scheduled_start) * 1000

        # 每个进程独立打乱 indices
        proc_indices = _MY_INDICES[:]
        random.seed(thread_id + query_rank * 1000)
        random.shuffle(proc_indices)

        qid_base = query_rank * _NUM_QUERIES * num_query_threads

        recall_results, remote_transfer_cnt, remote_transfer_data, distance_cnt, backtrack_cnt, pull_cnt, pull_data = {}, {}, {}, {}, {}, {}, {}

        PIPELINE_DEPTH = 16
        pending = {}
        sent, received = 0, 0

        for idx, i in enumerate(proc_indices):
            payload = _PAYLOAD_CLS()
            qid = i + thread_id * _NUM_QUERIES + qid_base
            sdk_client.query_send(qid, _QUERIES[i], payload, k=k)
            pending[qid] = i
            sent += 1
            while sent - received >= PIPELINE_DEPTH:
                ret_qid, results, ret_payload = sdk_client.query_recv()
                orig_idx = pending.pop(ret_qid)
                recall_results[orig_idx] = results
                remote_transfer_cnt[orig_idx] = ret_payload.remote_transfer_cnt
                remote_transfer_data[orig_idx] = ret_payload.remote_transfer_data
                distance_cnt[orig_idx] = ret_payload.distance_cnt
                backtrack_cnt[orig_idx] = ret_payload.backtrack_cnt
                pull_cnt[orig_idx] = ret_payload.pull_cnt
                pull_data[orig_idx] = ret_payload.pull_data
                received += 1
            if sent % 500 == 0:
                print(f"[pid={my_pid}] run={run_idx} tid={thread_id} "
                      f"sent={sent}/{len(proc_indices)} received={received}", flush=True)

        while received < sent:
            ret_qid, results, ret_payload = sdk_client.query_recv()
            orig_idx = pending.pop(ret_qid)
            recall_results[orig_idx] = results
            remote_transfer_cnt[orig_idx] = ret_payload.remote_transfer_cnt
            remote_transfer_data[orig_idx] = ret_payload.remote_transfer_data
            distance_cnt[orig_idx] = ret_payload.distance_cnt
            backtrack_cnt[orig_idx] = ret_payload.backtrack_cnt
            pull_cnt[orig_idx] = ret_payload.pull_cnt
            pull_data[orig_idx] = ret_payload.pull_data
            received += 1

        query_time = time.time() - actual_start
        print(f"[pid={my_pid}] _do_round run={run_idx} tid={thread_id} done, "
              f"time={query_time:.3f}s", flush=True)

        # 计算召回
        recall = {}
        for i, res in recall_results.items():
            results_set = set(res)
            true_ids = set(_GROUNDTRUTH[i][:k])
            correct = len(results_set.intersection(true_ids))
            recall[i] = correct / k

        # 每轮结束主动断开连接，模拟老代码 Process 结束时的清理
        sdk_client.disconnect()

        return {
            "thread_id": thread_id,
            "pid": my_pid,
            "query_time": query_time,
            "drift_ms": drift_ms,
            "recall": recall,
            "remote_transfer_cnt": remote_transfer_cnt,
            "remote_transfer_data": remote_transfer_data,
            "distance_cnt": distance_cnt,
            "backtrack_cnt": backtrack_cnt,
            "pull_cnt": pull_cnt,
            "pull_data": pull_data,
        }
    except Exception as e:
        print(f"[pid={my_pid}] _do_round run={run_idx} tid={thread_id} "
              f"ERROR: {e}", flush=True)
        raise


def barrier_wait(round_idx):
    """父进程的文件屏障：写 ready_<rank>_<round>，等 go_<round>。返回 scheduled_start。"""
    if args.barrier_dir is None:
        return time.time()
    ready_file = os.path.join(args.barrier_dir, f"ready_{args.query_rank}_{round_idx}")
    go_file = os.path.join(args.barrier_dir, f"go_{round_idx}")
    abort_file = os.path.join(args.barrier_dir, "abort")

    tmp_ready = ready_file + ".tmp"
    with open(tmp_ready, "w") as f:
        f.write(f"{time.time():.6f}\n")
    os.replace(tmp_ready, ready_file)
    print(f"[barrier] rank {args.query_rank} ready_{round_idx}, waiting for go_{round_idx}")

    barrier_start = time.time()
    while not os.path.exists(go_file):
        if os.path.exists(abort_file):
            print(f"[barrier] abort file detected, exiting")
            sys.exit(1)
        if time.time() - barrier_start > args.barrier_timeout:
            print(f"[barrier] TIMEOUT round {round_idx}, after {args.barrier_timeout}s")
            sys.exit(2)
        time.sleep(0.05)

    try:
        with open(go_file, "r") as f:
            scheduled_start = float(f.read().strip())
    except Exception:
        scheduled_start = time.time()
    print(f"[barrier] round {round_idx} GO received, scheduled_start={time.strftime('%H:%M:%S', time.localtime(scheduled_start))}")
    return scheduled_start


if __name__ == '__main__':
    print(f"Query client: rank={args.query_rank}/{args.query_world_size}")
    print(f"SDK entry: {sdk_ip}:{sdk_port} (auto-balanced across all entry endpoints by PID hash)")

    _QUERIES = read_vectors(args.query_path)
    _GROUNDTRUTH = read_groundtruth(args.groundtruth_path)
    _NUM_QUERIES = _QUERIES.shape[0]
    print(f"Queries shape: {_QUERIES.shape}, Groundtruth shape: {_GROUNDTRUTH.shape}")

    # 按 rank 分片
    all_indices = list(range(_NUM_QUERIES))
    random.seed(42)
    random.shuffle(all_indices)
    _MY_INDICES = [q for j, q in enumerate(all_indices) if (j % args.query_world_size) == args.query_rank]
    print(f"Queries assigned to this rank: {len(_MY_INDICES)}/{_NUM_QUERIES}")

    if args.barrier_dir is not None:
        os.makedirs(args.barrier_dir, exist_ok=True)

    qps_list = []
    query_times_list = []
    last_round_results = None

    # 多进程池：每个进程独立 PID → SDK sticky endpoint 自然分布到所有 worker 入口
    # 使用 spawn 避免 fork + pybind11/numpy 线程死锁；每个子进程独立加载数据+SDK
    ctx = mp.get_context('fork')
    pool = ctx.Pool(args.num_query_threads,
                    initializer=_pool_init,
                    initargs=(args.dtype, sdk_ip, sdk_port, args.comm_mode,
                              args.query_path, args.groundtruth_path,
                              args.query_rank, args.query_world_size))
    # Warmup：确保所有 pool worker 完成 _pool_init（spawn 是异步的）
    print(f"Warming up {args.num_query_threads} pool workers...", flush=True)
    warmup_t0 = time.time()
    warmup_pids = pool.map(_warmup, list(range(args.num_query_threads * 2)))
    unique_pids = sorted(set(warmup_pids))
    print(f"Pool ready: {len(unique_pids)} unique PIDs after {time.time()-warmup_t0:.2f}s", flush=True)
    try:
        for run_idx in range(args.num_runs):
            scheduled_start = barrier_wait(run_idx)

            round_args = [(run_idx, tid, scheduled_start, args.k, args.num_query_threads,
                           args.query_rank, args.query_world_size)
                          for tid in range(args.num_query_threads)]

            results_per_proc = pool.map(_do_round, round_args)

            run_max_t = max(r["query_time"] for r in results_per_proc)
            total_q = len(_MY_INDICES) * args.num_query_threads
            run_qps = total_q / run_max_t if run_max_t > 0 else 0.0
            qps_list.append(run_qps)
            query_times_list.append(run_max_t)
            drifts = [r["drift_ms"] for r in results_per_proc]
            print(f"  Run {run_idx+1}/{args.num_runs}: QPS={run_qps:.2f}, time={run_max_t:.4f}s, "
                  f"drift_ms=[min={min(drifts):+.2f}, max={max(drifts):+.2f}]")
            last_round_results = results_per_proc
    finally:
        pool.close()
        pool.join()

    # ---------- 聚合最后一轮统计 ----------
    all_recall = {}
    all_remote_transfer_cnt = {}
    all_remote_transfer_data = {}
    all_distance_cnt = {}
    all_backtrack_cnt = {}
    all_pull_cnt = {}
    all_pull_data = {}
    for s in last_round_results:
        all_recall.update(s["recall"])
        all_remote_transfer_cnt.update(s["remote_transfer_cnt"])
        all_remote_transfer_data.update(s["remote_transfer_data"])
        all_distance_cnt.update(s["distance_cnt"])
        all_backtrack_cnt.update(s["backtrack_cnt"])
        all_pull_cnt.update(s["pull_cnt"])
        all_pull_data.update(s["pull_data"])

    # ---------- QPS：>=3 轮去掉极值再平均 ----------
    if args.num_runs >= 3:
        sorted_qps = sorted(qps_list)
        avg_qps = sum(sorted_qps[1:-1]) / len(sorted_qps[1:-1])
    else:
        avg_qps = sum(qps_list) / len(qps_list) if qps_list else 0.0

    last_query_time = query_times_list[-1] if query_times_list else 0.0

    results_dir = os.path.join(project_root, "tmp", "query_results")
    os.makedirs(results_dir, exist_ok=True)
    results_path = f"{results_dir}/rank{args.query_rank}-of{args.query_world_size}.json"

    results = {
        "query_rank": args.query_rank,
        "query_world_size": args.query_world_size,
        "num_queries_assigned": len(_MY_INDICES),
        "num_queries_total": _NUM_QUERIES,
        "num_query_threads": args.num_query_threads,
        "k": args.k,
        "qps": avg_qps,
        "qps_runs": qps_list,
        "query_times_runs": query_times_list,
        "last_query_time": last_query_time,
        "recall": {str(k): v for k, v in all_recall.items()},
        "remote_transfer_cnt": {str(k): v for k, v in all_remote_transfer_cnt.items()},
        "remote_transfer_data": {str(k): v for k, v in all_remote_transfer_data.items()},
        "distance_cnt": {str(k): v for k, v in all_distance_cnt.items()},
        "backtrack_cnt": {str(k): v for k, v in all_backtrack_cnt.items()},
        "pull_cnt": {str(k): v for k, v in all_pull_cnt.items()},
        "pull_data": {str(k): v for k, v in all_pull_data.items()},
    }

    avg_recall = sum(all_recall.values()) / len(all_recall) if all_recall else 0
    results[f"avg_recall@{args.k}"] = avg_recall

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*50}")
    print(f"  Rank {args.query_rank}: QPS runs={[round(q,2) for q in qps_list]}, avg={avg_qps:.2f}")
    print(f"  Recall@{args.k}: {avg_recall:.4f}")
    print(f"  Result: {os.path.abspath(results_path)}")
    print(f"{'='*50}")
