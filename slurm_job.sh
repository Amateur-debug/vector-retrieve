#!/bin/bash
# =============================================================================
# SLURM 多机分布式作业脚本 — vector-retrieve (main.py + query.py)
# 参考: https://docs.hpc.hkust-gz.edu.cn/docs/hpc12/slurm/job2
#
# 架构（18 台机器）:
#   节点 0:   orchestrator (main.py 总控) + query 客户端 rank 0（64 线程发 query）
#   节点 1:   query 客户端 rank 1（64 线程发 query）
#   节点 2-17: 16 台 worker（每台 node_start.py，8 处理线程）
#
# 流程:
#   1. SLURM 分配 18 节点
#   2. 节点 0 运行 main.py：通过 SSH 在 16 台 worker 上启 node_start.py
#   3. main.py 加载/构建图，分区
#   4. main.py 通过 SSH 在 2 台 query host 上启 query.py（含本机）
#   5. 文件屏障：所有 query.py ready 后 main.py 发 go 信号
#   6. 各 query.py 同时开始发查询，结束后保存 per-rank 结果
#   7. main.py 聚合各 rank 的结果
# =============================================================================

# --------------- SBATCH 指令 ---------------
#SBATCH --job-name=vr-distributed
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64               # 预留足够的 CPU 给 worker/query srun 使用
#SBATCH --mem=256G
#SBATCH --time=7-00:00:00
#SBATCH --partition=i64m512ue 
#SBATCH --output=/hpc2hdd/home/xchen740/workspace/vector-retrieve/logs/slurm_%j.out
#SBATCH --error=/hpc2hdd/home/xchen740/workspace/vector-retrieve/logs/slurm_%j.err

# --------------- 可覆盖参数 ---------------
DATA_NAME="${DATA_NAME:-sift100m}"
K="${K:-10}"
M="${M:-10}"
EF_CONST="${EF_CONST:-200}"
EF_SEARCH="${EF_SEARCH:-15}"
LIMITED_LEVEL="${LIMITED_LEVEL:-5}"
PART_MODE="${PART_MODE:-k-means}"
REPLICATION="${REPLICATION:-2}"
COMM_MODE="${COMM_MODE:-tcp}"
BUILD_THREADS="${BUILD_THREADS:-16}"
SVR_THREADS="${SVR_THREADS:-8}"            # worker 节点处理线程数
# QUERY_THREADS 不再手动设置，自动 = (worker_num × svr_threads) / num_query_nodes
NUM_QUERY_NODES="${NUM_QUERY_NODES:-2}"    # query 客户端数量
CO_LOCATE="${CO_LOCATE:-0}"                # 1 = node0 同时跑 main.py + query rank 0 + worker 0
CO_LOCATE_MAIN_QUERY="${CO_LOCATE_MAIN_QUERY:-0}"  # 1 = node0 同时跑 main.py + query rank 0（不含 worker），节省 1 个节点
USE_SLURM="${USE_SLURM:-0}"                # 1 = main.py 用 srun 启动 query.py；0 = ssh
SAVE="${SAVE:-1}"
NUM_RUNS="${NUM_RUNS:-5}"
BASE_PORT="${BASE_PORT:-18080}"
DTYPE="${DTYPE:-uint8}"                   # 非空时覆盖 preset dtype（如 uint8）
PARTITION_DISTRIBUTION="${PARTITION_DISTRIBUTION:-nfs-load}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
FORCE_REPARTITION="${FORCE_REPARTITION:-0}"

# --------------- 初始化 ---------------
set -euo pipefail
export PATH="/opt/slurm/bin:$PATH"
PROJECT_ROOT="/hpc2hdd/home/xchen740/workspace/vector-retrieve"
cd "$PROJECT_ROOT"
mkdir -p logs

echo "============================================"
echo "  vector-retrieve SLURM Distributed Job"
echo "  Job ID:      ${SLURM_JOB_ID:-N/A}"
echo "  NNODES:      ${SLURM_NNODES:-N/A}"
echo "  Node list:   ${SLURM_JOB_NODELIST:-N/A}"
echo "  Dataset:     $DATA_NAME"
echo "  Started:     $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"

# --------------- 1. 激活 Conda ---------------
echo "[1/5] Activating conda..."
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate vector-retrieve

# --------------- 2. 编译 ---------------
echo "[2/5] Building C++ extension..."
BUILD_DIR="$PROJECT_ROOT/build"
mkdir -p "$BUILD_DIR"
SO_FILES=("$BUILD_DIR"/vector_retrieve*.so)
SO_PATH=""
if [ -e "${SO_FILES[0]}" ]; then
    SO_PATH="${SO_FILES[0]}"
fi

NEEDS_BUILD=0
if [ -z "$SO_PATH" ]; then
    NEEDS_BUILD=1
else
    for src in "$PROJECT_ROOT"/CMakeLists.txt "$PROJECT_ROOT"/source/cpp/*.h "$PROJECT_ROOT"/source/cpp/*.cpp; do
        if [ "$src" -nt "$SO_PATH" ]; then
            NEEDS_BUILD=1
            break
        fi
    done
fi

if [ "$NEEDS_BUILD" -eq 1 ]; then
    cd "$BUILD_DIR"
    rm -f CMakeCache.txt
    rm -rf CMakeFiles
    PYBIND11_CMAKE_DIR=$(python -c 'import pybind11, os; print(os.path.join(os.path.dirname(pybind11.__file__), "share", "cmake", "pybind11"))')
    CXX_COMPILER="${CXX:-$(command -v g++ || command -v c++ || true)}"
    if [ -z "$CXX_COMPILER" ] || [ ! -x "$CXX_COMPILER" ]; then
        echo "ERROR: no executable C++ compiler found in PATH" >&2
        exit 1
    fi
    cmake .. -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_COMPILER="$CXX_COMPILER" \
        -DPYTHON_EXECUTABLE="$(which python)" \
        -Dpybind11_DIR="$PYBIND11_CMAKE_DIR"
    make -j"${SLURM_CPUS_PER_TASK:-$(nproc)}"
    cd "$PROJECT_ROOT"
else
    echo "  Existing C++ extension is up to date: $SO_PATH"
fi
echo "  Build OK"

# --------------- 3. 解析 SLURM 节点 → 分组 ---------------
echo "[3/5] Splitting nodes: $NUM_QUERY_NODES query + N workers..."

NODE_HOSTNAMES=$(scontrol show hostname "$SLURM_JOB_NODELIST")
readarray -t NODE_HOST_ARRAY <<< "$(echo "$NODE_HOSTNAMES" | tr ' ' '\n')"
TOTAL_COUNT="${#NODE_HOST_ARRAY[@]}"

NODE_IPS=()
for h in "${NODE_HOST_ARRAY[@]}"; do
    ip=$(getent hosts "$h" | awk '{print $1}')
    [ -z "$ip" ] && { echo "  ERROR: Cannot resolve $h"; exit 1; }
    NODE_IPS+=("$ip")
done

# 节点分配
HOST_IP_FIRST="${NODE_IPS[0]}"
HOST_HOSTNAME_FIRST="${NODE_HOST_ARRAY[0]}"

if [ "$CO_LOCATE" = "1" ]; then
    # 共址模式：node 0 = main.py + query rank 0 + worker 0；node 1.. = 其余 worker
    NUM_QUERY_NODES=1
    QUERY_IPS=("${NODE_IPS[0]}")
    QUERY_HOSTS=("${NODE_HOST_ARRAY[0]}")
    WORKER_IPS=("${NODE_IPS[@]}")
    WORKER_HOSTS=("${NODE_HOST_ARRAY[@]}")
elif [ "$CO_LOCATE_MAIN_QUERY" = "1" ]; then
    # main + query0 共址模式：node 0 = main.py + query rank 0；node [1..NUM_QUERY_NODES-1] = 其他 query；剩下 = worker
    # 用于节省 1 个节点。total_nodes = NUM_QUERY_NODES + worker_count
    QUERY_IPS=()
    QUERY_HOSTS=()
    QUERY_IPS+=("${NODE_IPS[0]}")
    QUERY_HOSTS+=("${NODE_HOST_ARRAY[0]}")
    for ((i=1; i<NUM_QUERY_NODES && i<TOTAL_COUNT; i++)); do
        QUERY_IPS+=("${NODE_IPS[$i]}")
        QUERY_HOSTS+=("${NODE_HOST_ARRAY[$i]}")
    done

    WORKER_IPS=()
    WORKER_HOSTS=()
    for ((i=NUM_QUERY_NODES; i<TOTAL_COUNT; i++)); do
        WORKER_IPS+=("${NODE_IPS[$i]}")
        WORKER_HOSTS+=("${NODE_HOST_ARRAY[$i]}")
    done
else
    # 标准模式：node 0 = host (main.py 总控)，node [1..NUM_QUERY_NODES] = query，剩下 = worker
    QUERY_IPS=()
    QUERY_HOSTS=()
    for ((i=1; i<=NUM_QUERY_NODES && i<TOTAL_COUNT; i++)); do
        QUERY_IPS+=("${NODE_IPS[$i]}")
        QUERY_HOSTS+=("${NODE_HOST_ARRAY[$i]}")
    done

    WORKER_IPS=()
    WORKER_HOSTS=()
    for ((i=NUM_QUERY_NODES+1; i<TOTAL_COUNT; i++)); do
        WORKER_IPS+=("${NODE_IPS[$i]}")
        WORKER_HOSTS+=("${NODE_HOST_ARRAY[$i]}")
    done
fi
WORKER_COUNT="${#WORKER_IPS[@]}"

# QUERY_THREADS 自动计算：总并发 = worker_num × svr_threads，平均分到每个 query 节点
TOTAL_QUERY_CONCURRENCY=$((WORKER_COUNT * SVR_THREADS))
if [ "$NUM_QUERY_NODES" -gt 0 ]; then
    QUERY_THREADS=$((TOTAL_QUERY_CONCURRENCY / NUM_QUERY_NODES))
else
    QUERY_THREADS=$TOTAL_QUERY_CONCURRENCY
fi
[ "$QUERY_THREADS" -lt 1 ] && QUERY_THREADS=1

echo "  Total: $TOTAL_COUNT nodes"
echo "  Host (main.py only): $HOST_IP_FIRST ($HOST_HOSTNAME_FIRST)"
echo "  Query clients (${#QUERY_IPS[@]}): ${QUERY_IPS[*]}"
echo "  Workers ($WORKER_COUNT): ${WORKER_IPS[*]}"
echo "  Total concurrency: $WORKER_COUNT × $SVR_THREADS = $TOTAL_QUERY_CONCURRENCY"
echo "  Query threads per query node: $QUERY_THREADS"

# --------------- 4. 写 config.json ---------------
HOST_IP="$HOST_IP_FIRST"         # main.py 运行在独立的 host 节点
USER_NAME="${USER}"

# worker 节点（HNSW 处理）
NODES_JSON=""
for i in "${!WORKER_IPS[@]}"; do
    port=$((BASE_PORT + i))
    [ "$i" -gt 0 ] && NODES_JSON+=","
    NODES_JSON+=$'\n'"        {\"ip\": \"${WORKER_IPS[$i]}\", \"hostname\": \"${WORKER_HOSTS[$i]}\", \"port\": $port, \"user_name\": \"$USER_NAME\"}"
done

# query 节点（发查询）
QUERY_NODES_JSON=""
QUERY_BASE_PORT=$((BASE_PORT + 1000))
for i in "${!QUERY_IPS[@]}"; do
    qport=$((QUERY_BASE_PORT + i))
    [ "$i" -gt 0 ] && QUERY_NODES_JSON+=","
    QUERY_NODES_JSON+=$'\n'"        {\"ip\": \"${QUERY_IPS[$i]}\", \"hostname\": \"${QUERY_HOSTS[$i]}\", \"port\": $qport, \"user_name\": \"$USER_NAME\"}"
done

# layer_config: 所有层均使用全部 worker（[0..WORKER_COUNT-1]）
NUM_LAYERS=$((LIMITED_LEVEL + 1))
WORKER_LIST=""
for ((wi=0; wi<WORKER_COUNT; wi++)); do
    [ "$wi" -gt 0 ] && WORKER_LIST+=", "
    WORKER_LIST+="$wi"
done

LAYER_CONFIG_JSON=""
LAYER_FLAGS_JSON=""
for ((l=0; l<NUM_LAYERS; l++)); do
    if [ "$l" -gt 0 ]; then
        LAYER_CONFIG_JSON+=","
        LAYER_FLAGS_JSON+=", "
    fi
    LAYER_CONFIG_JSON+=$'\n'"        \"$l\": [$WORKER_LIST]"
    # 默认 layer_replicate_flags: 前 2 层 false（按 worker 划分），后面 true（全复制）
    if [ "$l" -lt 2 ]; then
        LAYER_FLAGS_JSON+="false"
    else
        LAYER_FLAGS_JSON+="true"
    fi
done

cat > "$PROJECT_ROOT/config.json" <<CFGEOF
{
    "host_node": {"ip": "$HOST_IP", "hostname": "$HOST_HOSTNAME_FIRST", "port": $BASE_PORT, "user_name": "$USER_NAME"},
    "work_nodes": [$NODES_JSON
    ],
    "query_nodes": [$QUERY_NODES_JSON
    ],
    "layer_config": {$LAYER_CONFIG_JSON
    },
    "layer_replicate_flags": [$LAYER_FLAGS_JSON]
}
CFGEOF

echo "  config.json:"
cat "$PROJECT_ROOT/config.json"
sync
sleep 1
# --------------- 5. 运行 main.py ---------------
echo "[5/5] Running main.py (host=$HOST_IP, workers=$WORKER_COUNT, query_clients=${#QUERY_IPS[@]})..."

SAVE_FLAG=""
[ "$SAVE" = "1" ] && SAVE_FLAG="--save"

SLURM_FLAG=""
[ "$USE_SLURM" = "1" ] && SLURM_FLAG="--slurm"

# DTYPE → data_name 后缀映射，必须与 batch.py DTYPE_NAME_SUFFIX 保持一致
# main.py 直接用 --data_name 拼 graph/results 文件名，所以后缀要加在 data_name 上
DTYPE_ARG=""
DTYPE_SUFFIX=""
EFFECTIVE_DATA_NAME="$DATA_NAME"
if [ -n "$DTYPE" ]; then
    DTYPE_ARG="--dtype $DTYPE"
    case "$DTYPE" in
        uint8) DTYPE_SUFFIX="-u8" ;;
        *)     DTYPE_SUFFIX="-${DTYPE}" ;;
    esac
    # 避免 DATA_NAME=sift100m-u8 + DTYPE=uint8 → sift100m-u8-u8 这种重复
    if [[ "$DATA_NAME" != *"$DTYPE_SUFFIX" ]]; then
        EFFECTIVE_DATA_NAME="${DATA_NAME}${DTYPE_SUFFIX}"
    fi
fi

LAYER_SUFFIX="-layer"
CLUSTER_BASE="./save/${EFFECTIVE_DATA_NAME}/node${WORKER_COUNT}/m${M}-efc${EF_CONST}-ll${LIMITED_LEVEL}-${PART_MODE}-rep${REPLICATION}${LAYER_SUFFIX}"
if [ "$FORCE_REPARTITION" = "1" ]; then
    echo "[force] Removing existing partitioned graphs: ${CLUSTER_BASE}-*.bin"
    rm -f ${CLUSTER_BASE}-*.bin
fi

set -x
python -u main.py \
    --data_name "$EFFECTIVE_DATA_NAME" \
    $DTYPE_ARG \
    --k "$K" \
    --m "$M" \
    --ef_construction "$EF_CONST" \
    --ef_search "$EF_SEARCH" \
    --limited_level "$LIMITED_LEVEL" \
    --node_num "$WORKER_COUNT" \
    --partition_mode "$PART_MODE" \
    --partition_distribution "$PARTITION_DISTRIBUTION" \
    --replication "$REPLICATION" \
    --num_construction_threads "$BUILD_THREADS" \
    --num_query_threads "$TOTAL_QUERY_CONCURRENCY" \
    --num_threads "$SVR_THREADS" \
    --comm_mode "$COMM_MODE" \
    --use_cpp True \
    --distributed \
    --layer \
    --graph_path "./save/hnsw-${EFFECTIVE_DATA_NAME}-m${M}-efc${EF_CONST}-ll${LIMITED_LEVEL}.bin" \
    --num_runs "$NUM_RUNS" \
    $SAVE_FLAG \
    $SLURM_FLAG \
    $EXTRA_ARGS
set +x

echo "============================================"
echo "  Done at $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
