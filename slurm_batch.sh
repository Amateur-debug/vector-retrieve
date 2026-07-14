#!/bin/bash
# =============================================================================
# 场景 4：batch query —— 一个 allocation 内串行跑 batch.py（调用 main.py）
#
# 节点分配：
#   node 0                 : orchestrator（被 batch.py 多次调用 main.py）
#   node 1..NUM_QUERY_NODES: query 客户端
#   其余 WORKER_COUNT 个    : worker
# batch.py 按 PARAM_GRID 笛卡尔积「串行」跑各组合：同 (node_num, m, ...) 复用服务端
# （--skip_prepare），跨组重建 cluster。产物：多个 results/cpp/*.json。
#
# 约束：单 allocation 的 worker 数固定 = WORKER_COUNT；PARAM_GRID["node_num"] 的
#       最大值必须 <= WORKER_COUNT（main.py 取前 node_num 个 worker）。
#       NUM_QUERY_NODES 也要 >= batch 里推导出的 query 节点数（8→2, 16→4）。
#       => 总节点数 = 1 + NUM_QUERY_NODES + WORKER_COUNT。
#
# 用法示例：
#   sbatch slurm_batch.sh                                  # 默认 grid, --skip
#   BATCH_ARGS="--skip --offset 5" sbatch slurm_batch.sh
#   WORKER_COUNT=16 NUM_QUERY_NODES=4 BASE_PORT=38320 sbatch -N21 slurm_batch.sh
# =============================================================================

#SBATCH --job-name=vr-batch
#SBATCH --nodes=11
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --time=7-00:00:00
#SBATCH --partition=i64m512ue
#SBATCH --qos=cpu2600
#SBATCH --output=/hpc2hdd/home/xchen740/workspace/vector-retrieve/logs/slurm_%j.out
#SBATCH --error=/hpc2hdd/home/xchen740/workspace/vector-retrieve/logs/slurm_%j.err

set -euo pipefail
export PATH="/opt/slurm/bin:$PATH"
PROJECT_ROOT="/hpc2hdd/home/xchen740/workspace/vector-retrieve"
cd "$PROJECT_ROOT"
mkdir -p logs save

# --------------- 可覆盖参数 ---------------
BASE_PORT="${BASE_PORT:-28180}"
NUM_QUERY_NODES="${NUM_QUERY_NODES:-2}"
WORKER_COUNT="${WORKER_COUNT:-8}"
BATCH_ARGS="${BATCH_ARGS:---skip}"

echo "============================================"
echo "  Batch QUERY: host + $NUM_QUERY_NODES query + $WORKER_COUNT worker | BATCH_ARGS=$BATCH_ARGS"
echo "  Job ID:    ${SLURM_JOB_ID:-N/A}   Nodelist: ${SLURM_JOB_NODELIST:-N/A}"
echo "  Started:   $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"

# --------------- 1. 激活 Conda ---------------
echo "[1/4] Activating conda..."
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate vector-retrieve

# --------------- 2. 编译 C++（按需）---------------
echo "[2/4] Building C++ extension (if needed)..."
BUILD_DIR="$PROJECT_ROOT/build"
mkdir -p "$BUILD_DIR"
SO_FILES=("$BUILD_DIR"/vector_retrieve*.so)
SO_PATH=""
[ -e "${SO_FILES[0]}" ] && SO_PATH="${SO_FILES[0]}"
NEEDS_BUILD=0
if [ -z "$SO_PATH" ]; then
    NEEDS_BUILD=1
else
    for src in "$PROJECT_ROOT"/CMakeLists.txt "$PROJECT_ROOT"/source/cpp/*.h "$PROJECT_ROOT"/source/cpp/*.cpp; do
        [ "$src" -nt "$SO_PATH" ] && { NEEDS_BUILD=1; break; }
    done
fi
if [ "$NEEDS_BUILD" -eq 1 ]; then
    cd "$BUILD_DIR"
    rm -f CMakeCache.txt && rm -rf CMakeFiles
    cmake .. -DCMAKE_BUILD_TYPE=Release \
        -DPYTHON_EXECUTABLE="$(which python)" \
        -Dpybind11_DIR="$(python -c 'import pybind11,os;print(os.path.join(os.path.dirname(pybind11.__file__),"share","cmake","pybind11"))')"
    make -j"${SLURM_CPUS_PER_TASK:-$(nproc)}"
    cd "$PROJECT_ROOT"
else
    echo "  Up to date: $SO_PATH"
fi

# --------------- 3. 解析节点 → 写 config.json ---------------
echo "[3/4] Writing job-specific config (host + $NUM_QUERY_NODES query + $WORKER_COUNT worker)..."
NODE_HOSTNAMES=$(scontrol show hostname "$SLURM_JOB_NODELIST")
readarray -t NODE_HOST_ARRAY <<< "$(echo "$NODE_HOSTNAMES" | tr ' ' '\n')"
TOTAL_COUNT="${#NODE_HOST_ARRAY[@]}"

REQUIRED_COUNT=$((1 + NUM_QUERY_NODES + WORKER_COUNT))
if [ "$TOTAL_COUNT" -lt "$REQUIRED_COUNT" ]; then
    echo "  ERROR: Need >=${REQUIRED_COUNT} nodes, got $TOTAL_COUNT"
    exit 1
fi

NODE_IPS=()
for h in "${NODE_HOST_ARRAY[@]}"; do
    ip=$(getent hosts "$h" | awk '{print $1}')
    [ -z "$ip" ] && { echo "  ERROR: Cannot resolve $h"; exit 1; }
    NODE_IPS+=("$ip")
done

HOST_IP="${NODE_IPS[0]}"
HOST_HOSTNAME="${NODE_HOST_ARRAY[0]}"

# query: index 1..NUM_QUERY_NODES
QUERY_NODES_JSON=""
QUERY_BASE_PORT=$((BASE_PORT + 1000))
for i in $(seq 1 $NUM_QUERY_NODES); do
    qport=$((QUERY_BASE_PORT + i - 1))
    [ "$i" -gt 1 ] && QUERY_NODES_JSON+=","
    QUERY_NODES_JSON+=$'\n'"        {\"ip\": \"${NODE_IPS[$i]}\", \"hostname\": \"${NODE_HOST_ARRAY[$i]}\", \"port\": $qport, \"user_name\": \"$USER\"}"
done

# worker: index (NUM_QUERY_NODES+1)..(NUM_QUERY_NODES+WORKER_COUNT)
NODES_JSON=""
for ((wi=0; wi<WORKER_COUNT; wi++)); do
    idx=$((NUM_QUERY_NODES + 1 + wi))
    port=$((BASE_PORT + wi))
    [ "$wi" -gt 0 ] && NODES_JSON+=","
    NODES_JSON+=$'\n'"        {\"ip\": \"${NODE_IPS[$idx]}\", \"hostname\": \"${NODE_HOST_ARRAY[$idx]}\", \"port\": $port, \"user_name\": \"$USER\"}"
done

CONFIG_PATH="$PROJECT_ROOT/config.${SLURM_JOB_ID:-manual}.${BASE_PORT}.json"
export VR_CONFIG_PATH="$CONFIG_PATH"

# 不写 layer_config：batch grid 的 node_num 可变，由 main.py 按各任务 node_num
# 生成默认 layer_config（避免固定 layer_config 引用未被选中的 worker）。
cat > "$CONFIG_PATH" <<CFGEOF
{
    "host_node": {"ip": "$HOST_IP", "hostname": "$HOST_HOSTNAME", "port": $BASE_PORT, "user_name": "$USER"},
    "work_nodes": [$NODES_JSON
    ],
    "query_nodes": [$QUERY_NODES_JSON
    ],
    "layer_replicate_flags": [false, false, true, true, true, true]
}
CFGEOF
echo "  VR_CONFIG_PATH=$CONFIG_PATH"
cat "$CONFIG_PATH"

# --------------- 4. 运行 batch.py（串行）---------------
echo "[4/4] Running batch.py --slurm $BATCH_ARGS ..."
set -x
python -u batch.py --slurm $BATCH_ARGS
set +x

echo "============================================"
echo "  Done at $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
