#!/bin/bash
# =============================================================================
# 场景 3：普通 query —— 多机分布式查询（调用 main.py）
#
# 节点分配（标准模式）：
#   node 0            : orchestrator (main.py 总控)
#   node 1..NUM_QUERY_NODES : query 客户端
#   其余节点          : worker（每台一个 HNSW 分片服务）
# main.py 加载/构建图 → 分区 → 用 srun 在各 query 节点启 query.py 并发发查询 → 聚合结果。
# 产物：results/cpp/results-*.json
#
# 并行：每个 job 用唯一 BASE_PORT + 独立 VR_CONFIG_PATH，可同时提交多个；也可单跑。
#
# 用法示例（worker 数 = NODE_NUM，总节点数需 = 1 + NUM_QUERY_NODES + NODE_NUM）：
#   M=30 EF_SEARCH=45 NODE_NUM=8  NUM_QUERY_NODES=2 BASE_PORT=18080 sbatch -N11 slurm_query.sh
#   M=30 EF_SEARCH=45 NODE_NUM=16 NUM_QUERY_NODES=4 BASE_PORT=19080 sbatch -N21 slurm_query.sh
# =============================================================================

#SBATCH --job-name=vr-query
#SBATCH --nodes=11
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --time=1-00:00:00
#SBATCH --partition=i64m512ue
#SBATCH --qos=i64m512ue
#SBATCH --output=/hpc2hdd/home/xchen740/workspace/vector-retrieve/logs/slurm_%j.out
#SBATCH --error=/hpc2hdd/home/xchen740/workspace/vector-retrieve/logs/slurm_%j.err

set -euo pipefail
export PATH="/opt/slurm/bin:$PATH"
PROJECT_ROOT="/hpc2hdd/home/xchen740/workspace/vector-retrieve"
cd "$PROJECT_ROOT"
mkdir -p logs save

# --------------- 可覆盖参数 ---------------
DATA_NAME="${DATA_NAME:-deep100m}"
DTYPE="${DTYPE:-float}"
K="${K:-10}"
M="${M:-10}"
EF_CONST="${EF_CONST:-200}"
EF_SEARCH="${EF_SEARCH:-15}"
LIMITED_LEVEL="${LIMITED_LEVEL:-5}"
NODE_NUM="${NODE_NUM:-8}"               # worker 数量（= config 里 work_nodes 数）
NUM_QUERY_NODES="${NUM_QUERY_NODES:-2}" # query 客户端数量
REPLICATION="${REPLICATION:-2}"
PART_MODE="${PART_MODE:-k-means}"
LAYER="${LAYER:-True}"                  # True/False
COMM_MODE="${COMM_MODE:-tcp}"
SVR_THREADS="${SVR_THREADS:-8}"         # 每个 worker 的处理线程数
NUM_RUNS="${NUM_RUNS:-5}"
BASE_PORT="${BASE_PORT:-18080}"
SAVE="${SAVE:-1}"
USE_SLURM="${USE_SLURM:-1}"             # 1 = main.py 用 srun 启动 query.py（slurm 内默认）
FORCE_REPARTITION="${FORCE_REPARTITION:-0}"
PART_TAG="${PART_TAG:-}"                # 分区路径标签(须与 build 时一致,如 w005)
# REP_SPEC 一行式(须与 build 一致):`repN-wM` 或 `repN`
REP_SPEC="${REP_SPEC:-}"
if [ -n "$REP_SPEC" ]; then
    _spec="${REP_SPEC#rep}"
    if [[ "$_spec" == *-w* ]]; then
        REPLICATION="${_spec%%-w*}"; PART_TAG="w${_spec##*-w}"
    elif [[ "$_spec" == *.* ]]; then
        REPLICATION=$(( ${_spec%.*} + 1 )); PART_TAG="r${_spec}"
    else
        REPLICATION="$_spec"; PART_TAG=""
    fi
fi
export VR_PART_TAG="$PART_TAG"

echo "============================================"
echo "  Distributed QUERY: data=$DATA_NAME M=$M efs=$EF_SEARCH node_num=$NODE_NUM rep=$REPLICATION layer=$LAYER"
echo "  Job ID:    ${SLURM_JOB_ID:-N/A}   Nodelist: ${SLURM_JOB_NODELIST:-N/A}"
echo "  Started:   $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"

# --------------- 1. 激活 Conda ---------------
echo "[1/5] Activating conda..."
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate vector-retrieve

# --------------- 2. 编译 C++（按需）---------------
echo "[2/5] Building C++ extension (if needed)..."
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

# --------------- 3. 解析节点 → host / query / worker ---------------
echo "[3/5] Splitting nodes: 1 host + $NUM_QUERY_NODES query + workers..."
NODE_HOSTNAMES=$(scontrol show hostname "$SLURM_JOB_NODELIST")
readarray -t NODE_HOST_ARRAY <<< "$(echo "$NODE_HOSTNAMES" | tr ' ' '\n')"
TOTAL_COUNT="${#NODE_HOST_ARRAY[@]}"

NODE_IPS=()
for h in "${NODE_HOST_ARRAY[@]}"; do
    ip=$(getent hosts "$h" | awk '{print $1}')
    [ -z "$ip" ] && { echo "  ERROR: Cannot resolve $h"; exit 1; }
    NODE_IPS+=("$ip")
done

HOST_IP="${NODE_IPS[0]}"
HOST_HOSTNAME="${NODE_HOST_ARRAY[0]}"

# query: index 1..NUM_QUERY_NODES；worker: 其余
QUERY_IPS=(); QUERY_HOSTS=()
for ((i=1; i<=NUM_QUERY_NODES && i<TOTAL_COUNT; i++)); do
    QUERY_IPS+=("${NODE_IPS[$i]}"); QUERY_HOSTS+=("${NODE_HOST_ARRAY[$i]}")
done
WORKER_IPS=(); WORKER_HOSTS=()
for ((i=NUM_QUERY_NODES+1; i<TOTAL_COUNT; i++)); do
    WORKER_IPS+=("${NODE_IPS[$i]}"); WORKER_HOSTS+=("${NODE_HOST_ARRAY[$i]}")
done
WORKER_COUNT="${#WORKER_IPS[@]}"

if [ "$WORKER_COUNT" -lt "$NODE_NUM" ]; then
    echo "  ERROR: need $NODE_NUM workers but only $WORKER_COUNT available"
    echo "         (total nodes=$TOTAL_COUNT, need 1 + $NUM_QUERY_NODES + $NODE_NUM = $((1 + NUM_QUERY_NODES + NODE_NUM)))"
    exit 1
fi
# 只用前 NODE_NUM 个 worker
WORKER_IPS=("${WORKER_IPS[@]:0:$NODE_NUM}")
WORKER_HOSTS=("${WORKER_HOSTS[@]:0:$NODE_NUM}")
WORKER_COUNT="$NODE_NUM"

# 总并发 = worker_num × svr_threads，平均分到每个 query 节点
TOTAL_QUERY_CONCURRENCY=$((WORKER_COUNT * SVR_THREADS))

echo "  Total nodes: $TOTAL_COUNT | host: $HOST_HOSTNAME | query: ${QUERY_HOSTS[*]} | workers ($WORKER_COUNT): ${WORKER_HOSTS[*]}"
echo "  Total query concurrency: $WORKER_COUNT × $SVR_THREADS = $TOTAL_QUERY_CONCURRENCY"

# --------------- 4. 写 config.json ---------------
CONFIG_PATH="$PROJECT_ROOT/config.${SLURM_JOB_ID:-manual}.${BASE_PORT}.json"
export VR_CONFIG_PATH="$CONFIG_PATH"

NODES_JSON=""
for i in "${!WORKER_IPS[@]}"; do
    port=$((BASE_PORT + i))
    [ "$i" -gt 0 ] && NODES_JSON+=","
    NODES_JSON+=$'\n'"        {\"ip\": \"${WORKER_IPS[$i]}\", \"hostname\": \"${WORKER_HOSTS[$i]}\", \"port\": $port, \"user_name\": \"$USER\"}"
done

QUERY_NODES_JSON=""
QUERY_BASE_PORT=$((BASE_PORT + 1000))
for i in "${!QUERY_IPS[@]}"; do
    qport=$((QUERY_BASE_PORT + i))
    [ "$i" -gt 0 ] && QUERY_NODES_JSON+=","
    QUERY_NODES_JSON+=$'\n'"        {\"ip\": \"${QUERY_IPS[$i]}\", \"hostname\": \"${QUERY_HOSTS[$i]}\", \"port\": $qport, \"user_name\": \"$USER\"}"
done

# layer_config：所有层均使用全部 worker；前 2 层按 worker 划分，其余全复制
WORKER_LIST=""
for ((wi=0; wi<WORKER_COUNT; wi++)); do
    [ "$wi" -gt 0 ] && WORKER_LIST+=", "
    WORKER_LIST+="$wi"
done
LAYER_CONFIG_JSON=""; LAYER_FLAGS_JSON=""
for ((l=0; l<=LIMITED_LEVEL; l++)); do
    if [ "$l" -gt 0 ]; then LAYER_CONFIG_JSON+=","; LAYER_FLAGS_JSON+=", "; fi
    LAYER_CONFIG_JSON+=$'\n'"        \"$l\": [$WORKER_LIST]"
    if [ "$l" -lt 2 ]; then LAYER_FLAGS_JSON+="false"; else LAYER_FLAGS_JSON+="true"; fi
done

cat > "$CONFIG_PATH" <<CFGEOF
{
    "host_node": {"ip": "$HOST_IP", "hostname": "$HOST_HOSTNAME", "port": $BASE_PORT, "user_name": "$USER"},
    "work_nodes": [$NODES_JSON
    ],
    "query_nodes": [$QUERY_NODES_JSON
    ],
    "layer_config": {$LAYER_CONFIG_JSON
    },
    "layer_replicate_flags": [$LAYER_FLAGS_JSON]
}
CFGEOF
echo "  VR_CONFIG_PATH=$CONFIG_PATH"
sync; sleep 1

# --------------- 5. 运行 main.py ---------------
SAVE_FLAG="";   [ "$SAVE" = "1" ] && SAVE_FLAG="--save"
SLURM_FLAG="";  [ "$USE_SLURM" = "1" ] && SLURM_FLAG="--slurm"
LAYER_FLAG="";  [ "$LAYER" = "True" ] && LAYER_FLAG="--layer"

LAYER_SUFFIX="";  [ "$LAYER" = "True" ] && LAYER_SUFFIX="-layer"
REP_FMT=$(python -c "r=float('$REPLICATION'); print(str(int(r)) if r==int(r) else str(r))")
TAG_SFX=""; [ -n "$PART_TAG" ] && TAG_SFX="-${PART_TAG}"
CLUSTER_BASE="./save/${DATA_NAME}/node${WORKER_COUNT}/m${M}-efc${EF_CONST}-ll${LIMITED_LEVEL}-${PART_MODE}-rep${REP_FMT}${LAYER_SUFFIX}${TAG_SFX}"
if [ "$FORCE_REPARTITION" = "1" ]; then
    echo "[force] Removing existing partitioned graphs: ${CLUSTER_BASE}-*.bin"
    rm -f ${CLUSTER_BASE}-*.bin
fi

echo "[5/5] Running main.py (workers=$WORKER_COUNT, query_clients=${#QUERY_IPS[@]})..."
set -x
python -u main.py \
    --data_name "$DATA_NAME" \
    --dtype "$DTYPE" \
    --k "$K" \
    --m "$M" \
    --ef_construction "$EF_CONST" \
    --ef_search "$EF_SEARCH" \
    --limited_level "$LIMITED_LEVEL" \
    --node_num "$WORKER_COUNT" \
    --partition_mode "$PART_MODE" \
    --partition_distribution nfs-load \
    --replication "$REPLICATION" \
    --num_construction_threads "${SLURM_CPUS_PER_TASK:-64}" \
    --num_query_threads "$TOTAL_QUERY_CONCURRENCY" \
    --num_threads "$SVR_THREADS" \
    --comm_mode "$COMM_MODE" \
    --distributed \
    --graph_path "./save/hnsw-${DATA_NAME}-m${M}-efc${EF_CONST}-ll${LIMITED_LEVEL}.bin" \
    --num_runs "$NUM_RUNS" \
    $LAYER_FLAG \
    $SAVE_FLAG \
    $SLURM_FLAG
set +x

echo "============================================"
echo "  Done at $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
