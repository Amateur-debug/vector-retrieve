#!/bin/bash
# =============================================================================
# SLURM 单 allocation 批量作业 — 在 21 节点的 allocation 内串行跑 batch.py
#
# 架构（21 台机器）:
#   节点 0:    main.py orchestrator（由 batch.py 多次调用）
#   节点 1-4:  query 客户端（rank 0-3）
#   节点 5-20: 16 台 worker（main.py 按 --node_num 选用前 N 台）
#
# query nodes 数量由 batch.py 根据 node_num 动态决定：
#   8 worker  → 2 query nodes
#   16 worker → 4 query nodes
#
# 流程:
#   1. SLURM 分配 21 节点
#   2. 写 config.json（host + 4 query + 16 worker）
#   3. 运行 batch.py --slurm：内部串行调 main.py 8 次
#      - 同 (node_num, m) 内复用服务端（--skip_prepare）
#      - 跨组重建 cluster；同 m 复用 graph 文件
# =============================================================================

#SBATCH --job-name=vr-batch
#SBATCH --nodes=21
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=7-00:00:00
#SBATCH --partition=i64m512ue
#SBATCH --qos=cpu2600
#SBATCH --output=/hpc2hdd/home/xchen740/workspace/vector-retrieve/logs/slurm_%j.out
#SBATCH --error=/hpc2hdd/home/xchen740/workspace/vector-retrieve/logs/slurm_%j.err

set -euo pipefail
export PATH="/opt/slurm/bin:$PATH"
PROJECT_ROOT="/hpc2hdd/home/xchen740/workspace/vector-retrieve"
cd "$PROJECT_ROOT"
mkdir -p logs

BASE_PORT="${BASE_PORT:-18080}"
NUM_QUERY_NODES=4

echo "============================================"
echo "  vector-retrieve SLURM Batch Job"
echo "  Job ID:    ${SLURM_JOB_ID:-N/A}"
echo "  NNODES:    ${SLURM_NNODES:-N/A}"
echo "  Nodelist:  ${SLURM_JOB_NODELIST:-N/A}"
echo "  Started:   $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"

# --------------- 1. 激活 Conda ---------------
echo "[1/4] Activating conda..."
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate vector-retrieve

# --------------- 2. 构建 C++ ---------------
echo "[2/4] Building C++ extension..."
BUILD_DIR="$PROJECT_ROOT/build"
SO_COUNT=$(find "$BUILD_DIR" -maxdepth 1 -name "vector_retrieve*.so" 2>/dev/null | wc -l)
if [ "$SO_COUNT" -eq 0 ]; then
    mkdir -p "$BUILD_DIR" && cd "$BUILD_DIR"
    cmake .. -DCMAKE_BUILD_TYPE=Release \
        -DPYTHON_EXECUTABLE="$(which python)" \
        -Dpybind11_DIR="$(python -c 'import pybind11,os;print(os.path.join(os.path.dirname(pybind11.__file__),"share","cmake","pybind11"))')"
    make -j"${SLURM_CPUS_PER_TASK:-$(nproc)}"
    cd "$PROJECT_ROOT"
fi
echo "  Build OK"

# --------------- 3. 解析节点 → 写 config.json ---------------
echo "[3/4] Writing config.json (host + 4 query + 16 worker)..."

NODE_HOSTNAMES=$(scontrol show hostname "$SLURM_JOB_NODELIST")
readarray -t NODE_HOST_ARRAY <<< "$(echo "$NODE_HOSTNAMES" | tr ' ' '\n')"
TOTAL_COUNT="${#NODE_HOST_ARRAY[@]}"

if [ "$TOTAL_COUNT" -lt 21 ]; then
    echo "ERROR: Need >=21 nodes, got $TOTAL_COUNT"
    exit 1
fi

NODE_IPS=()
for h in "${NODE_HOST_ARRAY[@]}"; do
    ip=$(getent hosts "$h" | awk '{print $1}')
    [ -z "$ip" ] && { echo "ERROR: Cannot resolve $h"; exit 1; }
    NODE_IPS+=("$ip")
done

HOST_IP="${NODE_IPS[0]}"
HOST_HOSTNAME="${NODE_HOST_ARRAY[0]}"
USER_NAME="${USER}"

# query nodes: index 1..NUM_QUERY_NODES
QUERY_NODES_JSON=""
QUERY_BASE_PORT=$((BASE_PORT + 1000))
for i in $(seq 1 $NUM_QUERY_NODES); do
    qport=$((QUERY_BASE_PORT + i - 1))
    [ "$i" -gt 1 ] && QUERY_NODES_JSON+=","
    QUERY_NODES_JSON+=$'\n'"        {\"ip\": \"${NODE_IPS[$i]}\", \"hostname\": \"${NODE_HOST_ARRAY[$i]}\", \"port\": $qport, \"user_name\": \"$USER_NAME\"}"
done

# worker nodes: index (NUM_QUERY_NODES+1)..(NUM_QUERY_NODES+16) (16 workers)
WORKER_COUNT=16
NODES_JSON=""
for ((wi=0; wi<WORKER_COUNT; wi++)); do
    idx=$((NUM_QUERY_NODES + 1 + wi))
    port=$((BASE_PORT + wi))
    [ "$wi" -gt 0 ] && NODES_JSON+=","
    NODES_JSON+=$'\n'"        {\"ip\": \"${NODE_IPS[$idx]}\", \"hostname\": \"${NODE_HOST_ARRAY[$idx]}\", \"port\": $port, \"user_name\": \"$USER_NAME\"}"
done

# limited_level=5 → 6 layers; flags = [false, false, true, true, true, true]
cat > "$PROJECT_ROOT/config.json" <<CFGEOF
{
    "host_node": {"ip": "$HOST_IP", "hostname": "$HOST_HOSTNAME", "port": $BASE_PORT, "user_name": "$USER_NAME"},
    "work_nodes": [$NODES_JSON
    ],
    "query_nodes": [$QUERY_NODES_JSON
    ],
    "layer_replicate_flags": [false, false, true, true, true, true]
}
CFGEOF

echo "  config.json:"
cat "$PROJECT_ROOT/config.json"

# --------------- 4. 运行 batch.py ---------------
echo "[4/4] Running batch.py --slurm ..."
set -x
python -u batch.py --slurm --skip
set +x

echo "============================================"
echo "  Done at $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
