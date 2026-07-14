#!/bin/bash
# =============================================================================
# 场景 2：单 node 64 核 —— 建「分区图」(partitioned graphs)
#
# 把完整 HNSW 图按 NODE_NUM 拆成各 worker 的本地图，非分布式、单节点。产物：
#   ./save/{DATA_NAME}/node{NODE_NUM}/m{M}-efc{EF_CONST}-ll{LIMITED_LEVEL}-{PART_MODE}-rep{REP}[-layer]-{i}.bin
# 完整图 (./save/hnsw-...bin) 若不存在，main.py 会先从原始数据建图再分区。
#
# 并行：不同 (M, rep, layer, node_num) 产出不同分区基路径 + 独立 VR_CONFIG_PATH，
#       可同时提交多个本脚本的 job。
#
# 用法示例：
#   NODE_NUM=8  M=30 REPLICATION=2 LAYER=True  sbatch slurm_build_partition.sh
#   NODE_NUM=16 M=30 REPLICATION=1 LAYER=False sbatch slurm_build_partition.sh
# =============================================================================

#SBATCH --job-name=vr-build-part
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --time=1-00:00:00
#SBATCH --partition=i64m512u
#SBATCH --qos=i64m512u
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
M="${M:-10}"
EF_CONST="${EF_CONST:-200}"
LIMITED_LEVEL="${LIMITED_LEVEL:-5}"
NODE_NUM="${NODE_NUM:-8}"
REPLICATION="${REPLICATION:-2}"
PART_MODE="${PART_MODE:-k-means}"
LAYER="${LAYER:-True}"               # True/False
BUILD_THREADS="${BUILD_THREADS:-64}"
PART_TAG="${PART_TAG:-}"             # 分区路径标签(区分复制策略,如 w005);空=默认
VR_REPL_BUDGET="${VR_REPL_BUDGET:-}" # 加权复制预算(>0 启用 targeted replication)
VR_REPL_FULL_ROUNDS="${VR_REPL_FULL_ROUNDS:-}" # 前 N 轮全量(默认 1),只对之后高跳轮加权
VR_REPL_BUDGETS="${VR_REPL_BUDGETS:-}" # 每跳独立预算,如 "0.85,0.1"(>=1 即该跳全量);优先于单值
VR_REPL_INDEG_THRESH="${VR_REPL_INDEG_THRESH:-}" # 全局阈值 τ(repN-wM):复制所有 indeg>=τ 的候选
VR_REPL_W="${VR_REPL_W:-}"           # 全局预算 M(0<M<1):repN-wM,内部 round0 直方图自动定 τ

# REP_SPEC 一行式命名:`repN-wM`(整体复制前N跳价值top M%)或 `repN`(全复制)。
# 例:REP_SPEC=rep2-w0.5 → REPLICATION=2 VR_REPL_W=0.5 PART_TAG=w0.5；REP_SPEC=rep2 → 全复制。
REP_SPEC="${REP_SPEC:-}"
if [ -n "$REP_SPEC" ]; then
    _spec="${REP_SPEC#rep}"                       # 去掉前缀 rep
    if [[ "$_spec" == *-w* ]]; then
        # 全局预算:repN-wM = 看 N 跳整体复制价值 top M%
        REPLICATION="${_spec%%-w*}"
        VR_REPL_W="${_spec##*-w}"
        PART_TAG="w${VR_REPL_W}"
    elif [[ "$_spec" == *.* ]]; then
        # 小数 repX.Y = 前 floor(X) 跳全量 + 下一跳价值 top 0.Y(per-hop)
        _full="${_spec%.*}"; _frac="0.${_spec#*.}"
        REPLICATION=$((_full + 1)); VR_REPL_FULL_ROUNDS="$_full"; VR_REPL_BUDGET="$_frac"; PART_TAG="r${_spec}"
    else
        REPLICATION="$_spec"; VR_REPL_W=""; PART_TAG=""   # 全复制 = 默认路径
    fi
fi
export VR_PART_TAG="$PART_TAG"
export VR_REPL_BUDGET
export VR_REPL_FULL_ROUNDS
export VR_REPL_BUDGETS
export VR_REPL_INDEG_THRESH
export VR_REPL_W

echo "============================================"
echo "  Build PARTITION graph: data=$DATA_NAME M=$M node_num=$NODE_NUM rep=$REPLICATION mode=$PART_MODE layer=$LAYER"
echo "  Job ID:  ${SLURM_JOB_ID:-N/A}   Node: $(hostname)"
echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
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

# rep 格式必须与 main.py 的 _rep_fmt 对齐（2.0->2, 1.5->1.5），否则 skip 检查失效。
REP_FMT=$(python -c "r=float('$REPLICATION'); print(str(int(r)) if r==int(r) else str(r))")
LAYER_SUFFIX=""; LAYER_FLAG=""
if [ "$LAYER" = "True" ]; then LAYER_SUFFIX="-layer"; LAYER_FLAG="--layer"; fi
TAG_SFX=""; [ -n "$PART_TAG" ] && TAG_SFX="-${PART_TAG}"
PART_BASE="./save/${DATA_NAME}/node${NODE_NUM}/m${M}-efc${EF_CONST}-ll${LIMITED_LEVEL}-${PART_MODE}-rep${REP_FMT}${LAYER_SUFFIX}${TAG_SFX}"

# --------------- 3. 检查是否已存在 ---------------
ALL_EXIST=1
for i in $(seq 0 $((NODE_NUM - 1))); do
    [ -f "${PART_BASE}-${i}.bin" ] || { ALL_EXIST=0; break; }
done
if [ "$ALL_EXIST" -eq 1 ]; then
    echo "[3/4] Partition files already exist (${PART_BASE}-*.bin), skipping."
    exit 0
fi
mkdir -p "./save/${DATA_NAME}/node${NODE_NUM}"

# --------------- 隔离 config ---------------
# 非分布式 layer 建图直接用 config 的 layer_config。绝不能继承根 config.json
# （那可能把所有层都指向 node0，导致 node0 文件巨大、查询退化为单节点）。
# 这里写 job 专属 config：layer_config 覆盖全部 NODE_NUM 个 worker。
echo "[3/4] Writing isolated config (layer_config -> [0..$((NODE_NUM-1))])..."
HOST_HOSTNAME=$(hostname)
HOST_IP=$(getent hosts "$HOST_HOSTNAME" | awk '{print $1}')
[ -z "$HOST_IP" ] && HOST_IP="127.0.0.1"

WORKER_LIST=""
for ((wi=0; wi<NODE_NUM; wi++)); do
    [ "$wi" -gt 0 ] && WORKER_LIST+=", "
    WORKER_LIST+="$wi"
done
LAYER_CONFIG_JSON=""
LAYER_FLAGS_JSON=""
for ((l=0; l<=LIMITED_LEVEL; l++)); do
    if [ "$l" -gt 0 ]; then LAYER_CONFIG_JSON+=","; LAYER_FLAGS_JSON+=", "; fi
    LAYER_CONFIG_JSON+=$'\n'"        \"$l\": [$WORKER_LIST]"
    # 前 2 层按 worker 划分(false)，其余全复制(true)
    if [ "$l" -lt 2 ]; then LAYER_FLAGS_JSON+="false"; else LAYER_FLAGS_JSON+="true"; fi
done

CONFIG_PATH="$PROJECT_ROOT/config.part.${SLURM_JOB_ID:-manual}.json"
export VR_CONFIG_PATH="$CONFIG_PATH"
cat > "$CONFIG_PATH" <<CFGEOF
{
    "host_node": {"ip": "$HOST_IP", "hostname": "$HOST_HOSTNAME", "port": 41000, "user_name": "$USER"},
    "work_nodes": [],
    "query_nodes": [],
    "layer_config": {$LAYER_CONFIG_JSON
    },
    "layer_replicate_flags": [$LAYER_FLAGS_JSON]
}
CFGEOF
echo "  VR_CONFIG_PATH=$CONFIG_PATH"

# --------------- 4. 分区（非分布式）---------------
GRAPH_PATH="./save/hnsw-${DATA_NAME}-m${M}-efc${EF_CONST}-ll${LIMITED_LEVEL}.bin"
echo "[4/4] Partitioning into $NODE_NUM nodes -> ${PART_BASE}-*.bin"
set -x
python -u main.py \
    --data_name "$DATA_NAME" \
    --dtype "$DTYPE" \
    --m "$M" \
    --ef_construction "$EF_CONST" \
    --ef_search 50 \
    --limited_level "$LIMITED_LEVEL" \
    --node_num "$NODE_NUM" \
    --num_query_nodes 0 \
    --num_query_threads 1 \
    --num_threads 1 \
    --num_construction_threads "$BUILD_THREADS" \
    --partition_mode "$PART_MODE" \
    --replication "$REPLICATION" \
    --partition_distribution nfs-load \
    --graph_path "$GRAPH_PATH" \
    --save \
    --skip_queries \
    $LAYER_FLAG
set +x

echo "Checking partition files..."
COUNT=$(ls ${PART_BASE}-*.bin 2>/dev/null | wc -l)
echo "  Files: $COUNT/$NODE_NUM"
[ "$COUNT" -eq "$NODE_NUM" ] && echo "Partition build done." || echo "WARNING: expected $NODE_NUM files, got $COUNT"
echo "============================================"
echo "  Done at $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
