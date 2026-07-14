#!/bin/bash
# =============================================================================
# 场景 1：单 node 64 核 —— 建「完整图」(full HNSW graph)
#
# 从原始数据建出完整 HNSW 图，非分布式、单节点。产物：
#   ./save/hnsw-{DATA_NAME}-m{M}-efc{EF_CONST}-ll{LIMITED_LEVEL}.bin
# 之后供 slurm_build_partition.sh / slurm_query.sh 复用做 partition。
#
# 并行：不同 M / dataset 产出不同 graph 文件 + 独立 VR_CONFIG_PATH，
#       可同时提交多个本脚本的 job，互不干扰。
#
# 用法示例：
#   sbatch slurm_build_full.sh                              # 默认 deep100m m=10
#   DATA_NAME=deep100m M=30 sbatch slurm_build_full.sh
#   DATA_NAME=sift100m-u8 DTYPE=uint8 M=20 sbatch slurm_build_full.sh
# =============================================================================

#SBATCH --job-name=vr-build-full
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=1-00:00:00
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
DATA_NAME="${DATA_NAME:-deep100m}"
DTYPE="${DTYPE:-float}"
M="${M:-10}"
EF_CONST="${EF_CONST:-200}"
LIMITED_LEVEL="${LIMITED_LEVEL:-5}"
BUILD_THREADS="${BUILD_THREADS:-64}"

echo "============================================"
echo "  Build FULL graph: data=$DATA_NAME dtype=$DTYPE M=$M efc=$EF_CONST ll=$LIMITED_LEVEL"
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

# --------------- 3. 写最小隔离 config ---------------
# main.py import 时会 json.load(VR_CONFIG_PATH)，即使非分布式也需要一个 config。
# build_only 在 partition 之前 return，所以这里只需占位（work/query 节点为空）。
echo "[3/4] Writing minimal isolated config..."
HOST_HOSTNAME=$(hostname)
HOST_IP=$(getent hosts "$HOST_HOSTNAME" | awk '{print $1}')
[ -z "$HOST_IP" ] && HOST_IP="127.0.0.1"
CONFIG_PATH="$PROJECT_ROOT/config.full.${SLURM_JOB_ID:-manual}.json"
export VR_CONFIG_PATH="$CONFIG_PATH"
cat > "$CONFIG_PATH" <<CFGEOF
{
    "host_node": {"ip": "$HOST_IP", "hostname": "$HOST_HOSTNAME", "port": 40000, "user_name": "$USER"},
    "work_nodes": [],
    "query_nodes": [],
    "layer_config": {"0": [0]},
    "layer_replicate_flags": [false]
}
CFGEOF
echo "  VR_CONFIG_PATH=$CONFIG_PATH"

# --------------- 4. 建图 ---------------
GRAPH_PATH="./save/hnsw-${DATA_NAME}-m${M}-efc${EF_CONST}-ll${LIMITED_LEVEL}.bin"
if [ -f "$GRAPH_PATH" ]; then
    echo "[4/4] Full graph already exists, skipping: $GRAPH_PATH"
    exit 0
fi

echo "[4/4] Building full graph -> $GRAPH_PATH"
set -x
python -u main.py \
    --data_name "$DATA_NAME" \
    --dtype "$DTYPE" \
    --m "$M" \
    --ef_construction "$EF_CONST" \
    --limited_level "$LIMITED_LEVEL" \
    --node_num 1 \
    --num_query_nodes 0 \
    --num_construction_threads "$BUILD_THREADS" \
    --graph_path "$GRAPH_PATH" \
    --build_only \
    --save
set +x

echo "============================================"
echo "  Done at $(date '+%Y-%m-%d %H:%M:%S')"
[ -f "$GRAPH_PATH" ] && echo "  OK: $GRAPH_PATH" || echo "  WARNING: graph not found!"
echo "============================================"
