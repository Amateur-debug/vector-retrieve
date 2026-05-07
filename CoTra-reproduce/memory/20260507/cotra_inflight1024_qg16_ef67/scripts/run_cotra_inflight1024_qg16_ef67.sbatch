#!/bin/bash
#SBATCH -J cotra_qg16
#SBATCH -p i64m512u
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 00:30:00
#SBATCH -o /hpc2hdd/home/rmeng603/workspace/CoTra/memory/20260507/cotra_inflight1024_qg16_ef67/logs/slurm-%j.out
#SBATCH -e /hpc2hdd/home/rmeng603/workspace/CoTra/memory/20260507/cotra_inflight1024_qg16_ef67/logs/slurm-%j.err

set -euo pipefail
ROOT=/hpc2hdd/home/rmeng603/workspace/CoTra
CASE=/hpc2hdd/home/rmeng603/workspace/CoTra/memory/20260507/cotra_inflight1024_qg16_ef67
DATA_ROOT=/hpc2hdd/home/rmeng603/vectordb/sift/100M/sift100m
INDEX_PREFIX=$ROOT/memory/20260424/sift100m_8node_experiment/results/index/merged_index
EF_LIST=${COTRA_EF_LIST:-66,67,67,67,67,67}
SEARCH_THREADS=${SEARCH_THREADS:-8}
QUERY_COUNT=${QUERY_COUNT:-10000}
DATASET_MILLIONS=${DATASET_MILLIONS:-100}
MEM_PORT=${MEM_PORT:-24816}
CONFIG_FILE=$CASE/scripts/slurm_${SLURM_JOB_ID}_ibs2.conf
EXPECTED_QUERY_GROUP_SIZE=16

mkdir -p "$CASE/logs" "$CASE/scripts"
ts() { date -Is; }
log_cmd() { echo "[$(ts)] $*" | tee -a "$CASE/commands_used.txt"; }
rdma_ip_for_host() {
  local h=${1%%.*}
  local n=${h#cpu1-}
  echo "10.120.30.${n}"
}
cleanup() {
  if [[ -n "${MEMPID:-}" ]]; then
    kill "$MEMPID" >/dev/null 2>&1 || true
    wait "$MEMPID" >/dev/null 2>&1 || true
  fi
  pkill -u "$USER" -f "memcached.*-p ${MEM_PORT}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

source /usr/local/Modules/init/bash
module load anaconda3 compilers/gcc-11.1.0
source /hpc2ssd/softwares/anaconda3/etc/profile.d/conda.sh
conda activate cotra-repro
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

actual_qgs=$(awk '/#define QUERY_GROUP_SIZE/ {gsub(/[()]/, "", $3); print $3}' "$ROOT/include/anns/anns_config.h")
if [[ "$actual_qgs" != "$EXPECTED_QUERY_GROUP_SIZE" ]]; then
  echo "Expected QUERY_GROUP_SIZE=$EXPECTED_QUERY_GROUP_SIZE, got $actual_qgs" >&2
  exit 3
fi

mapfile -t HOSTS < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
if [[ ${#HOSTS[@]} -ne 8 ]]; then
  echo "Expected 8 hosts, got ${#HOSTS[@]}: ${HOSTS[*]}" >&2
  exit 2
fi
MEM_IP=$(rdma_ip_for_host "${HOSTS[0]}")
{
  echo "$MEM_IP"
  echo "$MEM_PORT"
  for i in "${!HOSTS[@]}"; do
    echo "$(rdma_ip_for_host "${HOSTS[$i]}")=$i"
  done
} > "$CONFIG_FILE"
cp -f "$CONFIG_FILE" "$CASE/scripts/config.used"

{
  echo "CASE=$CASE"
  echo "SLURM_JOB_ID=$SLURM_JOB_ID"
  echo "HOSTS=${HOSTS[*]}"
  echo "CONFIG_FILE=$CONFIG_FILE"
  echo "MEM_IP=$MEM_IP"
  echo "MEM_PORT=$MEM_PORT"
  echo "EF_LIST=$EF_LIST"
  echo "SEARCH_THREADS=$SEARCH_THREADS"
  echo "QUERY_COUNT=$QUERY_COUNT"
  echo "QUERY_GROUP_SIZE=$actual_qgs"
  echo "EST_INFLIGHT_PER_NODE=$((SEARCH_THREADS * actual_qgs))"
  echo "EST_INFLIGHT_CLUSTER=$((8 * SEARCH_THREADS * actual_qgs))"
  echo "BINARY=$(ls -lh "$ROOT/build/tests/scala_anns")"
} | tee "$CASE/logs/job_info_${SLURM_JOB_ID}.txt"
log_cmd "job $SLURM_JOB_ID hosts=${HOSTS[*]} config=$CONFIG_FILE ef=$EF_LIST search_threads=$SEARCH_THREADS query_count=$QUERY_COUNT query_group_size=$actual_qgs estimated_inflight_cluster=$((8 * SEARCH_THREADS * actual_qgs))"

pkill -u "$USER" -f "memcached.*-p ${MEM_PORT}" >/dev/null 2>&1 || true
memcached -l "$MEM_IP" -p "$MEM_PORT" -c 10000 > "$CASE/logs/memcached_${SLURM_JOB_ID}_${HOSTS[0]}.log" 2>&1 &
MEMPID=$!
for _ in $(seq 1 30); do
  if timeout 2 bash -lc "</dev/tcp/${MEM_IP}/${MEM_PORT}" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
printf 'set serverNum 0 0 1\r\n0\r\nquit\r\n' | timeout 5 nc "$MEM_IP" "$MEM_PORT"
printf 'set clientNum 0 0 1\r\n0\r\nquit\r\n' | timeout 5 nc "$MEM_IP" "$MEM_PORT"
echo "MEMCACHED_INIT=OK"

export SEARCH_THREADS QUERY_COUNT DATASET_MILLIONS COTRA_EF_LIST="$EF_LIST" CONFIG_FILE
export OMP_NUM_THREADS=$SEARCH_THREADS
export MKL_NUM_THREADS=1
export LINE_BUFFER=1

SRUN_LOG_PATTERN=$CASE/logs/anns_job%j_%N_%t.log
log_cmd "srun scala_anns integrated query QUERY_GROUP_SIZE=$actual_qgs"
timeout 20m srun -N 8 -n 8 --ntasks-per-node=1 --kill-on-bad-exit=1 --cpus-per-task="$SEARCH_THREADS" \
  --output="$SRUN_LOG_PATTERN" \
  bash -lc '
    set -euo pipefail
    ROOT=/hpc2hdd/home/rmeng603/workspace/CoTra
    DATA_ROOT=/hpc2hdd/home/rmeng603/vectordb/sift/100M/sift100m
    INDEX_PREFIX=$ROOT/memory/20260424/sift100m_8node_experiment/results/index/merged_index
    source /usr/local/Modules/init/bash
    module load anaconda3 compilers/gcc-11.1.0
    source /hpc2ssd/softwares/anaconda3/etc/profile.d/conda.sh
    conda activate cotra-repro
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
    export OMP_NUM_THREADS="${SEARCH_THREADS}"
    export MKL_NUM_THREADS=1
    IB_DEV=${IB_DEV:-}
    if [[ -z "${IB_DEV}" ]] && command -v ibdev2netdev >/dev/null 2>&1; then
      IB_DEV=$(ibdev2netdev 2>/dev/null | awk '\''$0 ~ /==> ibs2/ && dev == "" {dev=$1} END {print dev}'\'')
    fi
    IB_DEV=${IB_DEV:-mlx5_2}
    echo "Requested search/query-injection threads: ${SEARCH_THREADS}"
    grep "^Cpus_allowed_list" /proc/self/status || true
    echo "COTRA_EF_LIST=${COTRA_EF_LIST}"
    echo "Using RDMA device: ${IB_DEV}"
    cd "$ROOT"
    exec stdbuf -o0 -e0 ./build/tests/scala_anns \
      --config_file "$CONFIG_FILE" \
      -d "$IB_DEV" \
      --app_type scala_v3 \
      --graph_type scalagraph_v3 \
      --data_type uint8 \
      --dist_fn l2 \
      --data_path "$DATA_ROOT/base.u8bin" \
      --query_path "$DATA_ROOT/queries.u8bin" \
      --gt_path "$DATA_ROOT/gt_100.bin" \
      --index_path_prefix "$INDEX_PREFIX" \
      -R 32 -L 100 -B 4 -M 8 -T "${SEARCH_THREADS}" \
      --scala_v3 -Q "${QUERY_COUNT}" -s "${DATASET_MILLIONS}" -t "${SEARCH_THREADS}"
  '
status=$?
echo "ANN_STATUS=$status"
exit "$status"
