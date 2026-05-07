#!/bin/bash
#SBATCH -J cotra_if1
#SBATCH -p i64m512u
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 00:20:00
#SBATCH -o /hpc2hdd/home/rmeng603/workspace/CoTra/memory/20260507/cotra_inflight_sweep_search_only/logs/slurm-%j.out
#SBATCH -e /hpc2hdd/home/rmeng603/workspace/CoTra/memory/20260507/cotra_inflight_sweep_search_only/logs/slurm-%j.err

set -euo pipefail
ROOT=/hpc2hdd/home/rmeng603/workspace/CoTra
CASE=/hpc2hdd/home/rmeng603/workspace/CoTra/memory/20260507/cotra_inflight_sweep_search_only
DATA_ROOT=/hpc2hdd/home/rmeng603/vectordb/sift/100M/sift100m
INDEX_PREFIX=$ROOT/memory/20260424/sift100m_8node_experiment/results/index/merged_index
QG=${QG:?Set QG to the desired COTRA_QUERY_GROUP_SIZE}
EF_LIST=${COTRA_EF_LIST:-66,67,67,67}
SEARCH_THREADS=${SEARCH_THREADS:-8}
QUERY_COUNT=${QUERY_COUNT:-10000}
DATASET_MILLIONS=${DATASET_MILLIONS:-100}
MEM_PORT=${MEM_PORT:-24833}
PER_RUN_TIMEOUT=${PER_RUN_TIMEOUT:-240}
CONFIG_FILE=$CASE/scripts/slurm_${SLURM_JOB_ID}_qg${QG}_ibs2.conf

mkdir -p "$CASE/logs" "$CASE/scripts" "$CASE/results"
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
cp -f "$CONFIG_FILE" "$CASE/scripts/config_qg${QG}_${SLURM_JOB_ID}.used"

EST=$((8 * SEARCH_THREADS * QG))
STATUS_FILE=$CASE/results/qg_single_run_status.tsv
if [[ ! -f "$STATUS_FILE" ]]; then
  printf 'jobid\tqg\testimated_inflight_cluster\tstatus\texit_code\tstarted_at\tfinished_at\n' > "$STATUS_FILE"
fi
{
  echo "CASE=$CASE"
  echo "SLURM_JOB_ID=$SLURM_JOB_ID"
  echo "QG=$QG"
  echo "EST_INFLIGHT_CLUSTER=$EST"
  echo "HOSTS=${HOSTS[*]}"
  echo "CONFIG_FILE=$CONFIG_FILE"
  echo "MEM_IP=$MEM_IP"
  echo "MEM_PORT=$MEM_PORT"
  echo "EF_LIST=$EF_LIST"
  echo "SEARCH_THREADS=$SEARCH_THREADS"
  echo "QUERY_COUNT=$QUERY_COUNT"
  echo "COTRA_SKIP_RECALL=1"
  echo "SOURCE_DEFAULT_QUERY_GROUP_SIZE=$(awk '/#define QUERY_GROUP_SIZE/ {gsub(/[()]/, "", $3); print $3}' "$ROOT/include/anns/anns_config.h")"
  echo "BINARY=$(ls -lh "$ROOT/build/tests/scala_anns")"
  echo "BINARY_SHA256=$(sha256sum "$ROOT/build/tests/scala_anns" | awk '{print $1}')"
} | tee "$CASE/logs/job_info_${SLURM_JOB_ID}_qg${QG}.txt"
log_cmd "job $SLURM_JOB_ID single-qg search-only qg=$QG estimated_inflight_cluster=$EST ef_list=$EF_LIST query_count=$QUERY_COUNT skip_recall=1 timeout=${PER_RUN_TIMEOUT}s"

pkill -u "$USER" -f "memcached.*-p ${MEM_PORT}" >/dev/null 2>&1 || true
memcached -l "$MEM_IP" -p "$MEM_PORT" -c 10000 > "$CASE/logs/memcached_${SLURM_JOB_ID}_qg${QG}_${HOSTS[0]}.log" 2>&1 &
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

export SEARCH_THREADS QUERY_COUNT DATASET_MILLIONS CONFIG_FILE
export OMP_NUM_THREADS=$SEARCH_THREADS
export MKL_NUM_THREADS=1
export LINE_BUFFER=1
export COTRA_SKIP_RECALL=1
export COTRA_QUERY_GROUP_SIZE="$QG"
export COTRA_EF_LIST="$EF_LIST"

STARTED=$(ts)
set +e
timeout "${PER_RUN_TIMEOUT}s" srun -N 8 -n 8 --ntasks-per-node=1 --kill-on-bad-exit=1 --cpus-per-task="$SEARCH_THREADS" \
  --output="$CASE/logs/anns_qg${QG}_job%j_%N_%t.log" \
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
    export COTRA_SKIP_RECALL=1
    echo "Requested search/query-injection threads: ${SEARCH_THREADS}"
    grep "^Cpus_allowed_list" /proc/self/status || true
    echo "COTRA_EF_LIST=${COTRA_EF_LIST}"
    echo "COTRA_QUERY_GROUP_SIZE=${COTRA_QUERY_GROUP_SIZE}"
    echo "COTRA_SKIP_RECALL=${COTRA_SKIP_RECALL}"
    IB_DEV=${IB_DEV:-}
    if [[ -z "${IB_DEV}" ]] && command -v ibdev2netdev >/dev/null 2>&1; then
      IB_DEV=$(ibdev2netdev 2>/dev/null | awk '\''$0 ~ /==> ibs2/ && dev == "" {dev=$1} END {print dev}'\'')
    fi
    IB_DEV=${IB_DEV:-mlx5_2}
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
rc=$?
set -e
FINISHED=$(ts)
if [[ $rc -eq 0 ]]; then
  status=OK
elif [[ $rc -eq 124 ]]; then
  status=TIMEOUT
else
  status=FAIL
fi
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$SLURM_JOB_ID" "$QG" "$EST" "$status" "$rc" "$STARTED" "$FINISHED" | tee -a "$STATUS_FILE"
log_cmd "job $SLURM_JOB_ID qg=$QG finished status=$status exit_code=$rc"
[[ $rc -eq 0 ]]
