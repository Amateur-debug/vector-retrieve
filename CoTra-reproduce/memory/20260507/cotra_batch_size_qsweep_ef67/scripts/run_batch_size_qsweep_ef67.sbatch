#!/bin/bash
#SBATCH -J cotra_bsz
#SBATCH -p i64m512u
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 02:00:00
#SBATCH -o /hpc2hdd/home/rmeng603/workspace/CoTra/memory/20260507/cotra_batch_size_qsweep_ef67/logs/slurm-%j.out
#SBATCH -e /hpc2hdd/home/rmeng603/workspace/CoTra/memory/20260507/cotra_batch_size_qsweep_ef67/logs/slurm-%j.err

set -uo pipefail
ROOT=/hpc2hdd/home/rmeng603/workspace/CoTra
CASE=/hpc2hdd/home/rmeng603/workspace/CoTra/memory/20260507/cotra_batch_size_qsweep_ef67
DATA_ROOT=/hpc2hdd/home/rmeng603/vectordb/sift/100M/sift100m
INDEX_PREFIX=$ROOT/memory/20260424/sift100m_8node_experiment/results/index/merged_index
QUERY_PATH=$CASE/data/queries_640k.u8bin
GT_PATH=$CASE/data/gt_100_640k_ids_only.bin
SEARCH_THREADS=${SEARCH_THREADS:-8}
DATASET_MILLIONS=${DATASET_MILLIONS:-100}
WARMUP_EF=${WARMUP_EF:-66}
MEASURE_EF=${MEASURE_EF:-67}
Q_LIST=${Q_LIST:-10000,20000,40000,80000,160000,320000,640000}
PER_Q_TIMEOUT=${PER_Q_TIMEOUT:-420}
STOP_ON_TIMEOUT=${STOP_ON_TIMEOUT:-0}
MEM_PORT=${MEM_PORT:-24711}
CONFIG_FILE=$CASE/scripts/slurm_${SLURM_JOB_ID}_ibs2.conf
STATUS_FILE=$CASE/results/status_${SLURM_JOB_ID}.tsv

mkdir -p "$CASE/logs" "$CASE/results" "$CASE/scripts"
ts() { date -Is; }
log_cmd() { echo "[$(ts)] $*" | tee -a "$CASE/commands_used.txt"; }
rdma_ip_for_host() {
  local h=${1%%.*}
  local n=${h#cpu1-}
  echo "10.120.30.${n}"
}
stop_memcached() {
  if [[ -n "${MEMPID:-}" ]]; then
    kill "$MEMPID" >/dev/null 2>&1 || true
    wait "$MEMPID" >/dev/null 2>&1 || true
    unset MEMPID
  fi
  pkill -u "$USER" -f "memcached.*-p ${MEM_PORT}" >/dev/null 2>&1 || true
}
cleanup() { stop_memcached; }
trap cleanup EXIT

source /usr/local/Modules/init/bash
module load anaconda3 compilers/gcc-11.1.0
source /hpc2ssd/softwares/anaconda3/etc/profile.d/conda.sh
conda activate cotra-repro
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

if [[ ! -x "$ROOT/build/tests/scala_anns" ]]; then
  echo "missing binary: $ROOT/build/tests/scala_anns" >&2
  exit 3
fi
if [[ ! -f "$QUERY_PATH" || ! -f "$GT_PATH" ]]; then
  echo "missing amplified query/gt: $QUERY_PATH $GT_PATH" >&2
  exit 4
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

git -C "$ROOT" rev-parse HEAD > "$CASE/cotra_git_head.txt" || true
git -C "$ROOT" status --short > "$CASE/cotra_git_status_before.txt" || true
git -C "$ROOT" diff > "$CASE/code_diff.patch" || true
cp -f "$ROOT/build/tests/scala_anns" "$CASE/results/scala_anns_${SLURM_JOB_ID}.snapshot" || true

{
  echo "CASE=$CASE"
  echo "HOSTS=${HOSTS[*]}"
  echo "CONFIG_FILE=$CONFIG_FILE"
  echo "MEM_IP=$MEM_IP"
  echo "MEM_PORT=$MEM_PORT"
  echo "WARMUP_EF=$WARMUP_EF MEASURE_EF=$MEASURE_EF"
  echo "Q_LIST=$Q_LIST"
  echo "PER_Q_TIMEOUT=$PER_Q_TIMEOUT"
  echo "STOP_ON_TIMEOUT=$STOP_ON_TIMEOUT"
  echo "SEARCH_THREADS=$SEARCH_THREADS"
  echo "QUERY_PATH=$QUERY_PATH"
  echo "GT_PATH=$GT_PATH"
  echo "INDEX_PREFIX=$INDEX_PREFIX"
} | tee "$CASE/results/run_${SLURM_JOB_ID}_settings.txt"

echo -e "timestamp\tq\tstatus\truntime_sec\tlog_glob" > "$STATUS_FILE"
log_cmd "job $SLURM_JOB_ID hosts=${HOSTS[*]} config=$CONFIG_FILE q_list=$Q_LIST ef_list=${WARMUP_EF},${MEASURE_EF} search_threads=$SEARCH_THREADS timeout=${PER_Q_TIMEOUT}s stop_on_timeout=$STOP_ON_TIMEOUT"

IFS=',' read -r -a Q_VALUES <<< "$Q_LIST"
for Q in "${Q_VALUES[@]}"; do
  Q=$(echo "$Q" | tr -d '[:space:]')
  [[ -z "$Q" ]] && continue
  if (( Q % 8 != 0 )); then
    echo "Q=$Q is not divisible by 8; current ScalaANN_v3 drops the remainder, skip" | tee -a "$CASE/error_log.md"
    echo -e "$(ts)\t$Q\tSKIP_NOT_DIVISIBLE_BY_8\t0\t" >> "$STATUS_FILE"
    continue
  fi

  stop_memcached
  memcached -l "$MEM_IP" -p "$MEM_PORT" -c 10000 > "$CASE/logs/memcached_${SLURM_JOB_ID}_q${Q}_${HOSTS[0]}.log" 2>&1 &
  MEMPID=$!
  ready=0
  for _ in $(seq 1 30); do
    if timeout 2 bash -lc "</dev/tcp/${MEM_IP}/${MEM_PORT}" >/dev/null 2>&1; then
      ready=1
      break
    fi
    sleep 1
  done
  if [[ "$ready" != 1 ]]; then
    echo "memcached not ready for Q=$Q" | tee -a "$CASE/error_log.md"
    echo -e "$(ts)\t$Q\tMEMCACHED_NOT_READY\t0\t" >> "$STATUS_FILE"
    continue
  fi
  printf 'set serverNum 0 0 1\r\n0\r\nquit\r\n' | timeout 5 nc "$MEM_IP" "$MEM_PORT" >/dev/null || true
  printf 'set clientNum 0 0 1\r\n0\r\nquit\r\n' | timeout 5 nc "$MEM_IP" "$MEM_PORT" >/dev/null || true

  export SEARCH_THREADS DATASET_MILLIONS CONFIG_FILE QUERY_PATH GT_PATH INDEX_PREFIX
  export COTRA_EF_LIST="${WARMUP_EF},${MEASURE_EF}"
  export QUERY_COUNT="$Q"
  export OMP_NUM_THREADS=$SEARCH_THREADS
  export MKL_NUM_THREADS=1
  export LINE_BUFFER=1
  SRUN_LOG_PATTERN=$CASE/logs/anns_q${Q}_job%j_%N_%t.log
  log_cmd "srun scala_anns integrated batch_size_Q=$Q per_rank=$((Q/8)) warmup=$WARMUP_EF measure=$MEASURE_EF"
  start_epoch=$(date +%s)
  timeout -k 30s "${PER_Q_TIMEOUT}s" srun -N 8 -n 8 --ntasks-per-node=1 --kill-on-bad-exit=1 --cpus-per-task="$SEARCH_THREADS" \
    --output="$SRUN_LOG_PATTERN" \
    bash -lc '
      set -euo pipefail
      ROOT=/hpc2hdd/home/rmeng603/workspace/CoTra
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
      echo "COTRA_EF_LIST=${COTRA_EF_LIST} QUERY_COUNT=${QUERY_COUNT}"
      echo "QUERY_PATH=${QUERY_PATH} GT_PATH=${GT_PATH}"
      echo "Using RDMA device: ${IB_DEV}"
      cd "$ROOT"
      exec stdbuf -o0 -e0 ./build/tests/scala_anns \
        --config_file "$CONFIG_FILE" \
        -d "$IB_DEV" \
        --app_type scala_v3 \
        --graph_type scalagraph_v3 \
        --data_type uint8 \
        --dist_fn l2 \
        --data_path /hpc2hdd/home/rmeng603/vectordb/sift/100M/sift100m/base.u8bin \
        --query_path "$QUERY_PATH" \
        --gt_path "$GT_PATH" \
        --index_path_prefix "$INDEX_PREFIX" \
        -R 32 -L 100 -B 4 -M 8 -T "${SEARCH_THREADS}" \
        --scala_v3 -Q "${QUERY_COUNT}" -s "${DATASET_MILLIONS}" -t "${SEARCH_THREADS}"
    '
  status=$?
  end_epoch=$(date +%s)
  runtime=$((end_epoch - start_epoch))
  if [[ $status -eq 124 || $status -eq 137 ]]; then
    echo "Q=$Q timed out after ${PER_Q_TIMEOUT}s" | tee -a "$CASE/error_log.md"
  elif [[ $status -ne 0 ]]; then
    echo "Q=$Q failed with status $status" | tee -a "$CASE/error_log.md"
  fi
  echo -e "$(ts)\t$Q\t$status\t$runtime\t$SRUN_LOG_PATTERN" >> "$STATUS_FILE"
  if [[ "$STOP_ON_TIMEOUT" == "1" && "$status" != "0" ]]; then
    log_cmd "stop sweep after Q=$Q status=$status"
    break
  fi
  sleep 3
done

log_cmd "batch-size sweep job $SLURM_JOB_ID finished"
exit 0
