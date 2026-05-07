#!/bin/bash
#SBATCH -J cotra_qmem
#SBATCH -p i64m512u
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -w cpu1-[90-97]
#SBATCH -t 00:55:00
#SBATCH -o /hpc2hdd/home/rmeng603/workspace/CoTra/memory/20260507/cotra_qscale_mem_sift100m/logs/slurm-%j.out
#SBATCH -e /hpc2hdd/home/rmeng603/workspace/CoTra/memory/20260507/cotra_qscale_mem_sift100m/logs/slurm-%j.err

set -uo pipefail
ROOT=/hpc2hdd/home/rmeng603/workspace/CoTra
CASE=$ROOT/memory/20260507/cotra_qscale_mem_sift100m
INDEX_PREFIX=$ROOT/memory/20260424/sift100m_8node_experiment/results/index/merged_index
QUERY_PATH=$CASE/data/queries_640k.u8bin
GT_PATH=$CASE/data/gt_100_640k_ids_only.bin
SEARCH_THREADS=${SEARCH_THREADS:-8}
DATASET_MILLIONS=${DATASET_MILLIONS:-100}
QUERY_GROUP_SIZE_VALUE=${COTRA_QUERY_GROUP_SIZE:-16}
EF_LIST=${COTRA_EF_LIST:-66,67,67,67,67,67}
QUERY_COUNT=${QUERY_COUNT:?set QUERY_COUNT, e.g. 160000}
RUN_TIMEOUT=${RUN_TIMEOUT:-1800}
MEM_PORT=${MEM_PORT:-24917}
MEM_SAMPLE_INTERVAL=${MEM_SAMPLE_INTERVAL:-1}
NODE_EXPECTED=${NODE_EXPECTED:-8}
CONFIG_FILE=$CASE/scripts/slurm_${SLURM_JOB_ID}_ibs2.conf
STATUS_FILE=$CASE/results/status_${SLURM_JOB_ID}.tsv
MEM_DIR=$CASE/mem_samples/q${QUERY_COUNT}_job${SLURM_JOB_ID}

mkdir -p "$CASE/logs" "$CASE/results" "$CASE/scripts" "$MEM_DIR"
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

if [[ ! -x "$ROOT/build/tests/scala_anns" ]]; then
  echo "missing binary: $ROOT/build/tests/scala_anns" >&2
  exit 3
fi
if [[ ! -f "$QUERY_PATH" || ! -f "$GT_PATH" ]]; then
  echo "missing query/gt: $QUERY_PATH $GT_PATH" >&2
  exit 4
fi
if (( QUERY_COUNT % NODE_EXPECTED != 0 )); then
  echo "QUERY_COUNT=$QUERY_COUNT is not divisible by $NODE_EXPECTED" | tee -a "$CASE/error_log.md"
  exit 5
fi
if (( QUERY_COUNT > 640000 )); then
  echo "QUERY_COUNT=$QUERY_COUNT exceeds current 640K query/GT cache" | tee -a "$CASE/error_log.md"
  exit 5
fi

mapfile -t HOSTS < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
if [[ ${#HOSTS[@]} -ne $NODE_EXPECTED ]]; then
  echo "Expected $NODE_EXPECTED hosts, got ${#HOSTS[@]}: ${HOSTS[*]}" >&2
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
cp -f "$CONFIG_FILE" "$CASE/scripts/config_${SLURM_JOB_ID}.used"
cp -f "$CONFIG_FILE" "$CASE/scripts/config.used"

git -C "$ROOT" rev-parse HEAD > "$CASE/cotra_git_head.txt" || true
git -C "$ROOT" status --short > "$CASE/cotra_git_status_at_job_${SLURM_JOB_ID}.txt" || true
git -C "$ROOT" diff > "$CASE/code_diff_at_job_${SLURM_JOB_ID}.patch" || true
cp -f "$ROOT/build/tests/scala_anns" "$CASE/results/scala_anns_${SLURM_JOB_ID}.snapshot" || true

echo -e "timestamp\tjob_id\tquery_count\tper_node_query_count\tquery_group_size\tef_list\tstatus\truntime_sec\tlog_glob\tmem_dir" > "$STATUS_FILE"
PER_NODE_QUERY_COUNT=$((QUERY_COUNT / NODE_EXPECTED))
{
  echo "CASE=$CASE"
  echo "JOB_ID=$SLURM_JOB_ID"
  echo "HOSTS=${HOSTS[*]}"
  echo "CONFIG_FILE=$CONFIG_FILE"
  echo "MEM_IP=$MEM_IP"
  echo "MEM_PORT=$MEM_PORT"
  echo "QUERY_COUNT=$QUERY_COUNT"
  echo "PER_NODE_QUERY_COUNT=$PER_NODE_QUERY_COUNT"
  echo "EF_LIST=$EF_LIST"
  echo "COTRA_QUERY_GROUP_SIZE=$QUERY_GROUP_SIZE_VALUE"
  echo "SEARCH_THREADS=$SEARCH_THREADS"
  echo "RUN_TIMEOUT=$RUN_TIMEOUT"
  echo "MEM_SAMPLE_INTERVAL=$MEM_SAMPLE_INTERVAL"
  echo "QUERY_PATH=$QUERY_PATH"
  echo "GT_PATH=$GT_PATH"
  echo "INDEX_PREFIX=$INDEX_PREFIX"
} | tee "$CASE/results/run_${SLURM_JOB_ID}_settings.txt"
log_cmd "job $SLURM_JOB_ID hosts=${HOSTS[*]} q=$QUERY_COUNT per_node_q=$PER_NODE_QUERY_COUNT qg=$QUERY_GROUP_SIZE_VALUE ef_list=$EF_LIST search_threads=$SEARCH_THREADS timeout=${RUN_TIMEOUT}s mem_interval=${MEM_SAMPLE_INTERVAL}s"

pkill -u "$USER" -f "memcached.*-p ${MEM_PORT}" >/dev/null 2>&1 || true
memcached -l "$MEM_IP" -p "$MEM_PORT" -c 10000 > "$CASE/logs/memcached_${SLURM_JOB_ID}_${HOSTS[0]}.log" 2>&1 &
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
  echo "memcached not ready on $MEM_IP:$MEM_PORT" | tee -a "$CASE/error_log.md"
  exit 6
fi
printf 'set serverNum 0 0 1\r\n0\r\nquit\r\n' | timeout 5 nc "$MEM_IP" "$MEM_PORT" >/dev/null || true
printf 'set clientNum 0 0 1\r\n0\r\nquit\r\n' | timeout 5 nc "$MEM_IP" "$MEM_PORT" >/dev/null || true

export ROOT CASE INDEX_PREFIX QUERY_PATH GT_PATH SEARCH_THREADS DATASET_MILLIONS CONFIG_FILE QUERY_COUNT MEM_DIR MEM_SAMPLE_INTERVAL
export COTRA_QUERY_GROUP_SIZE="$QUERY_GROUP_SIZE_VALUE"
export COTRA_EF_LIST="$EF_LIST"
export OMP_NUM_THREADS=$SEARCH_THREADS
export MKL_NUM_THREADS=1
export LINE_BUFFER=1

SRUN_LOG_PATTERN=$CASE/logs/anns_q${QUERY_COUNT}_qg${QUERY_GROUP_SIZE_VALUE}_job%j_%N_%t.log
log_cmd "srun scala_anns SIFT100M q=$QUERY_COUNT per_node_q=$PER_NODE_QUERY_COUNT qg=$QUERY_GROUP_SIZE_VALUE ef_list=$EF_LIST with per-node memory sampling"
start_epoch=$(date +%s)
timeout -k 30s "${RUN_TIMEOUT}s" srun -N "$NODE_EXPECTED" -n "$NODE_EXPECTED" --ntasks-per-node=1 --kill-on-bad-exit=1 --cpus-per-task="$SEARCH_THREADS" \
  --output="$SRUN_LOG_PATTERN" \
  bash -lc '
    set -uo pipefail
    ROOT=${ROOT:-/hpc2hdd/home/rmeng603/workspace/CoTra}
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
    NODE=${SLURMD_NODENAME:-$(hostname -s)}
    RANK=${SLURM_PROCID:-0}
    TASK_MEM_DIR="${MEM_DIR}"
    mkdir -p "${TASK_MEM_DIR}"
    MEM_FILE="${TASK_MEM_DIR}/${NODE}_rank${RANK}.tsv"
    PID_FILE="${TASK_MEM_DIR}/${NODE}_rank${RANK}.pid"
    echo -e "timestamp\tepoch_s\tnode\trank\tpid\tvmrss_kb\tvmhwm_kb\tvmsize_kb\tmemtotal_kb\tmemavailable_kb\tsystem_used_kb\tstate" > "${MEM_FILE}"
    sample_mem() {
      local pid=$1
      local rss=0 hwm=0 vsz=0 state=NA total=0 avail=0 used=0 now_iso now_s
      now_iso=$(date -Is)
      now_s=$(date +%s)
      if [[ -r "/proc/${pid}/status" ]]; then
        read -r rss hwm vsz state < <(awk '\''BEGIN{rss=0;hwm=0;vsz=0;state="NA"} /^State:/ {state=$2} /^VmRSS:/ {rss=$2} /^VmHWM:/ {hwm=$2} /^VmSize:/ {vsz=$2} END{print rss, hwm, vsz, state}'\'' "/proc/${pid}/status" 2>/dev/null)
      fi
      read -r total avail < <(awk '\''BEGIN{total=0;avail=0} /^MemTotal:/ {total=$2} /^MemAvailable:/ {avail=$2} END{print total, avail}'\'' /proc/meminfo)
      if [[ -n "$total" && -n "$avail" ]]; then
        used=$((total - avail))
      fi
      echo -e "${now_iso}\t${now_s}\t${NODE}\t${RANK}\t${pid}\t${rss}\t${hwm}\t${vsz}\t${total}\t${avail}\t${used}\t${state}" >> "${MEM_FILE}"
    }
    monitor_pid() {
      local pid=$1
      while kill -0 "${pid}" >/dev/null 2>&1; do
        sample_mem "${pid}"
        sleep "${MEM_SAMPLE_INTERVAL}"
      done
      sample_mem "${pid}" || true
    }
    echo "Requested search/query-injection threads: ${SEARCH_THREADS}"
    grep "^Cpus_allowed_list" /proc/self/status || true
    echo "COTRA_EF_LIST=${COTRA_EF_LIST} QUERY_COUNT=${QUERY_COUNT} COTRA_QUERY_GROUP_SIZE=${COTRA_QUERY_GROUP_SIZE}"
    echo "QUERY_PATH=${QUERY_PATH} GT_PATH=${GT_PATH}"
    echo "Using RDMA device: ${IB_DEV}"
    echo "MEM_SAMPLE_FILE=${MEM_FILE}"
    cd "$ROOT"
    stdbuf -o0 -e0 ./build/tests/scala_anns \
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
      --scala_v3 -Q "${QUERY_COUNT}" -s "${DATASET_MILLIONS}" -t "${SEARCH_THREADS}" &
    APP_PID=$!
    echo "${APP_PID}" > "${PID_FILE}"
    monitor_pid "${APP_PID}" &
    MON_PID=$!
    wait "${APP_PID}"
    APP_STATUS=$?
    kill "${MON_PID}" >/dev/null 2>&1 || true
    wait "${MON_PID}" >/dev/null 2>&1 || true
    sample_mem "${APP_PID}" || true
    exit "${APP_STATUS}"
  '
status=$?
end_epoch=$(date +%s)
runtime=$((end_epoch - start_epoch))
if [[ $status -ne 0 ]]; then
  echo "Q=$QUERY_COUNT failed with status $status" | tee -a "$CASE/error_log.md"
fi
echo -e "$(ts)\t$SLURM_JOB_ID\t$QUERY_COUNT\t$PER_NODE_QUERY_COUNT\t$QUERY_GROUP_SIZE_VALUE\t$EF_LIST\t$status\t$runtime\t$SRUN_LOG_PATTERN\t$MEM_DIR" >> "$STATUS_FILE"
exit "$status"
