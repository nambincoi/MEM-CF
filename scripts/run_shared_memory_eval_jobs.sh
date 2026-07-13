#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MEMCF_EVAL_ROOT="${MEMCF_EVAL_ROOT:-$ROOT/evaluation_results_memcf_shared}"
export MEMCF_MEMORY_ROOT="${MEMCF_MEMORY_ROOT:-$ROOT/agent_memory_memcf_shared}"
export N_USERS="${N_USERS:-300}"
export NUM_USER_SHARDS="${NUM_USER_SHARDS:-3}"
export PHASE=eval_only
export LOAD_SAVED_MEMORY=1
export RUN_NAME_SUFFIX="${RUN_NAME_SUFFIX:-sharedeval}"

# Run one eval shard per wave by default. Use EVAL_USER_SHARD_ID=0,1,2
# to keep concurrency near 4 datasets x 7 variants = 28 jobs.
EVAL_USER_SHARD_ID="${EVAL_USER_SHARD_ID:-0}"
if [ "$EVAL_USER_SHARD_ID" -lt 0 ] || [ "$EVAL_USER_SHARD_ID" -ge "$NUM_USER_SHARDS" ]; then
  echo "[ERROR] EVAL_USER_SHARD_ID must be in [0, $((NUM_USER_SHARDS - 1))], got $EVAL_USER_SHARD_ID" >&2
  exit 1
fi

DATASETS_STR="${DATASETS:-Video_Game Digital_Music_1000u CDs_and_Vinyl_1000u All_Beauty_1000u}"
VARIANTS_STR="${VARIANTS:-A1_same_user_only A2_candidate_item_only A3_neighbor_user_only A4_full_graph A5_full_graph_noharm A6_random_memory A7_shuffled_memory}"
PID_DIR="$MEMCF_EVAL_ROOT/_pids_eval_shared_shard${EVAL_USER_SHARD_ID}of${NUM_USER_SHARDS}"
mkdir -p "$PID_DIR"
read -r -a DATASETS_ARR <<< "$DATASETS_STR"
read -r -a VARIANTS_ARR <<< "$VARIANTS_STR"

for ds in "${DATASETS_ARR[@]}"; do
  memory_file="$MEMCF_MEMORY_ROOT/$ds/shared_nuser${N_USERS}_${NUM_USER_SHARDS}shards.memory.json"
  if [ ! -f "$memory_file" ]; then
    echo "[ERROR] missing merged memory file: $memory_file" >&2
    exit 1
  fi
  for variant in "${VARIANTS_ARR[@]}"; do
    log_dir="$MEMCF_EVAL_ROOT/$ds/logs"
    mkdir -p "$log_dir"
    MEMORY_FILE="$memory_file" USER_SHARD_ID="$EVAL_USER_SHARD_ID" NUM_USER_SHARDS="$NUM_USER_SHARDS" \
      nohup bash "$ROOT/scripts/run_ablation_one_dataset.sh" "$ds" "$variant" \
      > "$log_dir/${variant}_eval_shared_shard${EVAL_USER_SHARD_ID}of${NUM_USER_SHARDS}.nohup.log" 2>&1 &
    pid=$!
    echo "$pid" > "$PID_DIR/${ds}_${variant}_shard${EVAL_USER_SHARD_ID}of${NUM_USER_SHARDS}.pid"
    echo "STARTED eval $ds $variant shard=$EVAL_USER_SHARD_ID/$NUM_USER_SHARDS pid=$pid memory=$memory_file"
  done
done

cat <<EOF
PID dir: $PID_DIR
Check status:
for p in $PID_DIR/*.pid; do pid=\$(cat "\$p"); ps -p "\$pid" >/dev/null && echo RUNNING \$(basename "\$p") "\$pid" || echo DONE \$(basename "\$p") "\$pid"; done
EOF
