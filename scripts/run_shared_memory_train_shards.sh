#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MEMCF_EVAL_ROOT="${MEMCF_EVAL_ROOT:-$ROOT/evaluation_results_memcf_shared}"
export MEMCF_MEMORY_ROOT="${MEMCF_MEMORY_ROOT:-$ROOT/agent_memory_memcf_shared}"
export N_USERS="${N_USERS:-300}"
export NUM_USER_SHARDS="${NUM_USER_SHARDS:-3}"
export PHASE=train_only
export VARIANT_FOR_TRAIN="${VARIANT_FOR_TRAIN:-A4_full_graph}"
export RUN_NAME_SUFFIX="${RUN_NAME_SUFFIX:-sharedtrain}"

DATASETS_STR="${DATASETS:-Video_Game Digital_Music_1000u CDs_and_Vinyl_1000u All_Beauty_1000u}"
PID_DIR="$MEMCF_EVAL_ROOT/_pids_train_shards"
mkdir -p "$PID_DIR"
read -r -a DATASETS_ARR <<< "$DATASETS_STR"

for ds in "${DATASETS_ARR[@]}"; do
  for ((shard=0; shard<NUM_USER_SHARDS; shard++)); do
    log_dir="$MEMCF_EVAL_ROOT/$ds/logs"
    mkdir -p "$log_dir"
    export USER_SHARD_ID="$shard"
    nohup bash "$ROOT/scripts/run_ablation_one_dataset.sh" "$ds" "$VARIANT_FOR_TRAIN" \
      > "$log_dir/${VARIANT_FOR_TRAIN}_train_shard${shard}of${NUM_USER_SHARDS}.nohup.log" 2>&1 &
    pid=$!
    echo "$pid" > "$PID_DIR/${ds}_${VARIANT_FOR_TRAIN}_shard${shard}of${NUM_USER_SHARDS}.pid"
    echo "STARTED train $ds shard=$shard/$NUM_USER_SHARDS pid=$pid"
  done
done

cat <<EOF
PID dir: $PID_DIR
Check status:
for p in $PID_DIR/*.pid; do pid=\$(cat "\$p"); ps -p "\$pid" >/dev/null && echo RUNNING \$(basename "\$p") "\$pid" || echo DONE \$(basename "\$p") "\$pid"; done
EOF
