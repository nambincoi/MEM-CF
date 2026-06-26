#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export MEMCF_EVAL_ROOT="${MEMCF_EVAL_ROOT:-$ROOT/evaluation_results_memcf_mainpaper}"
export MEMCF_MEMORY_ROOT="${MEMCF_MEMORY_ROOT:-$ROOT/agent_memory_memcf_mainpaper}"
export RANKING_PROMPT_STYLE="${RANKING_PROMPT_STYLE:-compact_score}"
export N_USERS="${N_USERS:-100}"
export PHASE=train_only
export VARIANT_FOR_TRAIN="${VARIANT_FOR_TRAIN:-A4_full_graph}"
export RUN_NAME_SUFFIX="${RUN_NAME_SUFFIX:-mainpaper_train_once}"

DATASETS_STR="${DATASETS:-Video_Game Digital_Music_1000u All_Beauty_1000u CDs_and_Vinyl_1000u}"
PID_DIR="$MEMCF_EVAL_ROOT/_pids_train_once"
mkdir -p "$PID_DIR"
read -r -a DATASETS_ARR <<< "$DATASETS_STR"

for ds in "${DATASETS_ARR[@]}"; do
  memory_file="$MEMCF_MEMORY_ROOT/$ds/mainpaper_nuser${N_USERS}.memory.json"
  log_dir="$MEMCF_EVAL_ROOT/$ds/logs"
  mkdir -p "$log_dir" "$(dirname "$memory_file")"

  if [ -f "$memory_file" ] && [ "${FORCE_RETRAIN:-0}" != "1" ]; then
    echo "SKIP train $ds: existing memory file $memory_file"
    continue
  fi

  MEMORY_FILE="$memory_file" \
    nohup bash "$ROOT/scripts/run_ablation_one_dataset.sh" "$ds" "$VARIANT_FOR_TRAIN" \
    > "$log_dir/${VARIANT_FOR_TRAIN}_train_once_nuser${N_USERS}.nohup.log" 2>&1 &
  pid=$!
  echo "$pid" > "$PID_DIR/${ds}_${VARIANT_FOR_TRAIN}_train_once.pid"
  echo "STARTED train $ds variant=$VARIANT_FOR_TRAIN pid=$pid memory=$memory_file"
done

cat <<EOF
PID dir: $PID_DIR
Check status:
for p in $PID_DIR/*.pid; do pid=\$(cat "\$p"); ps -p "\$pid" >/dev/null && echo RUNNING \$(basename "\$p") "\$pid" || echo DONE \$(basename "\$p") "\$pid"; done
EOF
