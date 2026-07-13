#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export MEMCF_EVAL_ROOT="${MEMCF_EVAL_ROOT:-$ROOT/evaluation_results_memcf_cluster_eval}"
export MEMCF_MEMORY_ROOT="${MEMCF_MEMORY_ROOT:-$ROOT/agent_memory_memcf_mainpaper}"
export RANKING_PROMPT_STYLE="${RANKING_PROMPT_STYLE:-compact_score}"
export N_USERS="${N_USERS:-100}"
export PHASE=eval_only
export LOAD_SAVED_MEMORY=1
export RUN_NAME_SUFFIX="${RUN_NAME_SUFFIX:-cluster_eval}"

DATASETS_STR="${DATASETS:-Video_Game Digital_Music_1000u All_Beauty_1000u CDs_and_Vinyl_1000u}"
VARIANTS_STR="${VARIANTS:-A4_full_graph A14_hybrid_cluster A15_hybrid_cluster_strict A16_cluster_user_fixed A17_cluster_full_fixed A11_random_cluster A12_shuffled_cluster}"
PID_DIR="$MEMCF_EVAL_ROOT/_pids_cluster_eval"
mkdir -p "$PID_DIR"
read -r -a DATASETS_ARR <<< "$DATASETS_STR"
read -r -a VARIANTS_ARR <<< "$VARIANTS_STR"

for ds in "${DATASETS_ARR[@]}"; do
  memory_file="$MEMCF_MEMORY_ROOT/$ds/mainpaper_nuser${N_USERS}.memory.json"
  if [ ! -f "$memory_file" ]; then
    echo "[ERROR] missing memory file for $ds: $memory_file" >&2
    echo "Set MEMCF_MEMORY_ROOT to a root containing mainpaper_nuser${N_USERS}.memory.json, or train memory first." >&2
    exit 1
  fi

  for variant in "${VARIANTS_ARR[@]}"; do
    log_dir="$MEMCF_EVAL_ROOT/$ds/logs"
    mkdir -p "$log_dir"
    MEMORY_FILE="$memory_file" \
      nohup bash "$ROOT/scripts/run_ablation_one_dataset.sh" "$ds" "$variant" \
      > "$log_dir/${variant}_cluster_eval_nuser${N_USERS}.nohup.log" 2>&1 &
    pid=$!
    echo "$pid" > "$PID_DIR/${ds}_${variant}_cluster_eval.pid"
    echo "STARTED cluster-eval $ds $variant pid=$pid memory=$memory_file"
  done
done

cat <<MSG
PID dir: $PID_DIR
Check status:
for p in $PID_DIR/*.pid; do pid=\$(cat "\$p"); ps -p "\$pid" >/dev/null && echo RUNNING \$(basename "\$p") "\$pid" || echo DONE \$(basename "\$p") "\$pid"; done
MSG
