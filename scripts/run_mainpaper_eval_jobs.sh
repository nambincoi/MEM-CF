#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export MEMCF_EVAL_ROOT="${MEMCF_EVAL_ROOT:-$ROOT/evaluation_results_memcf_mainpaper}"
export MEMCF_MEMORY_ROOT="${MEMCF_MEMORY_ROOT:-$ROOT/agent_memory_memcf_mainpaper}"
export RANKING_PROMPT_STYLE="${RANKING_PROMPT_STYLE:-compact_score}"
export N_USERS="${N_USERS:-100}"
export PHASE=eval_only
export LOAD_SAVED_MEMORY=1
export RUN_NAME_SUFFIX="${RUN_NAME_SUFFIX:-mainpaper_eval_once}"

DATASETS_STR="${DATASETS:-Video_Game Digital_Music_1000u All_Beauty_1000u CDs_and_Vinyl_1000u}"
# Main-paper set: comparable to MemRec-level module ablations.
# A6_random_memory and A2/A3 path-only variants are intentionally kept for appendix runs.
VARIANTS_STR="${VARIANTS:-A0_no_memory A8_profile_only A1_same_user_only A4_full_graph A5_full_graph_noharm A7_shuffled_memory}"
PID_DIR="$MEMCF_EVAL_ROOT/_pids_eval_mainpaper"
mkdir -p "$PID_DIR"
read -r -a DATASETS_ARR <<< "$DATASETS_STR"
read -r -a VARIANTS_ARR <<< "$VARIANTS_STR"

for ds in "${DATASETS_ARR[@]}"; do
  memory_file="$MEMCF_MEMORY_ROOT/$ds/mainpaper_nuser${N_USERS}.memory.json"
  if [ ! -f "$memory_file" ]; then
    echo "[ERROR] missing memory file for $ds: $memory_file" >&2
    echo "Run scripts/run_mainpaper_train_memory_jobs.sh first, or set MEMCF_MEMORY_ROOT to an existing memory root." >&2
    exit 1
  fi

  for variant in "${VARIANTS_ARR[@]}"; do
    log_dir="$MEMCF_EVAL_ROOT/$ds/logs"
    mkdir -p "$log_dir"

    # No-memory does not need MEMORY_FILE, but passing it is harmless only for
    # memory variants. Keep it clean to avoid ambiguity in logs.
    if [ "$variant" = "A0_no_memory" ]; then
      nohup bash "$ROOT/scripts/run_ablation_one_dataset.sh" "$ds" "$variant" \
        > "$log_dir/${variant}_mainpaper_eval_nuser${N_USERS}.nohup.log" 2>&1 &
    else
      MEMORY_FILE="$memory_file" \
        nohup bash "$ROOT/scripts/run_ablation_one_dataset.sh" "$ds" "$variant" \
        > "$log_dir/${variant}_mainpaper_eval_nuser${N_USERS}.nohup.log" 2>&1 &
    fi
    pid=$!
    echo "$pid" > "$PID_DIR/${ds}_${variant}_mainpaper_eval.pid"
    echo "STARTED eval $ds $variant pid=$pid memory=$memory_file"
  done
done

cat <<EOF
PID dir: $PID_DIR
Check status:
for p in $PID_DIR/*.pid; do pid=\$(cat "\$p"); ps -p "\$pid" >/dev/null && echo RUNNING \$(basename "\$p") "\$pid" || echo DONE \$(basename "\$p") "\$pid"; done
EOF
