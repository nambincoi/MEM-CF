#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MEMCF_EVAL_ROOT="${MEMCF_EVAL_ROOT:-$ROOT/evaluation_results_memcf}"
export MEMCF_MEMORY_ROOT="${MEMCF_MEMORY_ROOT:-$ROOT/agent_memory_memcf}"
export RANKING_PROMPT_STYLE="${RANKING_PROMPT_STYLE:-compact_score}"

DATASETS_STR="${DATASETS:-Video_Game Digital_Music_1000u All_Beauty_1000u CDs_and_Vinyl_1000u}"
VARIANTS_STR="${VARIANTS:-A0_no_memory A1_same_user_only A2_candidate_item_only A3_neighbor_user_only A4_full_graph A5_full_graph_noharm A6_random_memory A7_shuffled_memory}"
PID_DIR="$MEMCF_EVAL_ROOT/_pids"
mkdir -p "$PID_DIR"

read -r -a DATASETS_ARR <<< "$DATASETS_STR"
read -r -a VARIANTS_ARR <<< "$VARIANTS_STR"

for ds in "${DATASETS_ARR[@]}"; do
  for variant in "${VARIANTS_ARR[@]}"; do
    log_dir="$MEMCF_EVAL_ROOT/$ds/logs"
    mkdir -p "$log_dir"
    nohup bash "$ROOT/scripts/run_ablation_one_dataset.sh" "$ds" "$variant" \
      > "$log_dir/${variant}.nohup.log" 2>&1 &
    pid=$!
    echo "$pid" > "$PID_DIR/${ds}_${variant}.pid"
    echo "STARTED $ds $variant pid=$pid"
  done
done

n_jobs=$((${#DATASETS_ARR[@]} * ${#VARIANTS_ARR[@]}))
echo "Launched $n_jobs jobs."
echo "PID files: $PID_DIR"
echo "Check status:"
echo "for p in $PID_DIR/*.pid; do pid=\$(cat \"\$p\"); ps -p \"\$pid\" >/dev/null && echo RUNNING \$(basename \"\$p\") \"\$pid\" || echo DONE \$(basename \"\$p\") \"\$pid\"; done"
echo "Logs: find $MEMCF_EVAL_ROOT -path '*/logs/*.log' -type f | sort"
