#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MEMCF_EVAL_ROOT="${MEMCF_EVAL_ROOT:-$ROOT/evaluation_results_memcf_strong}"
export MEMCF_MEMORY_ROOT="${MEMCF_MEMORY_ROOT:-$ROOT/agent_memory_memcf_strong}"
export RANKING_PROMPT_STYLE="${RANKING_PROMPT_STYLE:-compact_score}"
PID_DIR="$MEMCF_EVAL_ROOT/_pids"
mkdir -p "$PID_DIR"

DATASETS=(
  Video_Game
  Digital_Music_1000u
  CDs_and_Vinyl_1000u
  All_Beauty_1000u
)

VARIANTS=(
  A0_no_memory
  A1_safe_graph_no_noharm
  A2_safe_graph_noharm
  A3_safe_graph_k5_noharm
)

for ds in "${DATASETS[@]}"; do
  for variant in "${VARIANTS[@]}"; do
    log_dir="$MEMCF_EVAL_ROOT/$ds/logs"
    mkdir -p "$log_dir"
    nohup bash "$ROOT/scripts/run_ablation_one_dataset.sh" "$ds" "$variant" \
      > "$log_dir/${variant}.nohup.log" 2>&1 &
    pid=$!
    echo "$pid" > "$PID_DIR/${ds}_${variant}.pid"
    echo "STARTED $ds $variant pid=$pid"
  done
done

echo "PID files: $PID_DIR"
echo "Check status with:"
echo "for p in $PID_DIR/*.pid; do pid=\$(cat \"\$p\"); ps -p \"\$pid\" >/dev/null && echo RUNNING \$(basename \"\$p\") \"\$pid\" || echo DONE \$(basename \"\$p\") \"\$pid\"; done"
echo "Logs:"
echo "find $MEMCF_EVAL_ROOT -path '*/logs/*.log' -type f | sort"
