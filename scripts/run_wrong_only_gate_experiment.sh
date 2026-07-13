#!/usr/bin/env bash
set -euo pipefail

# Optional MEMCF experiment: reject memory facts whose only direct current
# candidate match is the past wrong item. This script is intentionally separate
# from 100-strong reproduction scripts so default behavior stays unchanged.

DATASETS=${DATASETS:-"Video_Game Digital_Music_1000u All_Beauty_1000u CDs_and_Vinyl_1000u"}
N_USERS=${N_USERS:-100}
MAX_POSITIVE_INTERACTIONS=${MAX_POSITIVE_INTERACTIONS:-5}
MAX_NEGATIVE_CANDIDATES=${MAX_NEGATIVE_CANDIDATES:-19}
MAX_ITERATIONS=${MAX_ITERATIONS:-1}
GRAPH_MEMORY_K=${GRAPH_MEMORY_K:-3}
NEIGHBOR_K=${NEIGHBOR_K:-10}
MIN_EVIDENCE_TERMS=${MIN_EVIDENCE_TERMS:-1}
RANKING_PROMPT_STYLE=${RANKING_PROMPT_STYLE:-compact_score}

MEMCF_EVAL_ROOT=${MEMCF_EVAL_ROOT:-/home/ubuntu/24nam.nh/video_games_data/evaluation_results_memcf_wrong_only_gate}
MEMCF_MEMORY_ROOT=${MEMCF_MEMORY_ROOT:-/home/ubuntu/24nam.nh/video_games_data/agent_memory_memcf_wrong_only_gate}
export MEMCF_EVAL_ROOT MEMCF_MEMORY_ROOT

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN=${PYTHON_BIN:-python3}

mkdir -p "$MEMCF_EVAL_ROOT/_pids"

for DS in $DATASETS; do
  mkdir -p "$MEMCF_EVAL_ROOT/$DS/logs"
  LOG="$MEMCF_EVAL_ROOT/$DS/logs/full_gk${GRAPH_MEMORY_K}_reject_wrong_only_nuser${N_USERS}.log"
  (
    cd "$PROJECT_ROOT"
    "$PYTHON_BIN" -u src/memcf/experiment.py \
      --data_name "$DS" \
      --use_memory \
      --number_of_users "$N_USERS" \
      --max_positive_interactions "$MAX_POSITIVE_INTERACTIONS" \
      --max_negative_candidates "$MAX_NEGATIVE_CANDIDATES" \
      --max_iterations "$MAX_ITERATIONS" \
      --graph_memory_k "$GRAPH_MEMORY_K" \
      --neighbor_k "$NEIGHBOR_K" \
      --min_evidence_terms "$MIN_EVIDENCE_TERMS" \
      --candidate_negative_mode candidate_hard \
      --ranking_prompt_style "$RANKING_PROMPT_STYLE" \
      --graph_retrieval_scope full \
      --no_harm_arbitration \
      --reject_wrong_only_memory \
      --run_name_suffix "reject_wrong_only_gk${GRAPH_MEMORY_K}" \
      2>&1 | tee "$LOG"
  ) &
  pid=$!
  echo "$pid" > "$MEMCF_EVAL_ROOT/_pids/${DS}_reject_wrong_only_gk${GRAPH_MEMORY_K}.pid"
  echo "STARTED $DS reject_wrong_only_gk${GRAPH_MEMORY_K} pid=$pid log=$LOG"
done

echo
echo "Check progress:"
echo "for p in $MEMCF_EVAL_ROOT/_pids/*.pid; do pid=\\$(cat \"\\$p\"); ps -p \"\\$pid\" >/dev/null && echo RUNNING \"\\$(basename \"\\$p\")\" \"\\$pid\" || echo DONE \"\\$(basename \"\\$p\")\" \"\\$pid\"; done"
