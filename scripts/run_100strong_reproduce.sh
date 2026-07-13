#!/usr/bin/env bash
set -euo pipefail

# Reproduce the best 100-user MEMCF behavior from:
#   evaluation_results_memcf_v2_strong_memrecold
# while keeping the current codebase's improved tracing/logging infrastructure.
#
# Important: this script intentionally DOES NOT enable strict candidate gates,
# random/shuffled controls, profile-only controls, or N=10 protocol.

ROOT=${MEMCF_ROOT:-/home/ubuntu/24nam.nh/video_games_data/MEMCF}
PY=${MEMCF_PYTHON:-/home/ubuntu/24nam.nh/venvs/agentrec_fail/bin/python3}

export MEMCF_DATA_ROOT=${MEMCF_DATA_ROOT:-/home/ubuntu/24nam.nh/video_games_data/runtime_data}
export MEMCF_EVAL_ROOT=${MEMCF_EVAL_ROOT:-/home/ubuntu/24nam.nh/video_games_data/evaluation_results_memcf_100strong_reproduce}
export MEMCF_MEMORY_ROOT=${MEMCF_MEMORY_ROOT:-/home/ubuntu/24nam.nh/video_games_data/agent_memory_memcf_100strong_reproduce}

export chat_api_base=${chat_api_base:-http://127.0.0.1:8000/v1}
export api_base=${api_base:-http://127.0.0.1:8000/v1}
export chat_model_name=${chat_model_name:-gpt-3.5-turbo-16k-0613}
export embedding_model_name=${embedding_model_name:-/home/ubuntu/qwen_embed}
export PYTHONUNBUFFERED=1

DATASETS=${DATASETS:-"Video_Game Digital_Music_1000u CDs_and_Vinyl_1000u All_Beauty_1000u"}
N_USERS=${N_USERS:-100}
MAX_POSITIVE_INTERACTIONS=${MAX_POSITIVE_INTERACTIONS:-5}
MAX_NEGATIVE_CANDIDATES=${MAX_NEGATIVE_CANDIDATES:-19}
MAX_ITERATIONS=${MAX_ITERATIONS:-1}
NEIGHBOR_K=${NEIGHBOR_K:-10}
MIN_EVIDENCE_TERMS=${MIN_EVIDENCE_TERMS:-1}
MAX_FAILURE_LESSONS_PER_USER=${MAX_FAILURE_LESSONS_PER_USER:-3}
MIN_LESSON_CONFIDENCE=${MIN_LESSON_CONFIDENCE:-0.25}
MAX_LESSON_RISK=${MAX_LESSON_RISK:-0.85}
RANKING_PROMPT_STYLE=${RANKING_PROMPT_STYLE:-compact_score}
CANDIDATE_NEGATIVE_MODE=${CANDIDATE_NEGATIVE_MODE:-candidate_hard}
export MEMCF_INCLUDE_DESCRIPTIONS_IN_PROMPT=${MEMCF_INCLUDE_DESCRIPTIONS_IN_PROMPT:-0}

PID_DIR="$MEMCF_EVAL_ROOT/_pids_100strong"
mkdir -p "$PID_DIR"

run_one() {
  local ds="$1"
  local tag="$2"
  shift 2

  local out_dir="$MEMCF_EVAL_ROOT/$ds"
  local trace_dir="$out_dir/traces/$tag"
  local log_dir="$out_dir/logs"
  mkdir -p "$trace_dir" "$log_dir" "$MEMCF_MEMORY_ROOT/$ds"

  local log="$log_dir/${tag}_nuser${N_USERS}.log"
  echo "[START] $ds $tag log=$log"

  (
    cd "$ROOT"
    "$PY" -u src/memcf/experiment.py \
      --data_name "$ds" \
      --number_of_users "$N_USERS" \
      --max_iterations "$MAX_ITERATIONS" \
      --max_positive_interactions "$MAX_POSITIVE_INTERACTIONS" \
      --max_negative_candidates "$MAX_NEGATIVE_CANDIDATES" \
      --candidate_negative_mode "$CANDIDATE_NEGATIVE_MODE" \
      --min_lesson_confidence "$MIN_LESSON_CONFIDENCE" \
      --max_lesson_risk "$MAX_LESSON_RISK" \
      --max_failure_lessons_per_user "$MAX_FAILURE_LESSONS_PER_USER" \
      --ranking_prompt_style "$RANKING_PROMPT_STYLE" \
      --neighbor_k "$NEIGHBOR_K" \
      --min_evidence_terms "$MIN_EVIDENCE_TERMS" \
      --trace_dir "$trace_dir" \
      "$@" \
      2>&1 | tee "$log"
  ) &

  echo $! > "$PID_DIR/${ds}_${tag}.pid"
}

for ds in $DATASETS; do
  # Same base reranker, no memory. This is the clean control.
  run_one "$ds" A0_no_memory \
    --no_use_memory \
    --run_name_suffix 100strong_A0_nomemory

  # 100-strong memory setting: graph k=3, no-harm arbitration on, no strict gate.
  run_one "$ds" A1_full_gk3_noharm \
    --use_memory \
    --graph_memory_k 3 \
    --graph_retrieval_scope full \
    --no_harm_arbitration \
    --run_name_suffix 100strong_A1_gk3_noharm

  # 100-strong alternative that was best for All_Beauty.
  run_one "$ds" A2_full_gk5_noharm \
    --use_memory \
    --graph_memory_k 5 \
    --graph_retrieval_scope full \
    --no_harm_arbitration \
    --run_name_suffix 100strong_A2_gk5_noharm
done

cat <<MSG

Started 100-strong reproduction jobs.
PID dir: $PID_DIR

Check progress:
for p in "$PID_DIR"/*.pid; do pid=\$(cat "\$p"); ps -p "\$pid" >/dev/null && echo RUNNING "\$(basename "\$p")" "\$pid" || echo DONE "\$(basename "\$p")" "\$pid"; done

Aggregate after completion:
python3 "$ROOT/scripts/aggregate_paper_tables.py" --root "$MEMCF_EVAL_ROOT" --markdown "$MEMCF_EVAL_ROOT/paper_table.md" --csv "$MEMCF_EVAL_ROOT/paper_table.csv"
cat "$MEMCF_EVAL_ROOT/paper_table.md"
MSG
