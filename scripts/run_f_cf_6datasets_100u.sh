#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export MEMCF_DATA_ROOT="${MEMCF_DATA_ROOT:-${AGENTICREC_DATA_ROOT:-$ROOT/data}}"
export MEMCF_EVAL_ROOT="${MEMCF_EVAL_ROOT:-${AGENTICREC_EVAL_ROOT:-$ROOT/evaluation_results_f_cf_100u}}"
export MEMCF_MEMORY_ROOT="${MEMCF_MEMORY_ROOT:-${AGENTICREC_MEMORY_ROOT:-$ROOT/agent_memory}}"
export AGENTICREC_DATA_ROOT="$MEMCF_DATA_ROOT"
export AGENTICREC_EVAL_ROOT="$MEMCF_EVAL_ROOT"
export AGENTICREC_MEMORY_ROOT="$MEMCF_MEMORY_ROOT"
export chat_api_base="${chat_api_base:-http://127.0.0.1:8000/v1}"
export api_base="${api_base:-$chat_api_base}"
export chat_model_name="${chat_model_name:-gpt-3.5-turbo-16k-0613}"
export PYTHONUNBUFFERED=1

export N_USERS="${N_USERS:-100}"
export PHASE="eval_only"
export LOAD_SAVED_MEMORY=1
export MAX_POSITIVE_INTERACTIONS="${MAX_POSITIVE_INTERACTIONS:-5}"
export MAX_NEGATIVE_CANDIDATES="${MAX_NEGATIVE_CANDIDATES:-19}"
export MAX_ITERATIONS="${MAX_ITERATIONS:-1}"
export CANDIDATE_NEGATIVE_MODE="${CANDIDATE_NEGATIVE_MODE:-candidate_hard}"
export RANKING_PROMPT_STYLE="${RANKING_PROMPT_STYLE:-compact_score}"
export FAILURE_CONSTRAINT_TIE_EPSILON="${FAILURE_CONSTRAINT_TIE_EPSILON:-0.0}"
export FAILURE_CONSTRAINT_MIN_CROSS_SUPPORT="${FAILURE_CONSTRAINT_MIN_CROSS_SUPPORT:-2}"
export FAILURE_CONSTRAINT_MIN_SHARED_ITEMS="${FAILURE_CONSTRAINT_MIN_SHARED_ITEMS:-1}"
export FAILURE_CONSTRAINT_MAX_CROSS_CORRECTIONS="${FAILURE_CONSTRAINT_MAX_CROSS_CORRECTIONS:-3}"
export FAILURE_CONSTRAINT_SAME_BUDGET="${FAILURE_CONSTRAINT_SAME_BUDGET:-32}"
export FAILURE_CONSTRAINT_CROSS_BUDGET="${FAILURE_CONSTRAINT_CROSS_BUDGET:-128}"
export SKIP_USER_CLUSTERS=1

DATASETS="${DATASETS:-Video_Game Digital_Music_1000u CDs_and_Vinyl_1000u Industrial_and_Scientific_1000u Prime_Pantry_1000u Software_1000u}"
VARIANTS="${VARIANTS:-F1_same_user_exact F2_shared_item_cross_only F3_same_plus_shared_cf F4_shuffled_neighbors F5_random_neighbors F6_cross_polarity_swapped F7_popularity_control}"
MAX_PARALLEL="${MAX_PARALLEL:-32}"
BASE_SUFFIX="${RUN_NAME_SUFFIX:-fcf100}"
PID_DIR="$MEMCF_EVAL_ROOT/_pids_f_cf_100u"
CACHE_ROOT="$MEMCF_EVAL_ROOT/_clean_score_cache"
mkdir -p "$PID_DIR" "$CACHE_ROOT"

pick_memory_file() {
  local ds="$1"
  local base="$MEMCF_MEMORY_ROOT/$ds"
  local candidates=(
    "$base/shared_nuser1000_8shards.memory.json"
    "$base/shared_nuser1000_1shards.memory.json"
    "$base/mainpaper_nuser100.memory.json"
    "$base/memcf_v2_graph_nuser1000_iter1_gk3_nk10_ev1.memory.json"
    "$base/memcf_v2_graph_nuser100_iter1_gk3_nk10_ev1.memory.json"
  )
  local file
  for file in "${candidates[@]}"; do
    if [ -s "$file" ]; then
      printf '%s\n' "$file"
      return 0
    fi
  done
  echo "[ERROR] missing memory file for $ds under $base" >&2
  return 1
}

running_jobs() {
  jobs -pr | wc -l | tr -d ' '
}

wait_for_slot() {
  while [ "$(running_jobs)" -ge "$MAX_PARALLEL" ]; do
    sleep 3
  done
}

launch() {
  local ds="$1"
  local variant="$2"
  local memory_file="$3"
  local suffix="${BASE_SUFFIX}_${variant}"
  local pid_file="$PID_DIR/${ds}_${variant}.pid"
  local log_dir="$MEMCF_EVAL_ROOT/$ds/logs"
  local nohup_log="$log_dir/${variant}_${suffix}.nohup.log"
  mkdir -p "$log_dir" "$CACHE_ROOT/$ds"
  MEMORY_FILE="$memory_file" \
  RANKING_SCORE_CACHE_DIR="$CACHE_ROOT/$ds" \
  RUN_NAME_SUFFIX="$suffix" \
    nohup bash scripts/run_ablation_one_dataset.sh "$ds" "$variant" \
      > "$nohup_log" 2>&1 &
  echo "$!" > "$pid_file"
  echo "STARTED $ds $variant pid=$(cat "$pid_file") log=$nohup_log"
}

# Stage 1 creates one clean Qwen score response per user. All F variants then
# replay their evidence policy over exactly these cached scores.
echo "STAGE 1/2: clean profile scoring once per dataset"
for ds in $DATASETS; do
  memory_file="$(pick_memory_file "$ds")"
  launch "$ds" F0_profile_base "$memory_file"
done
wait

echo "STAGE 2/2: paired offline CF/control replay (max_parallel=$MAX_PARALLEL)"
for ds in $DATASETS; do
  memory_file="$(pick_memory_file "$ds")"
  for variant in $VARIANTS; do
    wait_for_slot
    launch "$ds" "$variant" "$memory_file"
    sleep 0.1
  done
done
wait

echo "All F-family jobs completed."
echo "Result root: $MEMCF_EVAL_ROOT"
echo "PID dir: $PID_DIR"
echo "Clean score cache: $CACHE_ROOT"
