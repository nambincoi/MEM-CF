#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export MEMCF_DATA_ROOT="${MEMCF_DATA_ROOT:-${AGENTICREC_DATA_ROOT:-$ROOT/data}}"
export MEMCF_EVAL_ROOT="${MEMCF_EVAL_ROOT:-${AGENTICREC_EVAL_ROOT:-$ROOT/evaluation_results_g_cf_100u}}"
export MEMCF_MEMORY_ROOT="${MEMCF_MEMORY_ROOT:-${AGENTICREC_MEMORY_ROOT:-$ROOT/agent_memory}}"
export AGENTICREC_DATA_ROOT="$MEMCF_DATA_ROOT"
export AGENTICREC_EVAL_ROOT="$MEMCF_EVAL_ROOT"
export AGENTICREC_MEMORY_ROOT="$MEMCF_MEMORY_ROOT"
export chat_api_base="${chat_api_base:-http://127.0.0.1:8000/v1}"
export api_base="${api_base:-$chat_api_base}"
export chat_model_name="${chat_model_name:-gpt-3.5-turbo-16k-0613}"
export PYTHONUNBUFFERED=1

export N_USERS="${N_USERS:-100}"
export PHASE=eval_only
export LOAD_SAVED_MEMORY=1
export EVAL_SPLIT="${EVAL_SPLIT:-val}"
export MAX_POSITIVE_INTERACTIONS="${MAX_POSITIVE_INTERACTIONS:-5}"
export MAX_NEGATIVE_CANDIDATES="${MAX_NEGATIVE_CANDIDATES:-19}"
export CANDIDATE_NEGATIVE_MODE="${CANDIDATE_NEGATIVE_MODE:-candidate_hard}"
export RANKING_PROMPT_STYLE="${RANKING_PROMPT_STYLE:-compact_score}"
export GRAPH_MEMORY_K="${GRAPH_MEMORY_K:-3}"
export MAX_MEMORY_FACTS="${MAX_MEMORY_FACTS:-3}"
export MAX_MEMORY_FACT_WORDS="${MAX_MEMORY_FACT_WORDS:-55}"
export MEMORY_TOKEN_BUDGET="${MEMORY_TOKEN_BUDGET:-420}"
export FAILURE_CONSTRAINT_TIE_EPSILON="${FAILURE_CONSTRAINT_TIE_EPSILON:-0.0}"
export FAILURE_CONSTRAINT_MIN_CROSS_SUPPORT="${FAILURE_CONSTRAINT_MIN_CROSS_SUPPORT:-1}"
export FAILURE_CONSTRAINT_MIN_SHARED_ITEMS="${FAILURE_CONSTRAINT_MIN_SHARED_ITEMS:-1}"
export FAILURE_CONSTRAINT_MAX_CROSS_CORRECTIONS="${FAILURE_CONSTRAINT_MAX_CROSS_CORRECTIONS:-2}"
export FAILURE_CONSTRAINT_CROSS_BUDGET="${FAILURE_CONSTRAINT_CROSS_BUDGET:-2}"
export CF_SOURCE_BUDGET="${CF_SOURCE_BUDGET:-2}"
export CF_CONTROL_SEED="${CF_CONTROL_SEED:-2027}"
export SKIP_USER_CLUSTERS=1

DATASETS="${DATASETS:-Video_Game Digital_Music_1000u CDs_and_Vinyl_1000u Industrial_and_Scientific_1000u Prime_Pantry_1000u Software_1000u}"
CONTROL_VARIANTS="${CONTROL_VARIANTS:-G1_true_neighbor G2_shuffled_graph G3_random_neighbor G4_matched_random}"
MAX_PARALLEL="${MAX_PARALLEL:-24}"
BASE_SUFFIX="${RUN_NAME_SUFFIX:-gcf_val100_seed${CF_CONTROL_SEED}}"
PID_DIR="$MEMCF_EVAL_ROOT/_pids_g_cf"
CACHE_ROOT="$MEMCF_EVAL_ROOT/_same_user_score_cache"
mkdir -p "$PID_DIR" "$CACHE_ROOT"

python3 - <<PY
import urllib.request
urllib.request.urlopen("${chat_api_base%/}/models", timeout=5).read()
print("[OK] vLLM reachable: ${chat_api_base%/}/models")
PY

pick_memory_file() {
  local ds="$1" base="$MEMCF_MEMORY_ROOT/$1" file
  for file in \
    "$base/shared_nuser1000_8shards.memory.json" \
    "$base/shared_nuser1000_1shards.memory.json" \
    "$base/mainpaper_nuser100.memory.json"; do
    if [ -s "$file" ]; then printf '%s\n' "$file"; return 0; fi
  done
  echo "[ERROR] missing memory for $ds under $base" >&2
  return 1
}

running_jobs() { jobs -pr | wc -l | tr -d ' '; }
wait_for_slot() { while [ "$(running_jobs)" -ge "$MAX_PARALLEL" ]; do sleep 3; done; }

launch() {
  local ds="$1" variant="$2" memory_file="$3"
  local suffix="${BASE_SUFFIX}_${variant}"
  local log_dir="$MEMCF_EVAL_ROOT/$ds/logs"
  mkdir -p "$log_dir" "$CACHE_ROOT/$ds"
  MEMORY_FILE="$memory_file" \
  RANKING_SCORE_CACHE_DIR="$CACHE_ROOT/$ds" \
  RUN_NAME_SUFFIX="$suffix" \
    nohup bash scripts/run_ablation_one_dataset.sh "$ds" "$variant" \
      > "$log_dir/${variant}_${suffix}.nohup.log" 2>&1 &
  echo "$!" > "$PID_DIR/${ds}_${variant}.pid"
  echo "STARTED $ds $variant pid=$!"
}

echo "STAGE 1/2: create identical same-user prompt score caches"
for ds in $DATASETS; do
  memory_file="$(pick_memory_file "$ds")"
  launch "$ds" G0_same_only "$memory_file"
done
wait

echo "STAGE 2/2: replay true/shuffled/random/matched cross-user controls"
for ds in $DATASETS; do
  memory_file="$(pick_memory_file "$ds")"
  for variant in $CONTROL_VARIANTS; do
    wait_for_slot
    launch "$ds" "$variant" "$memory_file"
    sleep 0.1
  done
done
wait

summary_count=$(find "$MEMCF_EVAL_ROOT" -name '*.summary.json' -type f | wc -l | tr -d ' ')
echo "G-family runs completed: summaries=$summary_count expected=30"
echo "Result root: $MEMCF_EVAL_ROOT"
