#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export MEMCF_DATA_ROOT="${MEMCF_DATA_ROOT:-${AGENTICREC_DATA_ROOT:-$ROOT/data}}"
export MEMCF_EVAL_ROOT="${MEMCF_EVAL_ROOT:-${AGENTICREC_EVAL_ROOT:-$ROOT/evaluation_results_pairwise_cf_100u}}"
export MEMCF_MEMORY_ROOT="${MEMCF_MEMORY_ROOT:-${AGENTICREC_MEMORY_ROOT:-$ROOT/agent_memory}}"
export AGENTICREC_DATA_ROOT="$MEMCF_DATA_ROOT"
export AGENTICREC_EVAL_ROOT="$MEMCF_EVAL_ROOT"
export AGENTICREC_MEMORY_ROOT="$MEMCF_MEMORY_ROOT"
export chat_api_base="${chat_api_base:-http://127.0.0.1:8000/v1}"
export api_base="${api_base:-$chat_api_base}"
export chat_model_name="${chat_model_name:-gpt-3.5-turbo-16k-0613}"
export PYTHONUNBUFFERED=1

export N_USERS="${N_USERS:-100}"
export PHASE="${PHASE:-eval_only}"
export LOAD_SAVED_MEMORY="${LOAD_SAVED_MEMORY:-1}"
export MAX_POSITIVE_INTERACTIONS="${MAX_POSITIVE_INTERACTIONS:-5}"
export MAX_NEGATIVE_CANDIDATES="${MAX_NEGATIVE_CANDIDATES:-19}"
export MAX_ITERATIONS="${MAX_ITERATIONS:-1}"
export GRAPH_MEMORY_K="${GRAPH_MEMORY_K:-3}"
export NEIGHBOR_K="${NEIGHBOR_K:-10}"
export MIN_EVIDENCE_TERMS="${MIN_EVIDENCE_TERMS:-1}"
export MAX_MEMORY_FACTS="${MAX_MEMORY_FACTS:-3}"
export MAX_MEMORY_FACT_WORDS="${MAX_MEMORY_FACT_WORDS:-55}"
export MEMORY_TOKEN_BUDGET="${MEMORY_TOKEN_BUDGET:-420}"
export CANDIDATE_NEGATIVE_MODE="${CANDIDATE_NEGATIVE_MODE:-candidate_hard}"
export RANKING_PROMPT_STYLE="${RANKING_PROMPT_STYLE:-compact_score}"
export RUN_NAME_SUFFIX="${RUN_NAME_SUFFIX:-pairwise_cf_100u}"
BASE_RUN_NAME_SUFFIX="$RUN_NAME_SUFFIX"
export PAIRWISE_CF_ALPHA="${PAIRWISE_CF_ALPHA:-0.04}"
export PAIRWISE_CF_BETA="${PAIRWISE_CF_BETA:-0.04}"

DATASETS="${DATASETS:-Video_Game Digital_Music_1000u CDs_and_Vinyl_1000u Industrial_and_Scientific_1000u Prime_Pantry_1000u Software_1000u}"
VARIANTS="${VARIANTS:-A0_no_memory A1_same_user_only A4_full_graph C1_same_user_pairwise C2_full_graph_pairwise C3_full_graph_pairwise_prompt C4_full_graph_pairwise_strict C5_cross_user_pairwise}"
MAX_PARALLEL="${MAX_PARALLEL:-24}"
PID_DIR="$MEMCF_EVAL_ROOT/_pids_pairwise_cf_100u"
mkdir -p "$PID_DIR"

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
  local f
  for f in "${candidates[@]}"; do
    if [ -s "$f" ]; then
      printf '%s\n' "$f"
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
    sleep 10
  done
}

for DS in $DATASETS; do
  MEMORY_FILE="$(pick_memory_file "$DS")"
  export MEMORY_FILE
  mkdir -p "$MEMCF_EVAL_ROOT/$DS/logs"
  echo "================================================================================"
  echo "DATASET $DS"
  echo "MEMORY_FILE=$MEMORY_FILE"
  echo "================================================================================"
  for VAR in $VARIANTS; do
    wait_for_slot
    JOB_SUFFIX="${BASE_RUN_NAME_SUFFIX}_${VAR}"
    PID_FILE="$PID_DIR/${DS}_${VAR}.pid"
    NOHUP_LOG="$MEMCF_EVAL_ROOT/$DS/logs/${VAR}_${JOB_SUFFIX}.nohup.log"
    RUN_NAME_SUFFIX="$JOB_SUFFIX" nohup bash scripts/run_ablation_one_dataset.sh "$DS" "$VAR" > "$NOHUP_LOG" 2>&1 &
    echo "$!" > "$PID_FILE"
    echo "STARTED $DS $VAR suffix=$JOB_SUFFIX pid=$(cat "$PID_FILE") log=$NOHUP_LOG"
    sleep 0.2
  done
done

wait

echo "All pairwise-CF 100u jobs completed."
echo "PID dir: $PID_DIR"
echo "Result root: $MEMCF_EVAL_ROOT"
