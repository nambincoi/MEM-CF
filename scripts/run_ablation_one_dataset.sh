#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${1:?Usage: run_ablation_one_dataset.sh <dataset> <variant>}"
VARIANT="${2:?Usage: run_ablation_one_dataset.sh <dataset> <variant>}"

export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export MEMCF_DATA_ROOT="${MEMCF_DATA_ROOT:-${AGENTICREC_DATA_ROOT:-$ROOT/data}}"
export MEMCF_EVAL_ROOT="${MEMCF_EVAL_ROOT:-${AGENTICREC_EVAL_ROOT:-$ROOT/evaluation_results}}"
export MEMCF_MEMORY_ROOT="${MEMCF_MEMORY_ROOT:-${AGENTICREC_MEMORY_ROOT:-$ROOT/agent_memory}}"
export AGENTICREC_DATA_ROOT="$MEMCF_DATA_ROOT"
export AGENTICREC_EVAL_ROOT="$MEMCF_EVAL_ROOT"
export AGENTICREC_MEMORY_ROOT="$MEMCF_MEMORY_ROOT"
export chat_api_base="${chat_api_base:-http://127.0.0.1:8000/v1}"
export api_base="${api_base:-$chat_api_base}"
export chat_model_name="${chat_model_name:-gpt-3.5-turbo-16k-0613}"
export PYTHONUNBUFFERED=1

N_USERS="${N_USERS:-100}"
MAX_POS="${MAX_POSITIVE_INTERACTIONS:-5}"
MAX_NEG="${MAX_NEGATIVE_CANDIDATES:-19}"
MAX_ITER="${MAX_ITERATIONS:-1}"
NEIGHBOR_K="${NEIGHBOR_K:-10}"
MIN_EVIDENCE="${MIN_EVIDENCE_TERMS:-1}"
RANKING_PROMPT_STYLE="${RANKING_PROMPT_STYLE:-compact_score}"
MAX_MEMORY_FACTS="${MAX_MEMORY_FACTS:-3}"
MAX_MEMORY_FACT_WORDS="${MAX_MEMORY_FACT_WORDS:-55}"
MEMORY_TOKEN_BUDGET="${MEMORY_TOKEN_BUDGET:-420}"
GRAPH_MEMORY_K="${GRAPH_MEMORY_K:-3}"
CANDIDATE_NEGATIVE_MODE="${CANDIDATE_NEGATIVE_MODE:-candidate_hard}"
PHASE="${PHASE:-all}"

mkdir -p "$MEMCF_EVAL_ROOT/$DATASET/logs"
LOG_TAG="${VARIANT}_nuser${N_USERS}"
if [ -n "${NUM_USER_SHARDS:-}" ] && [ "${NUM_USER_SHARDS:-1}" != "1" ]; then
  LOG_TAG="${LOG_TAG}_shard${USER_SHARD_ID:-0}of${NUM_USER_SHARDS}"
fi
if [ -n "${RUN_NAME_SUFFIX:-}" ]; then
  SAFE_SUFFIX=$(printf "%s" "$RUN_NAME_SUFFIX" | tr -c 'A-Za-z0-9_.-' '_')
  LOG_TAG="${LOG_TAG}_${SAFE_SUFFIX}"
fi
LOG="$MEMCF_EVAL_ROOT/$DATASET/logs/${LOG_TAG}.log"

COMMON_ARGS=(
  --data_name "$DATASET"
  --number_of_users "$N_USERS"
  --max_positive_interactions "$MAX_POS"
  --max_negative_candidates "$MAX_NEG"
  --ranking_prompt_style "$RANKING_PROMPT_STYLE"
  --candidate_negative_mode "$CANDIDATE_NEGATIVE_MODE"
  --phase "$PHASE"
)

MEMORY_ARGS=(
  --use_memory
  --max_iterations "$MAX_ITER"
  --graph_memory_k "$GRAPH_MEMORY_K"
  --neighbor_k "$NEIGHBOR_K"
  --min_evidence_terms "$MIN_EVIDENCE"
  --max_memory_facts "$MAX_MEMORY_FACTS"
  --max_memory_fact_words "$MAX_MEMORY_FACT_WORDS"
  --memory_token_budget "$MEMORY_TOKEN_BUDGET"
)

EXTRA_ARGS=()
[ -n "${MEMORY_FILE:-}" ] && EXTRA_ARGS+=(--memory_file "$MEMORY_FILE")
[ -n "${ARTIFACT_ROOT:-}" ] && EXTRA_ARGS+=(--artifact_root "$ARTIFACT_ROOT")
[ -n "${USER_SHARD_ID:-}" ] && EXTRA_ARGS+=(--user_shard_id "$USER_SHARD_ID")
[ -n "${NUM_USER_SHARDS:-}" ] && EXTRA_ARGS+=(--num_user_shards "$NUM_USER_SHARDS")
[ -n "${RUN_NAME_SUFFIX:-}" ] && EXTRA_ARGS+=(--run_name_suffix "$RUN_NAME_SUFFIX")
[ "${LOAD_SAVED_MEMORY:-0}" = "1" ] && EXTRA_ARGS+=(--LOAD_SAVED_MEMORY)
[ "${DISABLE_TRACE:-0}" = "1" ] && EXTRA_ARGS+=(--disable_trace)

case "$VARIANT" in
  A0_no_memory)
    ARGS=("${COMMON_ARGS[@]}" --no_use_memory)
    ;;
  A1_same_user_only)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --graph_retrieval_scope same_user)
    ;;
  A2_candidate_item_only)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --graph_retrieval_scope candidate_item)
    ;;
  A3_neighbor_user_only)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --graph_retrieval_scope neighbor_user)
    ;;
  A4_full_graph)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --graph_retrieval_scope full)
    ;;
  A5_full_graph_noharm)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --graph_retrieval_scope full --no_harm_arbitration)
    ;;
  A6_random_memory)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --graph_retrieval_scope random_memory)
    ;;
  A7_shuffled_memory)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --graph_retrieval_scope shuffled_memory)
    ;;
  A8_profile_only)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --profile_only)
    ;;
  # Backward-compatible names from earlier runs.
  A1_safe_graph_no_noharm)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --graph_retrieval_scope full)
    ;;
  A2_safe_graph_noharm)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --graph_retrieval_scope full --no_harm_arbitration)
    ;;
  A3_safe_graph_k5_noharm)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --graph_retrieval_scope full --graph_memory_k 5 --no_harm_arbitration)
    ;;
  *)
    echo "Unknown variant: $VARIANT" >&2
    exit 2
    ;;
esac

ARGS+=("${EXTRA_ARGS[@]}")

echo "Dataset: $DATASET"
echo "Variant: $VARIANT"
echo "Log: $LOG"
echo "Args: ${ARGS[*]}"
python -m memcf "${ARGS[@]}" 2>&1 | tee "$LOG"
