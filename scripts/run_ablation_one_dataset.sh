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

mkdir -p "$MEMCF_EVAL_ROOT/$DATASET/logs"
LOG="$MEMCF_EVAL_ROOT/$DATASET/logs/${VARIANT}_nuser${N_USERS}.log"

case "$VARIANT" in
  A0_no_memory)
    ARGS=(
      --data_name "$DATASET"
      --number_of_users "$N_USERS"
      --no_use_memory
      --max_positive_interactions "$MAX_POS"
      --max_negative_candidates "$MAX_NEG"
      --ranking_prompt_style "$RANKING_PROMPT_STYLE"
    )
    ;;
  A1_safe_graph_no_noharm)
    ARGS=(
      --data_name "$DATASET"
      --number_of_users "$N_USERS"
      --use_memory
      --max_iterations "$MAX_ITER"
      --max_positive_interactions "$MAX_POS"
      --max_negative_candidates "$MAX_NEG"
      --ranking_prompt_style "$RANKING_PROMPT_STYLE"
      --graph_memory_k 3
      --neighbor_k "$NEIGHBOR_K"
      --min_evidence_terms "$MIN_EVIDENCE"
    )
    ;;
  A2_safe_graph_noharm)
    ARGS=(
      --data_name "$DATASET"
      --number_of_users "$N_USERS"
      --use_memory
      --max_iterations "$MAX_ITER"
      --max_positive_interactions "$MAX_POS"
      --max_negative_candidates "$MAX_NEG"
      --ranking_prompt_style "$RANKING_PROMPT_STYLE"
      --graph_memory_k 3
      --neighbor_k "$NEIGHBOR_K"
      --min_evidence_terms "$MIN_EVIDENCE"
      --no_harm_arbitration
    )
    ;;
  A3_safe_graph_k5_noharm)
    ARGS=(
      --data_name "$DATASET"
      --number_of_users "$N_USERS"
      --use_memory
      --max_iterations "$MAX_ITER"
      --max_positive_interactions "$MAX_POS"
      --max_negative_candidates "$MAX_NEG"
      --ranking_prompt_style "$RANKING_PROMPT_STYLE"
      --graph_memory_k 5
      --neighbor_k "$NEIGHBOR_K"
      --min_evidence_terms "$MIN_EVIDENCE"
      --no_harm_arbitration
    )
    ;;
  *)
    echo "Unknown variant: $VARIANT" >&2
    exit 2
    ;;
esac

echo "Dataset: $DATASET"
echo "Variant: $VARIANT"
echo "Log: $LOG"
python -m memcf "${ARGS[@]}" 2>&1 | tee "$LOG"
