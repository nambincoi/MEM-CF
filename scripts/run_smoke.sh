#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${1:-All_Beauty_1000u}"

export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export MEMCF_DATA_ROOT="${MEMCF_DATA_ROOT:-$ROOT/data}"
export MEMCF_EVAL_ROOT="${MEMCF_EVAL_ROOT:-$ROOT/evaluation_results}"
export MEMCF_MEMORY_ROOT="${MEMCF_MEMORY_ROOT:-$ROOT/agent_memory}"
export chat_api_base="${chat_api_base:-http://127.0.0.1:8000/v1}"
export api_base="${api_base:-$chat_api_base}"
export chat_model_name="${chat_model_name:-gpt-3.5-turbo-16k-0613}"

python -m memcf \
  --data_name "$DATASET" \
  --number_of_users 5 \
  --use_memory \
  --max_iterations 1 \
  --max_positive_interactions 5 \
  --max_negative_candidates 19 \
  --graph_memory_k 3 \
  --neighbor_k 10 \
  --min_evidence_terms 1 \
  --no_harm_arbitration
