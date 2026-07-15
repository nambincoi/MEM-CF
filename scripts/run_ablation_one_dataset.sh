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
EVAL_SPLIT="${EVAL_SPLIT:-test}"

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
  --eval_split "$EVAL_SPLIT"
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
[ -n "${RANKING_SCORE_CACHE_DIR:-}" ] && EXTRA_ARGS+=(--ranking_score_cache_dir "$RANKING_SCORE_CACHE_DIR")
[ "${SKIP_USER_CLUSTERS:-0}" = "1" ] && EXTRA_ARGS+=(--skip_user_clusters)
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
  A4_full_graph_llm_select)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --graph_retrieval_scope full --memory_selector llm --memory_selector_top_m "${MEMORY_SELECTOR_TOP_M:-12}" --memory_selector_top_k "${MEMORY_SELECTOR_TOP_K:-3}" --memory_selector_min_relevance "${MEMORY_SELECTOR_MIN_RELEVANCE:-0.60}")
    ;;
  A4_full_graph_llm_select_rejwrong)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --graph_retrieval_scope full --memory_selector llm --memory_selector_top_m "${MEMORY_SELECTOR_TOP_M:-12}" --memory_selector_top_k "${MEMORY_SELECTOR_TOP_K:-3}" --memory_selector_min_relevance "${MEMORY_SELECTOR_MIN_RELEVANCE:-0.70}" --reject_wrong_only_memory)
    ;;
  A4_full_graph_top1)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --graph_retrieval_scope full --graph_memory_k 1 --max_memory_facts 1)
    ;;
  A4_full_graph_same_user_first)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --graph_retrieval_scope same_user_first)
    ;;
  A4_full_graph_candidate_strict)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --graph_retrieval_scope candidate_strict --strict_memory_applicability --require_same_user_candidate_match --reject_wrong_only_memory)
    ;;
  A4_full_graph_cross_user_only)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --graph_retrieval_scope cross_user_only --strict_memory_applicability --reject_wrong_only_memory)
    ;;
  A5_full_graph_noharm)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --graph_retrieval_scope full --no_harm_arbitration)
    ;;
  A6_random_memory)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --graph_retrieval_scope random_memory)
    ;;
  A6_random_forced)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --graph_retrieval_scope random_memory --strict_memory_applicability --allow_random_memory_injection)
    ;;
  A6_random_gated)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --graph_retrieval_scope random_memory --strict_memory_applicability)
    ;;
  A6_random_forced_clean)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --graph_retrieval_scope random_memory_clean --strict_memory_applicability --allow_random_memory_injection)
    ;;
  A6_random_forced_clean_llm_select)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --graph_retrieval_scope random_memory_clean --strict_memory_applicability --allow_random_memory_injection --memory_selector llm --memory_selector_top_m "${MEMORY_SELECTOR_TOP_M:-12}" --memory_selector_top_k "${MEMORY_SELECTOR_TOP_K:-3}" --memory_selector_min_relevance "${MEMORY_SELECTOR_MIN_RELEVANCE:-0.60}")
    ;;
  A7_shuffled_memory)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --graph_retrieval_scope shuffled_memory)
    ;;
  A7_shuffled_forced)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --graph_retrieval_scope shuffled_memory --strict_memory_applicability --allow_random_memory_injection)
    ;;
  A7_shuffled_no_profile)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --graph_retrieval_scope shuffled_memory --strict_memory_applicability --require_same_user_candidate_match --disable_user_profile_in_eval_prompt)
    ;;
  A7_shuffled_forced_clean)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --graph_retrieval_scope shuffled_memory_clean --strict_memory_applicability --allow_random_memory_injection)
    ;;
  A7_shuffled_forced_clean_llm_select)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --graph_retrieval_scope shuffled_memory_clean --strict_memory_applicability --allow_random_memory_injection --memory_selector llm --memory_selector_top_m "${MEMORY_SELECTOR_TOP_M:-12}" --memory_selector_top_k "${MEMORY_SELECTOR_TOP_K:-3}" --memory_selector_min_relevance "${MEMORY_SELECTOR_MIN_RELEVANCE:-0.60}")
    ;;
  A8_profile_only)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --profile_only)
    ;;
  A9_same_user_curated)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --graph_retrieval_scope same_user --ranking_prompt_style compact_curated_score --reject_wrong_only_memory)
    ;;
  A9_full_graph_curated)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --graph_retrieval_scope full --ranking_prompt_style compact_curated_score --reject_wrong_only_memory)
    ;;
  A9_full_graph_curated_select)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --graph_retrieval_scope full --ranking_prompt_style compact_curated_score --memory_selector llm --memory_selector_top_m "${MEMORY_SELECTOR_TOP_M:-8}" --memory_selector_top_k "${MEMORY_SELECTOR_TOP_K:-3}" --memory_selector_min_relevance "${MEMORY_SELECTOR_MIN_RELEVANCE:-0.70}" --reject_wrong_only_memory)
    ;;
  A9_same_user_first_curated)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --graph_retrieval_scope same_user_first --ranking_prompt_style compact_curated_score --reject_wrong_only_memory)
    ;;
  A9_random_forced_curated)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --graph_retrieval_scope random_memory_clean --ranking_prompt_style compact_curated_score --strict_memory_applicability --allow_random_memory_injection)
    ;;
  A9_shuffled_forced_curated)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --graph_retrieval_scope shuffled_memory_clean --ranking_prompt_style compact_curated_score --strict_memory_applicability --allow_random_memory_injection)
    ;;
  A9_cluster_user)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --graph_retrieval_scope cluster_user)
    ;;
  A10_cluster_full)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --graph_retrieval_scope cluster_full)
    ;;
  A11_random_cluster)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --graph_retrieval_scope random_cluster --strict_memory_applicability --allow_random_memory_injection)
    ;;
  A12_shuffled_cluster)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --graph_retrieval_scope shuffled_cluster --strict_memory_applicability)
    ;;
  A13_cluster_full_noharm)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --graph_retrieval_scope cluster_full --no_harm_arbitration)
    ;;
  A14_hybrid_cluster)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --graph_retrieval_scope hybrid_cluster)
    ;;
  A15_hybrid_cluster_strict)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --graph_retrieval_scope hybrid_cluster_strict --strict_memory_applicability --reject_wrong_only_memory)
    ;;
  A16_cluster_user_fixed)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --graph_retrieval_scope cluster_user --reject_wrong_only_memory)
    ;;
  A17_cluster_full_fixed)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --graph_retrieval_scope cluster_full --reject_wrong_only_memory)
    ;;
  C1_same_user_pairwise)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --graph_retrieval_scope same_user --reject_wrong_only_memory --pairwise_cf_rerank --pairwise_cf_hide_memory_prompt --pairwise_cf_alpha "${PAIRWISE_CF_ALPHA:-0.04}" --pairwise_cf_beta "${PAIRWISE_CF_BETA:-0.04}")
    ;;
  C2_full_graph_pairwise)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --graph_retrieval_scope full --reject_wrong_only_memory --pairwise_cf_rerank --pairwise_cf_hide_memory_prompt --pairwise_cf_alpha "${PAIRWISE_CF_ALPHA:-0.04}" --pairwise_cf_beta "${PAIRWISE_CF_BETA:-0.04}")
    ;;
  C3_full_graph_pairwise_prompt)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --graph_retrieval_scope full --reject_wrong_only_memory --pairwise_cf_rerank --pairwise_cf_alpha "${PAIRWISE_CF_ALPHA:-0.04}" --pairwise_cf_beta "${PAIRWISE_CF_BETA:-0.04}")
    ;;
  C4_full_graph_pairwise_strict)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --graph_retrieval_scope candidate_strict --strict_memory_applicability --require_same_user_candidate_match --reject_wrong_only_memory --pairwise_cf_rerank --pairwise_cf_hide_memory_prompt --pairwise_cf_alpha "${PAIRWISE_CF_ALPHA:-0.04}" --pairwise_cf_beta "${PAIRWISE_CF_BETA:-0.04}")
    ;;
  C5_cross_user_pairwise)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --graph_retrieval_scope cross_user_only --strict_memory_applicability --reject_wrong_only_memory --pairwise_cf_rerank --pairwise_cf_hide_memory_prompt --pairwise_cf_alpha "${PAIRWISE_CF_ALPHA:-0.04}" --pairwise_cf_beta "${PAIRWISE_CF_BETA:-0.04}")
    ;;
  D0_profile_base)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --profile_only)
    ;;
  D1_same_user_exact)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --failure_constraint_mode same_exact --failure_constraint_tie_epsilon "${FAILURE_CONSTRAINT_TIE_EPSILON:-0.0}" --failure_constraint_min_context_terms "${FAILURE_CONSTRAINT_MIN_CONTEXT_TERMS:-1}" --failure_constraint_same_budget "${FAILURE_CONSTRAINT_SAME_BUDGET:-32}" --failure_constraint_cross_budget "${FAILURE_CONSTRAINT_CROSS_BUDGET:-128}")
    ;;
  D2_cross_user_exact)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --failure_constraint_mode cross_exact --failure_constraint_tie_epsilon "${FAILURE_CONSTRAINT_TIE_EPSILON:-0.0}" --failure_constraint_min_context_terms "${FAILURE_CONSTRAINT_MIN_CONTEXT_TERMS:-1}" --failure_constraint_same_budget "${FAILURE_CONSTRAINT_SAME_BUDGET:-32}" --failure_constraint_cross_budget "${FAILURE_CONSTRAINT_CROSS_BUDGET:-128}")
    ;;
  D3_full_partitioned)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --failure_constraint_mode full_partitioned --failure_constraint_tie_epsilon "${FAILURE_CONSTRAINT_TIE_EPSILON:-0.0}" --failure_constraint_min_context_terms "${FAILURE_CONSTRAINT_MIN_CONTEXT_TERMS:-1}" --failure_constraint_same_budget "${FAILURE_CONSTRAINT_SAME_BUDGET:-32}" --failure_constraint_cross_budget "${FAILURE_CONSTRAINT_CROSS_BUDGET:-128}")
    ;;
  D4_full_consensus)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --failure_constraint_mode full_consensus --failure_constraint_tie_epsilon "${FAILURE_CONSTRAINT_TIE_EPSILON:-0.0}" --failure_constraint_min_cross_support "${FAILURE_CONSTRAINT_MIN_CROSS_SUPPORT:-2}" --failure_constraint_min_context_terms "${FAILURE_CONSTRAINT_MIN_CONTEXT_TERMS:-1}" --failure_constraint_same_budget "${FAILURE_CONSTRAINT_SAME_BUDGET:-32}" --failure_constraint_cross_budget "${FAILURE_CONSTRAINT_CROSS_BUDGET:-128}")
    ;;
  D5_polarity_swapped)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --failure_constraint_mode polarity_swapped --failure_constraint_tie_epsilon "${FAILURE_CONSTRAINT_TIE_EPSILON:-0.0}" --failure_constraint_min_context_terms "${FAILURE_CONSTRAINT_MIN_CONTEXT_TERMS:-1}" --failure_constraint_same_budget "${FAILURE_CONSTRAINT_SAME_BUDGET:-32}" --failure_constraint_cross_budget "${FAILURE_CONSTRAINT_CROSS_BUDGET:-128}")
    ;;
  D6_popularity_control)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --failure_constraint_mode popularity --failure_constraint_tie_epsilon "${FAILURE_CONSTRAINT_TIE_EPSILON:-0.0}")
    ;;
  D7_shuffled_provenance)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --failure_constraint_mode shuffled_provenance --failure_constraint_tie_epsilon "${FAILURE_CONSTRAINT_TIE_EPSILON:-0.0}" --failure_constraint_min_context_terms "${FAILURE_CONSTRAINT_MIN_CONTEXT_TERMS:-1}" --failure_constraint_same_budget "${FAILURE_CONSTRAINT_SAME_BUDGET:-32}" --failure_constraint_cross_budget "${FAILURE_CONSTRAINT_CROSS_BUDGET:-128}")
    ;;
  F0_profile_base)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --profile_only)
    ;;
  F1_same_user_exact)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --failure_constraint_mode same_exact --failure_constraint_tie_epsilon "${FAILURE_CONSTRAINT_TIE_EPSILON:-0.0}" --failure_constraint_same_budget "${FAILURE_CONSTRAINT_SAME_BUDGET:-32}")
    ;;
  F2_shared_item_cross_only)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --failure_constraint_mode cf_shared_cross --failure_constraint_tie_epsilon "${FAILURE_CONSTRAINT_TIE_EPSILON:-0.0}" --failure_constraint_min_cross_support "${FAILURE_CONSTRAINT_MIN_CROSS_SUPPORT:-2}" --failure_constraint_min_shared_items "${FAILURE_CONSTRAINT_MIN_SHARED_ITEMS:-1}" --failure_constraint_max_cross_corrections "${FAILURE_CONSTRAINT_MAX_CROSS_CORRECTIONS:-3}" --failure_constraint_cross_budget "${FAILURE_CONSTRAINT_CROSS_BUDGET:-128}")
    ;;
  F3_same_plus_shared_cf)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --failure_constraint_mode cf_same_plus_shared --failure_constraint_tie_epsilon "${FAILURE_CONSTRAINT_TIE_EPSILON:-0.0}" --failure_constraint_min_cross_support "${FAILURE_CONSTRAINT_MIN_CROSS_SUPPORT:-2}" --failure_constraint_min_shared_items "${FAILURE_CONSTRAINT_MIN_SHARED_ITEMS:-1}" --failure_constraint_max_cross_corrections "${FAILURE_CONSTRAINT_MAX_CROSS_CORRECTIONS:-3}" --failure_constraint_same_budget "${FAILURE_CONSTRAINT_SAME_BUDGET:-32}" --failure_constraint_cross_budget "${FAILURE_CONSTRAINT_CROSS_BUDGET:-128}")
    ;;
  F4_shuffled_neighbors)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --failure_constraint_mode cf_shuffled_neighbors --failure_constraint_tie_epsilon "${FAILURE_CONSTRAINT_TIE_EPSILON:-0.0}" --failure_constraint_min_cross_support "${FAILURE_CONSTRAINT_MIN_CROSS_SUPPORT:-2}" --failure_constraint_min_shared_items "${FAILURE_CONSTRAINT_MIN_SHARED_ITEMS:-1}" --failure_constraint_max_cross_corrections "${FAILURE_CONSTRAINT_MAX_CROSS_CORRECTIONS:-3}" --failure_constraint_cross_budget "${FAILURE_CONSTRAINT_CROSS_BUDGET:-128}")
    ;;
  F5_random_neighbors)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --failure_constraint_mode cf_random_neighbors --failure_constraint_tie_epsilon "${FAILURE_CONSTRAINT_TIE_EPSILON:-0.0}" --failure_constraint_min_cross_support "${FAILURE_CONSTRAINT_MIN_CROSS_SUPPORT:-2}" --failure_constraint_min_shared_items "${FAILURE_CONSTRAINT_MIN_SHARED_ITEMS:-1}" --failure_constraint_max_cross_corrections "${FAILURE_CONSTRAINT_MAX_CROSS_CORRECTIONS:-3}" --failure_constraint_cross_budget "${FAILURE_CONSTRAINT_CROSS_BUDGET:-128}")
    ;;
  F6_cross_polarity_swapped)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --failure_constraint_mode cf_polarity_swapped --failure_constraint_tie_epsilon "${FAILURE_CONSTRAINT_TIE_EPSILON:-0.0}" --failure_constraint_min_cross_support "${FAILURE_CONSTRAINT_MIN_CROSS_SUPPORT:-2}" --failure_constraint_min_shared_items "${FAILURE_CONSTRAINT_MIN_SHARED_ITEMS:-1}" --failure_constraint_max_cross_corrections "${FAILURE_CONSTRAINT_MAX_CROSS_CORRECTIONS:-3}" --failure_constraint_cross_budget "${FAILURE_CONSTRAINT_CROSS_BUDGET:-128}")
    ;;
  F7_popularity_control)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --failure_constraint_mode popularity --failure_constraint_tie_epsilon "${FAILURE_CONSTRAINT_TIE_EPSILON:-0.0}")
    ;;
  AF0_profile_base)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --profile_only)
    ;;
  AF1_strong_same_user)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --graph_retrieval_scope same_user)
    ;;
  AF2_strong_cross_only)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --failure_constraint_mode cf_shared_cross --failure_constraint_tie_epsilon "${FAILURE_CONSTRAINT_TIE_EPSILON:-0.0}" --failure_constraint_min_cross_support "${FAILURE_CONSTRAINT_MIN_CROSS_SUPPORT:-2}" --failure_constraint_min_shared_items "${FAILURE_CONSTRAINT_MIN_SHARED_ITEMS:-1}" --failure_constraint_max_cross_corrections "${FAILURE_CONSTRAINT_MAX_CROSS_CORRECTIONS:-3}" --failure_constraint_cross_budget "${FAILURE_CONSTRAINT_CROSS_BUDGET:-128}")
    ;;
  AF3_strong_same_plus_cf)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --graph_retrieval_scope same_user --failure_constraint_with_prompt_memory --failure_constraint_mode cf_shared_cross --failure_constraint_tie_epsilon "${FAILURE_CONSTRAINT_TIE_EPSILON:-0.0}" --failure_constraint_min_cross_support "${FAILURE_CONSTRAINT_MIN_CROSS_SUPPORT:-2}" --failure_constraint_min_shared_items "${FAILURE_CONSTRAINT_MIN_SHARED_ITEMS:-1}" --failure_constraint_max_cross_corrections "${FAILURE_CONSTRAINT_MAX_CROSS_CORRECTIONS:-3}" --failure_constraint_cross_budget "${FAILURE_CONSTRAINT_CROSS_BUDGET:-128}")
    ;;
  AF4_strong_same_plus_shuffled)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --graph_retrieval_scope same_user --failure_constraint_with_prompt_memory --failure_constraint_mode cf_shuffled_neighbors --failure_constraint_tie_epsilon "${FAILURE_CONSTRAINT_TIE_EPSILON:-0.0}" --failure_constraint_min_cross_support "${FAILURE_CONSTRAINT_MIN_CROSS_SUPPORT:-2}" --failure_constraint_min_shared_items "${FAILURE_CONSTRAINT_MIN_SHARED_ITEMS:-1}" --failure_constraint_max_cross_corrections "${FAILURE_CONSTRAINT_MAX_CROSS_CORRECTIONS:-3}" --failure_constraint_cross_budget "${FAILURE_CONSTRAINT_CROSS_BUDGET:-128}")
    ;;
  AF5_strong_same_plus_random)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --graph_retrieval_scope same_user --failure_constraint_with_prompt_memory --failure_constraint_mode cf_random_neighbors --failure_constraint_tie_epsilon "${FAILURE_CONSTRAINT_TIE_EPSILON:-0.0}" --failure_constraint_min_cross_support "${FAILURE_CONSTRAINT_MIN_CROSS_SUPPORT:-2}" --failure_constraint_min_shared_items "${FAILURE_CONSTRAINT_MIN_SHARED_ITEMS:-1}" --failure_constraint_max_cross_corrections "${FAILURE_CONSTRAINT_MAX_CROSS_CORRECTIONS:-3}" --failure_constraint_cross_budget "${FAILURE_CONSTRAINT_CROSS_BUDGET:-128}")
    ;;
  AF6_strong_same_plus_polarity)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --graph_retrieval_scope same_user --failure_constraint_with_prompt_memory --failure_constraint_mode cf_polarity_swapped --failure_constraint_tie_epsilon "${FAILURE_CONSTRAINT_TIE_EPSILON:-0.0}" --failure_constraint_min_cross_support "${FAILURE_CONSTRAINT_MIN_CROSS_SUPPORT:-2}" --failure_constraint_min_shared_items "${FAILURE_CONSTRAINT_MIN_SHARED_ITEMS:-1}" --failure_constraint_max_cross_corrections "${FAILURE_CONSTRAINT_MAX_CROSS_CORRECTIONS:-3}" --failure_constraint_cross_budget "${FAILURE_CONSTRAINT_CROSS_BUDGET:-128}")
    ;;
  G0_same_only)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --graph_retrieval_scope same_user)
    ;;
  G1_true_neighbor)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --graph_retrieval_scope same_user --failure_constraint_with_prompt_memory --failure_constraint_mode g_true_neighbor --failure_constraint_tie_epsilon "${FAILURE_CONSTRAINT_TIE_EPSILON:-0.0}" --failure_constraint_min_cross_support "${FAILURE_CONSTRAINT_MIN_CROSS_SUPPORT:-1}" --failure_constraint_min_shared_items "${FAILURE_CONSTRAINT_MIN_SHARED_ITEMS:-1}" --failure_constraint_max_cross_corrections "${FAILURE_CONSTRAINT_MAX_CROSS_CORRECTIONS:-2}" --failure_constraint_cross_budget "${FAILURE_CONSTRAINT_CROSS_BUDGET:-2}" --cf_source_budget "${CF_SOURCE_BUDGET:-2}" --cf_control_seed "${CF_CONTROL_SEED:-2027}")
    ;;
  G2_shuffled_graph)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --graph_retrieval_scope same_user --failure_constraint_with_prompt_memory --failure_constraint_mode g_shuffled_graph --failure_constraint_tie_epsilon "${FAILURE_CONSTRAINT_TIE_EPSILON:-0.0}" --failure_constraint_min_cross_support "${FAILURE_CONSTRAINT_MIN_CROSS_SUPPORT:-1}" --failure_constraint_min_shared_items "${FAILURE_CONSTRAINT_MIN_SHARED_ITEMS:-1}" --failure_constraint_max_cross_corrections "${FAILURE_CONSTRAINT_MAX_CROSS_CORRECTIONS:-2}" --failure_constraint_cross_budget "${FAILURE_CONSTRAINT_CROSS_BUDGET:-2}" --cf_source_budget "${CF_SOURCE_BUDGET:-2}" --cf_control_seed "${CF_CONTROL_SEED:-2027}")
    ;;
  G3_random_neighbor)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --graph_retrieval_scope same_user --failure_constraint_with_prompt_memory --failure_constraint_mode g_random_neighbor --failure_constraint_tie_epsilon "${FAILURE_CONSTRAINT_TIE_EPSILON:-0.0}" --failure_constraint_min_cross_support "${FAILURE_CONSTRAINT_MIN_CROSS_SUPPORT:-1}" --failure_constraint_min_shared_items "${FAILURE_CONSTRAINT_MIN_SHARED_ITEMS:-1}" --failure_constraint_max_cross_corrections "${FAILURE_CONSTRAINT_MAX_CROSS_CORRECTIONS:-2}" --failure_constraint_cross_budget "${FAILURE_CONSTRAINT_CROSS_BUDGET:-2}" --cf_source_budget "${CF_SOURCE_BUDGET:-2}" --cf_control_seed "${CF_CONTROL_SEED:-2027}")
    ;;
  G4_matched_random)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --graph_retrieval_scope same_user --failure_constraint_with_prompt_memory --failure_constraint_mode g_matched_random --failure_constraint_tie_epsilon "${FAILURE_CONSTRAINT_TIE_EPSILON:-0.0}" --failure_constraint_min_cross_support "${FAILURE_CONSTRAINT_MIN_CROSS_SUPPORT:-1}" --failure_constraint_min_shared_items "${FAILURE_CONSTRAINT_MIN_SHARED_ITEMS:-1}" --failure_constraint_max_cross_corrections "${FAILURE_CONSTRAINT_MAX_CROSS_CORRECTIONS:-2}" --failure_constraint_cross_budget "${FAILURE_CONSTRAINT_CROSS_BUDGET:-2}" --cf_source_budget "${CF_SOURCE_BUDGET:-2}" --cf_control_seed "${CF_CONTROL_SEED:-2027}")
    ;;
  # Weak controls aligned with MemRec-style vanilla LLM.
  # These are additive variants; they do not change the main A0-A17 runs.
  B0_vanilla_llm)
    ARGS=("${COMMON_ARGS[@]}" --no_use_memory --ranking_prompt_style memrec_vanilla)
    ;;
  B0_anchor)
    ARGS=("${COMMON_ARGS[@]}" --no_use_memory --ranking_prompt_style weak_anchor_score)
    ;;
  B1_same_user_weak)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --graph_retrieval_scope same_user --ranking_prompt_style weak_memory_score)
    ;;
  B1_same_user_correction)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --graph_retrieval_scope same_user --ranking_prompt_style weak_memory_evidence_score)
    ;;
  B1_same_user_router)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --graph_retrieval_scope same_user --ranking_prompt_style weak_memory_router_score)
    ;;
  B1_same_user_anchor_router)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --graph_retrieval_scope same_user --ranking_prompt_style weak_anchor_router_score)
    ;;
  B4_full_graph_weak)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --graph_retrieval_scope full --ranking_prompt_style weak_memory_score)
    ;;
  B4_full_graph_correction)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --graph_retrieval_scope full --ranking_prompt_style weak_memory_evidence_score)
    ;;
  B4_full_graph_router)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --graph_retrieval_scope full --ranking_prompt_style weak_memory_router_score)
    ;;
  B4_full_graph_anchor_router)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --graph_retrieval_scope full --ranking_prompt_style weak_anchor_router_score)
    ;;
  B6_random_weak)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --graph_retrieval_scope random_memory --ranking_prompt_style weak_memory_score)
    ;;
  B6_random_correction)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --graph_retrieval_scope random_memory --ranking_prompt_style weak_memory_evidence_score)
    ;;
  B6_random_router)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --graph_retrieval_scope random_memory --ranking_prompt_style weak_memory_router_score)
    ;;
  B6_random_anchor_router)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --graph_retrieval_scope random_memory --ranking_prompt_style weak_anchor_router_score)
    ;;
  B7_shuffled_weak)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --graph_retrieval_scope shuffled_memory --ranking_prompt_style weak_memory_score)
    ;;
  B7_shuffled_correction)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --graph_retrieval_scope shuffled_memory --ranking_prompt_style weak_memory_evidence_score)
    ;;
  B7_shuffled_router)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --graph_retrieval_scope shuffled_memory --ranking_prompt_style weak_memory_router_score)
    ;;
  B7_shuffled_anchor_router)
    ARGS=("${COMMON_ARGS[@]}" "${MEMORY_ARGS[@]}" --graph_retrieval_scope shuffled_memory --ranking_prompt_style weak_anchor_router_score)
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
