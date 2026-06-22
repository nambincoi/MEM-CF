#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MEMCF_EVAL_ROOT="${MEMCF_EVAL_ROOT:-$ROOT/evaluation_results_memcf_strong}"
export MEMCF_MEMORY_ROOT="${MEMCF_MEMORY_ROOT:-$ROOT/agent_memory_memcf_strong}"
export RANKING_PROMPT_STYLE="${RANKING_PROMPT_STYLE:-compact_score}"
export DATASETS="${DATASETS:-Video_Game Digital_Music_1000u All_Beauty_1000u CDs_and_Vinyl_1000u}"
export VARIANTS="${VARIANTS:-A0_no_memory A1_same_user_only A4_full_graph A5_full_graph_noharm}"

bash "$ROOT/scripts/run_ablation_jobs.sh"
