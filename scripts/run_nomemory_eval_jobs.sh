#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MEMCF_EVAL_ROOT="${MEMCF_EVAL_ROOT:-$ROOT/evaluation_results_memcf_shared}"
export N_USERS="${N_USERS:-300}"
export PHASE=all
unset LOAD_SAVED_MEMORY MEMORY_FILE USER_SHARD_ID NUM_USER_SHARDS
export RUN_NAME_SUFFIX="${RUN_NAME_SUFFIX:-nomemory}"
export VARIANTS="A0_no_memory"
bash "$ROOT/scripts/run_ablation_jobs.sh"
