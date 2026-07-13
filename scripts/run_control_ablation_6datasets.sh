#!/usr/bin/env bash
set -euo pipefail

# Runs paper-control ablations without retraining memory.
# Defaults target the 6 datasets currently used for MEMCF-vs-MemRec analysis.

DATASETS_STR="${DATASETS:-Video_Game Digital_Music_1000u All_Beauty_1000u Industrial_and_Scientific_1000u Prime_Pantry_1000u Software_1000u}"
VARIANTS_STR="${VARIANTS:-A0_no_memory A8_profile_only A1_same_user_only A4_full_graph A6_random_forced A6_random_gated A7_shuffled_forced A7_shuffled_no_profile A4_full_graph_top1 A4_full_graph_same_user_first A4_full_graph_candidate_strict A4_full_graph_cross_user_only}"

export MEMCF_DATA_ROOT="${MEMCF_DATA_ROOT:-/home/ubuntu/24nam.nh/video_games_data/runtime_data}"
export MEMCF_EVAL_ROOT="${MEMCF_EVAL_ROOT:-/home/ubuntu/24nam.nh/video_games_data/evaluation_results_memcf_control_ablation_6datasets_100u}"
export MEMCF_MEMORY_ROOT="${MEMCF_MEMORY_ROOT:-/home/ubuntu/24nam.nh/video_games_data/agent_memory_memcf_control_ablation_6datasets_100u}"

export chat_api_base="${chat_api_base:-http://127.0.0.1:8000/v1}"
export api_base="${api_base:-http://127.0.0.1:8000/v1}"
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
export RUN_NAME_SUFFIX="${RUN_NAME_SUFFIX:-control_ablation_6ds_100u}"
export MAX_PARALLEL="${MAX_PARALLEL:-32}"

OLD_MEMORY_ROOT="${OLD_MEMORY_ROOT:-/home/ubuntu/24nam.nh/video_games_data/agent_memory_memcf_mainpaper_100u}"
NEW_MEMORY_ROOT="${NEW_MEMORY_ROOT:-/home/ubuntu/24nam.nh/video_games_data/agent_memory_memcf_3new_100u}"
PID_DIR="$MEMCF_EVAL_ROOT/_pids_control_ablation"
mkdir -p "$PID_DIR"

memory_file_for_dataset() {
  local ds="$1"
  case "$ds" in
    Video_Game|Digital_Music_1000u|All_Beauty_1000u)
      printf '%s/%s/mainpaper_nuser100.memory.json' "$OLD_MEMORY_ROOT" "$ds"
      ;;
    Industrial_and_Scientific_1000u|Prime_Pantry_1000u|Software_1000u)
      find "$NEW_MEMORY_ROOT/$ds" -maxdepth 1 -type f -name '*.memory.json' | sort | head -1
      ;;
    *)
      printf '%s/%s/mainpaper_nuser100.memory.json' "$MEMCF_MEMORY_ROOT" "$ds"
      ;;
  esac
}

wait_for_slot() {
  while true; do
    local running
    running=$(jobs -pr | wc -l)
    if [ "$running" -lt "$MAX_PARALLEL" ]; then
      break
    fi
    sleep 5
  done
}

echo "MEMCF control ablation launcher"
echo "EVAL_ROOT=$MEMCF_EVAL_ROOT"
echo "DATASETS=$DATASETS_STR"
echo "VARIANTS=$VARIANTS_STR"
echo "MAX_PARALLEL=$MAX_PARALLEL"

for DS in $DATASETS_STR; do
  export MEMORY_FILE
  MEMORY_FILE="$(memory_file_for_dataset "$DS")"
  if [ ! -f "$MEMORY_FILE" ]; then
    echo "ERROR missing MEMORY_FILE for $DS: $MEMORY_FILE" >&2
    exit 1
  fi
  for VAR in $VARIANTS_STR; do
    wait_for_slot
    mkdir -p "$MEMCF_EVAL_ROOT/$DS/logs"
    LOG="$MEMCF_EVAL_ROOT/$DS/logs/${VAR}_${RUN_NAME_SUFFIX}.nohup.log"
    PID_FILE="$PID_DIR/${DS}_${VAR}.pid"
    (
      cd /home/ubuntu/24nam.nh/video_games_data/MEMCF
      bash scripts/run_ablation_one_dataset.sh "$DS" "$VAR"
    ) > "$LOG" 2>&1 &
    echo "$!" > "$PID_FILE"
    echo "STARTED $DS $VAR pid=$(cat "$PID_FILE") log=$LOG"
    sleep 0.2
  done
done

wait
echo "All control ablation jobs finished."
