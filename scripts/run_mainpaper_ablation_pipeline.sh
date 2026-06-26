#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# This wrapper trains one shared memory pool per dataset, waits for training
# jobs, then evaluates the main-paper ablation variants from the same memory.
# For long server runs, running train/eval scripts separately is easier to
# monitor; this wrapper is provided for convenience.

bash "$ROOT/scripts/run_mainpaper_train_memory_jobs.sh"

PID_DIR="${MEMCF_EVAL_ROOT:-$ROOT/evaluation_results_memcf_mainpaper}/_pids_train_once"
if compgen -G "$PID_DIR/*.pid" > /dev/null; then
  echo "Waiting for train-once jobs in $PID_DIR"
  while true; do
    running=0
    for p in "$PID_DIR"/*.pid; do
      pid="$(cat "$p")"
      if ps -p "$pid" >/dev/null; then
        running=$((running + 1))
      fi
    done
    if [ "$running" -eq 0 ]; then
      break
    fi
    echo "Still training: $running jobs"
    sleep "${TRAIN_POLL_SECONDS:-30}"
  done
fi

bash "$ROOT/scripts/run_mainpaper_eval_jobs.sh"
