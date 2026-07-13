#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MEMCF_MEMORY_ROOT="${MEMCF_MEMORY_ROOT:-$ROOT/agent_memory_memcf_shared}"
export N_USERS="${N_USERS:-300}"
export NUM_USER_SHARDS="${NUM_USER_SHARDS:-3}"
DATASETS_STR="${DATASETS:-Video_Game Digital_Music_1000u CDs_and_Vinyl_1000u All_Beauty_1000u}"
read -r -a DATASETS_ARR <<< "$DATASETS_STR"

for ds in "${DATASETS_ARR[@]}"; do
  ds_dir="$MEMCF_MEMORY_ROOT/$ds"
  out="$ds_dir/shared_nuser${N_USERS}_${NUM_USER_SHARDS}shards.memory.json"
  mapfile -t files < <(find "$ds_dir" -maxdepth 1 -type f \
    -name "memcf_graph_nuser${N_USERS}_shard*of${NUM_USER_SHARDS}_*.memory.json" | sort)
  if [ "${#files[@]}" -eq 0 ]; then
    echo "[ERROR] no shard memory files found for $ds in $ds_dir" >&2
    exit 1
  fi
  echo "Merging $ds: ${#files[@]} shard files -> $out"
  python3 "$ROOT/scripts/merge_memory_shards.py" --output "$out" "${files[@]}"
done
