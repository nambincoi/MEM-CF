#!/usr/bin/env bash
set -euo pipefail

# Upload raw Amazon ratings + metadata files for the new MEMCF datasets.
# Run this from the local repo root: /home/hoangnam/Memrec

LOCAL_RAW_ROOT="${LOCAL_RAW_ROOT:-/home/hoangnam/Memrec/data copy}"
REMOTE_HOST="${REMOTE_HOST:-cecs}"
REMOTE_RAW_ROOT="${REMOTE_RAW_ROOT:-/home/ubuntu/24nam.nh/video_games_data/raw_amazon_new}"

if [[ ! -d "$LOCAL_RAW_ROOT" ]]; then
  echo "Missing local raw root: $LOCAL_RAW_ROOT" >&2
  exit 1
fi

if find "$LOCAL_RAW_ROOT" -maxdepth 1 -name "*.crdownload" | grep -q .; then
  echo "Found unfinished .crdownload file(s):" >&2
  find "$LOCAL_RAW_ROOT" -maxdepth 1 -name "*.crdownload" -print >&2
  echo "Finish the download and rename to .json.gz before uploading." >&2
  exit 2
fi

ssh "$REMOTE_HOST" "mkdir -p '$REMOTE_RAW_ROOT'"

for f in \
  "Industrial_and_Scientific.csv" \
  "meta_Industrial_and_Scientific.json.gz" \
  "Prime_Pantry.csv" \
  "meta_Prime_Pantry.json.gz" \
  "Software.csv" \
  "meta_Software.json.gz"
do
  if [[ ! -f "$LOCAL_RAW_ROOT/$f" ]]; then
    echo "Missing required file: $LOCAL_RAW_ROOT/$f" >&2
    exit 3
  fi
done

scp -O \
  "$LOCAL_RAW_ROOT/Industrial_and_Scientific.csv" \
  "$LOCAL_RAW_ROOT/meta_Industrial_and_Scientific.json.gz" \
  "$LOCAL_RAW_ROOT/Prime_Pantry.csv" \
  "$LOCAL_RAW_ROOT/meta_Prime_Pantry.json.gz" \
  "$LOCAL_RAW_ROOT/Software.csv" \
  "$LOCAL_RAW_ROOT/meta_Software.json.gz" \
  "$REMOTE_HOST:$REMOTE_RAW_ROOT/"

ssh "$REMOTE_HOST" "ls -lh '$REMOTE_RAW_ROOT'"
