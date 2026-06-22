#!/usr/bin/env python3
"""Merge MEMCF train-only memory shard artifacts into one memory file."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True, help="Merged memory JSON output path")
    ap.add_argument("memory_files", nargs="+", help="Shard memory JSON files")
    args = ap.parse_args()

    out = Path(args.output)
    seen_ids = set()
    lessons: List[Dict[str, Any]] = []
    user_profiles: Dict[str, Any] = {}
    inputs = []

    for raw in args.memory_files:
        path = Path(raw)
        payload = load_json(path)
        inputs.append(str(path))

        for lesson in payload.get("graph", {}).get("lessons", []):
            lesson_id = str(lesson.get("memory_id", ""))
            if not lesson_id:
                lesson_id = json.dumps(lesson, sort_keys=True, ensure_ascii=False)
            if lesson_id in seen_ids:
                continue
            seen_ids.add(lesson_id)
            lessons.append(lesson)

        for uid, profile in payload.get("user_profiles", {}).items():
            # User shards should be disjoint; keep first profile if duplicated.
            user_profiles.setdefault(str(uid), profile)

    merged = {
        "model": "MEMCF",
        "merged_at": datetime.now().isoformat(),
        "merge_inputs": inputs,
        "graph": {
            "lessons": lessons,
        },
        "user_profiles": user_profiles,
        "merge_stats": {
            "num_input_files": len(inputs),
            "num_lessons": len(lessons),
            "num_user_profiles": len(user_profiles),
        },
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(json.dumps(merged["merge_stats"], indent=2))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
