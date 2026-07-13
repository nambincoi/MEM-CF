#!/usr/bin/env python3
"""Aggregate MEMCF summary JSON files into CSV/Markdown tables for paper tracking."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def get(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def iter_summaries(root: Path) -> Iterable[Path]:
    yield from sorted(root.rglob("*.summary.json"))


def row_from_summary(path: Path, root: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        s = json.load(f)

    llm = s.get("llm_usage", {}) or {}
    runtime = s.get("runtime", {}) or {}
    diag = s.get("memory_diagnostics", {}) or {}
    artifact = s.get("memory_artifact", {}) or {}
    users = s.get("number_of_users_evaluated") or s.get("active_user_count") or 0
    users = int(users or 0)
    total_calls = int(llm.get("calls", llm.get("total_calls", 0)) or 0)
    total_tokens = int(llm.get("total_tokens", 0) or 0)
    total_seconds = runtime.get("total_seconds")

    return {
        "path": str(path.relative_to(root)),
        "dataset": s.get("dataset_name", path.parent.name),
        "run_name": s.get("run_name", path.stem.replace(".summary", "")),
        "phase": s.get("phase", ""),
        "use_memory": s.get("use_memory", ""),
        "scope": s.get("graph_retrieval_scope", ""),
        "failure_constraint_mode": s.get("failure_constraint_mode", ""),
        "failure_constraint_tie_epsilon": s.get("failure_constraint_tie_epsilon", ""),
        "no_harm": s.get("no_harm_arbitration", ""),
        "prompt_style": s.get("ranking_prompt_style", ""),
        "users": users,
        "recall@5": get(s, "metrics.recall@5"),
        "recall@10": get(s, "metrics.recall@10"),
        "recall@20": get(s, "metrics.recall@20"),
        "ndcg@5": get(s, "metrics.ndcg@5"),
        "ndcg@10": get(s, "metrics.ndcg@10"),
        "ndcg@20": get(s, "metrics.ndcg@20"),
        "base_ndcg@10": get(s, "baseline_metrics.ndcg@10"),
        "memory_lessons": s.get("num_graph_lessons", ""),
        "memory_file_mb": artifact.get("file_size_mb", ""),
        "memory_file_bytes": artifact.get("file_size_bytes", ""),
        "num_clusters": artifact.get("num_clusters", ""),
        "num_memory_source_users": artifact.get("num_memory_source_users", ""),
        "avg_users_per_cluster": artifact.get("avg_users_per_cluster", ""),
        "num_memory_user_edges": artifact.get("num_memory_user_edges", ""),
        "num_memory_item_edges": artifact.get("num_memory_item_edges", ""),
        "avg_retrieved": diag.get("avg_retrieved_memories", ""),
        "kept_total": diag.get("kept_total", ""),
        "users_with_kept": diag.get("users_with_kept_memory", ""),
        "no_harm_use_rate": diag.get("no_harm_memory_use_rate", ""),
        "rank_fallbacks": diag.get("rank_fallbacks", ""),
        "constraint_users": diag.get("failure_constraint_users", ""),
        "constraint_changed_users": diag.get("failure_constraint_changed_users", ""),
        "constraint_evidence": diag.get("failure_constraint_evidence", ""),
        "constraint_moved_candidates": diag.get("failure_constraint_moved_candidates", ""),
        "llm_calls": total_calls,
        "calls_per_user": (total_calls / users if users else None),
        "tokens": total_tokens,
        "tokens_per_user": (total_tokens / users if users else None),
        "seconds": total_seconds,
        "seconds_per_user": (float(total_seconds) / users if users and total_seconds is not None else None),
    }


def write_markdown(rows: List[Dict[str, Any]], out: Path) -> None:
    cols = [
        "dataset", "run_name", "scope", "failure_constraint_mode", "no_harm", "users",
        "ndcg@5", "ndcg@10", "ndcg@20", "recall@10",
        "memory_lessons", "memory_file_mb", "num_clusters", "num_memory_source_users", "avg_retrieved",
        "constraint_changed_users", "constraint_evidence",
        "llm_calls", "calls_per_user", "tokens_per_user", "seconds_per_user",
    ]
    with out.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("| " + " | ".join(["---"] * len(cols)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(fmt(row.get(c)) for c in cols) + " |\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Evaluation root containing *.summary.json")
    ap.add_argument("--csv", default=None, help="CSV output path")
    ap.add_argument("--markdown", default=None, help="Markdown output path")
    args = ap.parse_args()

    root = Path(args.root)
    rows = [row_from_summary(p, root) for p in iter_summaries(root)]
    rows.sort(key=lambda r: (str(r.get("dataset")), str(r.get("run_name"))))

    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
            if rows:
                writer.writeheader()
                writer.writerows(rows)
        print(f"Saved CSV: {out}")

    if args.markdown:
        out = Path(args.markdown)
        out.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(rows, out)
        print(f"Saved Markdown: {out}")

    print(json.dumps({"root": str(root), "num_summaries": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
