#!/usr/bin/env python3
"""Summarize D-family typed failure-constraint results and traces."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


VARIANTS = [
    "D0_profile_base",
    "D1_same_user_exact",
    "D2_cross_user_exact",
    "D3_full_partitioned",
    "D4_full_consensus",
    "D5_polarity_swapped",
    "D6_popularity_control",
    "D7_shuffled_provenance",
]

TRACE_TAGS = {
    "D0_profile_base": "dtyped100",
    "D1_same_user_exact": "d1same",
    "D2_cross_user_exact": "d2cross",
    "D3_full_partitioned": "d3full",
    "D4_full_consensus": "d4cons",
    "D5_polarity_swapped": "d5swap",
    "D6_popularity_control": "d6pop",
    "D7_shuffled_provenance": "d7shuf",
}


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def variant_for_path(path: Path) -> Optional[str]:
    return next((variant for variant in VARIANTS if variant in path.name), None)


def ndcg_at_10(ranking: List[str], ground_truth: str) -> float:
    try:
        rank = ranking.index(ground_truth) + 1
    except ValueError:
        return 0.0
    return 0.0 if rank > 10 else 1.0 / math.log2(rank + 1)


def latest_trace(dataset_dir: Path, variant: str) -> Optional[Path]:
    trace_root = dataset_dir / "traces"
    if not trace_root.exists():
        return None
    tag = TRACE_TAGS.get(variant, variant)
    matches = [path for path in trace_root.iterdir() if path.is_dir() and tag in path.name]
    if variant == "D0_profile_base":
        matches = [path for path in matches if "profileonly" in path.name]
    return max(matches, key=lambda path: path.stat().st_mtime) if matches else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--markdown")
    parser.add_argument("--json")
    args = parser.parse_args()

    root = Path(args.root)
    results: List[Dict[str, Any]] = []
    per_user: Dict[tuple[str, str], Dict[str, float]] = {}

    for dataset_dir in sorted(path for path in root.iterdir() if path.is_dir() and not path.name.startswith("_")):
        summaries: Dict[str, Dict[str, Any]] = {}
        for path in dataset_dir.glob("*.summary.json"):
            variant = variant_for_path(path)
            if variant:
                summaries[variant] = json.loads(path.read_text(encoding="utf-8"))

        for variant, summary in summaries.items():
            trace_dir = latest_trace(dataset_dir, variant)
            ranking_rows: Dict[str, Dict[str, Any]] = {}
            llm_rows: Dict[str, Dict[str, Any]] = {}
            if trace_dir:
                ranking_rows = {
                    str(row.get("user_id")): row
                    for row in read_jsonl(trace_dir / "ranking_result.jsonl")
                }
                for row in read_jsonl(trace_dir / "ranking_llm.jsonl"):
                    # Keep the final attempt: retries contain the ranking that
                    # was actually returned to evaluation.
                    llm_rows[str(row.get("user_id"))] = row

            user_metrics = {
                user_id: float((row.get("metrics") or {}).get("ndcg@10", 0.0))
                for user_id, row in ranking_rows.items()
            }
            per_user[dataset_dir.name, variant] = user_metrics

            causal_help = causal_hurt = causal_tie = 0
            evidence_rows = changed_users = moved_candidates = 0
            for user_id, row in ranking_rows.items():
                audit = (llm_rows.get(user_id, {}).get("failure_constraint_audit") or {})
                evidence_rows += int(audit.get("num_evidence_rows", 0) or 0)
                moved = int(audit.get("num_moved_candidates", 0) or 0)
                moved_candidates += moved
                changed_users += moved > 0
                ground_truth = str((row.get("ground_truth") or [""])[0])
                base = audit.get("base_ranking") or row.get("ranked_item_ids") or []
                constrained = audit.get("constrained_ranking") or row.get("ranked_item_ids") or []
                delta = ndcg_at_10(constrained, ground_truth) - ndcg_at_10(base, ground_truth)
                causal_help += delta > 1e-12
                causal_hurt += delta < -1e-12
                causal_tie += abs(delta) <= 1e-12

            metrics = summary.get("metrics", {}) or {}
            usage = summary.get("llm_usage", {}) or {}
            users = int(summary.get("number_of_users_evaluated", 0) or 0)
            results.append({
                "dataset": dataset_dir.name,
                "variant": variant,
                "users": users,
                "ndcg@5": metrics.get("ndcg@5"),
                "ndcg@10": metrics.get("ndcg@10"),
                "ndcg@20": metrics.get("ndcg@20"),
                "calls_per_user": (float(usage.get("calls", 0)) / users if users else None),
                "tokens_per_user": (float(usage.get("total_tokens", 0)) / users if users else None),
                "evidence_rows": evidence_rows,
                "changed_users": changed_users,
                "moved_candidates": moved_candidates,
                "causal_help": causal_help,
                "causal_hurt": causal_hurt,
                "causal_tie": causal_tie,
            })

    for row in results:
        base = per_user.get((row["dataset"], "D0_profile_base"), {})
        current = per_user.get((row["dataset"], row["variant"]), {})
        common = sorted(set(base) & set(current))
        deltas = [current[user_id] - base[user_id] for user_id in common]
        row["delta_vs_d0"] = sum(deltas) / len(deltas) if deltas else None
        row["wins_vs_d0"] = sum(delta > 1e-12 for delta in deltas)
        row["losses_vs_d0"] = sum(delta < -1e-12 for delta in deltas)
        row["ties_vs_d0"] = sum(abs(delta) <= 1e-12 for delta in deltas)

    output = {"root": str(root), "rows": results}
    if args.json:
        Path(args.json).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.markdown:
        columns = [
            "dataset", "variant", "users", "ndcg@10", "delta_vs_d0",
            "wins_vs_d0", "losses_vs_d0", "evidence_rows", "changed_users",
            "causal_help", "causal_hurt", "calls_per_user", "tokens_per_user",
        ]
        lines = [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join(["---"] * len(columns)) + " |",
        ]
        for row in results:
            values = []
            for column in columns:
                value = row.get(column)
                values.append(f"{value:.4f}" if isinstance(value, float) else str(value if value is not None else ""))
            lines.append("| " + " | ".join(values) + " |")
        Path(args.markdown).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"root": str(root), "num_rows": len(results)}, indent=2))


if __name__ == "__main__":
    main()
