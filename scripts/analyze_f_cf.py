#!/usr/bin/env python3
"""Analyze F-family collaborative failure-routing experiments."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


VARIANTS = [
    "F0_profile_base",
    "F1_same_user_exact",
    "F2_shared_item_cross_only",
    "F3_same_plus_shared_cf",
    "F4_shuffled_neighbors",
    "F5_random_neighbors",
    "F6_cross_polarity_swapped",
    "F7_popularity_control",
]
TRACE_TAGS = {
    "F0_profile_base": "profileonly",
    "F1_same_user_exact": "d1same",
    "F2_shared_item_cross_only": "f2shared",
    "F3_same_plus_shared_cf": "f3full",
    "F4_shuffled_neighbors": "f4shuf",
    "F5_random_neighbors": "f5rand",
    "F6_cross_polarity_swapped": "f6swap",
    "F7_popularity_control": "d6pop",
}


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def variant_for_path(path: Path) -> Optional[str]:
    return next((variant for variant in VARIANTS if variant in path.name), None)


def latest_trace(dataset_dir: Path, variant: str) -> Optional[Path]:
    trace_root = dataset_dir / "traces"
    if not trace_root.exists():
        return None
    tag = TRACE_TAGS[variant]
    matches = [path for path in trace_root.iterdir() if path.is_dir() and tag in path.name]
    return max(matches, key=lambda path: path.stat().st_mtime) if matches else None


def ndcg_at_10(ranking: List[str], ground_truth: str) -> float:
    try:
        rank = ranking.index(ground_truth) + 1
    except ValueError:
        return 0.0
    return 0.0 if rank > 10 else 1.0 / math.log2(rank + 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--markdown")
    parser.add_argument("--json")
    args = parser.parse_args()

    root = Path(args.root)
    rows: List[Dict[str, Any]] = []
    per_user: Dict[tuple[str, str], Dict[str, float]] = {}
    for dataset_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        for summary_path in dataset_dir.glob("*.summary.json"):
            variant = variant_for_path(summary_path)
            if not variant:
                continue
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            trace_dir = latest_trace(dataset_dir, variant)
            ranking_rows = {}
            llm_rows = {}
            if trace_dir:
                ranking_rows = {str(r.get("user_id")): r for r in read_jsonl(trace_dir / "ranking_result.jsonl")}
                for trace_row in read_jsonl(trace_dir / "ranking_llm.jsonl"):
                    llm_rows[str(trace_row.get("user_id"))] = trace_row
            per_user[dataset_dir.name, variant] = {
                uid: float((r.get("metrics") or {}).get("ndcg@10", 0.0))
                for uid, r in ranking_rows.items()
            }

            evidence = changed = help_count = hurt_count = true_cf = false_cf = 0
            for uid, result_row in ranking_rows.items():
                audit = (llm_rows.get(uid, {}).get("failure_constraint_audit") or {})
                evidence += int(audit.get("num_evidence_rows", 0) or 0)
                changed += int(audit.get("num_moved_candidates", 0) or 0) > 0
                active_evidence = [
                    e for signal in (audit.get("candidate_signals") or {}).values()
                    for e in (signal.get("evidence") or [])
                ]
                true_cf += sum(bool(e.get("cf_source_is_true_neighbor")) and not e.get("same_user") for e in active_evidence)
                false_cf += sum(e.get("cf_source_is_true_neighbor") is False and not e.get("same_user") for e in active_evidence)
                gt = str((result_row.get("ground_truth") or [""])[0])
                before = audit.get("base_ranking") or result_row.get("ranked_item_ids") or []
                after = audit.get("constrained_ranking") or result_row.get("ranked_item_ids") or []
                delta = ndcg_at_10(after, gt) - ndcg_at_10(before, gt)
                help_count += delta > 1e-12
                hurt_count += delta < -1e-12

            metrics = summary.get("metrics", {}) or {}
            users = int(summary.get("number_of_users_evaluated", 0) or 0)
            rows.append({
                "dataset": dataset_dir.name,
                "variant": variant,
                "users": users,
                "ndcg@10": metrics.get("ndcg@10"),
                "evidence_rows": evidence,
                "changed_users": changed,
                "causal_help": help_count,
                "causal_hurt": hurt_count,
                "true_cf_evidence": true_cf,
                "control_non_neighbor_evidence": false_cf,
            })

    for row in rows:
        for reference, field in (("F0_profile_base", "delta_vs_f0"), ("F1_same_user_exact", "delta_vs_f1")):
            base = per_user.get((row["dataset"], reference), {})
            current = per_user.get((row["dataset"], row["variant"]), {})
            common = sorted(set(base) & set(current))
            deltas = [current[uid] - base[uid] for uid in common]
            row[field] = sum(deltas) / len(deltas) if deltas else None
            row[field.replace("delta", "wins")] = sum(d > 1e-12 for d in deltas)
            row[field.replace("delta", "losses")] = sum(d < -1e-12 for d in deltas)

    output = {"root": str(root), "rows": rows}
    if args.json:
        Path(args.json).write_text(json.dumps(output, indent=2), encoding="utf-8")
    columns = [
        "dataset", "variant", "users", "ndcg@10", "delta_vs_f0", "delta_vs_f1",
        "causal_help", "causal_hurt", "changed_users", "evidence_rows",
        "true_cf_evidence", "control_non_neighbor_evidence",
    ]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in sorted(rows, key=lambda r: (r["dataset"], VARIANTS.index(r["variant"]))):
        values = []
        for column in columns:
            value = row.get(column)
            values.append(f"{value:.6f}" if isinstance(value, float) else str(value if value is not None else ""))
        lines.append("| " + " | ".join(values) + " |")
    markdown = "\n".join(lines) + "\n"
    if args.markdown:
        Path(args.markdown).write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
