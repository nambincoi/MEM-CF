#!/usr/bin/env python3
"""Analyze matched-budget G-family collaborative-filtering controls."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


VARIANTS = [
    "G0_same_only",
    "G1_true_neighbor",
    "G2_shuffled_graph",
    "G3_random_neighbor",
    "G4_matched_random",
]

MODE_TO_VARIANT = {
    "g_same_only": "G0_same_only",
    "g_true_neighbor": "G1_true_neighbor",
    "g_shuffled_graph": "G2_shuffled_graph",
    "g_random_neighbor": "G3_random_neighbor",
    "g_matched_random": "G4_matched_random",
}


def read_jsonl(path: Path):
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def variant_from_text(text: str):
    lowered = text.lower()
    return next((variant for variant in VARIANTS if variant.lower() in lowered), None)


def variant_from_summary(summary: dict, summary_path: Path):
    """Prefer explicit run metadata over potentially shortened filenames."""
    mode = str(summary.get("failure_constraint_mode", "")).strip().lower()
    return MODE_TO_VARIANT.get(mode) or variant_from_text(summary_path.name)


def trace_dir(dataset_dir: Path, summary: dict, variant: str):
    recorded = summary.get("trace_dir")
    if recorded:
        candidate = Path(recorded)
        if candidate.exists():
            return candidate
        local = dataset_dir / "traces" / candidate.name
        if local.exists():
            return local
    root = dataset_dir / "traces"
    matches = [p for p in root.iterdir() if p.is_dir() and variant.lower() in p.name.lower()] if root.exists() else []
    return max(matches, key=lambda p: p.stat().st_mtime) if matches else None


def bootstrap_ci(values, samples=10000, seed=2027):
    if not values:
        return None
    rng = random.Random(seed)
    count = len(values)
    means = sorted(
        sum(values[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(samples)
    )
    return [means[int(samples * 0.025)], means[int(samples * 0.975) - 1]]


def mean(values):
    return sum(values) / len(values) if values else None


def fmt(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, list):
        return "[" + ", ".join(f"{x:+.6f}" for x in value) + "]"
    return str(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--markdown")
    parser.add_argument("--json")
    args = parser.parse_args()
    root = Path(args.root)

    rows = []
    per_user = {}
    exposed_users = {}
    for dataset_dir in sorted(path for path in root.iterdir() if path.is_dir() and not path.name.startswith("_")):
        latest_summaries = {}
        for summary_path in dataset_dir.glob("*.summary.json"):
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            variant = variant_from_summary(summary, summary_path)
            if not variant:
                continue
            previous = latest_summaries.get(variant)
            if previous is None or summary_path.stat().st_mtime > previous.stat().st_mtime:
                latest_summaries[variant] = summary_path
        for variant, summary_path in latest_summaries.items():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            trace = trace_dir(dataset_dir, summary, variant)
            ranking_rows = read_jsonl(trace / "ranking_result.jsonl") if trace else []
            evidence_rows = read_jsonl(trace / "typed_failure_evidence.jsonl") if trace else []
            ranking_by_user = {
                str(row.get("user_id")): float((row.get("metrics") or {}).get("ndcg@10", 0.0))
                for row in ranking_rows
            }
            per_user[dataset_dir.name, variant] = ranking_by_user

            audits = [row.get("cf_control_audit") or {} for row in evidence_rows]
            exposed = [audit for audit in audits if int(audit.get("selected_evidence_rows", 0)) > 0]
            eligible = [audit for audit in audits if int(audit.get("desired_sources", 0)) > 0]
            equal_budget = [audit for audit in eligible if audit.get("equal_budget_satisfied") is True]
            exposed_users[dataset_dir.name, variant] = {
                str(audit.get("user_id")) for audit in exposed if audit.get("user_id") is not None
            }
            jaccards = [
                float(value)
                for audit in audits
                for value in (audit.get("selected_jaccard") or {}).values()
            ]
            result_metrics = summary.get("metrics") or {}
            rows.append({
                "dataset": dataset_dir.name,
                "variant": variant,
                "users": len(ranking_by_user) or summary.get("users_evaluated") or summary.get("number_of_users_evaluated"),
                "ndcg@10": result_metrics.get("ndcg@10", result_metrics.get("NDCG@10")),
                "cf_audited_users": len(audits),
                "cf_exposed_users": len(exposed),
                "cf_exposure_rate": len(exposed) / len(audits) if audits else 0.0,
                "cf_eligible_users": len(eligible),
                "cf_eligible_rate": len(eligible) / len(audits) if audits else 0.0,
                "equal_budget_rate": len(equal_budget) / len(eligible) if eligible else 0.0,
                "mean_selected_sources": mean([float(audit.get("selected_source_count", 0)) for audit in audits]),
                "mean_selected_jaccard": mean(jaccards),
                "trace_dir": str(trace) if trace else "",
            })

    references = ["G0_same_only", "G2_shuffled_graph", "G3_random_neighbor", "G4_matched_random"]
    for row in rows:
        current = per_user.get((row["dataset"], row["variant"]), {})
        for reference in references:
            baseline = per_user.get((row["dataset"], reference), {})
            common = sorted(set(current) & set(baseline))
            deltas = [current[user] - baseline[user] for user in common]
            key = reference.split("_", 1)[0].lower()
            row[f"delta_vs_{key}"] = mean(deltas)
            row[f"ci95_vs_{key}"] = bootstrap_ci(deltas)
            row[f"wins_vs_{key}"] = sum(delta > 1e-12 for delta in deltas)
            row[f"losses_vs_{key}"] = sum(delta < -1e-12 for delta in deltas)

            g1_exposed = exposed_users.get((row["dataset"], "G1_true_neighbor"), set())
            exposed_common = sorted(set(current) & set(baseline) & g1_exposed)
            exposed_deltas = [current[user] - baseline[user] for user in exposed_common]
            row[f"g1_exposed_delta_vs_{key}"] = mean(exposed_deltas)
            row[f"g1_exposed_ci95_vs_{key}"] = bootstrap_ci(exposed_deltas)

    columns = [
        "dataset", "variant", "users", "ndcg@10", "cf_eligible_rate", "cf_exposure_rate",
        "equal_budget_rate", "mean_selected_sources", "mean_selected_jaccard",
        "delta_vs_g0", "ci95_vs_g0", "delta_vs_g2", "delta_vs_g3", "delta_vs_g4",
        "g1_exposed_delta_vs_g0", "g1_exposed_delta_vs_g2",
        "g1_exposed_delta_vs_g3", "g1_exposed_delta_vs_g4",
    ]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in sorted(rows, key=lambda item: (item["dataset"], VARIANTS.index(item["variant"]))):
        lines.append("| " + " | ".join(fmt(row.get(column)) for column in columns) + " |")

    report = "\n".join(lines) + "\n"
    payload = {"rows": rows, "variants": VARIANTS}
    if args.markdown:
        Path(args.markdown).write_text(report, encoding="utf-8")
    if args.json:
        Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
