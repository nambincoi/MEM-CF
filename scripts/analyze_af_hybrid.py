#!/usr/bin/env python3
"""Summarize AF hybrid results and the incremental CF effect AF3-AF1."""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

VARIANTS = [
    "AF0_profile_base", "AF1_strong_same_user", "AF2_strong_cross_only",
    "AF3_strong_same_plus_cf", "AF4_strong_same_plus_shuffled",
    "AF5_strong_same_plus_random", "AF6_strong_same_plus_polarity",
]


def read_jsonl(path: Path):
    if not path.exists(): return []
    rows = []
    for line in path.open(encoding="utf-8", errors="replace"):
        try: rows.append(json.loads(line))
        except json.JSONDecodeError: pass
    return rows


def variant_from_name(name: str):
    low = name.lower()
    return next((v for v in VARIANTS if v.lower() in low), None)


def latest_trace(dataset_dir: Path, variant: str):
    root = dataset_dir / "traces"
    matches = [p for p in root.iterdir() if p.is_dir() and variant.lower() in p.name.lower()] if root.exists() else []
    return max(matches, key=lambda p: p.stat().st_mtime) if matches else None


def trace_from_summary(dataset_dir: Path, summary: dict, variant: str):
    """Prefer the recorded trace path because long run names may be hashed."""
    recorded = summary.get("trace_dir")
    if recorded:
        path = Path(recorded)
        if path.exists():
            return path
        local = dataset_dir / "traces" / path.name
        if local.exists():
            return local
    return latest_trace(dataset_dir, variant)


def bootstrap_ci(values, samples=10000):
    if not values: return None
    rng = random.Random(20260713); n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(samples))
    return [means[int(samples * .025)], means[int(samples * .975) - 1]]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--markdown"); ap.add_argument("--json")
    args = ap.parse_args(); root = Path(args.root); rows = []; per_user = {}
    for ds in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        for sp in ds.glob("*.summary.json"):
            variant = variant_from_name(sp.name)
            if not variant: continue
            summary = json.load(sp.open()); trace = trace_from_summary(ds, summary, variant)
            ranking = {}
            if trace:
                ranking = {str(r.get("user_id")): r for r in read_jsonl(trace / "ranking_result.jsonl")}
            per_user[ds.name, variant] = {u: float((r.get("metrics") or {}).get("ndcg@10", 0)) for u, r in ranking.items()}
            rows.append({"dataset": ds.name, "variant": variant, "users": len(ranking), "ndcg@10": (summary.get("metrics") or {}).get("ndcg@10")})

    for row in rows:
        for ref, field in [("AF0_profile_base", "delta_vs_af0"), ("AF1_strong_same_user", "delta_vs_af1")]:
            a = per_user.get((row["dataset"], row["variant"]), {}); b = per_user.get((row["dataset"], ref), {})
            common = sorted(set(a) & set(b)); delta = [a[u] - b[u] for u in common]
            row[field] = sum(delta) / len(delta) if delta else None
            row[field + "_wins"] = sum(x > 1e-12 for x in delta)
            row[field + "_losses"] = sum(x < -1e-12 for x in delta)
            row[field + "_ci95"] = bootstrap_ci(delta)

    columns = ["dataset", "variant", "users", "ndcg@10", "delta_vs_af0", "delta_vs_af1", "delta_vs_af1_wins", "delta_vs_af1_losses", "delta_vs_af1_ci95"]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in sorted(rows, key=lambda r: (r["dataset"], VARIANTS.index(r["variant"]))):
        vals=[]
        for c in columns:
            v=row.get(c)
            if isinstance(v,float): vals.append(f"{v:.6f}")
            elif isinstance(v,list): vals.append("[" + ", ".join(f"{x:+.6f}" for x in v) + "]")
            else: vals.append(str(v if v is not None else ""))
        lines.append("| " + " | ".join(vals) + " |")
    text="\n".join(lines)+"\n"
    if args.markdown: Path(args.markdown).write_text(text)
    if args.json: Path(args.json).write_text(json.dumps({"rows":rows}, indent=2))
    print(text)


if __name__ == "__main__": main()
