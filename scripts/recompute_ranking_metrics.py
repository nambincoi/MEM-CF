#!/usr/bin/env python3
"""Recompute ranking metrics from saved ranking JSON files.

This is a post-hoc utility: it does not call the LLM.  It reads saved
per-user rankings, recomputes Hit/Recall/NDCG at arbitrary cutoffs, and writes
CSV/Markdown tables.  Optionally, it can write the recomputed metrics back into
matching *.summary.json files.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def first_present(obj: Mapping[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for key in keys:
        if key in obj:
            return obj[key]
    return default


def as_str_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, tuple):
        return [str(x) for x in value]
    return [str(value)]


def dedup(values: Iterable[Any]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        s = str(value)
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def records_from_ranking(path: Path) -> List[Tuple[str, List[str], List[str]]]:
    obj = load_json(path)
    if isinstance(obj, list):
        rows = obj
    elif isinstance(obj, dict):
        if isinstance(obj.get("results"), list):
            rows = obj["results"]
        elif isinstance(obj.get("rankings"), list):
            rows = obj["rankings"]
        else:
            rows = list(obj.values())
    else:
        raise TypeError(f"Unsupported JSON root in {path}: {type(obj)}")

    records: List[Tuple[str, List[str], List[str]]] = []
    for i, row in enumerate(rows):
        if not isinstance(row, Mapping):
            continue
        user_id = str(first_present(row, ["user_id", "uid", "user"], f"__row_{i}"))
        gt = as_str_list(first_present(
            row,
            ["ground_truth_item_ids", "ground_truth", "answer", "answers", "gt_items", "target_item_ids"],
        ))
        pred = first_present(
            row,
            ["reranked_item_ids", "predictions", "ranking", "ranked_item_ids", "llm_ranking_list", "recommendations"],
        )
        if pred is None and isinstance(row.get("metrics"), Mapping):
            pred = row["metrics"].get("predictions")
        gt = dedup(gt)
        pred = dedup(as_str_list(pred))
        if gt and pred:
            records.append((user_id, gt, pred))
    return records


def hit_at_k(gt: Sequence[str], pred: Sequence[str], k: int) -> float:
    gt_set = set(gt)
    return 1.0 if any(item in gt_set for item in pred[:k]) else 0.0


def recall_at_k(gt: Sequence[str], pred: Sequence[str], k: int) -> float:
    gt_set = set(gt)
    if not gt_set:
        return 0.0
    return sum(1 for item in pred[:k] if item in gt_set) / len(gt_set)


def ndcg_at_k(gt: Sequence[str], pred: Sequence[str], k: int) -> float:
    gt_set = set(gt)
    if not gt_set:
        return 0.0
    dcg = 0.0
    for rank, item in enumerate(pred[:k], start=1):
        if item in gt_set:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(gt_set), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def mean(values: Sequence[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def compute_metrics(records: Sequence[Tuple[str, List[str], List[str]]], topks: Sequence[int]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for k in topks:
        metrics[f"hit@{k}"] = mean([hit_at_k(gt, pred, k) for _, gt, pred in records]) or 0.0
        metrics[f"recall@{k}"] = mean([recall_at_k(gt, pred, k) for _, gt, pred in records]) or 0.0
        metrics[f"ndcg@{k}"] = mean([ndcg_at_k(gt, pred, k) for _, gt, pred in records]) or 0.0
    return metrics


def find_ranking_for_summary(summary_path: Path) -> Optional[Path]:
    candidate = summary_path.with_name(summary_path.name.replace(".summary.json", ".json"))
    return candidate if candidate.exists() else None


def iter_inputs(root: Path) -> Iterable[Tuple[Optional[Path], Path]]:
    summaries = sorted(root.rglob("*.summary.json"))
    seen_rankings = set()
    for summary in summaries:
        ranking = find_ranking_for_summary(summary)
        if ranking:
            seen_rankings.add(ranking)
            yield summary, ranking

    # Also include ranking JSONs that do not have a summary next to them.
    for ranking in sorted(root.rglob("*.json")):
        if ranking.name.endswith(".summary.json") or ranking in seen_rankings:
            continue
        try:
            records = records_from_ranking(ranking)
        except Exception:
            continue
        if not records:
            continue
        yield None, ranking


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: List[Dict[str, Any]], path: Path, metric_cols: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["dataset", "run_name", "users"] + list(metric_cols)
    with path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("| " + " | ".join(["---"] * len(cols)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(fmt(row.get(c)) for c in cols) + " |\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path, help="Root containing ranking JSON files and summaries")
    parser.add_argument("--topk", default="1,3,5,10,20", help="Comma-separated cutoffs")
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--markdown", type=Path, default=None)
    parser.add_argument("--write_summary", action="store_true", help="Write recomputed metrics into matching *.summary.json files")
    args = parser.parse_args()

    topks = [int(x) for x in args.topk.split(",") if x.strip()]
    metric_cols = [f"{name}@{k}" for k in topks for name in ("hit", "recall", "ndcg")]

    rows: List[Dict[str, Any]] = []
    for summary_path, ranking_path in iter_inputs(args.root):
        records = records_from_ranking(ranking_path)
        if not records:
            continue
        metrics = compute_metrics(records, topks)

        summary: Dict[str, Any] = {}
        if summary_path:
            summary = load_json(summary_path)
            if args.write_summary:
                merged = dict(summary.get("metrics", {}) or {})
                merged.update(metrics)
                summary["metrics"] = merged
                summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        row: Dict[str, Any] = {
            "dataset": summary.get("dataset_name") or ranking_path.parent.name,
            "run_name": summary.get("run_name") or ranking_path.stem,
            "users": len(records),
            "summary_path": str(summary_path.relative_to(args.root)) if summary_path else "",
            "ranking_path": str(ranking_path.relative_to(args.root)),
        }
        row.update(metrics)
        rows.append(row)

    rows.sort(key=lambda r: (str(r.get("dataset")), str(r.get("run_name"))))

    if args.csv:
        write_csv(rows, args.csv)
        print(f"Saved CSV: {args.csv}")
    if args.markdown:
        write_markdown(rows, args.markdown, metric_cols)
        print(f"Saved Markdown: {args.markdown}")
    print(json.dumps({"root": str(args.root), "num_rankings": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
