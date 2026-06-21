#!/usr/bin/env python3
"""Paired statistical comparison for MEMCF ranking JSON files.

The script compares no-memory and memory rankings on matched users and reports:
- mean Recall/NDCG per top-k
- absolute and relative uplift
- paired win/loss/tie counts per user
- sign-test p-value
- paired bootstrap 95% confidence interval for mean delta
- paired randomization/permutation p-value for mean delta

It accepts the ranking JSON formats produced by MEMCF and similar candidate rerankers.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def first_present(obj: Mapping[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for key in keys:
        if key in obj:
            return obj[key]
    return default


def as_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(v) for v in x]
    if isinstance(x, tuple):
        return [str(v) for v in x]
    return [str(x)]


def dedup(seq: Iterable[Any]) -> List[str]:
    out: List[str] = []
    seen = set()
    for x in seq:
        sx = str(x)
        if sx not in seen:
            seen.add(sx)
            out.append(sx)
    return out


def normalize_record(rec: Mapping[str, Any]) -> Tuple[str, List[str], List[str]]:
    user_id = str(first_present(rec, ["user_id", "uid", "user"], ""))
    gt = as_list(first_present(rec, ["ground_truth_item_ids", "ground_truth", "answer", "answers", "gt_items", "target_item_ids"]))
    pred = first_present(rec, ["reranked_item_ids", "predictions", "ranking", "ranked_item_ids", "llm_ranking_list", "recommendations"])
    if pred is None and "metrics" in rec and "predictions" in rec["metrics"]:
        pred = rec["metrics"]["predictions"]
    return user_id, dedup(gt), dedup(as_list(pred))


def load_rankings(path: Path) -> Dict[str, Tuple[List[str], List[str]]]:
    obj = load_json(path)
    if isinstance(obj, dict):
        if "results" in obj and isinstance(obj["results"], list):
            records = obj["results"]
        elif "rankings" in obj and isinstance(obj["rankings"], list):
            records = obj["rankings"]
        else:
            records = list(obj.values())
    elif isinstance(obj, list):
        records = obj
    else:
        raise TypeError(f"Unsupported ranking JSON root in {path}: {type(obj)}")

    rankings: Dict[str, Tuple[List[str], List[str]]] = {}
    for i, rec in enumerate(records):
        if not isinstance(rec, Mapping):
            continue
        user_id, gt, pred = normalize_record(rec)
        if not user_id:
            user_id = f"__row_{i}"
        if gt and pred:
            rankings[user_id] = (gt, pred)
    return rankings


def recall_at_k(gt: Sequence[str], pred: Sequence[str], k: int) -> float:
    gt_set = set(gt)
    if not gt_set:
        return 0.0
    hits = sum(1 for item in pred[:k] if item in gt_set)
    return hits / len(gt_set)


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


def metric_value(metric: str, gt: Sequence[str], pred: Sequence[str], k: int) -> float:
    if metric == "recall":
        return recall_at_k(gt, pred, k)
    if metric == "ndcg":
        return ndcg_at_k(gt, pred, k)
    raise ValueError(metric)


def sign_test_p_value(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    smaller = min(wins, losses)
    # two-sided exact binomial with p=0.5
    prob = sum(math.comb(n, i) for i in range(0, smaller + 1)) / (2 ** n)
    return min(1.0, 2 * prob)


def percentile(xs: Sequence[float], q: float) -> float:
    if not xs:
        return float("nan")
    vals = sorted(xs)
    pos = (len(vals) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def bootstrap_ci(deltas: Sequence[float], samples: int, seed: int) -> Tuple[float, float]:
    if not deltas:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(deltas)
    means = []
    for _ in range(samples):
        means.append(sum(deltas[rng.randrange(n)] for _ in range(n)) / n)
    return percentile(means, 0.025), percentile(means, 0.975)


def permutation_p_value(deltas: Sequence[float], samples: int, seed: int) -> float:
    if not deltas:
        return 1.0
    observed = abs(mean(deltas))
    rng = random.Random(seed + 17)
    count = 0
    for _ in range(samples):
        perm_mean = sum((d if rng.random() < 0.5 else -d) for d in deltas) / len(deltas)
        if abs(perm_mean) >= observed:
            count += 1
    return (count + 1) / (samples + 1)


def compare(no_mem: Dict[str, Tuple[List[str], List[str]]], mem: Dict[str, Tuple[List[str], List[str]]], topks: Sequence[int], bootstrap: int, seed: int) -> Dict[str, Any]:
    users = sorted(set(no_mem) & set(mem))
    out: Dict[str, Any] = {"matched_users": len(users), "metrics": {}}
    for metric in ["recall", "ndcg"]:
        for k in topks:
            base_vals = []
            mem_vals = []
            deltas = []
            wins = losses = ties = 0
            for u in users:
                gt0, pred0 = no_mem[u]
                gt1, pred1 = mem[u]
                gt = gt0 or gt1
                b = metric_value(metric, gt, pred0, k)
                m = metric_value(metric, gt, pred1, k)
                d = m - b
                base_vals.append(b)
                mem_vals.append(m)
                deltas.append(d)
                eps = 1e-12
                if d > eps:
                    wins += 1
                elif d < -eps:
                    losses += 1
                else:
                    ties += 1
            base_mean = mean(base_vals) if base_vals else float("nan")
            mem_mean = mean(mem_vals) if mem_vals else float("nan")
            delta_mean = mean(deltas) if deltas else float("nan")
            ci_lo, ci_hi = bootstrap_ci(deltas, bootstrap, seed)
            key = f"{metric}@{k}"
            out["metrics"][key] = {
                "no_memory_mean": base_mean,
                "memory_mean": mem_mean,
                "absolute_uplift": delta_mean,
                "relative_uplift_percent": (delta_mean / base_mean * 100.0) if base_mean else float("nan"),
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "win_rate_excluding_ties": wins / (wins + losses) if (wins + losses) else float("nan"),
                "sign_test_p_two_sided": sign_test_p_value(wins, losses),
                "bootstrap_ci95": [ci_lo, ci_hi],
                "permutation_p_two_sided": permutation_p_value(deltas, bootstrap, seed),
            }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Paired MEMCF memory-vs-no-memory statistics")
    ap.add_argument("--no_memory_json", required=True, type=Path)
    ap.add_argument("--memory_json", required=True, type=Path)
    ap.add_argument("--name", default="MEMCF_vs_no_memory")
    ap.add_argument("--topk", default="5,10,20")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=2020)
    ap.add_argument("--output_json", type=Path, default=None)
    args = ap.parse_args()

    topks = [int(x) for x in args.topk.split(",") if x.strip()]
    no_mem = load_rankings(args.no_memory_json)
    mem = load_rankings(args.memory_json)
    result = {
        "comparison": args.name,
        "no_memory_json": str(args.no_memory_json),
        "memory_json": str(args.memory_json),
        **compare(no_mem, mem, topks, args.bootstrap, args.seed),
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
