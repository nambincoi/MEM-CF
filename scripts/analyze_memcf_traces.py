#!/usr/bin/env python3
"""Analyze MEMCF result JSONs and traces for memory-help/hurt diagnostics.

This script is intentionally read-only. It summarizes:
- summary metrics and LLM cost;
- per-user memory-vs-no-memory help/hurt counts;
- selected memory fact categories, especially wrong-only facts;
- trace-level source/path counts.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ndcg_rank(rank: Optional[int], k: int) -> float:
    if rank is None or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def ranking_records(path: Path) -> Dict[str, Tuple[List[str], List[str]]]:
    obj = load_json(path)
    rows = obj if isinstance(obj, list) else obj.get("results", [])
    out: Dict[str, Tuple[List[str], List[str]]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        user_id = str(row.get("user_id") or row.get("uid") or "")
        gt = [str(x) for x in row.get("ground_truth_item_ids", row.get("ground_truth", []))]
        pred = [str(x) for x in row.get("reranked_item_ids", row.get("predictions", []))]
        if user_id and gt and pred:
            out[user_id] = (gt, pred)
    return out


def rank_of(gt: Iterable[str], pred: List[str]) -> Optional[int]:
    gt_set = set(str(x) for x in gt)
    for i, item in enumerate(pred, start=1):
        if item in gt_set:
            return i
    return None


def infer_variant(path: Path) -> str:
    name = path.name
    lower = name.lower()
    if "hybrid_cluster_strict" in lower:
        return "hybrid_cluster_strict"
    if "hybrid_cluster" in lower:
        return "hybrid_cluster"
    if "cluster_user" in lower:
        return "cluster_user"
    if "cluster_full" in lower:
        return "cluster_full"
    if "random_cluster" in lower:
        return "random_cluster"
    if "shuffled_cluster" in lower:
        return "shuffled_cluster"
    if "nomemory" in name:
        return "A0_no_memory"
    if "gk3" in name:
        return "gk3"
    if "gk5" in name:
        return "gk5"
    if "same_user" in name:
        return "same_user"
    if "candidate_item" in name:
        return "candidate_item"
    if "neighbor_user" in name:
        return "neighbor_user"
    if "random" in name:
        return "random_memory"
    if "shuffled" in name:
        return "shuffled_memory"
    return path.stem.replace(".summary", "")


def find_ranking_for_summary(summary_path: Path) -> Optional[Path]:
    candidate = summary_path.with_name(summary_path.name.replace(".summary.json", ".json"))
    if candidate.exists():
        return candidate
    return None


def find_no_memory_ranking(dataset_dir: Path) -> Optional[Path]:
    candidates = [
        p for p in dataset_dir.glob("*.json")
        if "nomemory" in p.name and not p.name.endswith(".summary.json")
    ]
    return sorted(candidates)[0] if candidates else None


def trace_dir_from_summary(summary: Mapping[str, Any], dataset_dir: Path) -> Optional[Path]:
    trace = summary.get("trace_dir")
    if trace:
        p = Path(str(trace))
        if p.exists():
            return p
    # Fallback: match by run-name suffix if trace_dir was not recorded.
    return None


def summarize_trace(trace_dir: Optional[Path]) -> Dict[str, Any]:
    if not trace_dir or not trace_dir.exists():
        return {}
    selected = trace_dir / "memory_facts_selected.jsonl"
    out: Dict[str, Any] = {}
    source_counts = Counter()
    counts = Counter()
    users = 0
    if selected.exists():
        with selected.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                users += 1
                rows = obj.get("selected_rows") or obj.get("result", {}).get("selected_rows") or []
                rejected = obj.get("rejected_rows") or obj.get("result", {}).get("rejected_rows") or []
                for row in rows:
                    counts["selected_facts"] += 1
                    direct_correct = bool(row.get("direct_correct_candidate"))
                    direct_wrong = bool(row.get("direct_wrong_candidate"))
                    if direct_correct:
                        counts["selected_direct_correct"] += 1
                    if direct_wrong:
                        counts["selected_direct_wrong"] += 1
                    if direct_wrong and not direct_correct:
                        counts["selected_wrong_only"] += 1
                    if not direct_wrong and not direct_correct:
                        counts["selected_neither_direct"] += 1
                    for source in row.get("sources", []):
                        source_counts[f"selected_source_{source}"] += 1
                for row in rejected:
                    counts["rejected_facts"] += 1
                    direct_correct = bool(row.get("direct_correct_candidate"))
                    direct_wrong = bool(row.get("direct_wrong_candidate"))
                    if direct_wrong and not direct_correct:
                        counts["rejected_wrong_only"] += 1
    ranking_llm = trace_dir / "ranking_llm.jsonl"
    if ranking_llm.exists():
        valid = invalid = 0
        with ranking_llm.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if obj.get("parsed") or obj.get("parsed_ranking") or obj.get("scores"):
                    valid += 1
                if "error" in str(obj).lower() or "extract error" in str(obj).lower():
                    invalid += 1
        out["ranking_llm_records"] = valid + invalid
        out["ranking_llm_error_like_records"] = invalid
    out.update({k: int(v) for k, v in counts.items()})
    out.update({k: int(v) for k, v in source_counts.items()})
    out["memory_fact_users"] = users
    return out


def analyze_root(root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for summary_path in sorted(root.glob("*/*.summary.json")):
        dataset_dir = summary_path.parent
        dataset = dataset_dir.name
        summary = load_json(summary_path)
        ranking = find_ranking_for_summary(summary_path)
        no_memory = find_no_memory_ranking(dataset_dir)
        variant = infer_variant(summary_path)
        metrics = summary.get("metrics", {}) or {}
        diag = summary.get("memory_diagnostics", {}) or {}
        llm = summary.get("llm_usage", {}) or {}
        runtime = summary.get("runtime", {}) or {}
        row: Dict[str, Any] = {
            "dataset": dataset,
            "variant": variant,
            "summary_path": str(summary_path.relative_to(root)),
            "ranking_path": str(ranking.relative_to(root)) if ranking else "",
            "users": summary.get("number_of_users_evaluated"),
            "ndcg@5": metrics.get("ndcg@5"),
            "ndcg@10": metrics.get("ndcg@10"),
            "ndcg@20": metrics.get("ndcg@20"),
            "recall@5": metrics.get("recall@5"),
            "recall@10": metrics.get("recall@10"),
            "recall@20": metrics.get("recall@20"),
            "num_graph_lessons": summary.get("num_graph_lessons"),
            "no_harm_use_rate": diag.get("no_harm_memory_use_rate"),
            "rank_invalid_score_outputs": diag.get("rank_invalid_score_outputs"),
            "llm_calls": llm.get("calls", llm.get("total_calls")),
            "llm_tokens": llm.get("total_tokens"),
            "runtime_seconds": runtime.get("total_seconds"),
        }
        if ranking and no_memory and ranking != no_memory:
            base = ranking_records(no_memory)
            mem = ranking_records(ranking)
            users = sorted(set(base) & set(mem))
            help10 = hurt10 = same10 = 0
            help5 = hurt5 = same5 = 0
            delta10: List[float] = []
            delta5: List[float] = []
            top5_up = top5_down = 0
            for user in users:
                gt0, pred0 = base[user]
                gt1, pred1 = mem[user]
                gt = gt0 or gt1
                br = rank_of(gt, pred0)
                mr = rank_of(gt, pred1)
                d10 = ndcg_rank(mr, 10) - ndcg_rank(br, 10)
                d5 = ndcg_rank(mr, 5) - ndcg_rank(br, 5)
                delta10.append(d10)
                delta5.append(d5)
                if d10 > 1e-12:
                    help10 += 1
                elif d10 < -1e-12:
                    hurt10 += 1
                else:
                    same10 += 1
                if d5 > 1e-12:
                    help5 += 1
                elif d5 < -1e-12:
                    hurt5 += 1
                else:
                    same5 += 1
                if (br is None or br > 5) and mr is not None and mr <= 5:
                    top5_up += 1
                if br is not None and br <= 5 and (mr is None or mr > 5):
                    top5_down += 1
            row.update({
                "paired_users": len(users),
                "help@10": help10,
                "hurt@10": hurt10,
                "same@10": same10,
                "mean_delta_ndcg@10": sum(delta10) / len(delta10) if delta10 else None,
                "help@5": help5,
                "hurt@5": hurt5,
                "same@5": same5,
                "mean_delta_ndcg@5": sum(delta5) / len(delta5) if delta5 else None,
                "top5_up": top5_up,
                "top5_down": top5_down,
            })
        row.update(summarize_trace(trace_dir_from_summary(summary, dataset_dir)))
        rows.append(row)
    return rows


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


def write_markdown(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "dataset", "variant", "users", "ndcg@5", "ndcg@10", "ndcg@20",
        "mean_delta_ndcg@10", "help@10", "hurt@10", "top5_up", "top5_down",
        "selected_facts", "selected_wrong_only", "selected_direct_correct",
        "selected_source_cluster_user", "selected_source_random_cluster", "selected_source_shuffled_cluster",
        "no_harm_use_rate", "rank_invalid_score_outputs", "llm_calls", "llm_tokens",
    ]
    with path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("| " + " | ".join(["---"] * len(cols)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(fmt(row.get(c)) for c in cols) + " |\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--markdown", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    rows = analyze_root(args.root)
    print(json.dumps({"root": str(args.root), "rows": len(rows)}, indent=2))
    if args.csv:
        write_csv(rows, args.csv)
        print(f"Saved CSV: {args.csv}")
    if args.markdown:
        write_markdown(rows, args.markdown)
        print(f"Saved Markdown: {args.markdown}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Saved JSON: {args.json}")


if __name__ == "__main__":
    main()
