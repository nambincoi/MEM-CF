#!/usr/bin/env python3
"""Clean an existing MEMCF runtime dataset without changing its user split.

This is intended for fair reruns when the split/candidates are already fixed,
but item metadata contains HTML entities, raw list-like descriptions, or noisy
category fields. The script copies user sequence and negative files unchanged
and writes a cleaned items.json.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List


NOISE_RE = re.compile(r"<[^>]+>|&[A-Za-z]+;|a-size-|a-color-|span class|h1 class", re.I)


def flatten_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(flatten_text(x) for x in value if flatten_text(x))
    if isinstance(value, dict):
        return " ".join(flatten_text(x) for x in value.values() if flatten_text(x))
    return str(value)


def clean_text(value: Any, max_chars: int | None = None) -> str:
    text = flatten_text(value)
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(
        r"\b(Product Description|Product description|Amazon\.com|No Description Available|Media Type:|Artist:|Title:)\b",
        " ",
        text,
        flags=re.I,
    )
    text = re.sub(r"\s+", " ", text).strip(" []'\",")
    if max_chars and len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0].rstrip(" .,;:") + "..."
    return text


def clean_category(item: Dict[str, Any], fallback: str) -> str:
    raw = item.get("main_cat") or item.get("category") or item.get("categories") or fallback
    text = clean_text(raw, max_chars=120)
    if not text or text.lower() in {"unknown", "none", "nan", "[]"}:
        return fallback
    return text


def infer_fallback_category(dataset: str) -> str:
    name = dataset.lower()
    if "beauty" in name:
        return "All Beauty"
    if "cd" in name or "vinyl" in name:
        return "CDs and Vinyl"
    if "music" in name:
        return "Digital Music"
    if "video" in name or "game" in name:
        return "Video Games"
    return "Unknown"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def iter_referenced_items(sequences: Dict[str, Any], negatives: Dict[str, Any]) -> Iterable[str]:
    for user_id, seq in sequences.items():
        if isinstance(seq, dict):
            for key in ("train", "val", "test"):
                for item_id in seq.get(key, []) or []:
                    yield str(item_id)
        neg = negatives.get(user_id, {})
        if isinstance(neg, dict):
            for key in ("val_neg", "test_neg", "train_neg", "negatives"):
                for item_id in neg.get(key, []) or []:
                    yield str(item_id)


def audit_items(items: Dict[str, Dict[str, Any]], refs: List[str]) -> Dict[str, Any]:
    rows = [items[str(x)] for x in refs if str(x) in items]
    if not rows:
        return {"referenced_items": 0}
    title_noisy = sum(bool(NOISE_RE.search(str(r.get("title", "")))) for r in rows)
    desc_noisy = sum(bool(NOISE_RE.search(str(r.get("description", "")))) for r in rows)
    unknown_cat = sum((not r.get("main_cat") or str(r.get("main_cat")).lower() == "unknown") for r in rows)
    return {
        "referenced_items": len(rows),
        "title_noise_rate": title_noisy / len(rows),
        "description_noise_rate": desc_noisy / len(rows),
        "unknown_category_rate": unknown_cat / len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean MEMCF runtime dataset metadata while preserving split files.")
    parser.add_argument("--runtime_root", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out_dataset", default=None)
    parser.add_argument("--description_chars", type=int, default=260)
    parser.add_argument("--copy_alias_files", action="store_true")
    args = parser.parse_args()

    runtime_root = Path(args.runtime_root)
    in_dir = runtime_root / args.dataset
    out_dataset = args.out_dataset or f"{args.dataset}_clean"
    out_dir = runtime_root / out_dataset

    required = ["items.json", "user_sequences_10.json", "user_negatives_10.json"]
    missing = [name for name in required if not (in_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing {missing} in {in_dir}")

    items = load_json(in_dir / "items.json")
    sequences = load_json(in_dir / "user_sequences_10.json")
    negatives = load_json(in_dir / "user_negatives_10.json")
    refs = list(iter_referenced_items(sequences, negatives))
    before = audit_items(items, refs)

    fallback_category = infer_fallback_category(args.dataset)
    cleaned: Dict[str, Dict[str, Any]] = {}
    for item_id, item in items.items():
        if not isinstance(item, dict):
            item = {}
        title = clean_text(item.get("title") or f"Item {item_id}", max_chars=180) or f"Item {item_id}"
        description = clean_text(
            item.get("description") or item.get("feature") or item.get("description_short") or "",
            max_chars=args.description_chars,
        )
        category = clean_category(item, fallback=fallback_category)
        cleaned[str(item_id)] = {
            **item,
            "title": title,
            "description": description,
            "description_short": description,
            "main_cat": category,
            "category": category,
            "metadata_cleaned": True,
        }

    after = audit_items(cleaned, refs)

    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ["user_sequences_10.json", "user_negatives_10.json"]:
        shutil.copy2(in_dir / name, out_dir / name)
    if args.copy_alias_files:
        for name in ["user_sequences.json", "user_negatives.json", "user_sequences_5.json", "user_negatives_5.json"]:
            if (in_dir / name).exists():
                shutil.copy2(in_dir / name, out_dir / name)
    save_json(cleaned, out_dir / "items.json")
    audit = {
        "input_dir": str(in_dir),
        "output_dir": str(out_dir),
        "dataset": args.dataset,
        "out_dataset": out_dataset,
        "num_items": len(items),
        "num_users": len(sequences),
        "fallback_category": fallback_category,
        "before": before,
        "after": after,
    }
    save_json(audit, out_dir / "clean_audit.json")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
