#!/usr/bin/env python3
"""Build a MEMCF runtime dataset from raw Amazon ratings + metadata.

Input:
  - ratings CSV in Amazon review format: item_id,user_id,rating,timestamp
  - metadata JSONL or JSONL.GZ with Amazon fields such as asin/title/category

Output:
  - items.json
  - user_sequences_10.json
  - user_negatives_10.json
  - compatibility copies: user_sequences.json/user_negatives.json and _5 files

The split and negative protocol matches the existing MEMCF runtime datasets:
users are sorted chronologically, last two interactions are validation/test,
and each eval split gets popularity-weighted sampled negatives.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import html
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


ROOT_CATEGORY_STOPWORDS = {
    "amazon",
    "all",
    "all departments",
    "appliances",
    "apps & games",
    "arts, crafts & sewing",
    "automotive",
    "baby",
    "beauty",
    "books",
    "cds & vinyl",
    "cell phones & accessories",
    "clothing, shoes & jewelry",
    "digital music",
    "electronics",
    "grocery & gourmet food",
    "health & personal care",
    "home & kitchen",
    "industrial & scientific",
    "industrial and scientific",
    "movies & tv",
    "musical instruments",
    "office products",
    "patio, lawn & garden",
    "pet supplies",
    "prime pantry",
    "software",
    "sports & outdoors",
    "tools & home improvement",
    "toys & games",
    "video games",
}

NOISY_TEXT_RE = re.compile(
    r"<[^>]+>|a-size-|a-color-|nav-categ-image|sprite|logo|transparent-pixel",
    re.I,
)
SPACE_RE = re.compile(r"\s+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MEMCF runtime data from raw Amazon data.")
    parser.add_argument("--ratings", required=True, help="Raw ratings CSV path.")
    parser.add_argument("--meta", required=True, help="Raw metadata JSONL or JSONL.GZ path.")
    parser.add_argument("--output_dir", required=True, help="Runtime output directory.")
    parser.add_argument("--dataset_name", required=True, help="Name used in rebuild_audit.json.")
    parser.add_argument("--max_users", type=int, default=1000)
    parser.add_argument("--min_user_items", type=int, default=5)
    parser.add_argument("--neg_num", type=int, default=19)
    parser.add_argument("--user_sample_seed", type=int, default=42)
    parser.add_argument("--neg_seed", type=int, default=42)
    parser.add_argument("--description_chars", type=int, default=320)
    parser.add_argument(
        "--require_meta_gzip_valid",
        action="store_true",
        help="Fail before processing if --meta has .gz extension but is not valid gzip.",
    )
    return parser.parse_args()


def flatten_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(x for x in (flatten_text(v) for v in value) if x)
    if isinstance(value, dict):
        return " ".join(x for x in (flatten_text(v) for v in value.values()) if x)
    return str(value)


def clean_text(value: Any, max_chars: int | None = None) -> str:
    text = html.unescape(flatten_text(value))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(
        r"\b(Product Description|Amazon\.com|No Description Available|Editorial Reviews?)\b",
        " ",
        text,
        flags=re.I,
    )
    text = SPACE_RE.sub(" ", text).strip(" []'\",;/")
    if max_chars and len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0].rstrip(" .,;:") + "..."
    return text


def normalize_category_parts(raw_category: Any) -> List[str]:
    values = raw_category if isinstance(raw_category, list) else [raw_category]
    parts: List[str] = []
    for value in values:
        nested = value if isinstance(value, list) else [value]
        for piece in nested:
            text = clean_text(piece, max_chars=120)
            key = text.lower()
            if not text or key in ROOT_CATEGORY_STOPWORDS or key in {"unknown", "none", "nan", "null"}:
                continue
            if NOISY_TEXT_RE.search(text):
                continue
            parts.append(text)
    out: List[str] = []
    seen = set()
    for part in parts:
        key = part.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(part)
    return out


def normalize_main_cat(raw_main_cat: Any) -> str:
    raw = str(raw_main_cat or "")
    alt_match = re.search(r'alt=["\']([^"\']+)["\']', raw, flags=re.I)
    text = clean_text(alt_match.group(1) if alt_match else raw_main_cat, max_chars=120)
    if not text or text.lower() in ROOT_CATEGORY_STOPWORDS or NOISY_TEXT_RE.search(text):
        return ""
    return text


def build_description(row: Dict[str, Any], title: str, brand: str, categories: List[str], max_chars: int) -> str:
    bits: List[str] = []
    if brand:
        bits.append(f"Brand: {brand}.")
    if categories:
        bits.append(f"Category: {', '.join(categories[:4])}.")
    feature = clean_text(row.get("feature"), max_chars=220)
    description = clean_text(row.get("description"), max_chars=240)
    details = clean_text(row.get("details"), max_chars=180)
    for text in [description, feature, details]:
        if text:
            bits.append(text)
            break
    out = clean_text(" ".join(bits), max_chars=max_chars)
    return out or f"Item: {title}"


def load_ratings(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 4:
                continue
            item_id, user_id, rating, timestamp = row[:4]
            try:
                ts = int(float(timestamp))
            except ValueError:
                # Skip headers or malformed rows.
                continue
            rows.append(
                {
                    "item_id": str(item_id),
                    "user_id": str(user_id),
                    "rating": rating,
                    "timestamp": ts,
                }
            )
    return rows


def clean_ratings(rows: List[Dict[str, Any]], min_user_items: int, max_users: int, user_sample_seed: int) -> List[Dict[str, Any]]:
    rows = sorted(rows, key=lambda x: x["timestamp"])
    dedup: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in rows:
        dedup[(row["user_id"], row["item_id"])] = row
    cleaned_rows = sorted(dedup.values(), key=lambda x: (x["user_id"], x["timestamp"]))

    user2items: Dict[str, set[str]] = {}
    for row in cleaned_rows:
        user2items.setdefault(row["user_id"], set()).add(row["item_id"])
    valid_users = [u for u, items in user2items.items() if len(items) >= min_user_items]
    if max_users and len(valid_users) > max_users:
        valid_users = random.Random(user_sample_seed).sample(valid_users, max_users)
    valid_set = set(valid_users)
    return [row for row in cleaned_rows if row["user_id"] in valid_set]


def validate_gzip(path: Path) -> None:
    if path.suffix != ".gz":
        return
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for _ in range(3):
            if not f.readline():
                break


def iter_metadata(path: Path) -> Iterable[Dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def weighted_sample_without_replacement(rng: random.Random, population: List[str], weights: List[float], k: int) -> List[str]:
    chosen: List[str] = []
    items = list(population)
    ws = list(weights)
    for _ in range(min(k, len(items))):
        total = sum(ws)
        if total <= 0:
            idx = rng.randrange(len(items))
        else:
            target = rng.random() * total
            acc = 0.0
            idx = len(items) - 1
            for i, w in enumerate(ws):
                acc += w
                if acc >= target:
                    idx = i
                    break
        chosen.append(items.pop(idx))
        ws.pop(idx)
    return chosen


def create_splits(rows: List[Dict[str, Any]], neg_num: int, neg_seed: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    rows = sorted(rows, key=lambda x: (x["user_id"], x["timestamp"]))
    item_pop = Counter(row["item_id"] for row in rows)
    all_items = set(item_pop)
    rng = random.Random(neg_seed)

    user_rows: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        user_rows.setdefault(row["user_id"], []).append(row)

    sequences: Dict[str, Any] = {}
    negatives: Dict[str, Any] = {}
    skipped_too_few_negatives = 0
    for user_id, rows_for_user in user_rows.items():
        items = [row["item_id"] for row in rows_for_user]
        if len(items) < 3:
            continue
        interacted = set(items)
        candidate_negatives = list(all_items - interacted)
        if len(candidate_negatives) < neg_num:
            skipped_too_few_negatives += 1
            continue
        weights = [float(item_pop[i]) for i in candidate_negatives]
        sequences[str(user_id)] = {
            "train": [str(x) for x in items[:-2]],
            "val": [str(items[-2])],
            "test": [str(items[-1])],
        }
        negatives[str(user_id)] = {
            "val_neg": weighted_sample_without_replacement(rng, candidate_negatives, weights, neg_num),
            "test_neg": weighted_sample_without_replacement(rng, candidate_negatives, weights, neg_num),
        }
    return sequences, negatives


def referenced_item_ids(sequences: Dict[str, Any], negatives: Dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for row in sequences.values():
        out.update(str(x) for x in row.get("train", []))
        out.update(str(x) for x in row.get("val", []))
        out.update(str(x) for x in row.get("test", []))
    for row in negatives.values():
        out.update(str(x) for x in row.get("val_neg", []))
        out.update(str(x) for x in row.get("test_neg", []))
    return out


def build_items(meta_path: Path, refs: set[str], description_chars: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    items: Dict[str, Any] = {}
    stats = Counter()
    for row in iter_metadata(meta_path):
        asin = str(row.get("asin", "")).strip()
        if not asin or asin not in refs:
            continue
        title = clean_text(row.get("title"), max_chars=180)
        if not title:
            stats["missing_title"] += 1
            continue
        brand = clean_text(row.get("brand"), max_chars=120)
        main_cat = normalize_main_cat(row.get("main_cat"))
        categories = normalize_category_parts(row.get("category"))
        if not categories and main_cat:
            categories = [main_cat]
        if not categories:
            categories = ["Unknown"]
            stats["unknown_category"] += 1
        description = build_description(row, title, brand, categories, max_chars=description_chars)
        items[asin] = {
            "title": title,
            "brand": brand,
            "category": "; ".join(categories[:4]),
            "main_cat": categories[0],
            "categories": categories,
            "description": description,
            "description_short": description,
            "metadata_cleaned": True,
        }
        if brand:
            stats["with_brand"] += 1
        if description and description != f"Item: {title}":
            stats["with_rich_description"] += 1

    missing_refs = sorted(refs - set(items))
    for asin in missing_refs:
        items[asin] = {
            "title": f"Item {asin}",
            "brand": "",
            "category": "Unknown",
            "main_cat": "Unknown",
            "categories": ["Unknown"],
            "description": f"Item: Item {asin}",
            "description_short": f"Item: Item {asin}",
            "metadata_missing": True,
            "metadata_cleaned": True,
        }
    stats["missing_referenced_items"] = len(missing_refs)
    stats["items_written"] = len(items)
    return items, dict(stats)


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    ratings_path = Path(args.ratings)
    meta_path = Path(args.meta)
    if not ratings_path.exists():
        raise FileNotFoundError(ratings_path)
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)
    if meta_path.name.endswith(".crdownload"):
        raise RuntimeError(f"Metadata file still looks incomplete: {meta_path}")
    if args.require_meta_gzip_valid:
        validate_gzip(meta_path)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ratings_raw = load_ratings(ratings_path)
    ratings = clean_ratings(
        ratings_raw,
        min_user_items=args.min_user_items,
        max_users=args.max_users,
        user_sample_seed=args.user_sample_seed,
    )
    sequences, negatives = create_splits(ratings, neg_num=args.neg_num, neg_seed=args.neg_seed)
    refs = referenced_item_ids(sequences, negatives)
    items, item_stats = build_items(meta_path, refs, description_chars=args.description_chars)

    save_json(out_dir / "items.json", items)
    save_json(out_dir / "user_sequences_10.json", sequences)
    save_json(out_dir / "user_negatives_10.json", negatives)
    save_json(out_dir / "user_sequences.json", sequences)
    save_json(out_dir / "user_negatives.json", negatives)
    save_json(out_dir / "user_sequences_5.json", sequences)
    save_json(out_dir / "user_negatives_5.json", negatives)

    audit = {
        "dataset_name": args.dataset_name,
        "ratings_path": str(ratings_path.resolve()),
        "meta_path": str(meta_path.resolve()),
        "output_dir": str(out_dir.resolve()),
        "ratings_raw": len(ratings_raw),
        "ratings_after_user_filter": len(ratings),
        "users_written": len(sequences),
        "items_referenced": len(refs),
        "items_written": len(items),
        "max_users": args.max_users,
        "min_user_items": args.min_user_items,
        "neg_num": args.neg_num,
        "user_sample_seed": args.user_sample_seed,
        "neg_seed": args.neg_seed,
        "item_stats": item_stats,
    }
    save_json(out_dir / "rebuild_audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
