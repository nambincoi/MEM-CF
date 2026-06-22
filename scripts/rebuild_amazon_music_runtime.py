#!/usr/bin/env python3
"""Rebuild a MEMCF runtime dataset from raw Amazon ratings + metadata.

This is intended for music-like domains such as CDs/Vinyl and Digital Music,
where raw metadata often contains HTML, domain-root categories, and weak
descriptions. The output keeps MEMCF's canonical runtime format:

  items.json
  user_sequences_10.json
  user_negatives_10.json

The split logic stays close to the existing process_rating.py flow:
  - sort by timestamp
  - deduplicate user-item pairs
  - require at least 5 unique items per user
  - use popularity-weighted negative sampling

The main difference is metadata normalization:
  - clean HTML/entities from titles/categories
  - ignore useless root domain tokens such as "CDs & Vinyl"
  - preserve artist/brand and flattened subgenres
  - create a short description from feature/description/category/artist fields
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
    "cds & vinyl",
    "cds and vinyl",
    "digital music",
    "music",
    "amazon music",
}

NOISY_TEXT_RE = re.compile(
    r"<[^>]+>|a-size-|a-color-|nav-categ-image|amzn_music_logo|parentaladvisory",
    re.I,
)
SPACE_RE = re.compile(r"\s+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild MEMCF runtime dataset from raw Amazon ratings/metadata.")
    parser.add_argument("--ratings", required=True, help="Path to raw ratings CSV.")
    parser.add_argument("--meta", required=True, help="Path to raw metadata JSON(.gz).")
    parser.add_argument("--output_dir", required=True, help="Output runtime dataset directory.")
    parser.add_argument("--dataset_name", default="CDs_and_Vinyl_1000u_rebuilt_clean")
    parser.add_argument("--neg_num", type=int, default=19)
    parser.add_argument("--max_users", type=int, default=1000)
    parser.add_argument("--min_user_items", type=int, default=5)
    parser.add_argument("--user_sample_seed", type=int, default=42)
    parser.add_argument("--neg_seed", type=int, default=42)
    parser.add_argument("--description_chars", type=int, default=280)
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
        r"\b(Product Description|Amazon\.com|No Description Available|Artist:|Title:|Editorial Reviews?)\b",
        " ",
        text,
        flags=re.I,
    )
    text = SPACE_RE.sub(" ", text).strip(" []'\",;/")
    if max_chars and len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0].rstrip(" .,;:") + "..."
    return text


def normalize_category_parts(raw_category: Any) -> List[str]:
    parts: List[str] = []
    if isinstance(raw_category, list):
        iterable: Iterable[Any] = raw_category
    else:
        iterable = [raw_category]
    for value in iterable:
        if isinstance(value, list):
            nested = value
        else:
            nested = [value]
        for piece in nested:
            text = clean_text(piece, max_chars=120)
            text_lower = text.lower()
            if not text:
                continue
            if text_lower in ROOT_CATEGORY_STOPWORDS:
                continue
            if text_lower in {"unknown", "none", "nan", "null"}:
                continue
            parts.append(text)
    deduped: List[str] = []
    seen = set()
    for part in parts:
        key = part.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(part)
    return deduped


def normalize_main_cat(raw_main_cat: Any) -> str:
    text = clean_text(raw_main_cat, max_chars=120)
    alt_match = re.search(r'alt=["\']([^"\']+)["\']', str(raw_main_cat or ""), flags=re.I)
    if alt_match:
        text = clean_text(alt_match.group(1), max_chars=120)
    if not text:
        return ""
    if text.lower() in ROOT_CATEGORY_STOPWORDS:
        return ""
    return text


def build_music_description(
    title: str,
    artist: str,
    categories: List[str],
    main_cat: str,
    feature: Any,
    description: Any,
    max_chars: int,
) -> str:
    genre_bits = categories[:3]
    desc_candidates: List[str] = []
    if artist:
        desc_candidates.append(f"Artist: {artist}.")
    if genre_bits:
        desc_candidates.append(f"Genres: {', '.join(genre_bits)}.")
    elif main_cat:
        desc_candidates.append(f"Category: {main_cat}.")
    feature_text = clean_text(feature, max_chars=220)
    description_text = clean_text(description, max_chars=220)
    rich_text = description_text or feature_text
    if rich_text:
        desc_candidates.append(rich_text)
    joined = clean_text(" ".join(desc_candidates), max_chars=max_chars)
    if not joined:
        fallback_bits = [x for x in [artist, "; ".join(genre_bits), main_cat] if x]
        if fallback_bits:
            joined = clean_text(". ".join(fallback_bits), max_chars=max_chars)
    if not joined:
        joined = f"Music item: {title}"
    return joined


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
        rng = random.Random(user_sample_seed)
        valid_users = rng.sample(valid_users, max_users)
    valid_user_set = set(valid_users)
    return [row for row in cleaned_rows if row["user_id"] in valid_user_set]


def iter_metadata(path: Path) -> Iterable[Dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def build_items_dict(meta_path: Path, referenced_items: set[str], description_chars: int) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    items: Dict[str, Dict[str, Any]] = {}
    stats = Counter()
    for row in iter_metadata(meta_path):
        asin = str(row.get("asin", "")).strip()
        if not asin or asin not in referenced_items:
            continue
        title = clean_text(row.get("title"), max_chars=180)
        if not title:
            stats["missing_title"] += 1
            continue
        artist = clean_text(row.get("brand"), max_chars=120)
        main_cat = normalize_main_cat(row.get("main_cat"))
        categories = normalize_category_parts(row.get("category"))
        if not categories and main_cat:
            categories = [main_cat]
        if not categories:
            categories = ["Unknown"]
            stats["unknown_category"] += 1
        if NOISY_TEXT_RE.search(str(row.get("main_cat", ""))):
            stats["html_main_cat"] += 1
        description = build_music_description(
            title=title,
            artist=artist,
            categories=categories,
            main_cat=main_cat,
            feature=row.get("feature"),
            description=row.get("description"),
            max_chars=description_chars,
        )
        items[asin] = {
            "title": title,
            "brand": artist,
            "artist": artist,
            "category": "; ".join(categories[:4]),
            "main_cat": categories[0],
            "categories": categories,
            "description": description,
            "description_short": description,
            "metadata_cleaned": True,
        }
        if artist:
            stats["with_artist"] += 1
        if description and description != f"Music item: {title}":
            stats["with_rich_description"] += 1

    missing_refs = sorted(referenced_items - set(items))
    for asin in missing_refs:
        items[asin] = {
            "title": f"Item {asin}",
            "brand": "",
            "artist": "",
            "category": "Unknown",
            "main_cat": "Unknown",
            "categories": ["Unknown"],
            "description": f"Music item: Item {asin}",
            "description_short": f"Music item: Item {asin}",
            "metadata_missing": True,
            "metadata_cleaned": True,
        }
    stats["missing_referenced_items"] = len(missing_refs)
    stats["items_written"] = len(items)
    return items, dict(stats)


def weighted_sample_without_replacement(
    rng: random.Random,
    population: List[str],
    weights: List[float],
    k: int,
) -> List[str]:
    if k <= 0:
        return []
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
    item_popularity: Counter[str] = Counter(row["item_id"] for row in rows)
    pop_items = list(item_popularity.keys())
    all_items = set(pop_items)
    rng = random.Random(neg_seed)

    user_rows: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        user_rows.setdefault(row["user_id"], []).append(row)

    data_split: Dict[str, Any] = {}
    negative_samples: Dict[str, Any] = {}

    for user_id, user_df in user_rows.items():
        items = [row["item_id"] for row in user_df]
        if len(items) < 3:
            continue
        train_items = items[:-2]
        val_item = items[-2]
        test_item = items[-1]
        interacted = set(items)
        candidate_negatives = list(all_items - interacted)
        if len(candidate_negatives) < neg_num:
            continue
        cand_weights = [float(item_popularity[i]) for i in candidate_negatives]
        val_negatives = weighted_sample_without_replacement(rng, candidate_negatives, cand_weights, neg_num)
        test_negatives = weighted_sample_without_replacement(rng, candidate_negatives, cand_weights, neg_num)
        data_split[str(user_id)] = {
            "train": [str(x) for x in train_items],
            "val": [str(val_item)],
            "test": [str(test_item)],
        }
        negative_samples[str(user_id)] = {
            "val_neg": [str(x) for x in val_negatives],
            "test_neg": [str(x) for x in test_negatives],
        }
    return data_split, negative_samples


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


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.neg_seed)
    ratings = load_ratings(Path(args.ratings))
    ratings = clean_ratings(
        ratings,
        min_user_items=args.min_user_items,
        max_users=args.max_users,
        user_sample_seed=args.user_sample_seed,
    )
    sequences, negatives = create_splits(ratings, neg_num=args.neg_num, neg_seed=args.neg_seed)
    refs = referenced_item_ids(sequences, negatives)
    items, item_stats = build_items_dict(Path(args.meta), refs, description_chars=args.description_chars)

    save_json(out_dir / "items.json", items)
    save_json(out_dir / "user_sequences_10.json", sequences)
    save_json(out_dir / "user_negatives_10.json", negatives)
    save_json(out_dir / "user_sequences.json", sequences)
    save_json(out_dir / "user_negatives.json", negatives)
    save_json(out_dir / "user_sequences_5.json", sequences)
    save_json(out_dir / "user_negatives_5.json", negatives)

    audit = {
        "dataset_name": args.dataset_name,
        "ratings_path": str(Path(args.ratings).resolve()),
        "meta_path": str(Path(args.meta).resolve()),
        "output_dir": str(out_dir.resolve()),
        "users_written": len(sequences),
        "items_referenced": len(refs),
        "neg_num": args.neg_num,
        "max_users": args.max_users,
        "min_user_items": args.min_user_items,
        "item_stats": item_stats,
    }
    save_json(out_dir / "rebuild_audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
