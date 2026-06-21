import os
import json
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Any, Set, Tuple
from dataclasses import dataclass, asdict, field
# import google.generativeai as genai
from collections import defaultdict
from tqdm import tqdm
import random
import pickle
import time
import re
import hashlib
import urllib.request
import urllib.error
import argparse


def extract_json_object(raw_output: str) -> Dict[str, Any]:
    """Extract the first JSON object from an LLM response."""
    result_text = str(raw_output).strip()
    if "```json" in result_text:
        result_text = result_text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in result_text:
        result_text = result_text.split("```", 1)[1].split("```", 1)[0].strip()

    match = re.search(r"\{.*\}", result_text, re.DOTALL)
    json_str = match.group(0) if match else result_text
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # Local OpenAI-compatible models sometimes emit invalid backslash escapes
        # or trailing commas even when asked for strict JSON.
        json_str = re.sub(r"\\(?![\"\\/bfnrtu])", r"\\\\", json_str)
        json_str = escape_control_chars_in_json_strings(json_str)
        json_str = re.sub(r",\s*([}\]])", r"\1", json_str)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return json.loads(balance_json_object(json_str))


def escape_control_chars_in_json_strings(text: str) -> str:
    """Escape literal newlines/tabs inside JSON strings."""
    out = []
    in_string = False
    escaped = False
    for ch in str(text):
        if escaped:
            out.append(ch)
            escaped = False
            continue
        if ch == "\\":
            out.append(ch)
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            out.append(ch)
            continue
        if in_string and ch in {"\n", "\r", "\t"}:
            out.append({"\n": "\\n", "\r": "\\r", "\t": "\\t"}[ch])
        else:
            out.append(ch)
    return "".join(out)


def balance_json_object(text: str) -> str:
    """Best-effort close truncated JSON for local Qwen responses."""
    out = []
    stack = []
    in_string = False
    escaped = False
    for ch in str(text):
        out.append(ch)
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()
    if in_string:
        out.append('"')
    while stack:
        out.append("}" if stack.pop() == "{" else "]")
    return re.sub(r",\s*([}\]])", r"\1", "".join(out))


def clean_ranked_item_ids(ranked_ids: List[Any], candidate_items: List[Dict[str, Any]]) -> List[str]:
    """Return a valid candidate permutation: no duplicates, no hallucinated IDs, all candidates included."""
    all_candidate_ids = [str(c["item_id"]) for c in candidate_items]
    candidate_set = set(all_candidate_ids)
    cleaned: List[str] = []
    seen: Set[str] = set()

    if not isinstance(ranked_ids, list):
        ranked_ids = []

    for item_id in ranked_ids:
        item_id = str(item_id)
        if item_id in candidate_set and item_id not in seen:
            cleaned.append(item_id)
            seen.add(item_id)

    cleaned.extend(cid for cid in all_candidate_ids if cid not in seen)
    return cleaned


def add_candidate_aliases(candidate_items: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """Use C01/C02 aliases in prompts to reduce item-id hallucination."""
    aliased = []
    alias_to_item_id: Dict[str, str] = {}
    for idx, item in enumerate(candidate_items, 1):
        alias = f"C{idx:02d}"
        item_id = str(item["item_id"])
        alias_to_item_id[alias] = item_id
        row = dict(item)
        row["candidate_id"] = alias
        # Keep real item id out of the main output contract. The title/category
        # are enough for ranking; code maps Cxx back to item_id.
        aliased.append({
            "candidate_id": alias,
            "title": row.get("title", ""),
            "category": row.get("category", "Unknown"),
        })
    return aliased, alias_to_item_id


def parse_score_entries_from_text(raw_output: str, alias_to_item_id: Dict[str, str]) -> List[Dict[str, Any]]:
    """Recover candidate scores from malformed JSON text."""
    text = str(raw_output or "")
    valid_aliases = set(alias_to_item_id)
    entries: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    pattern = re.compile(
        r'"candidate_id"\s*:\s*"?(C\d{2})"?\s*,\s*"score"\s*:\s*"?([0-9]*\.?[0-9]+)"?',
        flags=re.IGNORECASE,
    )
    for match in pattern.finditer(text):
        alias = match.group(1).upper()
        if alias not in valid_aliases or alias in seen:
            continue
        seen.add(alias)
        score = max(0.0, min(1.0, float(match.group(2))))
        entries.append({
            "candidate_id": alias,
            "item_id": alias_to_item_id[alias],
            "score": score,
            "rationale": "Recovered from malformed score JSON",
        })
    return entries


def score_entries_to_ranking(
    raw_scores: List[Dict[str, Any]],
    alias_to_item_id: Dict[str, str],
) -> Tuple[List[str], Dict[str, Any]]:
    """Validate score rows and return a complete ranking."""
    alias_order = list(alias_to_item_id.keys())
    alias_index = {alias: idx for idx, alias in enumerate(alias_order)}
    seen: Set[str] = set()
    parsed: List[Dict[str, Any]] = []
    invalid_rows: List[Any] = []

    for row in raw_scores if isinstance(raw_scores, list) else []:
        if not isinstance(row, dict):
            invalid_rows.append(row)
            continue
        alias = str(row.get("candidate_id", "")).upper().strip()
        if not alias and row.get("item_id") is not None:
            item_id = str(row.get("item_id"))
            alias = next((a for a, iid in alias_to_item_id.items() if iid == item_id), "")
        if alias not in alias_to_item_id or alias in seen:
            invalid_rows.append(row)
            continue
        seen.add(alias)
        try:
            score = float(row.get("score", 0.0))
        except Exception:
            score = 0.0
        parsed.append({
            "candidate_id": alias,
            "item_id": alias_to_item_id[alias],
            "score": max(0.0, min(1.0, score)),
            "rationale": str(row.get("rationale", ""))[:180],
            "original_index": alias_index[alias],
        })

    missing_aliases = [alias for alias in alias_order if alias not in seen]
    for alias in missing_aliases:
        parsed.append({
            "candidate_id": alias,
            "item_id": alias_to_item_id[alias],
            "score": -1.0,
            "rationale": "Missing from LLM score output",
            "original_index": alias_index[alias],
        })

    parsed.sort(key=lambda x: (-x["score"], x["original_index"]))
    ranked = [row["item_id"] for row in parsed]
    validation = {
        "is_valid": len(missing_aliases) == 0 and len(invalid_rows) == 0,
        "raw_score_rows": len(raw_scores) if isinstance(raw_scores, list) else 0,
        "expected_rows": len(alias_order),
        "missing_candidate_ids": missing_aliases,
        "invalid_or_duplicate_rows": invalid_rows,
        "parsed_scores": parsed,
    }
    return ranked, validation


def deterministic_shuffle(values: List[Any], salt: str = "") -> List[Any]:
    """Shuffle reproducibly without depending on global random state consumed during training."""
    values = list(values)
    seed_material = salt + "||" + "||".join(str(v) for v in values)
    seed = int(hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed)
    rng.shuffle(values)
    return values


def dedupe_preserve_order(values: List[Any]) -> List[str]:
    seen = set()
    deduped: List[str] = []
    for value in values:
        key = str(value)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(key)
    return deduped


def make_jsonable(obj: Any) -> Any:
    """Convert dataclasses/numpy values into JSON-safe objects for traces."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (UserInteraction, BehaviorMemory, PairwiseUserState, PairwiseItemState)):
        return asdict(obj)
    if isinstance(obj, dict):
        return {str(k): make_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [make_jsonable(v) for v in obj]
    return obj


def interaction_to_trace(interaction: "UserInteraction") -> Dict[str, Any]:
    return make_jsonable(asdict(interaction))


def behavior_memory_to_trace(memory: "BehaviorMemory") -> Dict[str, Any]:
    data = make_jsonable(asdict(memory))
    data["embedding"] = None
    data["embedding_dim"] = int(len(memory.embedding)) if memory.embedding is not None else 0
    data["interaction_sequence"] = [interaction_to_trace(i) for i in memory.interaction_sequence]
    return data


GENERIC_MEMORY_TERMS = {
    "all", "beauty", "unknown", "category", "item", "items", "product", "products",
    "preference", "preferences", "user", "users", "recommendation", "recommendations",
    "wrong", "correct", "choice", "chosen", "preferred", "pattern", "future",
    "the", "and", "for", "with", "without", "this", "that", "these", "those",
    "system", "pack", "set", "edition", "standard", "new", "one", "two", "three",
    "likely", "intent", "needs", "need", "prioritize", "relevant", "similar",
    "based", "match", "matches", "matching", "current", "past", "history",
    "video", "game", "games", "gaming", "digital", "music", "album", "albums",
    "cds", "vinyl", "record", "records", "logo", "image", "audio",
    "beauty", "skin", "care", "shopping", "purchase", "purchases",
}

GENERIC_MEMORY_PHRASES = {
    "video game", "video games", "digital music", "cds and vinyl", "all beauty",
    "unknown category", "category mismatch", "user preference", "preferred item",
    "wrong choice", "future ranking", "current candidates",
}


def normalize_terms(text: str) -> List[str]:
    return [
        t for t in re.findall(r"[a-zA-Z0-9]+", str(text).lower())
        if len(t) >= 3 and t not in GENERIC_MEMORY_TERMS
    ]


def normalize_evidence_terms(values: Any) -> List[str]:
    """Normalize structured memory evidence terms and remove generic terms."""
    if values is None:
        return []
    if isinstance(values, str):
        raw_values = [values]
    elif isinstance(values, (list, tuple, set)):
        raw_values = list(values)
    else:
        raw_values = [str(values)]

    terms: List[str] = []
    seen: Set[str] = set()
    for value in raw_values:
        phrase = re.sub(r"\s+", " ", str(value).lower()).strip(" .,:;|")
        if not phrase or phrase in GENERIC_MEMORY_PHRASES:
            continue
        if 3 <= len(phrase) <= 40 and phrase not in GENERIC_MEMORY_TERMS and phrase not in seen:
            terms.append(phrase)
            seen.add(phrase)
        for token in normalize_terms(phrase):
            if token not in seen:
                terms.append(token)
                seen.add(token)
    return terms


def normalize_category(category: Any, fallback: str = "Unknown") -> str:
    """Normalize noisy Amazon metadata categories for prompt/retrieval use."""
    if isinstance(category, list):
        parts: List[str] = []
        for value in category:
            if isinstance(value, list):
                parts.extend(str(x) for x in value)
            else:
                parts.append(str(value))
        raw = " > ".join(p for p in parts if p)
    else:
        raw = str(category or "").strip()

    if not raw or raw.lower() in {"none", "nan", "[]", "unknown"}:
        return fallback

    alt_match = re.search(r'alt=["\']([^"\']+)["\']', raw, flags=re.IGNORECASE)
    if alt_match:
        raw = alt_match.group(1)
    raw = re.sub(r"<[^>]+>", " ", raw)
    raw = re.sub(r"\s+", " ", raw).strip(" /|>")
    return raw or fallback


def memory_text_is_too_generic(text: str, min_terms: int = 4) -> bool:
    return len(set(normalize_terms(text))) < min_terms


def collect_runtime_negative_pool(
    negative_data: Optional[Dict[str, Any]],
    valid_item_ids: Optional[Set[str]] = None,
    exclude_ids: Optional[Set[str]] = None,
) -> List[str]:
    negative_data = negative_data or {}
    values: List[str] = []
    for key in ("val_neg", "test_neg", "train_neg", "negatives"):
        raw = negative_data.get(key, [])
        if isinstance(raw, list):
            values.extend(str(x) for x in raw)
    values = dedupe_preserve_order(values)
    exclude_ids = set(str(x) for x in (exclude_ids or set()))
    if valid_item_ids is not None:
        values = [x for x in values if x in valid_item_ids and x not in exclude_ids]
    else:
        values = [x for x in values if x not in exclude_ids]
    return values


def item_category(item_info: Dict[str, Any], fallback: str = "Unknown") -> str:
    return normalize_category(
        item_info.get("main_cat")
        or item_info.get("category")
        or item_info.get("categories")
        or fallback,
        fallback=fallback,
    )


def item_title(item_info: Dict[str, Any], item_id: str) -> str:
    title = str(item_info.get("title") or "").strip()
    return title if title else f"Item {item_id}"


def ranking_validation(raw_ranked_ids: List[Any], candidate_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    candidate_ids = [str(c["item_id"]) for c in candidate_items]
    candidate_set = set(candidate_ids)
    raw_ids = [str(x) for x in raw_ranked_ids] if isinstance(raw_ranked_ids, list) else []
    seen: Set[str] = set()
    duplicate_ids: List[str] = []
    for item_id in raw_ids:
        if item_id in seen and item_id not in duplicate_ids:
            duplicate_ids.append(item_id)
        seen.add(item_id)
    hallucinated_ids = [item_id for item_id in raw_ids if item_id not in candidate_set]
    missing_ids = [item_id for item_id in candidate_ids if item_id not in set(raw_ids)]
    return {
        "is_valid": (
            len(raw_ids) == len(candidate_ids)
            and len(duplicate_ids) == 0
            and len(hallucinated_ids) == 0
            and len(missing_ids) == 0
        ),
        "raw_length": len(raw_ids),
        "expected_length": len(candidate_ids),
        "duplicate_ids": duplicate_ids,
        "hallucinated_ids": hallucinated_ids,
        "missing_ids": missing_ids,
    }


def memory_text(memory: "BehaviorMemory") -> str:
    return " ".join(
        [
            str(memory.behavior_explanation),
            str(memory.pattern_description),
            " ".join(str(k) for k in memory.keywords),
            " ".join(str(k) for k in getattr(memory, "applicable_when", [])),
            " ".join(str(k) for k in getattr(memory, "not_applicable_when", [])),
            str(getattr(memory, "wrong_item_type", "")),
            str(getattr(memory, "correct_item_type", "")),
            " ".join(str(k) for k in getattr(memory, "evidence_terms_required", [])),
        ]
    )


def build_retrieval_query(
    user_profile_text: str,
    candidate_items_info: List[Dict[str, Any]],
    mode: str,
) -> str:
    if mode == "user_only":
        return user_profile_text
    if mode != "candidate_aware":
        raise ValueError(f"Unsupported --memory_retrieval_mode={mode}")
    candidate_text = " ".join(
        f"{item.get('title', '')} {item.get('category', '')}" for item in candidate_items_info
    )
    return f"User recent history: {user_profile_text}\nCandidate items: {candidate_text}"


def term_matches_context(term: str, context: str) -> bool:
    term = re.sub(r"\s+", " ", str(term).lower()).strip()
    if not term:
        return False
    if " " in term:
        return term in context
    return re.search(rf"\b{re.escape(term)}\b", context) is not None


def gate_memory_records(
    memory_records: List[Dict[str, Any]],
    user_profile_text: str,
    candidate_items_info: List[Dict[str, Any]],
    gate_mode: str,
    similarity_threshold: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if gate_mode == "none":
        decisions = []
        for record in memory_records:
            decisions.append({
                "memory_id": record["memory"].thought_id,
                "decision": "keep",
                "reason": "memory_gate=none",
                "similarity": float(record["similarity"]),
                "matched_terms": [],
            })
        return memory_records, decisions
    if gate_mode not in {"rule", "strict_rule", "applicability"}:
        raise ValueError(f"Unsupported --memory_gate={gate_mode}")

    candidate_text = " ".join(
        f"{item.get('title', '')} {item.get('category', '')}" for item in candidate_items_info
    ).lower()
    history_text = str(user_profile_text).lower()
    combined_context = f"{history_text} {candidate_text}"
    candidate_categories = {
        str(item.get("category", "Unknown")).strip().lower()
        for item in candidate_items_info
        if item.get("category")
    }

    kept: List[Dict[str, Any]] = []
    decisions: List[Dict[str, Any]] = []
    for record in memory_records:
        memory = record["memory"]
        similarity = float(record["similarity"])
        text = memory_text(memory).lower()
        structured_terms = normalize_evidence_terms(
            list(getattr(memory, "evidence_terms_required", []) or [])
            + list(getattr(memory, "applicable_when", []) or [])
            + list(getattr(memory, "keywords", []) or [])
            + [getattr(memory, "wrong_item_type", ""), getattr(memory, "correct_item_type", "")]
        )
        text_terms = normalize_terms(text)
        candidate_terms = [
            term for term in (structured_terms + text_terms)
            if term not in GENERIC_MEMORY_TERMS
            and term not in GENERIC_MEMORY_PHRASES
            and len(term) >= 3
        ]
        seen_terms: Set[str] = set()
        candidate_terms = [t for t in candidate_terms if not (t in seen_terms or seen_terms.add(t))]
        matched_terms = [term for term in candidate_terms[:60] if term_matches_context(term, combined_context)]
        strong_matched_terms = [
            term for term in matched_terms
            if term not in GENERIC_MEMORY_TERMS
            and term not in GENERIC_MEMORY_PHRASES
            and len(term) >= 4
        ]
        not_applicable_terms = normalize_evidence_terms(getattr(memory, "not_applicable_when", []))
        matched_not_applicable_terms = [
            term for term in not_applicable_terms
            if term_matches_context(term, combined_context)
        ]

        category_mismatch_memory = (
            "category" in text
            and any(signal in text for signal in ["mismatch", "unrelated", "non-", "instead of"])
        )
        all_candidates_same_category = len(candidate_categories) <= 1

        decision = "keep"
        reason = "passed rule gate"
        min_strong_terms = int(os.getenv("MEMCF_STRICT_GATE_MIN_STRONG_TERMS", "2"))
        min_app_terms = int(os.getenv("MEMCF_APP_GATE_MIN_TERMS", "1"))
        generic_memory = (
            len(strong_matched_terms) == 0
            and sum(1 for term in ["mismatch", "category", "intent", "preference"] if term in text) >= 2
        )
        applicability_score = (
            len(strong_matched_terms)
            + 0.5 * len([t for t in matched_terms if t not in strong_matched_terms])
            + max(0.0, similarity - similarity_threshold)
        )
        if similarity < similarity_threshold:
            decision = "skip"
            reason = f"similarity {similarity:.4f} below threshold {similarity_threshold:.4f}"
        elif matched_not_applicable_terms:
            decision = "skip"
            reason = f"not_applicable_when matched current context: {matched_not_applicable_terms[:5]}"
        elif category_mismatch_memory and all_candidates_same_category and not matched_terms:
            decision = "skip"
            reason = "category-mismatch memory is not discriminative because candidate categories are identical"
        elif not matched_terms and similarity < similarity_threshold + 0.10:
            decision = "skip"
            reason = "no specific memory terms matched current user/candidates"
        elif gate_mode in {"strict_rule", "applicability"} and len(strong_matched_terms) < min_strong_terms:
            decision = "skip"
            reason = (
                f"{gate_mode} requires at least {min_strong_terms} strong matched terms; "
                f"found {len(strong_matched_terms)}"
            )
        elif gate_mode in {"strict_rule", "applicability"} and generic_memory:
            decision = "skip"
            reason = f"{gate_mode} rejected generic category/intent memory with no strong current evidence"
        elif gate_mode in {"strict_rule", "applicability"} and category_mismatch_memory and all_candidates_same_category:
            decision = "skip"
            reason = f"{gate_mode} rejected category-mismatch memory when all candidates have same category"
        elif gate_mode == "applicability" and applicability_score < min_app_terms:
            decision = "skip"
            reason = (
                f"applicability score {applicability_score:.2f} below required {min_app_terms}; "
                "memory lacks concrete evidence in current context"
            )

        item = {
            "memory_id": memory.thought_id,
            "decision": decision,
            "reason": reason,
            "similarity": similarity,
            "applicability_score": applicability_score,
            "matched_terms": matched_terms,
            "strong_matched_terms": strong_matched_terms,
            "matched_not_applicable_terms": matched_not_applicable_terms,
            "candidate_terms_checked": candidate_terms[:40],
            "gate_mode": gate_mode,
            "memory": behavior_memory_to_trace(memory),
        }
        decisions.append(item)
        if decision == "keep":
            kept.append(record)

    return kept, decisions


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
# # api_key = 'AIzaSyDNFlv44-Gl6QWVKFdwXEtOrlRtv4bBTu8'
# api_key = 'AIzaSyD98UePYe0I55FTfN9CjA3uShBhw9K02DY'

# genai.configure(api_key=api_key)
# print("✓ Gemini API configured: {}".format(api_key) )
# torch.manual_seed(42)
# torch.cuda.manual_seed_all(42)
# np.random.seed(42)
# random.seed(42)

# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
set_seed(42)
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except (ImportError, OSError) as e:
    print(f"Warning: transformers/torch not available: {e}")
    print("Will use OpenAI-compatible API endpoints if provided via env vars.")
    TRANSFORMERS_AVAILABLE = False
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except (ImportError, OSError) as e:
    print(f"Warning: sentence_transformers not available: {e}")
    print("Will use API embeddings or hash fallback")
    SENTENCE_TRANSFORMERS_AVAILABLE = False

@dataclass
class UserInteraction:
    """Represents a single user-item interaction"""
    item_id: str
    item_name: str
    item_category: str
    action_type: str  # 'purchase' for implicit feedback
    rating: Optional[float] = None
    timestamp: str = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

@dataclass
class BehaviorMemory:
    """
    Represents a generalized thought about user behavior patterns
    """
    thought_id: int
    interaction_sequence: List[UserInteraction]
    behavior_explanation: str
    pattern_description: str
    # extracted_preferences: List[str]
    keywords: List[str]
    embedding: np.ndarray
    applicable_when: List[str] = field(default_factory=list)
    not_applicable_when: List[str] = field(default_factory=list)
    wrong_item_type: str = ""
    correct_item_type: str = ""
    evidence_terms_required: List[str] = field(default_factory=list)
    specificity_score: float = 0.0
    overgeneralization_risk: float = 0.0
    links: List[int] = field(default_factory=list)
    timestamp: str = None
    evolution_count: int = 0  # Số lần đã evolve
    evolution_history: List[Dict[str, Any]] = field(default_factory=list)  # Lịch sử evolution
    max_evolutions: Optional[int] = None  # Giới hạn số lần evolve (None = unlimited)
    last_evolved_timestamp: Optional[str] = None  # Lần evolve cuối

    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    def can_evolve(self) -> bool:
        """Kiểm tra xem memory này còn được phép evolve không"""
        if self.max_evolutions is None:
            return True
        return self.evolution_count < self.max_evolutions
    def record_evolution(self, 
                        update_type: str,
                        old_values: Dict[str, Any],
                        new_values: Dict[str, Any],
                        reasoning: str) -> None:
        """Ghi lại một lần evolution"""
        self.evolution_count += 1
        self.last_evolved_timestamp = datetime.now().isoformat()
        
        self.evolution_history.append({
            'evolution_number': self.evolution_count,
            'timestamp': self.last_evolved_timestamp,
            'update_type': update_type,
            'old_values': old_values,
            'new_values': new_values,
            'reasoning': reasoning
        })
    def to_dict(self):
        data = asdict(self)
        data['embedding'] = self.embedding.tolist()
        data['interaction_sequence'] = [asdict(i) for i in self.interaction_sequence]
        return data
    
    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        data['embedding'] = np.array(data['embedding'])
        data['interaction_sequence'] = [UserInteraction(**i) for i in data['interaction_sequence']]
        # Backward compatibility with memories created before structured fields.
        data.setdefault('applicable_when', [])
        data.setdefault('not_applicable_when', [])
        data.setdefault('wrong_item_type', "")
        data.setdefault('correct_item_type', "")
        data.setdefault('evidence_terms_required', [])
        data.setdefault('specificity_score', 0.0)
        data.setdefault('overgeneralization_risk', 0.0)
        return cls(**data)


@dataclass
class PairwiseUserState:
    """pairwise user state used to bootstrap fail-interaction memory generation."""
    user_id: str
    short_term_memory: str = "I enjoy discovering new items."
    long_term_memory: List[str] = field(default_factory=list)
    interaction_history: List[str] = field(default_factory=list)

    def update_memory(self, new_memory: str):
        self.long_term_memory.append(self.short_term_memory)
        self.short_term_memory = new_memory

    def add_interaction(self, item_id: str):
        self.interaction_history.append(item_id)


@dataclass
class PairwiseItemState:
    """pairwise item state with mutable textual memory."""
    item_id: str
    title: str
    category: str
    memory: str


class TraceRecorder:
    """Small JSONL trace writer for reproducible MEMCF research runs."""

    def __init__(self, trace_dir: str, enabled: bool = True):
        self.trace_dir = trace_dir
        self.enabled = enabled
        self.counts: Dict[str, int] = defaultdict(int)
        if self.enabled:
            os.makedirs(self.trace_dir, exist_ok=True)

    def log(self, event_type: str, payload: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        self.counts[event_type] += 1
        row = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            **make_jsonable(payload),
        }
        path = os.path.join(self.trace_dir, f"{event_type}.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        events_path = os.path.join(self.trace_dir, "events.jsonl")
        with open(events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def write_manifest(self, payload: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        path = os.path.join(self.trace_dir, "manifest.json")
        data = {
            "trace_dir": self.trace_dir,
            "created_at": datetime.now().isoformat(),
            "event_counts": dict(self.counts),
            **make_jsonable(payload),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


class RecommendationMemorySystem:
    """A-Mem adapted for Amazon product recommendation with Memory Evolution"""
    
    def __init__(self, 
                 model_name: str = "Qwen/Qwen2.5-7B-Instruct",
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 use_gemini_embeddings: bool = None,
                 chat_api_base: Optional[str] = None,
                 embedding_api_base: Optional[str] = None,
                 api_key: Optional[str] = None,
                 chat_model_name: Optional[str] = None,
                 embedding_model_name: Optional[str] = None):
        _ = use_gemini_embeddings  # kept for backward compatibility

        self.llm_name = model_name
        self.embedding_model_name = embedding_model_name or os.getenv("embedding_model_name") or embedding_model
        self.chat_model_name = chat_model_name or os.getenv("chat_model_name") or model_name
        self.chat_api_base = (chat_api_base or os.getenv("chat_api_base") or os.getenv("api_base") or "").rstrip("/")
        self.embedding_api_base = (embedding_api_base or os.getenv("embedding_api_base") or os.getenv("api_base") or "").rstrip("/")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or "EMPTY"

        self.use_api_chat = bool(self.chat_api_base)
        # VIRAL-style embedding path: always use local SentenceTransformer.
        self.use_api_embedding = False

        self.tokenizer = None
        self.model = None
        self.embedding_model = None

        if not self.use_api_chat:
            if not TRANSFORMERS_AVAILABLE:
                raise RuntimeError(
                    "Local chat model requires transformers+torch, or set chat_api_base/api_base env vars."
                )
            self.tokenizer = AutoTokenizer.from_pretrained(self.llm_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.llm_name,
                dtype=torch.float16,
                device_map="auto"
            )

        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
        else:
            raise RuntimeError(
                "sentence_transformers is required for VIRAL-style embedding. "
                "Please install sentence-transformers."
            )


        self.behavior_memories: List[BehaviorMemory] = []
        self.user_interaction_history: List[UserInteraction] = []
        self.next_thought_id = 0
        self.trace_recorder: Optional[TraceRecorder] = None
        self.memory_diagnostics = defaultdict(float)

    def _trace(self, event_type: str, payload: Dict[str, Any]) -> None:
        if getattr(self, "trace_recorder", None) is not None:
            self.trace_recorder.log(event_type, payload)

    def _post_json(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        last_error = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=180) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="ignore")
                last_error = RuntimeError(f"API HTTP {e.code} at {url}: {body}")
            except Exception as e:
                last_error = e
            time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"API request failed after 3 attempts at {url}: {last_error}") from last_error

    def qwen_generate(
        self,
        prompt: str,
        role_prompt="You are a helpful AI assistant.",
        max_new_tokens=8000,
        json_schema: Optional[Dict[str, Any]] = None,
        json_mode: bool = False,
    ) -> str:
        if self.use_api_chat:
            endpoint = f"{self.chat_api_base}/chat/completions"
            temperature = float(os.getenv("MEMCF_TEMPERATURE", "0.0"))
            payload = {
                "model": self.chat_model_name,
                "messages": [
                    {"role": "system", "content": role_prompt},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": 1.0,
            }
            repetition_penalty = os.getenv("MEMCF_REPETITION_PENALTY", "1.05").strip()
            if repetition_penalty:
                try:
                    payload["repetition_penalty"] = float(repetition_penalty)
                except ValueError:
                    pass
            if json_schema is not None and os.getenv("MEMCF_USE_GUIDED_JSON", "0") == "1":
                # vLLM's OpenAI-compatible server accepts guided_json as an
                # extra request field. Keep this opt-in because older servers
                # may reject unknown structured-output parameters.
                payload["guided_json"] = json_schema
            elif json_mode and os.getenv("MEMCF_USE_RESPONSE_FORMAT_JSON", "0") == "1":
                # Some OpenAI-compatible Qwen endpoints support JSON mode.
                # The prompt/role must include the word JSON for those servers.
                payload["response_format"] = {"type": "json_object"}
            result = self._post_json(endpoint, payload)
            content = result["choices"][0]["message"]["content"]
            if isinstance(content, list):
                return "".join(
                    chunk.get("text", "") for chunk in content if isinstance(chunk, dict)
                )
            return str(content)

        messages = [
            {"role": "system", "content": role_prompt},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        gen_ids = outputs[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True)


    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

    def _create_embedding(self, text: str) -> np.ndarray:
        emb = self.embedding_model.encode(
            str(text),
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return emb.astype(np.float32)

    def _simple_hash_embedding(self, text: str, dim: int = 384) -> np.ndarray:
        np.random.seed(hash(text) % (2**32))
        embedding = np.random.randn(dim)
        embedding = embedding / np.linalg.norm(embedding)
        return embedding
    
    def add_interaction(self, 
                       item_id: str,
                       item_name: str,
                       item_category: str,
                       action_type: str = "purchase",
                       rating: Optional[float] = None,
                       metadata: Optional[Dict] = None) -> UserInteraction:
        interaction = UserInteraction(
            item_id=item_id,
            item_name=item_name,
            item_category=item_category,
            action_type=action_type,
            rating=rating,
            metadata=metadata or {}
        )
        
        self.user_interaction_history.append(interaction)
        
        return interaction
    
    def create_behavior_thought(self, 
                               interaction_window: List[UserInteraction],
                               k_neighbors: int = 10) -> BehaviorMemory:
        interaction_summary = []
        for interaction in interaction_window:
            summary = {
                "item": interaction.item_name,
                "category": interaction.item_category,
                "action": interaction.action_type
            }
            interaction_summary.append(summary)
        
        prompt = f"""Analyze this failed recommendation interaction.
        Input: {json.dumps(interaction_summary, indent=2)}

        Context:
        - The interaction contains a wrong choice and the preferred correct item.
        - Your job is to capture why the wrong choice happened and what correction rule should be applied next time.
        - This memory will be retrieved for future ranking. It must be specific enough to avoid being applied to unrelated users.

        Return ONLY a JSON object in this format:
        {{
        "behavior_explanation": "2-3 concise sentences explaining why the wrong choice was made versus the correct item",
        "pattern_description": "2-3 concise sentences describing a correction rule, including when it applies and when it should NOT apply",
        "applicable_when": ["specific title/type/category/attribute evidence required before using this memory"],
        "not_applicable_when": ["conditions where this memory should be ignored"],
        "wrong_item_type": "short concrete type/attribute of the wrong choice",
        "correct_item_type": "short concrete type/attribute of the preferred item",
        "evidence_terms_required": ["concrete evidence terms that must appear in future history/candidates before applying this memory"],
        "specificity_score": 0.0 to 1.0,
        "overgeneralization_risk": 0.0 to 1.0,
        "keywords": ["kw1", "kw2", ...] (5-8 concrete fail-interaction signals, no generic words)
        }}

        Requirements:
        - Ground every statement in the input interaction.
        - Emphasize contrast between wrong and correct choice.
        - Avoid generic shopping summaries like 'user preference', 'category mismatch', or 'prioritize relevant items' unless tied to concrete terms.
        - Do not claim the rule applies unless future candidates/history contain the applicable evidence."""

        try:
            # response = self.model.generate_content(prompt)
            response = self.qwen_generate(prompt=prompt, role_prompt='You are a behavioral memory modeling system.')
            # time.sleep(5)
            result = extract_json_object(response)
            self._trace("memory_create_llm", {
                "prompt": prompt,
                "role_prompt": "You are a behavioral memory modeling system.",
                "answer": response,
                "parsed": result,
                "interaction_window": [interaction_to_trace(i) for i in interaction_window],
            })
            
            behavior_explanation = result.get("behavior_explanation", "")
            pattern_description = result.get("pattern_description", "")
            applicable_when = result.get("applicable_when", [])
            not_applicable_when = result.get("not_applicable_when", [])
            wrong_item_type = result.get("wrong_item_type", "")
            correct_item_type = result.get("correct_item_type", "")
            evidence_terms_required = result.get("evidence_terms_required", [])
            try:
                specificity_score = float(result.get("specificity_score", 0.0))
            except Exception:
                specificity_score = 0.0
            try:
                overgeneralization_risk = float(result.get("overgeneralization_risk", 0.0))
            except Exception:
                overgeneralization_risk = 0.0
            if applicable_when:
                pattern_description += f" Applicable when: {json.dumps(applicable_when, ensure_ascii=False)}."
            if not_applicable_when:
                pattern_description += f" Do not apply when: {json.dumps(not_applicable_when, ensure_ascii=False)}."
            if evidence_terms_required:
                pattern_description += f" Evidence required: {json.dumps(evidence_terms_required, ensure_ascii=False)}."
            if wrong_item_type or correct_item_type:
                pattern_description += (
                    f" Wrong item type: {wrong_item_type}. "
                    f"Correct item type: {correct_item_type}."
                )
            keywords = result.get("keywords", [])
            for extra_kw in [wrong_item_type, correct_item_type] + list(evidence_terms_required or []):
                if extra_kw and extra_kw not in keywords:
                    keywords.append(extra_kw)
            # extracted_preferences = result.get("extracted_preferences", [])
            
        except Exception as e:
            print(f"Error in behavior analysis: {e}")
            self._trace("memory_create_error", {
                "error": str(e),
                "prompt": prompt,
                "interaction_window": [interaction_to_trace(i) for i in interaction_window],
            })
            behavior_explanation = f"A failed interaction occurred with {len(interaction_window)} compared items."
            pattern_description = "Correction rule is unclear; prefer signals from the preferred item over the wrong choice."
            # extracted_preferences = []
            keywords = [i.item_category for i in interaction_window[:3]]
            applicable_when = []
            not_applicable_when = []
            wrong_item_type = ""
            correct_item_type = ""
            evidence_terms_required = []
            specificity_score = 0.0
            overgeneralization_risk = 1.0
        
        combined_text = f"{behavior_explanation} {pattern_description} {' '.join(keywords)}"
        embedding = self._create_embedding(combined_text)
        
        behavior_memory = BehaviorMemory(
            thought_id=self.next_thought_id,
            interaction_sequence=interaction_window.copy(),
            behavior_explanation=behavior_explanation,
            pattern_description=pattern_description,
            # extracted_preferences=extracted_preferences,
            keywords=keywords,
            embedding=embedding,
            applicable_when=applicable_when if isinstance(applicable_when, list) else [str(applicable_when)],
            not_applicable_when=not_applicable_when if isinstance(not_applicable_when, list) else [str(not_applicable_when)],
            wrong_item_type=str(wrong_item_type or ""),
            correct_item_type=str(correct_item_type or ""),
            evidence_terms_required=evidence_terms_required if isinstance(evidence_terms_required, list) else [str(evidence_terms_required)],
            specificity_score=max(0.0, min(1.0, specificity_score)),
            overgeneralization_risk=max(0.0, min(1.0, overgeneralization_risk)),
        )
        
        self.next_thought_id += 1
        self._trace("memory_created", {
            "memory": behavior_memory_to_trace(behavior_memory),
        })
        return behavior_memory
    
    def link_behavior_memories(self, 
                               new_memory: BehaviorMemory,
                               k: int = 5, wo_link=False) -> List[int]:
        """Link new behavior memory with similar past patterns"""
        if len(self.behavior_memories) == 0:
            return []
        
        similarities = []
        for memory in self.behavior_memories:
            sim = self._cosine_similarity(new_memory.embedding, memory.embedding)
            similarities.append((memory.thought_id, sim, memory))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        nearest_k = similarities[:min(k, len(similarities))]
        
        if len(nearest_k) == 0:
            return []
        if wo_link:
            linked = [thought_id for thought_id, _, _ in nearest_k]
            self._trace("memory_link_decision", {
                "new_memory": behavior_memory_to_trace(new_memory),
                "wo_link": True,
                "linked_thought_ids": linked,
                "reasoning": "wo_link enabled; using nearest memories without LLM link filtering.",
            })
            return linked
        
        nearest_info = []
        for thought_id, sim, memory in nearest_k:
            nearest_info.append({
                "thought_id": thought_id,
                "behavior_explanation": memory.behavior_explanation,
                "pattern": memory.pattern_description,
                # "preferences": memory.extracted_preferences,
                "similarity": float(sim)
            })
        
        prompt = f"""Determine if the new fail-interaction memory should be linked to past fail memories.
        New Pattern:
        - Behavior: {new_memory.behavior_explanation}
        - Pattern: {new_memory.pattern_description}

        Similar Past Patterns:
        {json.dumps(nearest_info, indent=2)}

        Link ONLY if:
        - They share a similar error/correction pattern (same mismatch type or same correction signal).
        - They imply a consistent fix strategy across users or interactions.
        - Their wrong-vs-correct contrast is semantically aligned.
        Do NOT link if they describe unrelated failure reasons.

        Return JSON:
        {{
        "should_link": true/false,
        "linked_thought_ids": [list of IDs],
        "reasoning": "1-2 sentences explaining shared fail/correction evidence"
        }}
        Keep reasoning concise and specific."""

        try:
            # response = self.model.generate_content(prompt)
            # time.sleep(3)
            response = self.qwen_generate(prompt=prompt, role_prompt='You are a behavioral memory modeling system.')

            result = extract_json_object(response)
            self._trace("memory_link_llm", {
                "prompt": prompt,
                "role_prompt": "You are a behavioral memory modeling system.",
                "answer": response,
                "parsed": result,
                "new_memory": behavior_memory_to_trace(new_memory),
                "nearest_info": nearest_info,
            })
            
            if result.get("should_link", False):
                linked = result.get("linked_thought_ids", [])
            else:
                linked = []
            self._trace("memory_link_decision", {
                "new_memory": behavior_memory_to_trace(new_memory),
                "linked_thought_ids": linked,
                "reasoning": result.get("reasoning", ""),
            })
            return linked
                
        except Exception as e:
            print(f"Error in linking: {e}")
            linked = [thought_id for thought_id, sim, _ in nearest_k if sim > 0.65]
            self._trace("memory_link_error", {
                "error": str(e),
                "new_memory": behavior_memory_to_trace(new_memory),
                "nearest_info": nearest_info,
                "fallback_linked_thought_ids": linked,
            })
            return linked
    
    def evolve_behavior_memories(self,
                                new_memory: BehaviorMemory,
                                linked_ids: List[int],
                                max_evolutions_per_memory: Optional[int] = None) -> None:
        """Evolve existing behavior memories based on new patterns (Section 3.3)"""
        if len(linked_ids) == 0:
            return

        linked_memories = [m for m in self.behavior_memories if m.thought_id in linked_ids]
        if len(linked_memories) == 0:
            return
        # ============ LỌC MEMORIES CÒN CÓ THỂ EVOLVE ============
        evolvable_memories = []
        for mem in linked_memories:
            if max_evolutions_per_memory is not None:
                mem.max_evolutions = max_evolutions_per_memory
            
            if mem.can_evolve():
                evolvable_memories.append(mem)
            else:
                print(f"  ⚠ Memory {mem.thought_id} reached max evolutions ({mem.evolution_count}), skipping...")
        
        if len(evolvable_memories) == 0:
            print("  → No memories available for evolution (all reached max)")
            return
        
        
        mem_info = []
        for mem in evolvable_memories:
            mem_info.append({
                "thought_id": mem.thought_id,
                "behavior_explanation": mem.behavior_explanation,
                "pattern": mem.pattern_description,
                "evolution_count": mem.evolution_count 
            })
        

# Return ONLY JSON."""
        prompt = f"""Determine if past fail memories should be updated using a new fail case.
        New Pattern:
        - Behavior: {new_memory.behavior_explanation}
        - Pattern: {new_memory.pattern_description}

        Linked Past Patterns (with evolution history):
        {json.dumps(mem_info, indent=2)}

        Update Guidelines:
        - Update when the new fail case provides clearer correction evidence for an existing fail pattern.
        - Refine wording toward a stronger wrong-vs-correct contrast.
        - Prefer updates that improve future error avoidance rules.
        - Skip updates when the new fail case is unrelated.

        Return JSON:
        {{
        "should_evolve": true/false,
        "updates": [
            {{
            "thought_id": ID,
            "behavior_explanation": "updated text or null",
            "new_pattern": "updated text or null",
            "reasoning": "1 sentence explaining how the fail-correction rule is refined"
            }}
        ]
        }}
        Ensure updates are grounded in input data and reasoning is concise."""

        try:
            # response = self.model.generate_content(prompt)
            # time.sleep(3)
            response = self.qwen_generate(prompt=prompt, role_prompt='You are a behavioral memory modeling system.')
            result = extract_json_object(response)
            self._trace("memory_evolve_llm", {
                "prompt": prompt,
                "role_prompt": "You are a behavioral memory modeling system.",
                "answer": response,
                "parsed": result,
                "new_memory": behavior_memory_to_trace(new_memory),
                "linked_ids": linked_ids,
                "linked_memories": mem_info,
            })
            
            if result.get("should_evolve", False):
                updates = result.get("updates", [])
                
                for update in updates:
                    thought_id = update.get("thought_id")
                    memory = next((m for m in self.behavior_memories if m.thought_id == thought_id), None)
                    
                    if memory:
                        # ============ GHI LẠI GIÁ TRỊ CŨ ============
                        old_values = {
                            'behavior_explanation': memory.behavior_explanation,
                            'pattern_description': memory.pattern_description,
                            # 'extracted_preferences': memory.extracted_preferences.copy()
                        }
                    
                        updated = False
                        update_type = []
                        if update.get("behavior_explanation"):
                            memory.behavior_explanation = update["behavior_explanation"]
                            updated = True
                            update_type.append("behavior_explanation")

                        if update.get("new_pattern"):
                            memory.pattern_description = update["new_pattern"]
                            updated = True
                            update_type.append("pattern")
                        
                        # if update.get("additional_preferences"):
                        #     memory.extracted_preferences.extend(update["additional_preferences"])
                        #     memory.extracted_preferences = list(set(memory.extracted_preferences))
                        #     updated = True
                        #     update_type.append("preferences")
                        
                        # Regenerate embedding if updated
                        if updated:
                            # combined_text = f"{memory.behavior_explanation} {memory.pattern_description} {' '.join(memory.keywords)} {' '.join(memory.extracted_preferences)}"
                            combined_text = f"{memory.behavior_explanation} {memory.pattern_description} {' '.join(memory.keywords)}"
                            memory.embedding = self._create_embedding(combined_text)
                            new_values = {
                                'behavior_explanation': memory.behavior_explanation,
                                'pattern_description': memory.pattern_description,
                                # 'extracted_preferences': memory.extracted_preferences.copy()
                            }
                            
                            memory.record_evolution(
                                update_type=", ".join(update_type),
                                old_values=old_values,
                                new_values=new_values,
                                reasoning=update.get("reasoning", "")
                            )
                            self._trace("memory_evolved", {
                                "new_memory": behavior_memory_to_trace(new_memory),
                                "evolved_thought_id": thought_id,
                                "update_type": ", ".join(update_type),
                                "old_values": old_values,
                                "new_values": new_values,
                                "reasoning": update.get("reasoning", ""),
                                "evolution_count": memory.evolution_count,
                            })
                        
        except Exception as e:
            print(f"Error in memory evolution: {e}")
            self._trace("memory_evolve_error", {
                "error": str(e),
                "new_memory": behavior_memory_to_trace(new_memory),
                "linked_ids": linked_ids,
            })
    
    def add_behavior_memory(self,
                           interaction_window: List[UserInteraction],
                           k_neighbors: int = 5) -> BehaviorMemory:
        """Complete A-Mem pipeline: Create, Link, and Evolve"""
        # Step 1: Create behavior thought
        behavior_memory = self.create_behavior_thought(interaction_window, k_neighbors)
        
        # Step 2: Link with similar patterns
        linked_ids = self.link_behavior_memories(behavior_memory, k_neighbors)
        behavior_memory.links = linked_ids
        
        # Update bidirectional links
        for thought_id in linked_ids:
            memory = next((m for m in self.behavior_memories if m.thought_id == thought_id), None)
            if memory and behavior_memory.thought_id not in memory.links:
                memory.links.append(behavior_memory.thought_id)
        
        # Step 3: Evolve existing memories based on new pattern
        self.evolve_behavior_memories(behavior_memory, linked_ids)
        
        # Add to collection
        self.behavior_memories.append(behavior_memory)
        return behavior_memory
    
    def retrieve_relevant_memory_records(self, query_text: str, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve top-k memory records with similarity scores."""
        if len(self.behavior_memories) == 0:
            return []
        
        profile_embedding = self._create_embedding(query_text)
        
        similarities = []
        for memory in self.behavior_memories:
            sim = self._cosine_similarity(profile_embedding, memory.embedding)
            similarities.append((memory, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_records = [
            {"memory": mem, "similarity": float(sim)}
            for mem, sim in similarities[:k]
        ]
        self._trace("memory_retrieval", {
            "query_text": query_text,
            "k": k,
            "retrieved": [
                {
                    "similarity": float(sim),
                    "memory": behavior_memory_to_trace(mem),
                }
                for mem, sim in similarities[:k]
            ],
        })
        return top_records

    def retrieve_relevant_memories(self, user_profile_text: str, k: int = 5) -> List[BehaviorMemory]:
        """Backward-compatible retrieval API returning only memories."""
        return [record["memory"] for record in self.retrieve_relevant_memory_records(user_profile_text, k=k)]

    def record_memory_diagnostics(self, retrieved: int, kept: int, skipped: int) -> None:
        self.memory_diagnostics["eval_users"] += 1
        self.memory_diagnostics["retrieved_total"] += retrieved
        self.memory_diagnostics["kept_total"] += kept
        self.memory_diagnostics["skipped_total"] += skipped
        if kept > 0:
            self.memory_diagnostics["users_with_kept_memory"] += 1
    
    def llm_ranking(self,
                   train_items: List[Dict],
                   candidate_items: List[Dict],
                   retrieved_memories: Optional[List[BehaviorMemory]],
                   prompt_sample: str,
                   ranking_prompt_style: str = "memcf",
                   trace_context: Optional[Dict[str, Any]] = None) -> List[str]:
        """Score candidates with the LLM, then sort locally.

        Previous versions asked Qwen to output a full permutation of raw item IDs.
        Traces showed frequent duplicates/missing IDs. This score-based path asks
        for C01..C20 candidate scores and maps them back to item IDs in code.
        """
        user_profile = [
            {"title": item["title"], "category": item["category"]}
            for item in train_items
        ]
        candidate_info = [
            {"item_id": item["item_id"], "title": item["title"], "category": item["category"]}
            for item in candidate_items
        ]
        aliased_candidates, alias_to_item_id = add_candidate_aliases(candidate_info)
        valid_candidate_aliases = list(alias_to_item_id.keys())

        memory_thoughts = []
        if retrieved_memories:
            for mem in retrieved_memories:
                memory_thoughts.append({
                    "memory_id": mem.thought_id,
                    "behavior_explanation": mem.behavior_explanation,
                    "pattern": mem.pattern_description,
                    "applicable_when": getattr(mem, "applicable_when", []),
                    "not_applicable_when": getattr(mem, "not_applicable_when", []),
                    "wrong_item_type": getattr(mem, "wrong_item_type", ""),
                    "correct_item_type": getattr(mem, "correct_item_type", ""),
                    "evidence_terms_required": getattr(mem, "evidence_terms_required", []),
                    "specificity_score": getattr(mem, "specificity_score", 0.0),
                    "overgeneralization_risk": getattr(mem, "overgeneralization_risk", 0.0),
                    "keywords": getattr(mem, "keywords", [])[:10],
                })

        if ranking_prompt_style == "compact_score":
            prompt = build_compact_score_prompt(
                history_items=user_profile[-10:],
                aliased_candidates=aliased_candidates,
                prompt_sample=prompt_sample,
                memory_payload=memory_thoughts if retrieved_memories else None,
                user_profile_payload=None,
            )
        elif retrieved_memories:
            prompt = f"""
You are scoring candidate items for a recommender system.

Important memory policy:
- Retrieved memories may be irrelevant.
- Use a memory only if its applicable_when or evidence_terms_required directly appears in the current user history or candidate items.
- If a memory conflicts with recent user history or candidate facts, ignore the memory.
- If no memory is clearly applicable, score exactly as you would from user history and candidate facts only.
- Memories are weak evidence, not hard rules.

Inputs:
User Recent History (last interactions; prioritize most recent):
{json.dumps(user_profile[-10:], ensure_ascii=False, indent=2)}

Retrieved Fail-Correction Memories:
{json.dumps(memory_thoughts, ensure_ascii=False, indent=2)}

Candidate Items (use candidate_id only in output):
{json.dumps(aliased_candidates, ensure_ascii=False, indent=2)}

Output requirements:
- Return ONLY valid JSON. No markdown.
- Output one score row for every candidate_id exactly once.
- Score is a number from 0.0 to 1.0.
- Rationale must be <= 8 words.

JSON format:
{{
  "scores": [
    {{"candidate_id": "C01", "score": 0.0, "rationale": "short reason"}}
  ],
  "reasoning": "one short sentence"
}}
"""
        else:
            prompt = f"""
You are scoring candidate items for a recommender system based only on user history and candidate facts.
{prompt_sample}

Inputs:
User Recent History (last interactions; prioritize most recent):
{json.dumps(user_profile[-10:], ensure_ascii=False, indent=2)}

Candidate Items (use candidate_id only in output):
{json.dumps(aliased_candidates, ensure_ascii=False, indent=2)}

Output requirements:
- Return ONLY valid JSON. No markdown.
- Output one score row for every candidate_id exactly once.
- Score is a number from 0.0 to 1.0.
- Rationale must be <= 8 words.

JSON format:
{{
  "scores": [
    {{"candidate_id": "C01", "score": 0.0, "rationale": "short reason"}}
  ],
  "reasoning": "one short sentence"
}}
"""

        score_json_schema = {
            "type": "object",
            "properties": {
                "scores": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "candidate_id": {"type": "string", "enum": valid_candidate_aliases},
                            "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                            "rationale": {"type": "string"},
                        },
                        "required": ["candidate_id", "score", "rationale"],
                        "additionalProperties": False,
                    },
                    "minItems": len(valid_candidate_aliases),
                    "maxItems": len(valid_candidate_aliases),
                },
            },
            "required": ["scores"],
            "additionalProperties": False,
        }
        if ranking_prompt_style != "compact_score":
            score_json_schema["properties"]["reasoning"] = {"type": "string"}
            score_json_schema["required"] = ["scores", "reasoning"]

        max_retries = int(os.getenv("MEMCF_RANK_RETRIES", "1"))
        current_prompt = prompt
        last_error: Optional[str] = None
        final_ranked: Optional[List[str]] = None
        final_validation: Optional[Dict[str, Any]] = None
        final_result: Dict[str, Any] = {}
        raw_response = ""

        for attempt in range(max_retries + 1):
            try:
                raw_response = self.qwen_generate(
                    prompt=current_prompt,
                    role_prompt=(
                        "You are a deterministic recommender scorer. "
                        "Return JSON only and follow the provided JSON schema exactly."
                    ),
                    max_new_tokens=int(os.getenv("MEMCF_RANK_MAX_TOKENS", "1400")),
                    json_schema=score_json_schema,
                    json_mode=True,
                )
                try:
                    result = extract_json_object(raw_response)
                    raw_scores = result.get("scores", [])
                except Exception as parse_error:
                    last_error = str(parse_error)
                    result = {
                        "scores": parse_score_entries_from_text(raw_response, alias_to_item_id),
                        "reasoning": "Recovered score rows from malformed JSON",
                    }
                    raw_scores = result.get("scores", [])

                ranked_ids, validation = score_entries_to_ranking(raw_scores, alias_to_item_id)
                final_ranked = ranked_ids
                final_validation = validation
                final_result = result

                self.memory_diagnostics["rank_score_calls"] += 1
                self.memory_diagnostics["rank_missing_score_rows"] += len(validation["missing_candidate_ids"])
                self.memory_diagnostics["rank_invalid_score_rows"] += len(validation["invalid_or_duplicate_rows"])
                if validation["is_valid"]:
                    self.memory_diagnostics["rank_valid_score_outputs"] += 1
                else:
                    self.memory_diagnostics["rank_invalid_score_outputs"] += 1

                self._trace("ranking_llm", {
                    **(trace_context or {}),
                    "attempt": attempt,
                    "ranking_mode": "score_based_candidate_alias",
                    "prompt": current_prompt,
                    "role_prompt": (
                        "You are a deterministic recommender scorer. "
                        "Return JSON only and follow the provided JSON schema exactly."
                    ),
                    "answer": raw_response,
                    "parsed": result,
                    "score_validation": validation,
                    "raw_output_valid": validation["is_valid"],
                    "cleaned_ranked_item_ids": ranked_ids,
                    "candidate_items": candidate_info,
                    "aliased_candidate_items": aliased_candidates,
                    "alias_to_item_id": alias_to_item_id,
                    "train_items": user_profile[-10:],
                    "retrieved_memories": [
                        behavior_memory_to_trace(mem) for mem in (retrieved_memories or [])
                    ],
                    "use_retrieved_memories": retrieved_memories is not None,
                })

                if validation["is_valid"] or attempt >= max_retries:
                    if not validation["is_valid"]:
                        self._trace("ranking_retry_exhausted", {
                            **(trace_context or {}),
                            "attempts": attempt + 1,
                            "final_validation": validation,
                            "cleaned_ranked_item_ids": ranked_ids,
                        })
                    return ranked_ids

                current_prompt = f"""{prompt}

The previous answer was invalid:
{json.dumps(validation, ensure_ascii=False, indent=2)}

Retry now. Return ONLY valid JSON with exactly one score row for every candidate_id.
"""
            except Exception as e:
                last_error = str(e)
                self.memory_diagnostics["rank_attempt_errors"] += 1
                self._trace("ranking_attempt_error", {
                    **(trace_context or {}),
                    "attempt": attempt,
                    "error": last_error,
                    "prompt": current_prompt,
                    "candidate_items": candidate_info,
                    "aliased_candidate_items": aliased_candidates,
                    "retrieved_memories": [
                        behavior_memory_to_trace(mem) for mem in (retrieved_memories or [])
                    ],
                })
                if attempt >= max_retries:
                    break
                current_prompt = f"""{prompt}

The previous answer could not be parsed because:
{last_error}

Retry now. Return ONLY valid JSON with exactly one score row for every candidate_id.
"""

        print(f"Error in LLM scoring/ranking: {last_error}")
        fallback_ids = [str(item["item_id"]) for item in candidate_items]
        self.memory_diagnostics["rank_fallbacks"] += 1
        self._trace("ranking_error", {
            **(trace_context or {}),
            "error": last_error,
            "prompt": prompt,
            "answer": raw_response,
            "parsed": final_result,
            "score_validation": final_validation,
            "candidate_items": candidate_info,
            "aliased_candidate_items": aliased_candidates,
            "fallback_ranked_item_ids": fallback_ids,
            "retrieved_memories": [
                behavior_memory_to_trace(mem) for mem in (retrieved_memories or [])
            ],
        })
        return fallback_ids

    def get_evolution_statistics(self) -> Dict[str, Any]:
        """Phân tích thống kê về evolution của các memories"""
        if not self.behavior_memories:
            return {}
        
        evolution_counts = [m.evolution_count for m in self.behavior_memories]
        
        stats = {
            'total_memories': len(self.behavior_memories),
            'total_evolutions': sum(evolution_counts),
            'avg_evolutions_per_memory': np.mean(evolution_counts),
            'max_evolutions': max(evolution_counts),
            'min_evolutions': min(evolution_counts),
            'std_evolutions': np.std(evolution_counts),
            'memories_never_evolved': sum(1 for c in evolution_counts if c == 0),
            'memories_evolved_once': sum(1 for c in evolution_counts if c == 1),
            'memories_evolved_multiple': sum(1 for c in evolution_counts if c > 1),
            'evolution_distribution': {
                f'{i}_times': sum(1 for c in evolution_counts if c == i)
                for i in range(max(evolution_counts) + 1)
            }
        }
        
        # Top memories theo evolution count
        top_evolved = sorted(
            [(m.thought_id, m.evolution_count, m.behavior_explanation) 
            for m in self.behavior_memories],
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        stats['top_10_most_evolved'] = [
            {
                'thought_id': tid,
                'evolution_count': count,
                'behavior': behavior[:100]  # Truncate
            }
            for tid, count, behavior in top_evolved
        ]
        
        return stats

    def print_evolution_report(self):
        """In báo cáo evolution"""
        stats = self.get_evolution_statistics()
        
        print("\n" + "="*80)
        print("MEMORY EVOLUTION REPORT")
        print("="*80)
        print(f"Total Memories: {stats['total_memories']}")
        print(f"Total Evolutions: {stats['total_evolutions']}")
        print(f"Average Evolutions per Memory: {stats['avg_evolutions_per_memory']:.2f}")
        print(f"Max Evolutions: {stats['max_evolutions']}")
        print(f"Min Evolutions: {stats['min_evolutions']}")
        print(f"Std Deviation: {stats['std_evolutions']:.2f}")
        print("-"*80)
        print(f"Never Evolved: {stats['memories_never_evolved']}")
        print(f"Evolved Once: {stats['memories_evolved_once']}")
        print(f"Evolved Multiple Times: {stats['memories_evolved_multiple']}")
        print("-"*80)
        print("Evolution Distribution:")
        for times, count in stats['evolution_distribution'].items():
            if count > 0:
                print(f"  {times}: {count} memories")
        print("-"*80)
        print("Top 10 Most Evolved Memories:")
        for item in stats['top_10_most_evolved']:
            print(f"  ID {item['thought_id']}: {item['evolution_count']} evolutions")
            print(f"    → {item['behavior']}")

    def save_memory(self, filepath: str, format: str = 'json') -> None:
        """
        Lưu memory system ra file (chứa memories của TẤT CẢ users)
        
        Args:
            filepath: Đường dẫn file để lưu
            format: Định dạng file ('json' hoặc 'pickle')
        """
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        if format == 'json':
            memories_dict = [mem.to_dict() for mem in self.behavior_memories]
            interactions_dict = [asdict(interaction) for interaction in self.user_interaction_history]
            
            data = {
                'behavior_memories': memories_dict,
                'user_interaction_history': interactions_dict,
                'next_thought_id': self.next_thought_id,
                'metadata': {
                    'num_memories': len(self.behavior_memories),
                    'num_interactions': len(self.user_interaction_history),
                    'save_timestamp': datetime.now().isoformat()
                }
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            file_size_mb = os.path.getsize(filepath) / (1024*1024)
            print(f"✓ Memory saved to {filepath}")
            print(f"  - Format: JSON")
            print(f"  - Memories: {len(self.behavior_memories)}")
            print(f"  - Interactions: {len(self.user_interaction_history)}")
            print(f"  - File size: {file_size_mb:.2f} MB")
            
        elif format == 'pickle':
            data = {
                'behavior_memories': self.behavior_memories,
                'user_interaction_history': self.user_interaction_history,
                'next_thought_id': self.next_thought_id,
                'metadata': {
                    'num_memories': len(self.behavior_memories),
                    'num_interactions': len(self.user_interaction_history),
                    'save_timestamp': datetime.now().isoformat()
                }
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            
            file_size_mb = os.path.getsize(filepath) / (1024*1024)
            print(f"✓ Memory saved to {filepath}")
            print(f"  - Format: Pickle")
            print(f"  - Memories: {len(self.behavior_memories)}")
            print(f"  - Interactions: {len(self.user_interaction_history)}")
            print(f"  - File size: {file_size_mb:.2f} MB")
        
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'json' or 'pickle'")

    def load_memory(self, filepath: str, format: str = None) -> None:
        """
        Tải memory system từ file
        
        Args:
            filepath: Đường dẫn file để đọc
            format: Định dạng file ('json' hoặc 'pickle'). Nếu None, tự động detect từ extension
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        if format is None:
            if filepath.endswith('.json'):
                format = 'json'
            elif filepath.endswith('.pkl') or filepath.endswith('.pickle'):
                format = 'pickle'
            else:
                try:
                    with open(filepath, 'r') as f:
                        json.load(f)
                    format = 'json'
                except:
                    format = 'pickle'
        
        if format == 'json':
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.behavior_memories = [
                BehaviorMemory.from_dict(mem_dict) 
                for mem_dict in data['behavior_memories']
            ]
            
            self.user_interaction_history = [
                UserInteraction(**interaction_dict)
                for interaction_dict in data['user_interaction_history']
            ]
            
            self.next_thought_id = data['next_thought_id']
            
            print(f"✓ Memory loaded from {filepath}")
            print(f"  - Format: JSON")
            print(f"  - Memories: {len(self.behavior_memories)}")
            print(f"  - Interactions: {len(self.user_interaction_history)}")
            
        elif format == 'pickle':
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.behavior_memories = data['behavior_memories']
            self.user_interaction_history = data['user_interaction_history']
            self.next_thought_id = data['next_thought_id']
            
            print(f"✓ Memory loaded from {filepath}")
            print(f"  - Format: Pickle")
            print(f"  - Memories: {len(self.behavior_memories)}")
            print(f"  - Interactions: {len(self.user_interaction_history)}")
        
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'json' or 'pickle'")

import os
from typing import Dict, List

def save_all_users_ranking_results(all_results: List[Dict],
                                  items_meta: Dict,
                                  output_file: str = "all_users_ranking_results.json"):
    """
    Lưu toàn bộ kết quả ranking của tất cả users vào 1 file JSON duy nhất.
    
    Args:
        all_results: List các dict chứa thông tin của từng user
        items_meta: Metadata items để lấy title, category,...
        output_file: Tên file output (sẽ tự tạo thư mục nếu cần)
    """
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    def get_item_info(item_id: str) -> Dict:
        if item_id in items_meta:
            info = items_meta[item_id]
            return {
                "item_id": item_id,
                "title": item_title(info, item_id),
                "category": item_category(info),
                # "brand": info.get("brand", ""),
                # "price": info.get("price", None)
            }
        else:
            return {
                "item_id": item_id,
                "title": f"Unknown Item {item_id}",
                "category": "Unknown",
                # "brand": "",
                # "price": None
            }
    
    # Chuyển đổi chi tiết items cho tất cả users
    final_results = []
    for res in all_results:
        user_result = {
            "user_id": res["user_id"],
            "num_candidates": len(res["candidates"]),
            "ground_truth_item_ids": res["ground_truth"],
            "candidate_item_ids": res["candidates"],
            "reranked_item_ids": res["predictions"],
            # "ground_truth_items": [get_item_info(iid) for iid in res["ground_truth"]],
            "candidate_items": [get_item_info(iid) for iid in res["candidates"]],
            "reranked_items": [get_item_info(iid) for iid in res["predictions"]],
            "metrics": res["metrics"],  # thêm metrics của user này
            "baseline_metrics": res["baseline_metrics"]
        }
        final_results.append(user_result)
    
    # Lưu vào 1 file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Saved ranking results of {len(final_results)} users to {output_file}")
    print(f"   File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")


def load_data(items_path: str, sequences_path: str, negatives_path: str):
    """Load Amazon dataset"""
    print("Loading data...")
    
    with open(items_path, 'r', encoding="utf-8") as f:
        items_meta = json.load(f)
    for item_id, item_info in list(items_meta.items()):
        if not isinstance(item_info, dict):
            item_info = {"title": str(item_info)}
            items_meta[item_id] = item_info
        item_info["title"] = item_title(item_info, str(item_id))
        item_info["main_cat"] = item_category(item_info)
        item_info["category_normalized"] = True
    
    with open(sequences_path, 'r', encoding="utf-8") as f:
        user_sequences = json.load(f)
    
    with open(negatives_path, 'r', encoding="utf-8") as f:
        user_negatives = json.load(f)
    
    print(f"Loaded {len(items_meta)} items")
    print(f"Loaded {len(user_sequences)} users")
    
    return items_meta, user_sequences, user_negatives

def calculate_recall_at_k(predictions: List[str], ground_truth: List[str], k: int) -> float:
    """Calculate Recall@K"""
    top_k = predictions[:k]
    hits = len(set(top_k) & set(ground_truth))
    return hits / len(ground_truth) if ground_truth else 0.0

def calculate_ndcg_at_k(predictions: List[str], ground_truth: List[str], k: int) -> float:
    """Calculate NDCG@K"""
    top_k = predictions[:k]
    
    # DCG
    dcg = 0.0
    for i, item in enumerate(top_k):
        if item in ground_truth:
            dcg += 1.0 / np.log2(i + 2)
    
    # IDCG
    idcg = sum([1.0 / np.log2(i + 2) for i in range(min(len(ground_truth), k))])
    
    return dcg / idcg if idcg > 0 else 0.0


def init_pairwise_item_states(items_meta: Dict[str, Dict[str, Any]]) -> Dict[str, PairwiseItemState]:
    """Initialize item states for pairwise failure training."""
    item_states: Dict[str, PairwiseItemState] = {}
    for item_id, item_info in items_meta.items():
        title = item_title(item_info, str(item_id))
        category = item_category(item_info)
        memory = f"The item is called '{title}'. The category is: '{category}'."
        item_states[item_id] = PairwiseItemState(
            item_id=item_id,
            title=title,
            category=category,
            memory=memory,
        )
    return item_states


def get_or_create_user_state(user_states: Dict[str, PairwiseUserState], user_id: str) -> PairwiseUserState:
    if user_id not in user_states:
        user_states[user_id] = PairwiseUserState(user_id=user_id)
    return user_states[user_id]


def _extract_json_from_llm_output(raw_output: str) -> Dict[str, Any]:
    cleaned = raw_output.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*", "", cleaned).rstrip("```").strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in LLM output")
    json_str = match.group(0)
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # Local OpenAI-compatible models sometimes emit invalid backslash escapes.
        json_str = re.sub(r"\\(?![\"\\/bfnrtu])", r"\\\\", json_str)
        json_str = re.sub(r",\s*([}\]])", r"\1", json_str)
        return json.loads(json_str)


def autonomous_pairwise_interaction(
    memory_system: RecommendationMemorySystem,
    user_state: PairwiseUserState,
    pos_item: PairwiseItemState,
    neg_item: PairwiseItemState,
) -> Tuple[str, str]:
    """pairwise autonomous interaction: choose between positive/negative item."""
    prompt = f"""You are an enthusiast. Here is your self-introduction: "{user_state.short_term_memory}"

Now, you are considering to select an item from two candidates:
1. Title: {neg_item.title}, Description: {neg_item.memory}
2. Title: {pos_item.title}, Description: {pos_item.memory}
\n\n Please select the item that aligns best with your preferences and explain your choice while rejecting the other. \n Follow these steps:\n 1. Extract your preferences and dislikes from your self-introduction. \n 2. Evaluate the two items based on your preferences and how they relate to the item features.\n 3. Explain your choice, detailing the relationship between your preferences/dislikes and the item features

\n\n Important notes:
\n 1. Do not fabricate your preferences! If your self-introduction lacks relevant details, use common knowledge to guide your decision, such as item popularity. \n 2. Select one candidate, not both. \n 3. Your explanation should be specific; general preferences like genre are insufficient. Focus on the item's finer attributes and be concise! \n 4. Base your explanation on facts. If your self-introduction doesn't specify preferences, you cannot claim your decision was influenced by them."

Output format:
Chosen Item: [1 or 2]
Explanation: [Your detailed reasoning]

Important: You must choose one of these two candidates."""

    response = memory_system.qwen_generate(prompt=prompt)
    chosen_item_id = pos_item.item_id
    if "Chosen Item: 1" in response or "chosen item: 1" in response.lower():
        chosen_item_id = neg_item.item_id
    memory_system._trace("autonomous_choice_llm", {
        "user_id": user_state.user_id,
        "prompt": prompt,
        "role_prompt": "You are a helpful AI assistant.",
        "answer": response,
        "positive_item": asdict(pos_item),
        "negative_item": asdict(neg_item),
        "chosen_item_id": chosen_item_id,
        "is_failure": chosen_item_id != pos_item.item_id,
        "user_state": asdict(user_state),
    })
    return chosen_item_id, response


def corrective_pairwise_reflection(
    memory_system: RecommendationMemorySystem,
    user_state: PairwiseUserState,
    pos_item: PairwiseItemState,
    neg_item: PairwiseItemState,
    chosen_item_id: str,
    explanation: str,
) -> None:
    """pairwise reflection update for user memory and item memories."""
    if chosen_item_id == pos_item.item_id:
        return

    user_prompt = f"""You are an enthusiast with these preferences: "{user_state.short_term_memory}"

Recently, you chose between two items:
1. Title: {neg_item.title}, Description: {neg_item.memory}
2. Title: {pos_item.title}, Description: {pos_item.memory}

You selected item 1, but you discovered you actually prefer item 2 instead.
Your previous explanation was: "{explanation}"

This indicates an incorrect choice, and your previous judgment about your preferences was mistaken. Your task now is to update your self-introduction with your new preferences and dislikes. \n Follow these steps: \n 1. Analyze misconceptions in your previous judgment and correct them.\n 2. Identify new preferences from '{pos_item.title}' and dislikes from '{neg_item.title}'. \n 3. Summarize your past preferences, merging them with new insights and removing conflicting parts.\n 4. Update your self-introduction, starting with new preferences, then summarizing past ones, followed by dislikes. \n\n Important notes: 1. Keep it under 150 words.  \n 2. Be concise and clear. \n 3. Describe only the features of items you prefer or dislike, without mentioning your thought process. \n 4. Your self-introduction should be specific and personalized; avoid generic preferences."

Output format:
My updated self-introduction: [Your updated preferences in under 150 words]

Important: Focus on what features you like and dislike, be specific and personalized."""

    new_user_memory = memory_system.qwen_generate(prompt=user_prompt)
    memory_system._trace("reflection_user_memory_llm", {
        "user_id": user_state.user_id,
        "prompt": user_prompt,
        "role_prompt": "You are a helpful AI assistant.",
        "answer": new_user_memory,
        "old_user_memory": user_state.short_term_memory,
        "positive_item": asdict(pos_item),
        "negative_item": asdict(neg_item),
        "chosen_item_id": chosen_item_id,
    })
    if "My updated self-introduction:" in new_user_memory:
        new_user_memory = new_user_memory.split("My updated self-introduction:")[1].strip()
    old_user_memory = user_state.short_term_memory
    user_state.update_memory(new_user_memory)
    memory_system._trace("reflection_user_memory_updated", {
        "user_id": user_state.user_id,
        "old_user_memory": old_user_memory,
        "new_user_memory": user_state.short_term_memory,
        "positive_item_id": pos_item.item_id,
        "negative_item_id": neg_item.item_id,
    })

    item_prompt = f"""A user with these preferences browsed items: "{user_state.short_term_memory}"

The user considered two items:
1. Title: {pos_item.title}, Description: {pos_item.memory}
2. Title: {neg_item.title}, Description: {neg_item.memory}

The user initially chose item 2 but actually prefers item 1, indicating the descriptions may be misleading.

Your task is to update the descriptions of these items based on these insights. \n Follow these steps:\n 1. Analyze the user's preferences and dislikes from the self-description. \n 2. Explore the chosen item's features that align with preferences and oppose dislikes, and examine the rejected item's features that align with dislikes and oppose preferences. Highlight the differences thoroughly. \n 3. Incorporate new features into the previous descriptions, preserving key information while being concise.\n\n Important notes: \n 1. Your output should be in the following format: 'The updated description of the first item is: [updated description]. \\n The updated description of the second item is: [updated description].'. \n 2. Each updated description cannot exceed 50 words; be concise and clear! \n 3. In your updated descriptions, refer to preferences collectively, avoiding individual references. For example, say 'the user with ... preferences/dislikes'.\n 4. New features should reflect user preferences, and the updated descriptions must not contradict the inherent characteristics of the items, e.g., do not describe a thriller as having a predictably happy ending.

Update the description of item 1 to better reflect why users with these preferences would like it.

Output format (STRICT JSON, no extra text):
{{
  "item_1": "<updated description, single paragraph>",
  "item_2": "<updated description, single paragraph>"
}}

Important: Make it specific and aligned with user preferences."""

    new_item_memory = memory_system.qwen_generate(prompt=item_prompt)
    memory_system._trace("reflection_item_memory_llm", {
        "user_id": user_state.user_id,
        "prompt": item_prompt,
        "role_prompt": "You are a helpful AI assistant.",
        "answer": new_item_memory,
        "old_positive_item_memory": pos_item.memory,
        "old_negative_item_memory": neg_item.memory,
        "positive_item": asdict(pos_item),
        "negative_item": asdict(neg_item),
    })

    # Item reflection should improve local item descriptions, but it should not
    # block fail-memory creation when the local LLM emits malformed JSON.
    try:
        data = _extract_json_from_llm_output(new_item_memory)
        item1_desc = data["item_1"].strip()
        item2_desc = data["item_2"].strip()
    except Exception as e:
        print(f"  ⚠ Item reflection JSON parse failed; keeping previous item memories: {e}")
        memory_system._trace("reflection_error", {
            "user_id": user_state.user_id,
            "error": str(e),
            "positive_item_id": pos_item.item_id,
            "negative_item_id": neg_item.item_id,
            "raw_answer": new_item_memory,
        })
        item1_desc = pos_item.memory
        item2_desc = neg_item.memory

    # 5. Update memories
    old_positive_item_memory = pos_item.memory
    old_negative_item_memory = neg_item.memory
    pos_item.memory = item1_desc
    neg_item.memory = item2_desc
    memory_system._trace("reflection_item_memory_updated", {
        "user_id": user_state.user_id,
        "positive_item_id": pos_item.item_id,
        "negative_item_id": neg_item.item_id,
        "old_positive_item_memory": old_positive_item_memory,
        "old_negative_item_memory": old_negative_item_memory,
        "new_positive_item_memory": pos_item.memory,
        "new_negative_item_memory": neg_item.memory,
    })


def train_memory_from_fail_interactions(
    user_id: str,
    user_data: Dict,
    memory_system: RecommendationMemorySystem,
    user_states: Dict[str, PairwiseUserState],
    item_states: Dict[str, PairwiseItemState],
    items_meta: Dict[str, Dict[str, Any]],
    negative_data: Optional[Dict[str, Any]] = None,
    max_iterations: int = 1,
    max_positive_interactions: Optional[int] = None,
    candidate_negative_mode: str = "random",
    min_lesson_confidence: float = 0.25,
    max_lesson_risk: float = 0.85,
    max_failure_lessons_per_user: int = 3,
) -> List[BehaviorMemory]:
    """
    Hybrid training:
    - pairwise initialization and interaction loop.
    - Create behavior memories ONLY from failed interactions.
    """
    train_items = user_data["train"]
    if max_positive_interactions and max_positive_interactions > 0:
        train_items = train_items[-max_positive_interactions:]
    else:
        train_items = train_items[-30:]
    if len(train_items) == 0:
        return []

    user_state = get_or_create_user_state(user_states, user_id)
    all_item_ids = list(item_states.keys())
    if not all_item_ids:
        return []

    # temp memory system must be local for this user
    memory_system.user_interaction_history = []
    memory_system.behavior_memories = []
    memory_system.next_thought_id = 0

    new_memories: List[BehaviorMemory] = []
    for pos_item_id in train_items:
        if pos_item_id not in item_states:
            continue

        neg_item_id = choose_training_negative_item_id(
            user_id=str(user_id),
            pos_item_id=str(pos_item_id),
            user_data=user_data,
            negative_data=negative_data,
            items_meta=items_meta,
            all_item_ids=all_item_ids,
            mode=candidate_negative_mode,
            max_positive_interactions=max_positive_interactions,
        )
        if not neg_item_id:
            continue

        pos_item = item_states[pos_item_id]
        neg_item = item_states[neg_item_id]

        for _ in range(max_iterations):
            chosen_item_id, explanation = autonomous_pairwise_interaction(
                memory_system=memory_system,
                user_state=user_state,
                pos_item=pos_item,
                neg_item=neg_item,
            )

            if chosen_item_id == pos_item_id:
                user_state.add_interaction(pos_item_id)
                break

            try:
                corrective_pairwise_reflection(
                    memory_system=memory_system,
                    user_state=user_state,
                    pos_item=pos_item,
                    neg_item=neg_item,
                    chosen_item_id=chosen_item_id,
                    explanation=explanation,
                )
            except Exception as e:
                print(f"  ⚠ Reflection failed for user {user_id}, item {pos_item_id}: {e}")
                continue

            # Memory unit is one failed interaction pair instead of sliding windows.
            fail_window = [
                UserInteraction(
                    item_id=neg_item.item_id,
                    item_name=neg_item.title,
                    item_category=neg_item.category,
                    action_type="wrong_choice",
                    metadata={"user_id": user_id, "role": "chosen_wrong"},
                ),
                UserInteraction(
                    item_id=pos_item.item_id,
                    item_name=pos_item.title,
                    item_category=pos_item.category,
                    action_type="preferred_item",
                    metadata={"user_id": user_id, "role": "ground_truth"},
                ),
            ]
            try:
                fail_memory = memory_system.create_behavior_thought(fail_window)
                passed_gate, gate_reason = behavior_memory_passes_quality_gate(
                    fail_memory,
                    fail_window,
                    min_confidence=min_lesson_confidence,
                    max_risk=max_lesson_risk,
                )
                memory_system._trace("memory_quality_gate", {
                    "user_id": user_id,
                    "positive_item_id": pos_item.item_id,
                    "negative_item_id": neg_item.item_id,
                    "passed": passed_gate,
                    "reason": gate_reason,
                    "min_lesson_confidence": min_lesson_confidence,
                    "max_lesson_risk": max_lesson_risk,
                    "memory": behavior_memory_to_trace(fail_memory),
                })
                if not passed_gate:
                    continue
                memory_system._trace("fail_memory_from_wrong_choice", {
                    "user_id": user_id,
                    "positive_item_id": pos_item.item_id,
                    "negative_item_id": neg_item.item_id,
                    "chosen_wrong_item_id": neg_item.item_id,
                    "preferred_item_id": pos_item.item_id,
                    "memory": behavior_memory_to_trace(fail_memory),
                    "choice_explanation": explanation,
                })
                new_memories.append(fail_memory)
                if max_failure_lessons_per_user > 0 and len(new_memories) >= max_failure_lessons_per_user:
                    memory_system._trace("memory_generation_limit_reached", {
                        "user_id": user_id,
                        "max_failure_lessons_per_user": max_failure_lessons_per_user,
                        "current_count": len(new_memories),
                    })
                    return new_memories
            except Exception as e:
                print(f"  ⚠ Fail-memory creation error for user {user_id}: {e}")
                memory_system._trace("fail_memory_error", {
                    "user_id": user_id,
                    "positive_item_id": pos_item.item_id,
                    "negative_item_id": neg_item.item_id,
                    "error": str(e),
                })

    return new_memories

def evaluate_user(user_data: Dict,
                 negative_data: Dict,
                 items_meta: Dict,
                 memory_system: RecommendationMemorySystem,
                 eval_type: str = 'test', use_memory = True, k_memories: int = 5, sample_user_list: List = None, negative_data_sample_list: List = None,
                 max_positive_interactions: Optional[int] = None, max_negative_candidates: Optional[int] = None,
                 user_id: Optional[str] = None,
                 memory_retrieval_mode: str = "user_only",
                 memory_gate: str = "none",
                 memory_similarity_threshold: float = 0.35,
                 no_harm_arbitration: bool = False,
                 no_harm_min_applicability: float = 1.0,
                 ranking_prompt_style: str = "memcf") -> Dict[str, float]:
    """Evaluate for a single user with LLM-based ranking"""
    
    # Get ground truth and candidates
    if eval_type == 'val':
        ground_truth = user_data['val']
        negatives = negative_data.get('val_neg', [])
    else:  # test
        ground_truth = user_data['test']
        negatives = negative_data.get('test_neg', [])
    if max_negative_candidates and max_negative_candidates > 0:
        negatives = negatives[:max_negative_candidates]

    if sample_user_list is not None:
        ground_truth_sample_fewshot = []
        negatives_sample_fewshot = []
        for i in range(len(sample_user_list)):
            sample_user_data = sample_user_list[i]
            negative_data_sample = negative_data_sample_list[i]

            ground_truth_sample = sample_user_data.get('val', [])
            ground_truth_sample_fewshot.append(ground_truth_sample)

            negatives_sample = negative_data_sample.get('val_neg', [])
            if max_negative_candidates and max_negative_candidates > 0:
                negatives_sample = negatives_sample[:max_negative_candidates]
            negatives_sample_fewshot.append(negatives_sample)

    # Prepare train items for user profile
    train_items_info = []
    user_profile_texts = []
    train_history_for_profile = user_data['train']
    if max_positive_interactions and max_positive_interactions > 0:
        train_history_for_profile = train_history_for_profile[-max_positive_interactions:]
    else:
        train_history_for_profile = train_history_for_profile[-10:]
    for item_id in train_history_for_profile:
        if item_id in items_meta:
            item_info = items_meta[item_id]
            title = item_title(item_info, str(item_id))
            category = item_category(item_info)
            
            train_items_info.append({
                'item_id': item_id,
                'title': title,
                'category': category
            })
            user_profile_texts.append(f"{title} {category}")
    
    # Create user profile text for retrieval
    user_profile_text = " ".join(user_profile_texts)

    # create sample for fewshot ranking
    prompt_sample = ''
    if sample_user_list is not None:
        prompt_sample = 'Learn from the following examples:\n'
        for i in range(len(sample_user_list)):
            sample_user_data = sample_user_list[i]
            sample_train_items_info = []
            sample_history = sample_user_data['train']
            if max_positive_interactions and max_positive_interactions > 0:
                sample_history = sample_history[-max_positive_interactions:]
            else:
                sample_history = sample_history[-10:]
            for item_id in sample_history:
                if item_id in items_meta:
                    item_info = items_meta[item_id]
                    title = item_title(item_info, str(item_id))
                    category = item_category(item_info)
                    
                    sample_train_items_info.append({
                        'item_id': item_id,
                        'title': title,
                        'category': category
                    })
            sample_user_profile_texts = []
            for item in sample_train_items_info:
                sample_user_profile_texts.append(f"{item['title']} {item['category']}")
            sample_user_profile_text = " ".join(sample_user_profile_texts)

            candidates_sample = deterministic_shuffle(
                ground_truth_sample_fewshot[i] + negatives_sample_fewshot[i],
                salt=f"fewshot_{i}",
            )
            candidate_items_info_sample = []
            for item_id in candidates_sample:
                if item_id in items_meta:
                    item_info = items_meta[item_id]
                    candidate_items_info_sample.append({
                        'item_id': item_id,
                        'title': item_title(item_info, str(item_id)),
                        'category': item_category(item_info)
                    })
                else:
                    candidate_items_info_sample.append({
                        'item_id': item_id,
                        'title': f'Item {item_id}',
                        'category': 'Unknown'
                    })
            prompt_sample += f"""
            Example {i+1}:
            Other user Recent History: {sample_user_profile_text}
            Candidate Items: {json.dumps(candidate_items_info_sample, indent=2)}
            You should set the true items "{json.dumps(ground_truth_sample_fewshot[i], indent=2)}" at the top of the ranking.\n
            """
        # user_profile_text += " " + sample_user_profile_text
    
    # Combine ground truth and negatives as candidates
    candidates = deterministic_shuffle(ground_truth + negatives, salt=f"{eval_type}_candidates")
    candidate_items_info = []
    for item_id in candidates:
        if item_id in items_meta:
            item_info = items_meta[item_id]
            candidate_items_info.append({
                'item_id': item_id,
                'title': item_title(item_info, str(item_id)),
                'category': item_category(item_info)
            })
        else:
            candidate_items_info.append({
                'item_id': item_id,
                'title': f'Item {item_id}',
                'category': 'Unknown'
            })
    
    retrieval_query_text = None
    retrieved_memory_records: List[Dict[str, Any]] = []
    gate_decisions: List[Dict[str, Any]] = []
    if use_memory:
        retrieval_query_text = build_retrieval_query(
            user_profile_text=user_profile_text,
            candidate_items_info=candidate_items_info,
            mode=memory_retrieval_mode,
        )
        retrieved_memory_records = memory_system.retrieve_relevant_memory_records(retrieval_query_text, k=k_memories)
        kept_memory_records, gate_decisions = gate_memory_records(
            memory_records=retrieved_memory_records,
            user_profile_text=user_profile_text,
            candidate_items_info=candidate_items_info,
            gate_mode=memory_gate,
            similarity_threshold=memory_similarity_threshold,
        )
        for decision in gate_decisions:
            memory_system._trace("memory_gate_decision", {
                "user_id": user_id,
                "eval_type": eval_type,
                "retrieval_mode": memory_retrieval_mode,
                "memory_gate": memory_gate,
                **decision,
            })
        memory_system.record_memory_diagnostics(
            retrieved=len(retrieved_memory_records),
            kept=len(kept_memory_records),
            skipped=len(retrieved_memory_records) - len(kept_memory_records),
        )
        retrieved_memories = [record["memory"] for record in kept_memory_records]
        if len(retrieved_memories) == 0:
            retrieved_memories = None
    else:
        retrieved_memories = None
    no_memory_predictions = None
    memory_predictions = None
    selected_ranking_source = "memory" if retrieved_memories else "no_memory"
    arbitration_decision: Dict[str, Any] = {}

    if use_memory and no_harm_arbitration:
        memory_system.memory_diagnostics["no_harm_users"] += 1
        no_memory_predictions = memory_system.llm_ranking(
            train_items_info,
            candidate_items_info,
            None,
            prompt_sample,
            ranking_prompt_style=ranking_prompt_style,
            trace_context={
                "user_id": user_id,
                "eval_type": eval_type,
                "use_memory": False,
                "ranking_path": "no_harm_no_memory_candidate",
                "memory_retrieval_mode": memory_retrieval_mode,
                "memory_gate": memory_gate,
                "memory_similarity_threshold": memory_similarity_threshold,
                "retrieval_query_text": retrieval_query_text,
                "ground_truth": ground_truth,
                "fixed_candidates": candidates,
            },
        )
        if retrieved_memories:
            memory_predictions = memory_system.llm_ranking(
                train_items_info,
                candidate_items_info,
                retrieved_memories,
                prompt_sample,
                ranking_prompt_style=ranking_prompt_style,
                trace_context={
                    "user_id": user_id,
                    "eval_type": eval_type,
                    "use_memory": True,
                    "ranking_path": "no_harm_memory_candidate",
                    "memory_retrieval_mode": memory_retrieval_mode,
                    "memory_gate": memory_gate,
                    "memory_similarity_threshold": memory_similarity_threshold,
                    "retrieval_query_text": retrieval_query_text,
                    "ground_truth": ground_truth,
                    "fixed_candidates": candidates,
                },
            )
            kept_decisions = [d for d in gate_decisions if d.get("decision") == "keep"]
            max_applicability = max(
                [float(d.get("applicability_score", 0.0)) for d in kept_decisions] or [0.0]
            )
            max_strong_terms = max(
                [len(d.get("strong_matched_terms", [])) for d in kept_decisions] or [0]
            )
            use_memory_ranking = (
                max_applicability >= no_harm_min_applicability
                and max_strong_terms > 0
            )
            if use_memory_ranking:
                predictions = memory_predictions
                selected_ranking_source = "memory"
                memory_system.memory_diagnostics["no_harm_used_memory"] += 1
            else:
                predictions = no_memory_predictions
                selected_ranking_source = "no_memory"
                memory_system.memory_diagnostics["no_harm_fallback_no_memory"] += 1
            arbitration_decision = {
                "enabled": True,
                "selected_ranking_source": selected_ranking_source,
                "max_applicability": max_applicability,
                "max_strong_terms": max_strong_terms,
                "no_harm_min_applicability": no_harm_min_applicability,
                "reason": (
                    "memory passed no-harm evidence threshold"
                    if use_memory_ranking
                    else "fallback to no-memory: insufficient memory applicability evidence"
                ),
            }
        else:
            predictions = no_memory_predictions
            selected_ranking_source = "no_memory"
            memory_system.memory_diagnostics["no_harm_fallback_no_memory"] += 1
            arbitration_decision = {
                "enabled": True,
                "selected_ranking_source": selected_ranking_source,
                "reason": "fallback to no-memory: no kept retrieved memories",
            }
        memory_system._trace("no_harm_arbitration", {
            "user_id": user_id,
            "eval_type": eval_type,
            "decision": arbitration_decision,
            "gate_decisions": gate_decisions,
            "no_memory_predictions": no_memory_predictions,
            "memory_predictions": memory_predictions,
            "selected_predictions": predictions,
        })
    else:
        # Use LLM to rank candidates
        predictions = memory_system.llm_ranking(
            train_items_info,
            candidate_items_info,
            retrieved_memories,
            prompt_sample,
            ranking_prompt_style=ranking_prompt_style,
            trace_context={
                "user_id": user_id,
                "eval_type": eval_type,
                "use_memory": use_memory,
                "ranking_path": "single_path",
                "memory_retrieval_mode": memory_retrieval_mode,
                "memory_gate": memory_gate,
                "memory_similarity_threshold": memory_similarity_threshold,
                "retrieval_query_text": retrieval_query_text,
                "ground_truth": ground_truth,
                "fixed_candidates": candidates,
            },
        )
    baseline_metric = {
        'recall@5': calculate_recall_at_k(candidates, ground_truth, 5),
        'recall@10': calculate_recall_at_k(candidates, ground_truth, 10),
        'recall@20': calculate_recall_at_k(candidates, ground_truth, 20),
        'ndcg@5': calculate_ndcg_at_k(candidates, ground_truth, 5),
        'ndcg@10': calculate_ndcg_at_k(candidates, ground_truth, 10),
        'ndcg@20': calculate_ndcg_at_k(candidates, ground_truth, 20),
    }
    # Calculate metrics
    metrics = {
        'recall@5': calculate_recall_at_k(predictions, ground_truth, 5),
        'recall@10': calculate_recall_at_k(predictions, ground_truth, 10),
        'recall@20': calculate_recall_at_k(predictions, ground_truth, 20),
        'ndcg@5': calculate_ndcg_at_k(predictions, ground_truth, 5),
        'ndcg@10': calculate_ndcg_at_k(predictions, ground_truth, 10),
        'ndcg@20': calculate_ndcg_at_k(predictions, ground_truth, 20),
    }
    memory_system._trace("ranking_result", {
        "user_id": user_id,
        "eval_type": eval_type,
        "use_memory": use_memory,
        "ground_truth": ground_truth,
        "candidate_item_ids": candidates,
        "ranked_item_ids": predictions,
        "metrics": metrics,
        "baseline_metrics": baseline_metric,
        "memory_retrieval_mode": memory_retrieval_mode,
        "memory_gate": memory_gate,
        "memory_similarity_threshold": memory_similarity_threshold,
        "ranking_prompt_style": ranking_prompt_style,
        "no_harm_arbitration": arbitration_decision,
        "selected_ranking_source": selected_ranking_source,
        "retrieval_query_text": retrieval_query_text,
        "gate_decisions": gate_decisions,
        "retrieved_memories": [
            behavior_memory_to_trace(mem) for mem in (retrieved_memories or [])
        ],
    })
    
    return baseline_metric,metrics, candidates, predictions, ground_truth

def parse_args():
    parser = argparse.ArgumentParser(description="Experiment configuration")

    # Basic config
    parser.add_argument("--data_name", type=str, default="Video_Game")

    parser.add_argument("--use_memory", action="store_true", default=True)
    parser.add_argument("--no_use_memory", action="store_false", dest="use_memory")

    parser.add_argument("--LOAD_SAVED_MEMORY", action="store_true", default=False)

    # Hyperparameters for training
    parser.add_argument("--wo_evolving", action="store_true", default=True)
    parser.add_argument("--with_evolving", action="store_false", dest="wo_evolving")
    parser.add_argument("--wo_link", action="store_true", default=False)

    parser.add_argument("--max_evolutions_per_memory", type=int, default=None)
    parser.add_argument("--window_size", type=int, default=5)
    parser.add_argument("--link_size", type=int, default=5)
    parser.add_argument("--max_iterations", type=int, default=1)

    # Hyperparameter for ranking
    parser.add_argument("--k_memories", type=int, default=1)
    parser.add_argument("--memory_retrieval_mode", type=str, default="user_only",
                        choices=["user_only", "candidate_aware"],
                        help="Memory retrieval query: user history only or user history plus candidate set.")
    parser.add_argument("--memory_gate", type=str, default="none",
                        choices=["none", "rule", "strict_rule", "applicability"],
                        help="Whether to filter retrieved memories before ranking.")
    parser.add_argument("--memory_similarity_threshold", type=float, default=0.35,
                        help="Minimum retrieval similarity for rule-based memory gate.")
    parser.add_argument("--no_harm_arbitration", action="store_true", default=False,
                        help="Run no-memory and memory ranking, then use memory only if applicability evidence is strong.")
    parser.add_argument("--no_harm_min_applicability", type=float, default=1.0,
                        help="Minimum applicability score required to use memory ranking under --no_harm_arbitration.")

    # Hyperparameter for few-shot ranking (LLM ranking)
    parser.add_argument("--fewshot_ranking", action="store_true", default=False)
    parser.add_argument("--k_shot", type=int, default=3)

    # Other
    parser.add_argument("--number_of_users", type=int, default=100)
    parser.add_argument("--max_positive_interactions", type=int, default=0,
                        help="If >0, use only the latest N positive train interactions per user.")
    parser.add_argument("--max_negative_candidates", type=int, default=0,
                        help="If >0, use only the first N negative candidates per user during evaluation.")
    parser.add_argument("--candidate_negative_mode", type=str, default="candidate_hard",
                        choices=["random", "candidate_hard"],
                        help="Training negative sampling: random global item or user-runtime hard negative.")
    parser.add_argument("--min_lesson_confidence", type=float, default=0.25,
                        help="Minimum confidence/specificity required to keep a new fail lesson.")
    parser.add_argument("--max_lesson_risk", type=float, default=0.85,
                        help="Maximum overgeneralization risk allowed for a new fail lesson.")
    parser.add_argument("--max_failure_lessons_per_user", type=int, default=3,
                        help="Maximum kept failure lessons per user during training. <=0 means no cap.")
    parser.add_argument("--ranking_prompt_style", type=str, default="compact_score",
                        choices=["memcf", "compact_score"],
                        help="Prompt style for ranking.")
    parser.add_argument("--trace_dir", type=str, default=None,
                        help="Directory for JSONL traces. Default: evaluation_results/<dataset>/traces/<run_name>.")
    parser.add_argument("--disable_trace", action="store_false", dest="trace_enabled",
                        help="Disable JSONL traces for this run.")
    parser.set_defaults(trace_enabled=True)
    return parser.parse_args()

def main():
    # Paths to data files
    args = parse_args()

    data_name = args.data_name
    use_memory = args.use_memory
    LOAD_SAVED_MEMORY = args.LOAD_SAVED_MEMORY

    wo_evolving = args.wo_evolving
    wo_link = args.wo_link
    max_evolutions_per_memory = args.max_evolutions_per_memory
    window_size = args.window_size
    link_size = args.link_size
    max_iterations = args.max_iterations

    k_memories = args.k_memories
    memory_retrieval_mode = args.memory_retrieval_mode
    memory_gate = args.memory_gate
    memory_similarity_threshold = args.memory_similarity_threshold
    no_harm_arbitration = args.no_harm_arbitration
    no_harm_min_applicability = args.no_harm_min_applicability
    fewshot_ranking = args.fewshot_ranking
    k_shot = args.k_shot

    number_of_users = args.number_of_users
    max_positive_interactions = args.max_positive_interactions
    max_negative_candidates = args.max_negative_candidates
    candidate_negative_mode = args.candidate_negative_mode
    min_lesson_confidence = args.min_lesson_confidence
    max_lesson_risk = args.max_lesson_risk
    max_failure_lessons_per_user = args.max_failure_lessons_per_user
    ranking_prompt_style = args.ranking_prompt_style
    trace_enabled = args.trace_enabled

    base_dir = (
        os.getenv("MEMCF_ROOT")
        or os.getenv("AGENTICREC_CFMEMORY_ROOT")
        or os.path.dirname(os.path.abspath(__file__))
    )
    data_root = (
        os.getenv("MEMCF_DATA_ROOT")
        or os.getenv("AGENTICREC_DATA_ROOT")
        or os.path.join(base_dir, "data")
    )
    eval_root = (
        os.getenv("MEMCF_EVAL_ROOT")
        or os.getenv("AGENTICREC_EVAL_ROOT")
        or os.path.join(base_dir, "evaluation_results")
    )
    memory_root = (
        os.getenv("MEMCF_MEMORY_ROOT")
        or os.getenv("AGENTICREC_MEMORY_ROOT")
        or os.path.join(base_dir, "agent_memory")
    )

    items_path = os.path.join(data_root, data_name, "items.json")
    sequences_path = os.path.join(data_root, data_name, "user_sequences_10.json")
    negatives_path = os.path.join(data_root, data_name, "user_negatives_10.json")
    
    if use_memory:
        if wo_evolving:
            output_file = os.path.join(eval_root, data_name, f"nuser{number_of_users}_fail_interactions_no_evolving_k{k_memories}_iter{max_iterations}_memory.json")
            memory_file_path = os.path.join(memory_root, data_name, f"nuser{number_of_users}_fail_interactions_no_evolving_iter{max_iterations}.json")
        elif wo_link:
            output_file = os.path.join(eval_root, data_name, f"nuser{number_of_users}_fail_interactions_no_link_k{k_memories}_iter{max_iterations}_memory_maxevolution{str(max_evolutions_per_memory)}.json")
            memory_file_path = os.path.join(memory_root, data_name, f"nuser{number_of_users}_fail_interactions_no_link_iter{max_iterations}_maxevolution{str(max_evolutions_per_memory)}.json")
        else:
            output_file = os.path.join(eval_root, data_name, f"nuser{number_of_users}_global_fail_interactions_{k_memories}_iter{max_iterations}_link{link_size}_memory_maxevolution{str(max_evolutions_per_memory)}.json")
            memory_file_path = os.path.join(memory_root, data_name, f"nuser{number_of_users}_global_fail_interactions_iter{max_iterations}_link{link_size}_maxevolution{str(max_evolutions_per_memory)}.json")
    else:
        if fewshot_ranking:
            output_file = os.path.join(eval_root, data_name, f"nuser{number_of_users}_fewshot_{k_shot}_users_ranking_no_memory.json")
        else:
            output_file = os.path.join(eval_root, data_name, f"nuser{number_of_users}_zeroshot_users_ranking_no_memory.json")

    if use_memory and (memory_retrieval_mode != "user_only" or memory_gate != "none"):
        phase2_suffix = (
            f"_retrieval{memory_retrieval_mode}_gate{memory_gate}"
            f"_thr{str(memory_similarity_threshold).replace('.', 'p')}"
        )
        output_file = output_file.replace(".json", f"{phase2_suffix}.json")
    if use_memory and no_harm_arbitration:
        output_file = output_file.replace(
            ".json",
            f"_noharm_minapp{str(no_harm_min_applicability).replace('.', 'p')}.json",
        )

    run_name = os.path.splitext(os.path.basename(output_file))[0]
    trace_dir = args.trace_dir or os.path.join(
        eval_root,
        data_name,
        "traces",
        f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    trace_recorder = TraceRecorder(trace_dir=trace_dir, enabled=trace_enabled)
    trace_recorder.write_manifest({
        "run_name": run_name,
        "output_file": output_file,
        "data_name": data_name,
        "number_of_users": number_of_users,
        "use_memory": use_memory,
        "load_saved_memory": LOAD_SAVED_MEMORY,
        "wo_evolving": wo_evolving,
        "wo_link": wo_link,
        "max_iterations": max_iterations,
        "k_memories": k_memories,
        "memory_retrieval_mode": memory_retrieval_mode,
        "memory_gate": memory_gate,
        "memory_similarity_threshold": memory_similarity_threshold,
        "no_harm_arbitration": no_harm_arbitration,
        "no_harm_min_applicability": no_harm_min_applicability,
        "max_positive_interactions": max_positive_interactions,
        "max_negative_candidates": max_negative_candidates,
        "candidate_negative_mode": candidate_negative_mode,
        "min_lesson_confidence": min_lesson_confidence,
        "max_lesson_risk": max_lesson_risk,
        "max_failure_lessons_per_user": max_failure_lessons_per_user,
        "ranking_prompt_style": ranking_prompt_style,
        "phase": "phase1_correctness",
    })
    if trace_enabled:
        print(f"✓ Trace enabled: {trace_dir}")
    # Load data
    items_meta, user_sequences, user_negatives = load_data(
        items_path, sequences_path, negatives_path
    )

    print(f"Total users loaded: {len(user_sequences)}")
    
    # Get first 100 users
    user_ids = list(user_sequences.keys())[: number_of_users]
    if not use_memory and fewshot_ranking:
        sample_user_ids = list(user_sequences.keys())[number_of_users:]
    
    global_memory = RecommendationMemorySystem(use_gemini_embeddings=True)
    global_memory.trace_recorder = trace_recorder
    user_states: Dict[str, PairwiseUserState] = {}
    item_states = init_pairwise_item_states(items_meta)

    if use_memory:
        if LOAD_SAVED_MEMORY and os.path.exists(memory_file_path):
            print("\n" + "="*80)
            print("LOADING SAVED MEMORY SYSTEM")
            print("="*80)
            global_memory.load_memory(memory_file_path)
        else:
            print("\n" + "="*80)
            print("PHASE 1: TRAINING WITH CROSS-USER EVOLVING ONLY")
            print("="*80)
            
            global_memory = RecommendationMemorySystem(use_gemini_embeddings=False)
            global_memory.trace_recorder = trace_recorder
            for user_id in user_ids:
                profile = initialize_user_memory_from_history_v2(
                    memory_system=global_memory,
                    user_id=str(user_id),
                    user_data=user_sequences[user_id],
                    items_meta=items_meta,
                    max_positive_interactions=max_positive_interactions,
                )
                user_states[str(user_id)] = PairwiseUserState(
                    user_id=str(user_id),
                    short_term_memory=profile.profile,
                )
                global_memory._trace("user_memory_initialized", {
                    "user_id": user_id,
                    "profile": asdict(profile),
                })
            
            # shuffled_user_ids = user_ids.copy()
            # random.shuffle(shuffled_user_ids)
            
            for user_id in tqdm(user_ids, desc="Cross-user Training"):
                user_data = user_sequences[user_id]
                effective_train_len = (
                    min(len(user_data['train']), max_positive_interactions)
                    if max_positive_interactions and max_positive_interactions > 0
                    else min(len(user_data['train']), 30)
                )
                print(f"\nProcessing user {user_id} ({effective_train_len}/{len(user_data['train'])} train interactions)")
                
                # Tạo temp system chỉ để generate new memories
                # temp_system = RecommendationMemorySystem(use_gemini_embeddings=False)
                temp_system = RecommendationMemorySystem.__new__(RecommendationMemorySystem)
                temp_system.llm_name = global_memory.llm_name
                temp_system.embedding_model_name = global_memory.embedding_model_name
                temp_system.chat_model_name = global_memory.chat_model_name
                temp_system.chat_api_base = global_memory.chat_api_base
                temp_system.embedding_api_base = global_memory.embedding_api_base
                temp_system.api_key = global_memory.api_key
                temp_system.use_api_chat = global_memory.use_api_chat
                temp_system.use_api_embedding = global_memory.use_api_embedding
                temp_system.model = getattr(global_memory, "model", None)
                temp_system.tokenizer = getattr(global_memory, "tokenizer", None)
                temp_system.embedding_model = getattr(global_memory, "embedding_model", None)
                temp_system.trace_recorder = trace_recorder

                # reset memory data only
                temp_system.behavior_memories = []
                temp_system.user_interaction_history = []
                temp_system.next_thought_id = 0
                temp_system.memory_diagnostics = defaultdict(float)
                
                # Chỉ tạo new memories từ user này (không evolve nội bộ)
                try:
                    new_memories = train_memory_from_fail_interactions(
                        user_id=user_id,
                        user_data=user_data,
                        memory_system=temp_system,
                        user_states=user_states,
                        item_states=item_states,
                        items_meta=items_meta,
                        negative_data=user_negatives.get(user_id, {}),
                        max_iterations=max_iterations,
                        max_positive_interactions=max_positive_interactions,
                        candidate_negative_mode=candidate_negative_mode,
                        min_lesson_confidence=min_lesson_confidence,
                        max_lesson_risk=max_lesson_risk,
                        max_failure_lessons_per_user=max_failure_lessons_per_user,
                    )
                except Exception as e:
                    print(f"\nError evaluating user {user_id}: {e}")
                    continue
                if not new_memories:
                    print("  → No new memories generated, skipping...")
                    continue
                
                print(f"  → Generated {len(new_memories)} fail-interaction memories")
               
                max_global_id = global_memory.next_thought_id - 1 if global_memory.behavior_memories else -1

                for new_mem in new_memories:
                    # Offset thought_id của memory mới
                    local_thought_id = new_mem.thought_id
                    new_mem.thought_id += (max_global_id + 1)
                    trace_recorder.log("global_memory_candidate", {
                        "user_id": user_id,
                        "local_thought_id": local_thought_id,
                        "global_thought_id": new_mem.thought_id,
                        "wo_evolving": wo_evolving,
                        "wo_link": wo_link,
                        "memory": behavior_memory_to_trace(new_mem),
                    })
                    
                    # Link với global (linked_ids là id cũ trong global)
                    if not wo_evolving:
                        try:
                            linked_ids = global_memory.link_behavior_memories(new_mem, k=link_size, wo_link=wo_link)
                            new_mem.links = linked_ids  # vẫn là id cũ, đúng
                            
                            # Evolve global dựa trên new_mem
                            global_memory.evolve_behavior_memories(new_mem, linked_ids, max_evolutions_per_memory=max_evolutions_per_memory)
                        except Exception as e:
                            print(f"\nError evolving memory for user {user_id}: {e}")
                    
                    # Add vào global
                    global_memory.behavior_memories.append(new_mem)
                    
                    # Update next_id
                    global_memory.next_thought_id = new_mem.thought_id + 1
                    trace_recorder.log("global_memory_added", {
                        "user_id": user_id,
                        "global_thought_id": new_mem.thought_id,
                        "links": new_mem.links,
                        "memory_pool_size": len(global_memory.behavior_memories),
                    })
                
                print(f"  → Global memory pool now has {len(global_memory.behavior_memories)} memories")
                global_memory.save_memory(memory_file_path, format='json')
                print(f"  → Overwritten common global memory file: {memory_file_path}")
        
        print("\n" + "="*80)
        print("SAVING GLOBAL CROSS-USER EVOLVING MEMORY")
        print("="*80)
        global_memory.save_memory(memory_file_path, format='json')
        global_memory.print_evolution_report()
        stats = global_memory.get_evolution_statistics()
        stats_file = memory_file_path.replace('.json', '_evolution_stats.json')
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"✓ Evolution statistics saved to {stats_file}")
            

    
    # PHASE 2: EVALUATE ON VALIDATION SET
    print("\n" + "="*80)
    print("PHASE 2: VALIDATION SET EVALUATION")
    print("="*80)
    
    val_metrics = {
        'recall@5': [], 'recall@10': [], 'recall@20': [],
        'ndcg@5': [], 'ndcg@10': [], 'ndcg@20': []
    }
    baseline_metrics = {
        'recall@5': [], 'recall@10': [], 'recall@20': [],
        'ndcg@5': [], 'ndcg@10': [], 'ndcg@20': []
    }
    
    all_user_results = []
    
    if not use_memory:
        global_memory = RecommendationMemorySystem(use_gemini_embeddings=True)
        global_memory.trace_recorder = trace_recorder
    for user_id in tqdm(user_ids, desc="Validation"):
        try:
            user_data = user_sequences[user_id]
            negative_data = user_negatives.get(user_id, {})

            # sample_user_id = random.choice(sample_user_ids) if not use_memory and fewshot_ranking else None
            sample_user_id_list = random.sample(sample_user_ids, k_shot) if not use_memory and fewshot_ranking else None
            if sample_user_id_list is not None:
                sample_user_list = []
                negative_data_sample_list = []
                for id in sample_user_id_list:
                    sample_user_data = user_sequences[id] if id else None
                    negative_data_sample = user_negatives.get(id, {}) if id else None   
                    sample_user_list.append(sample_user_data)
                    negative_data_sample_list.append(negative_data_sample)
            else:
                sample_user_list = None
                negative_data_sample_list = None
            # print(sample_user_data)

            baseline_metric, metrics, candidates, predictions, ground_truth = evaluate_user(
                user_data, negative_data, 
                items_meta, global_memory, eval_type='test', use_memory=use_memory, k_memories=k_memories, sample_user_list=sample_user_list, negative_data_sample_list=negative_data_sample_list,
                max_positive_interactions=max_positive_interactions, max_negative_candidates=max_negative_candidates,
                user_id=user_id,
                memory_retrieval_mode=memory_retrieval_mode,
                memory_gate=memory_gate,
                memory_similarity_threshold=memory_similarity_threshold,
                no_harm_arbitration=no_harm_arbitration,
                no_harm_min_applicability=no_harm_min_applicability,
                ranking_prompt_style=ranking_prompt_style,
            )
            # Lưu tạm thông tin user này
            all_user_results.append({
                "user_id": user_id,
                "ground_truth": ground_truth,
                "candidates": candidates,
                "predictions": predictions,
                "metrics": metrics,
                "baseline_metrics": baseline_metric
            })
            for key in val_metrics:
                val_metrics[key].append(metrics[key])
                baseline_metrics[key].append(baseline_metric[key])
                
        except Exception as e:
            print(f"\nError evaluating user {user_id}: {e}")
            continue
    
    save_all_users_ranking_results(
        all_results=all_user_results,
        items_meta=items_meta,
        output_file=output_file
    )
    # Print validation results
    print("\nValidation Results:")
    print("-" * 80)
    for metric in ['recall@5', 'recall@10', 'recall@20','ndcg@5', 'ndcg@10', 'ndcg@20']:
        if len(baseline_metrics[metric]) > 0:
            mean_val = np.mean(baseline_metrics[metric])
            print(f"Baseline {metric:10s}: {mean_val:.4f}")
        else:
            print(f"Baseline {metric:10s}: N/A")
    print("-" * 80)
    for metric in ['recall@5', 'recall@10', 'recall@20','ndcg@5', 'ndcg@10', 'ndcg@20']:
        if len(val_metrics[metric]) > 0:
            mean_val = np.mean(val_metrics[metric])
            print(f"{metric:12s}: {mean_val:.4f}")
        else:
            print(f"{metric:12s}: N/A")

    diag = getattr(global_memory, "memory_diagnostics", defaultdict(float))
    diag_users = float(diag.get("eval_users", 0.0))
    retrieved_total = float(diag.get("retrieved_total", 0.0))
    kept_total = float(diag.get("kept_total", 0.0))
    skipped_total = float(diag.get("skipped_total", 0.0))
    memory_diagnostics = {
        "retrieval_mode": memory_retrieval_mode,
        "memory_gate": memory_gate,
        "memory_similarity_threshold": memory_similarity_threshold,
        "eval_users_with_memory_retrieval": int(diag_users),
        "retrieved_total": int(retrieved_total),
        "kept_total": int(kept_total),
        "skipped_total": int(skipped_total),
        "avg_retrieved_memories": retrieved_total / diag_users if diag_users else 0.0,
        "avg_kept_memories": kept_total / diag_users if diag_users else 0.0,
        "avg_skipped_memories": skipped_total / diag_users if diag_users else 0.0,
        "gate_keep_rate": kept_total / retrieved_total if retrieved_total else 0.0,
        "gate_skip_rate": skipped_total / retrieved_total if retrieved_total else 0.0,
        "users_with_kept_memory": int(diag.get("users_with_kept_memory", 0.0)),
        "rank_score_calls": int(diag.get("rank_score_calls", 0.0)),
        "rank_valid_score_outputs": int(diag.get("rank_valid_score_outputs", 0.0)),
        "rank_invalid_score_outputs": int(diag.get("rank_invalid_score_outputs", 0.0)),
        "rank_valid_score_rate": (
            float(diag.get("rank_valid_score_outputs", 0.0)) / float(diag.get("rank_score_calls", 0.0))
            if float(diag.get("rank_score_calls", 0.0)) else 0.0
        ),
        "rank_missing_score_rows": int(diag.get("rank_missing_score_rows", 0.0)),
        "rank_invalid_score_rows": int(diag.get("rank_invalid_score_rows", 0.0)),
        "rank_attempt_errors": int(diag.get("rank_attempt_errors", 0.0)),
        "rank_fallbacks": int(diag.get("rank_fallbacks", 0.0)),
        "no_harm_users": int(diag.get("no_harm_users", 0.0)),
        "no_harm_used_memory": int(diag.get("no_harm_used_memory", 0.0)),
        "no_harm_fallback_no_memory": int(diag.get("no_harm_fallback_no_memory", 0.0)),
        "no_harm_memory_use_rate": (
            float(diag.get("no_harm_used_memory", 0.0)) / float(diag.get("no_harm_users", 0.0))
            if float(diag.get("no_harm_users", 0.0)) else 0.0
        ),
    }

    summary = {
        "model": "MEMCF",
        "dataset": data_name,
        "number_of_users_requested": number_of_users,
        "number_of_users_evaluated": len(all_user_results),
        "use_memory": use_memory,
        "load_saved_memory": LOAD_SAVED_MEMORY,
        "wo_evolving": wo_evolving,
        "wo_link": wo_link,
        "max_iterations": max_iterations,
        "k_memories": k_memories,
        "memory_retrieval_mode": memory_retrieval_mode,
        "memory_gate": memory_gate,
        "memory_similarity_threshold": memory_similarity_threshold,
        "no_harm_arbitration": no_harm_arbitration,
        "no_harm_min_applicability": no_harm_min_applicability,
        "trace_enabled": trace_enabled,
        "trace_dir": trace_dir if trace_enabled else None,
        "max_positive_interactions": max_positive_interactions,
        "max_negative_candidates": max_negative_candidates,
        "candidate_negative_mode": candidate_negative_mode,
        "min_lesson_confidence": min_lesson_confidence,
        "max_lesson_risk": max_lesson_risk,
        "max_failure_lessons_per_user": max_failure_lessons_per_user,
        "ranking_prompt_style": ranking_prompt_style,
        "phase1_correctness": {
            "clean_ranked_item_ids": True,
            "drop_hallucinated_item_ids": True,
            "deduplicate_ranked_item_ids": True,
            "append_missing_candidates": True,
            "deterministic_candidate_order": True,
            "api_temperature": float(os.getenv("MEMCF_TEMPERATURE", "0.0")),
            "rank_max_tokens": int(os.getenv("MEMCF_RANK_MAX_TOKENS", "1500")),
            "rank_retries": int(os.getenv("MEMCF_RANK_RETRIES", "1")),
            "strict_gate_min_strong_terms": int(os.getenv("MEMCF_STRICT_GATE_MIN_STRONG_TERMS", "2")),
            "normalize_categories": True,
            "strict_output_validation": True,
            "retry_invalid_rankings": True,
            "score_based_ranking": True,
            "candidate_aliases": True,
            "structured_memory_fields": True,
            "applicability_gate_available": True,
            "no_harm_arbitration_available": True,
        },
        "baseline_metrics": {
            metric: (float(np.mean(baseline_metrics[metric])) if len(baseline_metrics[metric]) > 0 else None)
            for metric in ['recall@5', 'recall@10', 'recall@20', 'ndcg@5', 'ndcg@10', 'ndcg@20']
        },
        "metrics": {
            metric: (float(np.mean(val_metrics[metric])) if len(val_metrics[metric]) > 0 else None)
            for metric in ['recall@5', 'recall@10', 'recall@20', 'ndcg@5', 'ndcg@10', 'ndcg@20']
        },
        "memory_diagnostics": memory_diagnostics,
    }
    summary_file = output_file.replace(".json", ".summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"✓ Saved MEMCF summary to {summary_file}")
    trace_recorder.write_manifest({
        "run_name": run_name,
        "output_file": output_file,
        "summary_file": summary_file,
        "completed_at": datetime.now().isoformat(),
        "summary": summary,
    })

@dataclass
class UserMemoryProfile:
    """Stable user profile initialized from observed train history."""
    user_id: str
    profile: str
    facets: List[str] = field(default_factory=list)
    evidence_item_ids: List[str] = field(default_factory=list)
    source: str = "history_init"
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

    def to_prompt_dict(self) -> Dict[str, Any]:
        return {
            "profile": self.profile,
            "facets": self.facets[:8],
            "evidence_item_ids": self.evidence_item_ids[:10],
        }


@dataclass
class FailureEvent:
    """Full trace object for one failed pairwise interaction."""
    event_id: str
    source_user_id: str
    recent_history: List[Dict[str, Any]]
    user_memory_before: str
    user_memory_after: str
    wrong_item: Dict[str, Any]
    correct_item: Dict[str, Any]
    model_wrong_reasoning: str
    failure_type: str
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


@dataclass
class FailureLesson:
    """Compact graph-retrievable memory derived from a FailureEvent."""
    memory_id: str
    source_user_id: str
    source_event_id: str
    lesson: str
    prefer: str
    avoid: str
    applies_if: List[str] = field(default_factory=list)
    do_not_apply_if: List[str] = field(default_factory=list)
    evidence_terms: List[str] = field(default_factory=list)
    wrong_item_id: str = ""
    correct_item_id: str = ""
    wrong_item_title: str = ""
    correct_item_title: str = ""
    wrong_item_category: str = ""
    correct_item_category: str = ""
    source_user_preference: str = ""
    history_item_ids: List[str] = field(default_factory=list)
    confidence: float = 0.5
    overgeneralization_risk: float = 0.5
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

    def short_facet(self) -> str:
        if self.lesson:
            return self.lesson.strip()
        prefer = self.prefer.strip() or "items matching concrete history signals"
        avoid = self.avoid.strip() or "items matching only superficial signals"
        return f"User likes {prefer}, often prefers it over {avoid}, and should not be matched by generic category alone."

    def safe_fact(self) -> str:
        """Factual memory sentence for ranking prompts, avoiding extra LLM analysis."""
        pref = re.sub(r"\s+", " ", str(self.source_user_preference or self.prefer or "similar observed history")).strip()
        correct = self.correct_item_title or self.prefer or self.correct_item_id
        wrong = self.wrong_item_title or self.avoid or self.wrong_item_id
        if len(pref) > 180:
            pref = pref[:177].rstrip() + "..."
        return (
            f"A user with preference/history '{pref}' preferred/bought "
            f"'{correct}' instead of '{wrong}'."
        )


@dataclass
class GraphRetrievedLesson:
    lesson: FailureLesson
    score: float
    sources: List[str]
    paths: List[str]
    matched_evidence_terms: List[str] = field(default_factory=list)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _item_info_for_prompt(item_id: str, items_meta: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    item_id = str(item_id)
    info = items_meta.get(item_id, {})
    return {
        "item_id": item_id,
        "title": item_title(info, item_id) if isinstance(info, dict) else f"Item {item_id}",
        "category": item_category(info) if isinstance(info, dict) else "Unknown",
    }


def _history_item_infos(item_ids: List[str], items_meta: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [_item_info_for_prompt(str(item_id), items_meta) for item_id in item_ids if str(item_id) in items_meta]


def _context_text_from_items(items: List[Dict[str, Any]]) -> str:
    return " ".join(f"{x.get('title', '')} {x.get('category', '')}" for x in items).lower()


def _item_tokens_for_hard_negative(item_id: str, items_meta: Dict[str, Dict[str, Any]]) -> Set[str]:
    info = _item_info_for_prompt(str(item_id), items_meta)
    text = f"{info.get('title', '')} {info.get('category', '')}"
    return set(normalize_terms(text))


def choose_training_negative_item_id(
    user_id: str,
    pos_item_id: str,
    user_data: Dict[str, Any],
    negative_data: Optional[Dict[str, Any]],
    items_meta: Dict[str, Dict[str, Any]],
    all_item_ids: List[str],
    mode: str = "random",
    max_positive_interactions: Optional[int] = None,
) -> Optional[str]:
    valid_item_ids = set(str(x) for x in all_item_ids)
    pos_item_id = str(pos_item_id)
    runtime_pool = collect_runtime_negative_pool(
        negative_data=negative_data,
        valid_item_ids=valid_item_ids,
        exclude_ids={pos_item_id},
    )
    if mode != "candidate_hard":
        base_pool = runtime_pool or [str(iid) for iid in all_item_ids if str(iid) != pos_item_id]
        if not base_pool:
            return None
        return deterministic_shuffle(base_pool, salt=f"randneg::{user_id}::{pos_item_id}")[0]

    if not runtime_pool:
        fallback_pool = [str(iid) for iid in all_item_ids if str(iid) != pos_item_id]
        if not fallback_pool:
            return None
        return deterministic_shuffle(fallback_pool, salt=f"hardneg_fallback::{user_id}::{pos_item_id}")[0]

    history_ids = [str(x) for x in user_data.get("train", [])]
    if max_positive_interactions and max_positive_interactions > 0:
        history_ids = history_ids[-max_positive_interactions:]
    else:
        history_ids = history_ids[-10:]
    anchor_tokens: Set[str] = set()
    for hid in history_ids:
        anchor_tokens.update(_item_tokens_for_hard_negative(hid, items_meta))
    anchor_tokens.update(_item_tokens_for_hard_negative(pos_item_id, items_meta))

    pos_category = _item_info_for_prompt(pos_item_id, items_meta).get("category", "Unknown")
    shuffled_pool = deterministic_shuffle(runtime_pool, salt=f"hardneg_pool::{user_id}::{pos_item_id}")
    best_item_id = shuffled_pool[0]
    best_score = -1.0
    for neg_item_id in shuffled_pool:
        neg_tokens = _item_tokens_for_hard_negative(neg_item_id, items_meta)
        neg_category = _item_info_for_prompt(neg_item_id, items_meta).get("category", "Unknown")
        overlap = len(anchor_tokens & neg_tokens)
        category_bonus = 0.5 if pos_category != "Unknown" and pos_category == neg_category else 0.0
        score = float(overlap) + category_bonus
        if score > best_score:
            best_score = score
            best_item_id = neg_item_id
    return best_item_id


class MemoryGraphIndex:
    """Graph-scoped for retrieving fail lessons.

    Nodes:
    - users
    - items
    - failure lessons

    Edges:
    - user -> train/history item
    - lesson -> source user
    - lesson -> wrong/correct/history item evidence

    Retrieval is graph-scoped first, text-gated second. It does not search the
    global memory pool by embedding similarity.
    """

    def __init__(self, user_sequences: Dict[str, Dict[str, Any]]):
        self.items_by_user: Dict[str, Set[str]] = defaultdict(set)
        self.users_by_item: Dict[str, Set[str]] = defaultdict(set)
        self.memories_by_user: Dict[str, Set[str]] = defaultdict(set)
        self.memories_by_item: Dict[str, Set[str]] = defaultdict(set)
        self.lessons: Dict[str, FailureLesson] = {}
        for user_id, user_data in user_sequences.items():
            for item_id in user_data.get("train", []):
                sid = str(item_id)
                self.items_by_user[str(user_id)].add(sid)
                self.users_by_item[sid].add(str(user_id))

    def add_lesson(self, lesson: FailureLesson) -> None:
        self.lessons[lesson.memory_id] = lesson
        self.memories_by_user[lesson.source_user_id].add(lesson.memory_id)
        item_ids = set(str(x) for x in lesson.history_item_ids)
        if lesson.wrong_item_id:
            item_ids.add(str(lesson.wrong_item_id))
        if lesson.correct_item_id:
            item_ids.add(str(lesson.correct_item_id))
        for item_id in item_ids:
            self.memories_by_item[item_id].add(lesson.memory_id)

    def similar_users(self, user_id: str, top_k: int = 10) -> List[Tuple[str, int]]:
        user_id = str(user_id)
        counts: Dict[str, int] = defaultdict(int)
        for item_id in self.items_by_user.get(user_id, set()):
            for other_user in self.users_by_item.get(item_id, set()):
                if other_user != user_id:
                    counts[other_user] += 1
        return sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:top_k]

    def retrieve(
        self,
        user_id: str,
        recent_history_ids: List[str],
        candidate_ids: List[str],
        current_context_text: str,
        top_k: int = 3,
        neighbor_k: int = 10,
        min_evidence_terms: int = 1,
    ) -> List[GraphRetrievedLesson]:
        user_id = str(user_id)
        scores: Dict[str, float] = defaultdict(float)
        sources: Dict[str, Set[str]] = defaultdict(set)
        paths: Dict[str, List[str]] = defaultdict(list)
        candidate_set = set(str(x) for x in candidate_ids)
        history_set = set(str(x) for x in recent_history_ids)

        for mid in self.memories_by_user.get(user_id, set()):
            scores[mid] += 3.0
            sources[mid].add("same_user")
            paths[mid].append(f"user:{user_id}->memory:{mid}")

        for item_id in candidate_set:
            for mid in self.memories_by_item.get(item_id, set()):
                scores[mid] += 2.0
                sources[mid].add("candidate_item")
                paths[mid].append(f"candidate_item:{item_id}->memory:{mid}")

        for item_id in history_set:
            for mid in self.memories_by_item.get(item_id, set()):
                scores[mid] += 1.5
                sources[mid].add("history_item")
                paths[mid].append(f"history_item:{item_id}->memory:{mid}")

        for other_user, shared_count in self.similar_users(user_id, top_k=neighbor_k):
            for mid in self.memories_by_user.get(other_user, set()):
                scores[mid] += 1.0 + 0.2 * min(shared_count, 5)
                sources[mid].add("neighbor_user")
                paths[mid].append(f"user:{user_id}->shared_items:{shared_count}->user:{other_user}->memory:{mid}")

        retrieved: List[GraphRetrievedLesson] = []
        for mid, score in scores.items():
            lesson = self.lessons.get(mid)
            if lesson is None:
                continue
            evidence_terms = normalize_evidence_terms(
                list(lesson.evidence_terms or [])
                + list(lesson.applies_if or [])
                + [lesson.prefer, lesson.avoid]
            )
            matched_terms = [
                term for term in evidence_terms
                if term_matches_context(term, current_context_text)
            ]
            same_user = "same_user" in sources[mid]
            candidate_item = "candidate_item" in sources[mid]
            if not same_user and not candidate_item and len(matched_terms) < min_evidence_terms:
                continue
            score += 0.25 * len(matched_terms)
            score += 0.2 * _safe_float(lesson.confidence, 0.5)
            score -= 0.5 * _safe_float(lesson.overgeneralization_risk, 0.5)
            retrieved.append(GraphRetrievedLesson(
                lesson=lesson,
                score=score,
                sources=sorted(sources[mid]),
                paths=paths[mid],
                matched_evidence_terms=matched_terms,
            ))
        retrieved.sort(key=lambda r: (-r.score, r.lesson.memory_id))
        return retrieved[:top_k]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lessons": [asdict(lesson) for lesson in self.lessons.values()],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], user_sequences: Dict[str, Dict[str, Any]]) -> "MemoryGraphIndex":
        graph = cls(user_sequences)
        for row in data.get("lessons", []):
            graph.add_lesson(FailureLesson(**row))
        return graph


def behavior_memory_passes_quality_gate(
    memory: "BehaviorMemory",
    interaction_window: List["UserInteraction"],
    min_confidence: float,
    max_risk: float,
) -> Tuple[bool, str]:
    specificity = _safe_float(getattr(memory, "specificity_score", 0.0), 0.0)
    risk = _safe_float(getattr(memory, "overgeneralization_risk", 1.0), 1.0)
    wrong_ids = [
        str(x.item_id) for x in interaction_window
        if str((x.metadata or {}).get("role", "")) in {"chosen_wrong", "wrong_choice"}
    ]
    correct_ids = [
        str(x.item_id) for x in interaction_window
        if str((x.metadata or {}).get("role", "")) in {"ground_truth", "preferred_item"}
    ]
    concrete_terms = normalize_evidence_terms(
        list(getattr(memory, "evidence_terms_required", []) or [])
        + list(getattr(memory, "keywords", []) or [])
        + [getattr(memory, "wrong_item_type", ""), getattr(memory, "correct_item_type", "")]
    )
    combined_text = " ".join([
        str(getattr(memory, "behavior_explanation", "")),
        str(getattr(memory, "pattern_description", "")),
        str(getattr(memory, "wrong_item_type", "")),
        str(getattr(memory, "correct_item_type", "")),
        " ".join(concrete_terms[:8]),
    ])
    if specificity < min_confidence:
        return False, f"low_specificity:{specificity:.2f}"
    if risk > max_risk:
        return False, f"high_risk:{risk:.2f}"
    if not wrong_ids or not correct_ids:
        return False, "missing_wrong_or_correct_item_ids"
    if len(concrete_terms) < 2:
        return False, "too_few_concrete_terms"
    if memory_text_is_too_generic(combined_text):
        return False, "memory_text_too_generic"
    return True, "accepted"


def failure_lesson_passes_quality_gate_v2(
    lesson: "FailureLesson",
    min_confidence: float,
    max_risk: float,
) -> Tuple[bool, str]:
    confidence = _safe_float(getattr(lesson, "confidence", 0.0), 0.0)
    risk = _safe_float(getattr(lesson, "overgeneralization_risk", 1.0), 1.0)
    concrete_terms = normalize_evidence_terms(
        list(getattr(lesson, "evidence_terms", []) or [])
        + list(getattr(lesson, "applies_if", []) or [])
        + [getattr(lesson, "prefer", ""), getattr(lesson, "avoid", "")]
    )
    combined_text = " ".join([
        str(getattr(lesson, "lesson", "")),
        str(getattr(lesson, "prefer", "")),
        str(getattr(lesson, "avoid", "")),
        " ".join(concrete_terms[:8]),
    ])
    if confidence < min_confidence:
        return False, f"low_confidence:{confidence:.2f}"
    if risk > max_risk:
        return False, f"high_risk:{risk:.2f}"
    if not getattr(lesson, "wrong_item_id", "") or not getattr(lesson, "correct_item_id", ""):
        return False, "missing_wrong_or_correct_item_ids"
    if len(concrete_terms) < 2:
        return False, "too_few_concrete_terms"
    if memory_text_is_too_generic(combined_text):
        return False, "lesson_text_too_generic"
    return True, "accepted"


def _format_compact_history_lines(items: List[Dict[str, Any]]) -> str:
    if not items:
        return "- No user history."
    return "\n".join(
        f"- title: {str(item.get('title', '')).strip() or 'Unknown'}; category: {str(item.get('category', 'Unknown')).strip() or 'Unknown'}"
        for item in items[-10:]
    )


def _format_compact_candidate_lines(items: List[Dict[str, Any]]) -> str:
    return "\n".join(
        f"- candidate_id: {item.get('candidate_id')}, title: {str(item.get('title', '')).strip() or 'Unknown'}, category: {str(item.get('category', 'Unknown')).strip() or 'Unknown'}"
        for item in items
    )


def build_compact_score_prompt(
    history_items: List[Dict[str, Any]],
    aliased_candidates: List[Dict[str, Any]],
    prompt_sample: str = "",
    memory_payload: Optional[List[Any]] = None,
    user_profile_payload: Optional[Dict[str, Any]] = None,
) -> str:
    using_memory = bool(memory_payload)
    intro = (
        "You are scoring candidate items for a recommender system based on user history, candidate facts, and optional memory facts.\n"
        if using_memory else
        "You are scoring candidate items for a recommender system based only on user history and candidate facts.\n"
    )
    parts = [intro]
    if prompt_sample:
        parts.append(str(prompt_sample).strip() + "\n")
    parts.append("Inputs:\n")
    if user_profile_payload:
        parts.append("User Memory Profile:\n")
        parts.append(json.dumps(user_profile_payload, ensure_ascii=False, indent=2) + "\n")
    parts.append("User recent history:\n")
    parts.append(_format_compact_history_lines(history_items) + "\n")
    if memory_payload:
        parts.append("Memory Facts:\n")
        for row in memory_payload[:5]:
            if isinstance(row, dict):
                text = (
                    row.get("lesson")
                    or row.get("behavior_explanation")
                    or row.get("pattern")
                    or row.get("pattern_description")
                    or json.dumps(row, ensure_ascii=False)
                )
            else:
                text = str(row)
            cleaned_text = re.sub(r"\s+", " ", str(text)).strip()[:220]
            parts.append(f"- {cleaned_text}\n")
    parts.append("Candidate Items:\n")
    parts.append(_format_compact_candidate_lines(aliased_candidates) + "\n")
    parts.append(
        "\nOutput requirements:\n"
        "- Return ONLY valid compact JSON. No markdown.\n"
        "- Output one score row for every candidate_id exactly once.\n"
        "- Score is a number from 0.0 to 1.0.\n"
        "- Rationale must be <= 8 words.\n"
        "- Base scoring on recent history and candidate facts.\n"
        "- If history is weak, prefer broader category relevance.\n"
        "\nJSON format:\n"
        "{\n"
        '  "scores": [\n'
        '    {"candidate_id": "C01", "score": 0.0, "rationale": "short reason"}\n'
        "  ],\n"
        '  "reasoning": "one short sentence"\n'
        "}\n"
    )
    return "".join(parts)


def initialize_user_memory_from_history_v2(
    memory_system: RecommendationMemorySystem,
    user_id: str,
    user_data: Dict[str, Any],
    items_meta: Dict[str, Dict[str, Any]],
    max_positive_interactions: Optional[int] = None,
) -> UserMemoryProfile:
    history_ids = user_data.get("train", [])
    if max_positive_interactions and max_positive_interactions > 0:
        history_ids = history_ids[-max_positive_interactions:]
    else:
        history_ids = history_ids[-10:]
    history_items = _history_item_infos(history_ids, items_meta)
    if not history_items:
        return UserMemoryProfile(
            user_id=str(user_id),
            profile="The user has no usable recent history.",
            facets=[],
            evidence_item_ids=[],
            source="empty_history",
        )

    prompt = f"""Create a compact user memory from observed recommendation history.

Observed positive history:
{json.dumps(history_items, ensure_ascii=False, indent=2)}

Return ONLY valid JSON:
{{
  "profile": "under 80 words describing concrete preferences grounded in history",
  "facets": [
    "User likes ... and often prefers ...",
    "User does not show evidence for ..."
  ],
  "evidence_item_ids": ["item id from history"]
}}

Rules:
- Use only evidence from the observed history.
- Do not infer broad demographic traits.
- Prefer concrete titles, subgenres, product attributes, artist/style/format signals, or use-case signals.
- Avoid generic phrases like 'likes products in this category'."""

    try:
        raw = memory_system.qwen_generate(
            prompt=prompt,
            role_prompt="You summarize user preference evidence for a recommender system.",
            max_new_tokens=700,
            json_mode=True,
        )
        parsed = extract_json_object(raw)
        profile = str(parsed.get("profile") or "").strip()
        facets = [str(x).strip() for x in parsed.get("facets", []) if str(x).strip()]
        evidence = [str(x).strip() for x in parsed.get("evidence_item_ids", []) if str(x).strip()]
        if not profile:
            raise ValueError("empty profile")
        result = UserMemoryProfile(
            user_id=str(user_id),
            profile=profile,
            facets=facets[:8],
            evidence_item_ids=evidence[:10],
            source="llm_history_init",
        )
        memory_system._trace("user_memory_init_llm", {
            "user_id": user_id,
            "prompt": prompt,
            "answer": raw,
            "parsed": parsed,
            "profile": asdict(result),
        })
        return result
    except Exception as e:
        titles = [x["title"] for x in history_items[:5]]
        categories = sorted({x["category"] for x in history_items if x.get("category")})
        profile = (
            "The user has positive history with "
            + ", ".join(titles[:4])
            + (f". Observed categories: {', '.join(categories[:3])}." if categories else ".")
        )
        result = UserMemoryProfile(
            user_id=str(user_id),
            profile=profile[:500],
            facets=[f"User has positive history with {title}." for title in titles[:3]],
            evidence_item_ids=[str(x["item_id"]) for x in history_items[:10]],
            source="fallback_history_init",
        )
        memory_system._trace("user_memory_init_error", {
            "user_id": user_id,
            "error": str(e),
            "prompt": prompt,
            "fallback_profile": asdict(result),
        })
        return result


def make_failure_event_v2(
    user_id: str,
    user_data: Dict[str, Any],
    items_meta: Dict[str, Dict[str, Any]],
    pos_item: PairwiseItemState,
    neg_item: PairwiseItemState,
    explanation: str,
    user_memory_before: str,
    user_memory_after: str,
    max_positive_interactions: Optional[int] = None,
) -> FailureEvent:
    history_ids = user_data.get("train", [])
    if max_positive_interactions and max_positive_interactions > 0:
        history_ids = history_ids[-max_positive_interactions:]
    else:
        history_ids = history_ids[-10:]
    recent_history = _history_item_infos(history_ids, items_meta)
    seed = f"{user_id}|{pos_item.item_id}|{neg_item.item_id}|{hashlib.md5(str(explanation).encode()).hexdigest()[:8]}"
    event_id = hashlib.md5(seed.encode("utf-8")).hexdigest()[:16]
    return FailureEvent(
        event_id=event_id,
        source_user_id=str(user_id),
        recent_history=recent_history,
        user_memory_before=user_memory_before,
        user_memory_after=user_memory_after,
        wrong_item=asdict(neg_item),
        correct_item=asdict(pos_item),
        model_wrong_reasoning=explanation,
        failure_type="wrong_choice_between_positive_and_negative",
    )


def create_failure_lesson_v2(
    memory_system: RecommendationMemorySystem,
    event: FailureEvent,
) -> Optional[FailureLesson]:
    prompt = f"""Convert this failed recommendation event into one compact reusable memory.

Failed event JSON:
{json.dumps(asdict(event), ensure_ascii=False, indent=2)}

Return ONLY valid JSON:
{{
  "lesson": "one sentence: User likes ..., often prefers ..., and not ...",
  "prefer": "short concrete positive preference inferred from correct item/history",
  "avoid": "short concrete negative/superficial signal from wrong item",
  "applies_if": ["specific evidence terms required before using this memory"],
  "do_not_apply_if": ["conditions where this memory should be ignored"],
  "evidence_terms": ["concrete words/phrases from history/candidates that must match"],
  "confidence": 0.0,
  "overgeneralization_risk": 0.0
}}

Rules:
- Preserve the identity of the user, wrong item, and correct item in the reasoning internally, but make the lesson short.
- The lesson must be grounded in the event, not a generic rule.
- If evidence is weak, set confidence low and overgeneralization_risk high.
- Do not say the future user likes something unless it appears in history or the correct item."""

    try:
        raw = memory_system.qwen_generate(
            prompt=prompt,
            role_prompt="You extract reusable failure-correction memories for recommendation.",
            max_new_tokens=900,
            json_mode=True,
        )
        parsed = extract_json_object(raw)
        lesson_text = str(parsed.get("lesson") or "").strip()
        prefer = str(parsed.get("prefer") or "").strip()
        avoid = str(parsed.get("avoid") or "").strip()
        applies_if = normalize_evidence_terms(parsed.get("applies_if", []))
        do_not_apply_if = normalize_evidence_terms(parsed.get("do_not_apply_if", []))
        evidence_terms = normalize_evidence_terms(parsed.get("evidence_terms", []) + applies_if)
        confidence = max(0.0, min(1.0, _safe_float(parsed.get("confidence"), 0.5)))
        risk = max(0.0, min(1.0, _safe_float(parsed.get("overgeneralization_risk"), 0.5)))
        if not lesson_text:
            lesson_text = f"User likes {prefer}, often prefers it over {avoid}, and not generic category matches."
        if len(normalize_terms(lesson_text)) < 3:
            raise ValueError("lesson too generic")
        memory_id = hashlib.md5(f"{event.event_id}|{lesson_text}".encode("utf-8")).hexdigest()[:16]
        history_item_ids = [str(x.get("item_id")) for x in event.recent_history if x.get("item_id")]
        lesson = FailureLesson(
            memory_id=memory_id,
            source_user_id=event.source_user_id,
            source_event_id=event.event_id,
            lesson=lesson_text[:500],
            prefer=prefer[:250],
            avoid=avoid[:250],
            applies_if=applies_if[:12],
            do_not_apply_if=do_not_apply_if[:12],
            evidence_terms=evidence_terms[:16],
            wrong_item_id=str(event.wrong_item.get("item_id", "")),
            correct_item_id=str(event.correct_item.get("item_id", "")),
            wrong_item_title=str(event.wrong_item.get("title", ""))[:300],
            correct_item_title=str(event.correct_item.get("title", ""))[:300],
            wrong_item_category=str(event.wrong_item.get("category", ""))[:120],
            correct_item_category=str(event.correct_item.get("category", ""))[:120],
            source_user_preference=str(event.user_memory_before or event.user_memory_after or "")[:500],
            history_item_ids=history_item_ids[:20],
            confidence=confidence,
            overgeneralization_risk=risk,
        )
        memory_system._trace("failure_lesson_llm", {
            "event_id": event.event_id,
            "prompt": prompt,
            "answer": raw,
            "parsed": parsed,
            "lesson": asdict(lesson),
        })
        memory_system._trace("failure_lesson_created", {
            "event": asdict(event),
            "lesson": asdict(lesson),
        })
        return lesson
    except Exception as e:
        wrong_title = str(event.wrong_item.get("title", "wrong item"))
        correct_title = str(event.correct_item.get("title", "correct item"))
        history_terms = normalize_evidence_terms([
            x.get("title", "") for x in event.recent_history[:5]
        ] + [correct_title, wrong_title])
        lesson_text = (
            f"User likes signals similar to {correct_title}, often prefers them over "
            f"{wrong_title}, and not superficial category matches."
        )
        memory_id = hashlib.md5(f"{event.event_id}|fallback".encode("utf-8")).hexdigest()[:16]
        lesson = FailureLesson(
            memory_id=memory_id,
            source_user_id=event.source_user_id,
            source_event_id=event.event_id,
            lesson=lesson_text[:500],
            prefer=correct_title[:250],
            avoid=wrong_title[:250],
            applies_if=history_terms[:8],
            do_not_apply_if=[],
            evidence_terms=history_terms[:12],
            wrong_item_id=str(event.wrong_item.get("item_id", "")),
            correct_item_id=str(event.correct_item.get("item_id", "")),
            wrong_item_title=str(event.wrong_item.get("title", ""))[:300],
            correct_item_title=str(event.correct_item.get("title", ""))[:300],
            wrong_item_category=str(event.wrong_item.get("category", ""))[:120],
            correct_item_category=str(event.correct_item.get("category", ""))[:120],
            source_user_preference=str(event.user_memory_before or event.user_memory_after or "")[:500],
            history_item_ids=[str(x.get("item_id")) for x in event.recent_history if x.get("item_id")][:20],
            confidence=0.35,
            overgeneralization_risk=0.75,
        )
        memory_system._trace("failure_lesson_error", {
            "event": asdict(event),
            "error": str(e),
            "fallback_lesson": asdict(lesson),
        })
        return lesson


def read_graph_lessons_as_facets_v2(
    memory_system: RecommendationMemorySystem,
    user_profile: Optional[UserMemoryProfile],
    train_items: List[Dict[str, Any]],
    candidate_items: List[Dict[str, Any]],
    retrieved_lessons: List[GraphRetrievedLesson],
    trace_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Select safe factual memory snippets without another LLM call.

    The evaluation prompt receives facts only, not a regenerated analytical
    memory. This is intentionally conservative for Qwen-sized models:
    - same_user memories can be used directly;
    - candidate_item memories need evidence in the current user's history/profile;
    - neighbor_user/history_item memories also need history/profile evidence.
    """
    if not retrieved_lessons:
        return {
            "use_memory": False,
            "facets": [],
            "memory_facts": [],
            "used_memory_ids": [],
            "rejected_memory_ids": [],
            "reason": "no graph lessons retrieved",
        }

    history_profile_text = (
        _context_text_from_items(train_items[-10:])
        + " "
        + (user_profile.profile.lower() if user_profile else "")
        + " "
        + " ".join(user_profile.facets if user_profile else []).lower()
    )
    candidate_text = _context_text_from_items(candidate_items)

    used_facts: List[str] = []
    used_ids: List[str] = []
    rejected: List[Dict[str, Any]] = []
    selected_rows: List[Dict[str, Any]] = []
    all_rows: List[Dict[str, Any]] = []
    weak_gate_terms = {
        "like", "likes", "liked", "specific", "history", "brands", "brand",
        "shows", "show", "interest", "interests", "preference", "preferences",
        "mentions", "includes", "recent", "particular", "frequently", "engages",
        "beauty", "product", "products", "category", "item", "items",
    }

    for r in retrieved_lessons:
        lesson = r.lesson
        evidence_terms = normalize_evidence_terms(
            list(lesson.evidence_terms or [])
            + list(lesson.applies_if or [])
            + [lesson.prefer, lesson.correct_item_title, lesson.correct_item_category]
        )
        concrete_terms = [
            term for term in evidence_terms
            if term not in weak_gate_terms and len(str(term).strip()) >= 4
        ]
        history_matches = [
            term for term in concrete_terms
            if term_matches_context(term, history_profile_text)
        ]
        candidate_matches = [
            term for term in concrete_terms
            if term_matches_context(term, candidate_text)
        ]
        strong_history_matches = [
            term for term in history_matches
            if (" " in term or len(term) >= 6)
        ]
        same_user = "same_user" in r.sources
        candidate_item_path = "candidate_item" in r.sources
        strong_graph_path = same_user or "history_item" in r.sources

        row = {
            "memory_id": lesson.memory_id,
            "safe_fact": lesson.safe_fact(),
            "sources": r.sources,
            "paths": r.paths[:5],
            "history_profile_matches": history_matches,
            "strong_history_matches": strong_history_matches,
            "candidate_matches": candidate_matches,
            "confidence": lesson.confidence,
            "overgeneralization_risk": lesson.overgeneralization_risk,
            "correct_item_id": lesson.correct_item_id,
            "wrong_item_id": lesson.wrong_item_id,
            "correct_item_title": lesson.correct_item_title,
            "wrong_item_title": lesson.wrong_item_title,
        }
        all_rows.append(row)

        # Main safety rule: avoid candidate-only activation. If only candidate
        # item triggered retrieval, require evidence from the current user's
        # own history/profile before showing this fact to the ranker.
        accept = False
        reason = ""
        if same_user:
            accept = True
            reason = "same_user memory"
        elif candidate_item_path and strong_history_matches:
            accept = True
            reason = "candidate_item path plus strong user-history evidence"
        elif strong_graph_path and strong_history_matches:
            accept = True
            reason = "history/neighbor graph path plus strong user-history evidence"
        else:
            reason = "rejected: no strong current user-history evidence"

        if lesson.overgeneralization_risk >= 0.85 and not same_user:
            accept = False
            reason = "rejected: high overgeneralization risk"

        if accept:
            used_ids.append(lesson.memory_id)
            used_facts.append(lesson.safe_fact())
            row["accept_reason"] = reason
            selected_rows.append(row)
        else:
            row["reject_reason"] = reason
            rejected.append(row)
        if len(used_facts) >= 3:
            break

    result = {
        "use_memory": bool(used_facts),
        # Keep key name `facets` for backward compatibility with existing code,
        # but the content is now factual memory snippets, not model-generated facets.
        "facets": used_facts,
        "memory_facts": used_facts,
        "used_memory_ids": used_ids,
        "rejected_memory_ids": [x["memory_id"] for x in rejected],
        "reason": "deterministic safe factual memory selection" if used_facts else "no safe factual memory passed history evidence gate",
    }
    memory_system._trace("memory_facts_selected", {
        **(trace_context or {}),
        "result": result,
        "selected_rows": selected_rows,
        "rejected_rows": rejected,
        "retrieved_graph_lessons": all_rows,
        "history_profile_text": history_profile_text[:2000],
    })
    return result

def llm_ranking_v2(
    memory_system: RecommendationMemorySystem,
    train_items: List[Dict[str, Any]],
    candidate_items: List[Dict[str, Any]],
    user_profile: Optional[UserMemoryProfile],
    memory_facets: Optional[List[str]],
    prompt_sample: str = "",
    ranking_prompt_style: str = "memcf",
    trace_context: Optional[Dict[str, Any]] = None,
) -> List[str]:
    candidate_info = [
        {"item_id": str(item["item_id"]), "title": item["title"], "category": item["category"]}
        for item in candidate_items
    ]
    aliased_candidates, alias_to_item_id = add_candidate_aliases(candidate_info)
    valid_candidate_aliases = list(alias_to_item_id.keys())
    profile_block = user_profile.to_prompt_dict() if user_profile else {}
    facets = [str(x) for x in (memory_facets or []) if str(x).strip()]

    if ranking_prompt_style == "compact_score":
        prompt = build_compact_score_prompt(
            history_items=train_items[-10:],
            aliased_candidates=aliased_candidates,
            prompt_sample=prompt_sample,
            memory_payload=facets if facets else None,
            user_profile_payload=profile_block if profile_block else None,
        )
    elif facets:
        prompt = f"""
You are scoring candidate items for a recommender system.

Inputs:
User Memory Profile initialized from observed history:
{json.dumps(profile_block, ensure_ascii=False, indent=2)}

User Recent History:
{json.dumps(train_items[-10:], ensure_ascii=False, indent=2)}

Safe Graph Memory Facts (factual snippets from prior failures):
{json.dumps(facets, ensure_ascii=False, indent=2)}

Candidate Items (use candidate_id only in output):
{json.dumps(aliased_candidates, ensure_ascii=False, indent=2)}

Memory policy:
- The graph memory facts are weak evidence from prior observed failures.
- Use a memory fact only when it matches current history or candidate facts.
- If a memory fact conflicts with item facts, ignore it.
- Do not overgeneralize from a single failure memory.

Output requirements:
- Return ONLY valid JSON. No markdown.
- Output one score row for every candidate_id exactly once.
- Score is a number from 0.0 to 1.0.
- Rationale must be <= 8 words.

JSON format:
{{
  "scores": [
    {{"candidate_id": "C01", "score": 0.0, "rationale": "short reason"}}
  ],
  "reasoning": "one short sentence"
}}
"""
    else:
        prompt = f"""
You are scoring candidate items for a recommender system based only on user history and candidate facts.
{prompt_sample}

Inputs:
User Memory Profile initialized from observed history:
{json.dumps(profile_block, ensure_ascii=False, indent=2)}

User Recent History:
{json.dumps(train_items[-10:], ensure_ascii=False, indent=2)}

Candidate Items (use candidate_id only in output):
{json.dumps(aliased_candidates, ensure_ascii=False, indent=2)}

Output requirements:
- Return ONLY valid JSON. No markdown.
- Output one score row for every candidate_id exactly once.
- Score is a number from 0.0 to 1.0.
- Rationale must be <= 8 words.

JSON format:
{{
  "scores": [
    {{"candidate_id": "C01", "score": 0.0, "rationale": "short reason"}}
  ],
  "reasoning": "one short sentence"
}}
"""

    score_json_schema = {
        "type": "object",
        "properties": {
            "scores": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "candidate_id": {"type": "string", "enum": valid_candidate_aliases},
                        "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "rationale": {"type": "string"},
                    },
                    "required": ["candidate_id", "score", "rationale"],
                    "additionalProperties": False,
                },
                "minItems": len(valid_candidate_aliases),
                "maxItems": len(valid_candidate_aliases),
            },
        },
        "required": ["scores"],
        "additionalProperties": False,
    }
    if ranking_prompt_style != "compact_score":
        score_json_schema["properties"]["reasoning"] = {"type": "string"}
        score_json_schema["required"] = ["scores", "reasoning"]

    max_retries = int(os.getenv("MEMCF_RANK_RETRIES", "1"))
    current_prompt = prompt
    for attempt in range(max_retries + 1):
        try:
            raw_response = memory_system.qwen_generate(
                prompt=current_prompt,
                role_prompt=(
                    "You are a deterministic recommender scorer. "
                    "Return JSON only and follow the provided JSON schema exactly."
                ),
                max_new_tokens=int(os.getenv("MEMCF_RANK_MAX_TOKENS", "1400")),
                json_schema=score_json_schema,
                json_mode=True,
            )
            try:
                result = extract_json_object(raw_response)
                raw_scores = result.get("scores", [])
            except Exception:
                result = {
                    "scores": parse_score_entries_from_text(raw_response, alias_to_item_id),
                    "reasoning": "Recovered score rows from malformed JSON",
                }
                raw_scores = result.get("scores", [])
            ranked_ids, validation = score_entries_to_ranking(raw_scores, alias_to_item_id)
            memory_system.memory_diagnostics["rank_score_calls"] += 1
            if validation["is_valid"]:
                memory_system.memory_diagnostics["rank_valid_score_outputs"] += 1
            else:
                memory_system.memory_diagnostics["rank_invalid_score_outputs"] += 1
            memory_system._trace("ranking_llm", {
                **(trace_context or {}),
                "attempt": attempt,
                "memcf_graph": True,
                "ranking_mode": "score_based_candidate_alias_graph_facets",
                "prompt": current_prompt,
                "answer": raw_response,
                "parsed": result,
                "score_validation": validation,
                "cleaned_ranked_item_ids": ranked_ids,
                "candidate_items": candidate_info,
                "aliased_candidate_items": aliased_candidates,
                "alias_to_item_id": alias_to_item_id,
                "train_items": train_items[-10:],
                "user_memory_profile": asdict(user_profile) if user_profile else None,
                "memory_facts": facets,
                "use_graph_memory_facts": bool(facets),
            })
            if validation["is_valid"] or attempt >= max_retries:
                return ranked_ids
            current_prompt = f"""{prompt}

The previous answer was invalid:
{json.dumps(validation, ensure_ascii=False, indent=2)}

Retry now. Return ONLY valid JSON with exactly one score row for every candidate_id.
"""
        except Exception as e:
            memory_system.memory_diagnostics["rank_attempt_errors"] += 1
            memory_system._trace("ranking_attempt_error", {
                **(trace_context or {}),
                "attempt": attempt,
                "error": str(e),
                "prompt": current_prompt,
                "candidate_items": candidate_info,
                "memory_facts": facets,
            })
            if attempt >= max_retries:
                break
    memory_system.memory_diagnostics["rank_fallbacks"] += 1
    return [str(item["item_id"]) for item in candidate_items]


def train_memory_graph_from_fail_interactions_v2(
    user_id: str,
    user_data: Dict[str, Any],
    negative_data: Optional[Dict[str, Any]],
    items_meta: Dict[str, Dict[str, Any]],
    memory_system: RecommendationMemorySystem,
    user_states: Dict[str, PairwiseUserState],
    item_states: Dict[str, PairwiseItemState],
    graph: MemoryGraphIndex,
    max_iterations: int = 1,
    max_positive_interactions: Optional[int] = None,
    candidate_negative_mode: str = "random",
    min_lesson_confidence: float = 0.25,
    max_lesson_risk: float = 0.85,
    max_failure_lessons_per_user: int = 3,
) -> List[FailureLesson]:
    train_items = user_data.get("train", [])
    if max_positive_interactions and max_positive_interactions > 0:
        train_items = train_items[-max_positive_interactions:]
    else:
        train_items = train_items[-30:]
    if not train_items:
        return []

    user_state = get_or_create_user_state(user_states, str(user_id))
    all_item_ids = list(item_states.keys())
    new_lessons: List[FailureLesson] = []
    for pos_item_id in train_items:
        pos_item_id = str(pos_item_id)
        if pos_item_id not in item_states:
            continue
        neg_item_id = choose_training_negative_item_id(
            user_id=str(user_id),
            pos_item_id=str(pos_item_id),
            user_data=user_data,
            negative_data=negative_data,
            items_meta=items_meta,
            all_item_ids=all_item_ids,
            mode=candidate_negative_mode,
            max_positive_interactions=max_positive_interactions,
        )
        if not neg_item_id:
            continue
        pos_item = item_states[pos_item_id]
        neg_item = item_states[neg_item_id]

        for _ in range(max_iterations):
            user_memory_before = user_state.short_term_memory
            chosen_item_id, explanation = autonomous_pairwise_interaction(
                memory_system=memory_system,
                user_state=user_state,
                pos_item=pos_item,
                neg_item=neg_item,
            )
            memory_system._trace("autonomous_choice_result", {
                "user_id": user_id,
                "positive_item_id": pos_item_id,
                "negative_item_id": neg_item_id,
                "chosen_item_id": chosen_item_id,
                "is_failure": chosen_item_id != pos_item_id,
            })
            if chosen_item_id == pos_item_id:
                user_state.add_interaction(pos_item_id)
                break
            try:
                corrective_pairwise_reflection(
                    memory_system=memory_system,
                    user_state=user_state,
                    pos_item=pos_item,
                    neg_item=neg_item,
                    chosen_item_id=chosen_item_id,
                    explanation=explanation,
                )
            except Exception as e:
                print(f"  ⚠ Reflection failed for user {user_id}, item {pos_item_id}: {e}")
                memory_system._trace("reflection_error", {
                    "user_id": user_id,
                    "positive_item_id": pos_item_id,
                    "negative_item_id": neg_item_id,
                    "error": str(e),
                })
                continue

            event = make_failure_event_v2(
                user_id=str(user_id),
                user_data=user_data,
                items_meta=items_meta,
                pos_item=pos_item,
                neg_item=neg_item,
                explanation=explanation,
                user_memory_before=user_memory_before,
                user_memory_after=user_state.short_term_memory,
                max_positive_interactions=max_positive_interactions,
            )
            memory_system._trace("failure_event_created", asdict(event))
            lesson = create_failure_lesson_v2(memory_system, event)
            if lesson is None:
                continue
            passed_gate, gate_reason = failure_lesson_passes_quality_gate_v2(
                lesson,
                min_confidence=min_lesson_confidence,
                max_risk=max_lesson_risk,
            )
            memory_system._trace("memory_quality_gate", {
                "user_id": user_id,
                "positive_item_id": pos_item.item_id,
                "negative_item_id": neg_item.item_id,
                "passed": passed_gate,
                "reason": gate_reason,
                "min_lesson_confidence": min_lesson_confidence,
                "max_lesson_risk": max_lesson_risk,
                "lesson": asdict(lesson),
            })
            if not passed_gate:
                continue
            graph.add_lesson(lesson)
            new_lessons.append(lesson)
            memory_system._trace("global_memory_added", {
                "memory_type": "FailureLesson",
                "lesson": asdict(lesson),
                "graph_edges": {
                    "source_user": lesson.source_user_id,
                    "wrong_item_id": lesson.wrong_item_id,
                    "correct_item_id": lesson.correct_item_id,
                    "history_item_ids": lesson.history_item_ids,
                },
            })
            if max_failure_lessons_per_user > 0 and len(new_lessons) >= max_failure_lessons_per_user:
                memory_system._trace("memory_generation_limit_reached", {
                    "user_id": user_id,
                    "max_failure_lessons_per_user": max_failure_lessons_per_user,
                    "current_count": len(new_lessons),
                })
                return new_lessons
    return new_lessons


def evaluate_user_v2(
    user_id: str,
    user_data: Dict[str, Any],
    negative_data: Dict[str, Any],
    items_meta: Dict[str, Dict[str, Any]],
    memory_system: RecommendationMemorySystem,
    graph: MemoryGraphIndex,
    user_profiles: Dict[str, UserMemoryProfile],
    eval_type: str = "test",
    use_memory: bool = True,
    graph_memory_k: int = 3,
    neighbor_k: int = 10,
    min_evidence_terms: int = 1,
    max_positive_interactions: Optional[int] = None,
    max_negative_candidates: Optional[int] = None,
    no_harm_arbitration: bool = False,
    ranking_prompt_style: str = "memcf",
) -> Tuple[Dict[str, float], Dict[str, float], List[str], List[str], List[str]]:
    if eval_type == "val":
        ground_truth = [str(x) for x in user_data.get("val", [])]
        negatives = [str(x) for x in negative_data.get("val_neg", [])]
    else:
        ground_truth = [str(x) for x in user_data.get("test", [])]
        negatives = [str(x) for x in negative_data.get("test_neg", [])]
    if max_negative_candidates and max_negative_candidates > 0:
        negatives = negatives[:max_negative_candidates]
    candidates = deterministic_shuffle(ground_truth + negatives, salt=f"{eval_type}_candidates")

    train_history = [str(x) for x in user_data.get("train", [])]
    if max_positive_interactions and max_positive_interactions > 0:
        train_history_for_prompt = train_history[-max_positive_interactions:]
    else:
        train_history_for_prompt = train_history[-10:]
    train_items_info = _history_item_infos(train_history_for_prompt, items_meta)
    candidate_items_info = [_item_info_for_prompt(item_id, items_meta) for item_id in candidates]
    user_profile = user_profiles.get(str(user_id))

    retrieved_graph_lessons: List[GraphRetrievedLesson] = []
    memory_reader_result = {"use_memory": False, "facets": [], "reason": "memory disabled"}
    if use_memory:
        current_context_text = (
            _context_text_from_items(train_items_info)
            + " "
            + _context_text_from_items(candidate_items_info)
            + " "
            + (user_profile.profile.lower() if user_profile else "")
        )
        retrieved_graph_lessons = graph.retrieve(
            user_id=str(user_id),
            recent_history_ids=train_history_for_prompt,
            candidate_ids=candidates,
            current_context_text=current_context_text,
            top_k=graph_memory_k,
            neighbor_k=neighbor_k,
            min_evidence_terms=min_evidence_terms,
        )
        memory_system.record_memory_diagnostics(
            retrieved=len(retrieved_graph_lessons),
            kept=len(retrieved_graph_lessons),
            skipped=0,
        )
        memory_system._trace("graph_memory_retrieval", {
            "user_id": user_id,
            "eval_type": eval_type,
            "graph_memory_k": graph_memory_k,
            "neighbor_k": neighbor_k,
            "min_evidence_terms": min_evidence_terms,
            "candidate_item_ids": candidates,
            "recent_history_ids": train_history_for_prompt,
            "retrieved": [
                {
                    "lesson": asdict(r.lesson),
                    "score": r.score,
                    "sources": r.sources,
                    "paths": r.paths,
                    "matched_evidence_terms": r.matched_evidence_terms,
                }
                for r in retrieved_graph_lessons
            ],
        })
        memory_reader_result = read_graph_lessons_as_facets_v2(
            memory_system=memory_system,
            user_profile=user_profile,
            train_items=train_items_info,
            candidate_items=candidate_items_info,
            retrieved_lessons=retrieved_graph_lessons,
            trace_context={"user_id": user_id, "eval_type": eval_type},
        )

    memory_facets = memory_reader_result.get("facets", []) if memory_reader_result.get("use_memory") else []
    selected_ranking_source = "graph_memory_facts" if memory_facets else "no_memory"
    no_memory_predictions = None
    memory_predictions = None
    arbitration = {"enabled": False, "selected_ranking_source": selected_ranking_source}
    if use_memory and no_harm_arbitration:
        memory_system.memory_diagnostics["no_harm_users"] += 1
        no_memory_predictions = llm_ranking_v2(
            memory_system, train_items_info, candidate_items_info, user_profile, [],
            ranking_prompt_style=ranking_prompt_style,
            trace_context={"user_id": user_id, "eval_type": eval_type, "ranking_path": "v2_no_harm_no_memory"},
        )
        if memory_facets:
            memory_predictions = llm_ranking_v2(
                memory_system, train_items_info, candidate_items_info, user_profile, memory_facets,
                ranking_prompt_style=ranking_prompt_style,
                trace_context={"user_id": user_id, "eval_type": eval_type, "ranking_path": "v2_no_harm_memory"},
            )
            # Conservative rule: use memory only if the reader accepted facets and
            # at least one retrieved lesson has direct same_user/candidate_item path.
            strong_path = any(
                ("same_user" in r.sources or "candidate_item" in r.sources)
                for r in retrieved_graph_lessons
            )
            if strong_path:
                predictions = memory_predictions
                selected_ranking_source = "graph_memory_facts"
                memory_system.memory_diagnostics["no_harm_used_memory"] += 1
            else:
                predictions = no_memory_predictions
                selected_ranking_source = "no_memory"
                memory_system.memory_diagnostics["no_harm_fallback_no_memory"] += 1
            arbitration = {
                "enabled": True,
                "selected_ranking_source": selected_ranking_source,
                "strong_graph_path": strong_path,
                "reason": "use memory only for same_user/candidate_item graph evidence",
            }
        else:
            predictions = no_memory_predictions
            selected_ranking_source = "no_memory"
            memory_system.memory_diagnostics["no_harm_fallback_no_memory"] += 1
            arbitration = {
                "enabled": True,
                "selected_ranking_source": selected_ranking_source,
                "reason": "fallback to no-memory: no accepted facets",
            }
        memory_system._trace("no_harm_arbitration", {
            "user_id": user_id,
            "eval_type": eval_type,
            "decision": arbitration,
            "no_memory_predictions": no_memory_predictions,
            "memory_predictions": memory_predictions,
            "selected_predictions": predictions,
            "memory_reader_result": memory_reader_result,
        })
    else:
        predictions = llm_ranking_v2(
            memory_system, train_items_info, candidate_items_info, user_profile, memory_facets,
            ranking_prompt_style=ranking_prompt_style,
            trace_context={
                "user_id": user_id,
                "eval_type": eval_type,
                "ranking_path": "v2_single_path",
                "use_memory": use_memory,
                "selected_ranking_source": selected_ranking_source,
            },
        )

    baseline_metric = {
        "recall@5": calculate_recall_at_k(candidates, ground_truth, 5),
        "recall@10": calculate_recall_at_k(candidates, ground_truth, 10),
        "recall@20": calculate_recall_at_k(candidates, ground_truth, 20),
        "ndcg@5": calculate_ndcg_at_k(candidates, ground_truth, 5),
        "ndcg@10": calculate_ndcg_at_k(candidates, ground_truth, 10),
        "ndcg@20": calculate_ndcg_at_k(candidates, ground_truth, 20),
    }
    metrics = {
        "recall@5": calculate_recall_at_k(predictions, ground_truth, 5),
        "recall@10": calculate_recall_at_k(predictions, ground_truth, 10),
        "recall@20": calculate_recall_at_k(predictions, ground_truth, 20),
        "ndcg@5": calculate_ndcg_at_k(predictions, ground_truth, 5),
        "ndcg@10": calculate_ndcg_at_k(predictions, ground_truth, 10),
        "ndcg@20": calculate_ndcg_at_k(predictions, ground_truth, 20),
    }
    memory_system._trace("ranking_result", {
        "user_id": user_id,
        "eval_type": eval_type,
        "use_memory": use_memory,
        "ground_truth": ground_truth,
        "candidate_item_ids": candidates,
        "ranked_item_ids": predictions,
        "metrics": metrics,
        "baseline_metrics": baseline_metric,
        "selected_ranking_source": selected_ranking_source,
        "ranking_prompt_style": ranking_prompt_style,
        "memory_reader_result": memory_reader_result,
        "retrieved_graph_lessons": [
            {
                "lesson": asdict(r.lesson),
                "score": r.score,
                "sources": r.sources,
                "paths": r.paths,
                "matched_evidence_terms": r.matched_evidence_terms,
            }
            for r in retrieved_graph_lessons
        ],
        "no_memory_predictions": no_memory_predictions,
        "memory_predictions": memory_predictions,
    })
    return baseline_metric, metrics, candidates, predictions, ground_truth


def parse_args_v2():
    parser = argparse.ArgumentParser(description="MEMCF graph-memory experiment")
    parser.add_argument("--data_name", type=str, default="Video_Game")
    parser.add_argument("--use_memory", action="store_true", default=True)
    parser.add_argument("--no_use_memory", action="store_false", dest="use_memory")
    parser.add_argument("--LOAD_SAVED_MEMORY", action="store_true", default=False)
    parser.add_argument("--max_iterations", type=int, default=1)
    parser.add_argument("--number_of_users", type=int, default=100)
    parser.add_argument("--max_positive_interactions", type=int, default=5)
    parser.add_argument("--max_negative_candidates", type=int, default=19)
    parser.add_argument("--graph_memory_k", type=int, default=3)
    parser.add_argument("--neighbor_k", type=int, default=10)
    parser.add_argument("--min_evidence_terms", type=int, default=1)
    parser.add_argument("--no_harm_arbitration", action="store_true", default=False)
    parser.add_argument("--candidate_negative_mode", type=str, default="candidate_hard",
                        choices=["random", "candidate_hard"])
    parser.add_argument("--min_lesson_confidence", type=float, default=0.25)
    parser.add_argument("--max_lesson_risk", type=float, default=0.85)
    parser.add_argument("--max_failure_lessons_per_user", type=int, default=3)
    parser.add_argument("--ranking_prompt_style", type=str, default="compact_score",
                        choices=["memcf", "compact_score"])
    parser.add_argument("--trace_dir", type=str, default=None)
    parser.add_argument("--disable_trace", action="store_false", dest="trace_enabled")
    parser.set_defaults(trace_enabled=True)
    return parser.parse_args()


def save_v2_memory(path: str, graph: MemoryGraphIndex, user_profiles: Dict[str, UserMemoryProfile]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "model": "MEMCF",
        "saved_at": datetime.now().isoformat(),
        "graph": graph.to_dict(),
        "user_profiles": {uid: asdict(profile) for uid, profile in user_profiles.items()},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(make_jsonable(payload), f, ensure_ascii=False, indent=2)
    print(f"✓ Saved MEMCF graph memory to {path}")


def load_v2_memory(path: str, user_sequences: Dict[str, Dict[str, Any]]) -> Tuple[MemoryGraphIndex, Dict[str, UserMemoryProfile]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    graph = MemoryGraphIndex.from_dict(payload.get("graph", {}), user_sequences)
    user_profiles = {
        str(uid): UserMemoryProfile(**profile)
        for uid, profile in payload.get("user_profiles", {}).items()
    }
    return graph, user_profiles


def main_v2():
    args = parse_args_v2()
    random.seed(2020)
    np.random.seed(2020)

    data_name = args.data_name
    use_memory = args.use_memory
    number_of_users = args.number_of_users
    max_positive_interactions = args.max_positive_interactions
    max_negative_candidates = args.max_negative_candidates
    candidate_negative_mode = args.candidate_negative_mode
    min_lesson_confidence = args.min_lesson_confidence
    max_lesson_risk = args.max_lesson_risk
    max_failure_lessons_per_user = args.max_failure_lessons_per_user
    ranking_prompt_style = args.ranking_prompt_style

    base_dir = os.getenv("AGENTICREC_CFMEMORY_ROOT", os.path.dirname(os.path.abspath(__file__)))
    data_root = os.getenv("AGENTICREC_DATA_ROOT", os.path.join(base_dir, "data"))
    eval_root = os.getenv("AGENTICREC_EVAL_ROOT", os.path.join(base_dir, "evaluation_results"))
    memory_root = os.getenv("AGENTICREC_MEMORY_ROOT", os.path.join(base_dir, "agent_memory"))
    data_dir = os.path.join(data_root, data_name)
    eval_dir = os.path.join(eval_root, data_name)
    memory_dir = os.path.join(memory_root, data_name)
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(memory_dir, exist_ok=True)

    items_path = os.path.join(data_dir, "items.json")
    sequences_path = os.path.join(data_dir, "user_sequences_10.json")
    negatives_path = os.path.join(data_dir, "user_negatives_10.json")
    items_meta, user_sequences, user_negatives = load_data(items_path, sequences_path, negatives_path)
    user_ids = list(user_sequences.keys())[:number_of_users]
    print(f"Total users loaded: {len(user_sequences)}")
    print(f"MEMCF selected users: {len(user_ids)}")

    run_name = (
        f"memcf_graph_nuser{number_of_users}_iter{args.max_iterations}"
        f"_gk{args.graph_memory_k}_nk{args.neighbor_k}_ev{args.min_evidence_terms}"
    )
    if not use_memory:
        run_name = f"memcf_nomemory_nuser{number_of_users}"

    trace_dir = args.trace_dir or os.path.join(
        eval_dir,
        "traces",
        f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    trace_recorder = TraceRecorder(trace_dir, enabled=args.trace_enabled)
    if args.trace_enabled:
        print(f"✓ Trace enabled: {trace_dir}")

    memory_system = RecommendationMemorySystem(use_gemini_embeddings=True)
    memory_system.trace_recorder = trace_recorder
    item_states = init_pairwise_item_states(items_meta)
    user_states: Dict[str, PairwiseUserState] = {}
    graph = MemoryGraphIndex(user_sequences)
    user_profiles: Dict[str, UserMemoryProfile] = {}
    memory_file_path = os.path.join(memory_dir, f"{run_name}.memory.json")

    if use_memory:
        if args.LOAD_SAVED_MEMORY and os.path.exists(memory_file_path):
            print(f"Loading MEMCF graph memory from {memory_file_path}")
            graph, user_profiles = load_v2_memory(memory_file_path, user_sequences)
            for uid, profile in user_profiles.items():
                user_states[uid] = PairwiseUserState(user_id=uid, short_term_memory=profile.profile)
        else:
            print("\n" + "=" * 80)
            print("PHASE 0: INITIALIZE USER MEMORY FROM HISTORY")
            print("=" * 80)
            for user_id in tqdm(user_ids, desc="Init user memories"):
                profile = initialize_user_memory_from_history_v2(
                    memory_system=memory_system,
                    user_id=str(user_id),
                    user_data=user_sequences[user_id],
                    items_meta=items_meta,
                    max_positive_interactions=max_positive_interactions,
                )
                user_profiles[str(user_id)] = profile
                user_states[str(user_id)] = PairwiseUserState(
                    user_id=str(user_id),
                    short_term_memory=profile.profile,
                )
                memory_system._trace("user_memory_initialized", {
                    "user_id": user_id,
                    "profile": asdict(profile),
                })

            print("\n" + "=" * 80)
            print("PHASE 1: PAIRWISE FAILURE TRAINING -> GRAPH LESSONS")
            print("=" * 80)
            total_lessons = 0
            for user_id in tqdm(user_ids, desc="Graph failure training"):
                print(f"\nProcessing user {user_id}")
                lessons = train_memory_graph_from_fail_interactions_v2(
                    user_id=str(user_id),
                    user_data=user_sequences[user_id],
                    negative_data=user_negatives.get(user_id, {}),
                    items_meta=items_meta,
                    memory_system=memory_system,
                    user_states=user_states,
                    item_states=item_states,
                    graph=graph,
                    max_iterations=args.max_iterations,
                    max_positive_interactions=max_positive_interactions,
                    candidate_negative_mode=candidate_negative_mode,
                    min_lesson_confidence=min_lesson_confidence,
                    max_lesson_risk=max_lesson_risk,
                    max_failure_lessons_per_user=max_failure_lessons_per_user,
                )
                total_lessons += len(lessons)
                print(f"  → Generated {len(lessons)} graph failure lessons")
            print(f"Total MEMCF graph lessons: {total_lessons}")
            save_v2_memory(memory_file_path, graph, user_profiles)
    else:
        print("MEMCF no-memory run: skipping user-memory init and failure-memory training.")

    print("\n" + "=" * 80)
    print("PHASE 2: VALIDATION SET EVALUATION")
    print("=" * 80)
    all_user_results = []
    val_metrics = defaultdict(list)
    baseline_metrics = defaultdict(list)
    for user_id in tqdm(user_ids, desc="Validation"):
        if user_id not in user_sequences or user_id not in user_negatives:
            continue
        baseline_metric, metrics, candidates, predictions, ground_truth = evaluate_user_v2(
            user_id=str(user_id),
            user_data=user_sequences[user_id],
            negative_data=user_negatives[user_id],
            items_meta=items_meta,
            memory_system=memory_system,
            graph=graph,
            user_profiles=user_profiles,
            eval_type="test",
            use_memory=use_memory,
            graph_memory_k=args.graph_memory_k,
            neighbor_k=args.neighbor_k,
            min_evidence_terms=args.min_evidence_terms,
            max_positive_interactions=max_positive_interactions,
            max_negative_candidates=max_negative_candidates,
            no_harm_arbitration=args.no_harm_arbitration,
            ranking_prompt_style=ranking_prompt_style,
        )
        for metric_name, value in metrics.items():
            val_metrics[metric_name].append(value)
        for metric_name, value in baseline_metric.items():
            baseline_metrics[metric_name].append(value)
        all_user_results.append({
            "user_id": user_id,
            "candidates": candidates,
            "predictions": predictions,
            "ground_truth": ground_truth,
            "metrics": metrics,
            "baseline_metrics": baseline_metric,
        })

    if use_memory:
        output_file = os.path.join(eval_dir, f"{run_name}.json")
    else:
        output_file = os.path.join(eval_dir, "memcf_zeroshot_users_ranking_no_memory.json")
    save_all_users_ranking_results(all_user_results, items_meta, output_file)

    print("\nValidation Results:")
    print("-" * 80)
    for metric in ["recall@5", "recall@10", "recall@20", "ndcg@5", "ndcg@10", "ndcg@20"]:
        if baseline_metrics[metric]:
            print(f"Baseline {metric:10s}: {np.mean(baseline_metrics[metric]):.4f}")
        else:
            print(f"Baseline {metric:10s}: N/A")
    print("-" * 80)
    for metric in ["recall@5", "recall@10", "recall@20", "ndcg@5", "ndcg@10", "ndcg@20"]:
        if val_metrics[metric]:
            print(f"{metric:12s}: {np.mean(val_metrics[metric]):.4f}")
        else:
            print(f"{metric:12s}: N/A")

    diag = getattr(memory_system, "memory_diagnostics", defaultdict(float))
    summary = {
        "model": "MEMCF",
        "dataset": data_name,
        "number_of_users_requested": number_of_users,
        "number_of_users_evaluated": len(all_user_results),
        "use_memory": use_memory,
        "load_saved_memory": args.LOAD_SAVED_MEMORY,
        "max_iterations": args.max_iterations,
        "max_positive_interactions": max_positive_interactions,
        "max_negative_candidates": max_negative_candidates,
        "candidate_negative_mode": candidate_negative_mode,
        "min_lesson_confidence": min_lesson_confidence,
        "max_lesson_risk": max_lesson_risk,
        "max_failure_lessons_per_user": max_failure_lessons_per_user,
        "ranking_prompt_style": ranking_prompt_style,
        "graph_memory_k": args.graph_memory_k,
        "neighbor_k": args.neighbor_k,
        "min_evidence_terms": args.min_evidence_terms,
        "no_harm_arbitration": args.no_harm_arbitration,
        "trace_enabled": args.trace_enabled,
        "trace_dir": trace_dir if args.trace_enabled else None,
        "memory_file": memory_file_path if use_memory else None,
        "num_graph_lessons": len(graph.lessons),
        "num_user_profiles": len(user_profiles),
        "baseline_metrics": {
            metric: (float(np.mean(baseline_metrics[metric])) if baseline_metrics[metric] else None)
            for metric in ["recall@5", "recall@10", "recall@20", "ndcg@5", "ndcg@10", "ndcg@20"]
        },
        "metrics": {
            metric: (float(np.mean(val_metrics[metric])) if val_metrics[metric] else None)
            for metric in ["recall@5", "recall@10", "recall@20", "ndcg@5", "ndcg@10", "ndcg@20"]
        },
        "memory_diagnostics": {
            "eval_users_with_memory_retrieval": int(diag.get("eval_users", 0.0)),
            "retrieved_total": int(diag.get("retrieved_total", 0.0)),
            "kept_total": int(diag.get("kept_total", 0.0)),
            "users_with_kept_memory": int(diag.get("users_with_kept_memory", 0.0)),
            "avg_retrieved_memories": (
                float(diag.get("retrieved_total", 0.0)) / float(diag.get("eval_users", 0.0))
                if float(diag.get("eval_users", 0.0)) else 0.0
            ),
            "rank_score_calls": int(diag.get("rank_score_calls", 0.0)),
            "rank_valid_score_outputs": int(diag.get("rank_valid_score_outputs", 0.0)),
            "rank_invalid_score_outputs": int(diag.get("rank_invalid_score_outputs", 0.0)),
            "rank_fallbacks": int(diag.get("rank_fallbacks", 0.0)),
            "no_harm_users": int(diag.get("no_harm_users", 0.0)),
            "no_harm_used_memory": int(diag.get("no_harm_used_memory", 0.0)),
            "no_harm_fallback_no_memory": int(diag.get("no_harm_fallback_no_memory", 0.0)),
        },
    }
    summary_file = output_file.replace(".json", ".summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(make_jsonable(summary), f, ensure_ascii=False, indent=2)
    print(f"✓ Saved MEMCF summary to {summary_file}")
    trace_recorder.write_manifest({
        "run_name": run_name,
        "output_file": output_file,
        "summary_file": summary_file,
        "memory_file": memory_file_path if use_memory else None,
        "completed_at": datetime.now().isoformat(),
        "summary": summary,
    })


if __name__ == "__main__":
    main_v2()
