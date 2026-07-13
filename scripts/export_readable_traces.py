#!/usr/bin/env python3
"""Export MEMCF/MemRec JSONL traces into beginner-readable Markdown.

This script is intentionally stdlib-only so it can run on the server without
installing dependencies.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

IMPORTANT_FILES = [
    "ranking_llm.jsonl",
    "memory_facts_selected.jsonl",
    "graph_memory_retrieval.jsonl",
    "failure_lesson_llm.jsonl",
    "failure_lesson_created.jsonl",
    "memory_create_llm.jsonl",
    "memory_created.jsonl",
    "autonomous_choice_llm.jsonl",
    "autonomous_choice_result.jsonl",
    "reflection_user_memory_llm.jsonl",
    "reflection_item_memory_llm.jsonl",
    "no_harm_arbitration.jsonl",
    "ranking_result.jsonl",
    "llm_call.jsonl",
    "events.jsonl",
]

FILE_DESCRIPTIONS = {
    "ranking_llm.jsonl": "Full evaluation ranking prompt, raw LLM answer, and parsed ranking if available.",
    "memory_facts_selected.jsonl": "Final memory facts selected/injected into the evaluation prompt.",
    "graph_memory_retrieval.jsonl": "Graph retrieval candidates, selected memory paths, and retrieval scores.",
    "failure_lesson_llm.jsonl": "Prompt/answer for converting a failed event into a compact lesson.",
    "failure_lesson_created.jsonl": "Parsed failure lesson object after quality gates.",
    "memory_create_llm.jsonl": "Legacy prompt/answer for creating a behavior memory from a fail case.",
    "memory_created.jsonl": "Legacy parsed behavior memory object.",
    "autonomous_choice_llm.jsonl": "Training pairwise choice prompt/answer that can generate fail interactions.",
    "autonomous_choice_result.jsonl": "Parsed pairwise choice result during training.",
    "reflection_user_memory_llm.jsonl": "Prompt/answer for updating user memory after a wrong choice.",
    "reflection_item_memory_llm.jsonl": "Prompt/answer for updating item memories after a wrong choice.",
    "no_harm_arbitration.jsonl": "No-harm verifier deciding whether memory ranking or no-memory ranking should be used.",
    "ranking_result.jsonl": "Per-user ranked output and metric metadata.",
    "llm_call.jsonl": "Per-call token/runtime usage; normally excludes full prompt text.",
    "events.jsonl": "Chronological event stream; useful as a fallback when split files are missing.",
    "*.conversations.jsonl": "MemRec conversation log with messages and model responses.",
}


def read_jsonl(path: Path, limit: Optional[int] = None) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return
    n = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                yield {"_parse_error": str(e), "_raw_line": line[:2000]}
            n += 1
            if limit is not None and n >= limit:
                break


def count_lines(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            return sum(1 for _ in f)
    except Exception:
        return 0


def short_text(value: Any, limit: int = 1200) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False, indent=2)
    else:
        text = str(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if len(text) > limit:
        return text[:limit].rstrip() + f"\n... [truncated {len(text)-limit} chars]"
    return text


def md_code(value: Any, limit: int = 2500, lang: str = "text") -> str:
    text = short_text(value, limit=limit)
    text = text.replace("```", "` ` `")
    return f"```{lang}\n{text}\n```"


def item_title(x: Any) -> str:
    if isinstance(x, dict):
        iid = x.get("item_id") or x.get("id") or x.get("candidate_id") or "?"
        title = x.get("title") or x.get("item") or x.get("name") or ""
        category = x.get("category") or x.get("tags") or ""
        return f"{iid}: {title}" + (f" [{category}]" if category else "")
    return str(x)


def record_user_id(obj: Dict[str, Any]) -> str:
    meta = obj.get("metadata") if isinstance(obj.get("metadata"), dict) else {}
    return str(obj.get("user_id") or obj.get("user") or meta.get("user_id") or obj.get("id") or "unknown")


def record_kind(obj: Dict[str, Any]) -> str:
    meta = obj.get("metadata") if isinstance(obj.get("metadata"), dict) else {}
    return str(obj.get("event_type") or obj.get("call_type") or meta.get("stage") or meta.get("call_type") or "record")


def find_trace_dirs(root: Path) -> List[Path]:
    if root.is_file():
        return [root.parent]
    if any((root / name).exists() for name in IMPORTANT_FILES) or list(root.glob("*.conversations.jsonl")):
        return [root]
    dirs = []
    for p in root.rglob("*.jsonl"):
        parent = p.parent
        if parent not in dirs:
            dirs.append(parent)
    return sorted(dirs)


def collect_records(trace: Path, sample_per_file: int) -> Dict[str, List[Dict[str, Any]]]:
    records = {}
    for p in sorted(trace.glob("*.jsonl")):
        name = p.name
        if name in IMPORTANT_FILES or name.endswith(".conversations.jsonl"):
            records[name] = list(read_jsonl(p, sample_per_file))
    return records


def render_summary(trace: Path) -> str:
    files = sorted(trace.glob("*.jsonl"))
    lines = [f"# Trace Summary: `{trace.name}`", ""]
    lines.append(f"Source trace directory: `{trace}`")
    lines.append("")
    lines.append("## File Inventory")
    lines.append("")
    lines.append("| File | Records | Meaning |")
    lines.append("| --- | ---: | --- |")
    for p in files:
        desc = FILE_DESCRIPTIONS.get(p.name, "Additional trace stream.")
        if p.name.endswith(".conversations.jsonl"):
            desc = FILE_DESCRIPTIONS["*.conversations.jsonl"]
        lines.append(f"| `{p.name}` | {count_lines(p)} | {desc} |")
    return "\n".join(lines) + "\n"


def render_llm_record(obj: Dict[str, Any], file_name: str, idx: int, prompt_chars: int, answer_chars: int) -> str:
    user = record_user_id(obj)
    kind = record_kind(obj)
    title = f"## {idx}. `{file_name}` - {kind} - user `{user}`"
    parts = [title, ""]

    for key in ["timestamp", "eval_type", "variant", "eval_variant", "ranking_prompt_style", "use_memory", "graph_retrieval_scope", "memory_retrieval_mode", "memory_gate"]:
        if key in obj:
            parts.append(f"- `{key}`: `{short_text(obj.get(key), 200)}`")
    if "ground_truth" in obj:
        parts.append(f"- `ground_truth`: `{short_text(obj.get('ground_truth'), 300)}`")
    if "fixed_candidates" in obj:
        parts.append(f"- `candidate_count`: `{len(obj.get('fixed_candidates') or [])}`")
    if "candidate_items" in obj:
        cands = obj.get("candidate_items") or []
        parts.append("- Candidate preview:")
        for c in cands[:8]:
            parts.append(f"  - {item_title(c)}")
    if "train_items" in obj:
        hist = obj.get("train_items") or []
        parts.append("- User history preview:")
        for h in hist[:6]:
            parts.append(f"  - {item_title(h)}")
    parts.append("")

    prompt = obj.get("prompt")
    if prompt is None and "messages" in obj:
        prompt = obj.get("messages")
    if prompt is not None:
        parts.append("### Prompt")
        parts.append(md_code(prompt, limit=prompt_chars, lang="text"))
        parts.append("")

    if obj.get("role_prompt"):
        parts.append("### System / Role Prompt")
        parts.append(md_code(obj.get("role_prompt"), limit=1200, lang="text"))
        parts.append("")

    answer = obj.get("answer")
    if answer is None:
        answer = obj.get("response")
    if answer is not None:
        parts.append("### LLM Answer")
        parts.append(md_code(answer, limit=answer_chars, lang="text"))
        parts.append("")

    for key in ["parsed", "scores", "ranking", "cleaned_ranked_item_ids", "raw_ranked_item_ids", "selected_rows", "rejected_rows", "retrieved_memories", "memory", "parsed_lesson", "result"]:
        if key in obj and obj.get(key) not in (None, ""):
            parts.append(f"### `{key}`")
            parts.append(md_code(obj.get(key), limit=2500, lang="json"))
            parts.append("")

    return "\n".join(parts).rstrip() + "\n"


def render_beginner_guide() -> str:
    return """# Beginner Guide To Reading These Traces

## What To Read First

1. `index.md`: file inventory and links.
2. `ranking_llm_samples.md`: exact evaluation prompts and LLM answers.
3. `memory_facts_selected_samples.md`: what memories were actually inserted into the prompt.
4. `graph_memory_retrieval_samples.md`: where those memories came from.
5. `failure_lesson_llm_samples.md` or `memory_create_llm_samples.md`: how memories were created from failed training cases.
6. `llm_usage.md`: LLM call/token/runtime cost.

## How To Interpret One Evaluation Case

- `User recent history`: what the user liked before evaluation.
- `Candidate Items`: one true held-out item plus negatives.
- `Memory Facts`: short memories retrieved from previous failures.
- `LLM Answer`: JSON scores/ranking generated by the model.
- `ranking_result`: whether the true item moved up/down.

## What Usually Matters For MEMCF

- A useful memory should mention concrete prefer/avoid evidence that appears in the current user history or current candidates.
- A harmful memory is often generic, from a wrong user/item path, or supports an item that is not semantically close to the current candidate set.
- Random/shuffled controls are meant to test whether memory provenance matters.
"""


def render_llm_usage(trace: Path) -> str:
    p = trace / "llm_call.jsonl"
    lines = ["# LLM Usage", ""]
    if not p.exists():
        lines.append("No `llm_call.jsonl` found.")
        return "\n".join(lines) + "\n"
    stats = defaultdict(lambda: Counter())
    total = Counter()
    for obj in read_jsonl(p):
        key = str(obj.get("call_type") or obj.get("event_type") or "unknown")
        stats[key]["calls"] += 1
        for k in ["prompt_tokens", "completion_tokens", "total_tokens", "seconds", "error"]:
            v = obj.get(k)
            if isinstance(v, (int, float)):
                stats[key][k] += v
                total[k] += v
        total["calls"] += 1
    lines.append("| Call type | Calls | Prompt tokens | Completion tokens | Total tokens | Seconds | Errors |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for key, c in sorted(stats.items()):
        lines.append(f"| `{key}` | {int(c['calls'])} | {int(c['prompt_tokens'])} | {int(c['completion_tokens'])} | {int(c['total_tokens'])} | {c['seconds']:.1f} | {int(c['error'])} |")
    lines.append(f"| **TOTAL** | **{int(total['calls'])}** | **{int(total['prompt_tokens'])}** | **{int(total['completion_tokens'])}** | **{int(total['total_tokens'])}** | **{total['seconds']:.1f}** | **{int(total['error'])}** |")
    return "\n".join(lines) + "\n"


def export_trace(trace: Path, out_root: Path, sample_per_file: int, prompt_chars: int, answer_chars: int) -> Dict[str, Any]:
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", trace.name)[:160] or "trace"
    out = out_root / safe_name
    out.mkdir(parents=True, exist_ok=True)

    (out / "README.md").write_text(render_beginner_guide(), encoding="utf-8")
    (out / "index.md").write_text(render_summary(trace), encoding="utf-8")
    (out / "llm_usage.md").write_text(render_llm_usage(trace), encoding="utf-8")

    files_written = ["README.md", "index.md", "llm_usage.md"]
    for p in sorted(trace.glob("*.jsonl")):
        if p.name not in IMPORTANT_FILES and not p.name.endswith(".conversations.jsonl"):
            continue
        recs = list(read_jsonl(p, sample_per_file))
        if not recs:
            continue
        parts = [f"# Samples From `{p.name}`", "", FILE_DESCRIPTIONS.get(p.name, "Trace samples."), ""]
        for i, obj in enumerate(recs, 1):
            parts.append(render_llm_record(obj, p.name, i, prompt_chars, answer_chars))
            parts.append("\n---\n")
        fname = p.name.replace(".jsonl", "") + "_samples.md"
        (out / fname).write_text("\n".join(parts), encoding="utf-8")
        files_written.append(fname)

    return {"trace": str(trace), "out": str(out), "files": files_written}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace", action="append", default=[], help="Trace directory or JSONL file. Can be repeated.")
    ap.add_argument("--root", action="append", default=[], help="Root directory containing trace dirs. Can be repeated.")
    ap.add_argument("--out", required=True, help="Output directory for readable markdown traces.")
    ap.add_argument("--sample_per_file", type=int, default=5, help="Number of records exported per trace file.")
    ap.add_argument("--prompt_chars", type=int, default=6000, help="Max prompt chars per sample.")
    ap.add_argument("--answer_chars", type=int, default=4000, help="Max answer chars per sample.")
    args = ap.parse_args()

    inputs = [Path(x).expanduser() for x in args.trace]
    for root in args.root:
        inputs.extend(find_trace_dirs(Path(root).expanduser()))
    if not inputs:
        raise SystemExit("Provide --trace or --root")

    out_root = Path(args.out).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = []
    seen = set()
    for trace in inputs:
        trace = trace.resolve()
        if trace in seen:
            continue
        seen.add(trace)
        if not trace.exists():
            print(f"SKIP missing: {trace}")
            continue
        if trace.is_file():
            trace = trace.parent
        print(f"EXPORT {trace}")
        manifest.append(export_trace(trace, out_root, args.sample_per_file, args.prompt_chars, args.answer_chars))

    (out_root / "EXPORT_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Readable Trace Export", "", f"Trace dirs exported: {len(manifest)}", ""]
    for m in manifest:
        lines.append(f"- [`{Path(m['out']).name}`]({Path(m['out']).name}/index.md) from `{m['trace']}`")
    (out_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"DONE: {out_root}")


if __name__ == "__main__":
    main()
