# MEMCF Trace Format

MEMCF writes JSONL traces to:

```text
$MEMCF_EVAL_ROOT/<dataset>/traces/<run_name_timestamp>/
```

Each event type is saved as `<event_type>.jsonl`. The global chronological stream is saved as `events.jsonl`.

## Core Training Traces

| File | Meaning |
| --- | --- |
| `user_memory_initialized.jsonl` | Initial user memory built from user history. |
| `autonomous_choice_result.jsonl` | Pairwise choice result during failure training. |
| `failure_event_created.jsonl` | Structured failure record when the model chooses the negative item. |
| `failure_lesson_llm.jsonl` | Prompt/response for distilling a failure into a memory lesson. |
| `failure_lesson_created.jsonl` | Parsed compact memory lesson. |
| `global_memory_added.jsonl` | Memory lesson added to the global graph memory pool. |

## Core Evaluation Traces

| File | Meaning |
| --- | --- |
| `graph_memory_retrieval.jsonl` | Graph retrieval candidates and selected facts per user. |
| `memory_facts_selected.jsonl` | Final short memory facts injected into the ranking prompt. |
| `ranking_llm.jsonl` | Full ranking prompt, raw LLM response, and parsed ranking. |
| `ranking_attempt_error.jsonl` | Parsing or request failures during ranking. |
| `ranking_result.jsonl` | Per-user ranking, answer item, and metrics metadata. |
| `no_harm_arbitration.jsonl` | Decision between memory and no-memory ranking when enabled. |

## Quick Inspection

```bash
TRACE=/path/to/trace_dir
find "$TRACE" -maxdepth 1 -type f -name "*.jsonl" -printf "%f\t" -exec wc -l {} \;
```

Show the first two memory retrieval records:

```bash
python3 - <<'PY'
import json
from pathlib import Path
trace = Path('/path/to/trace_dir')
for line in (trace / 'graph_memory_retrieval.jsonl').open():
    print(json.dumps(json.loads(line), ensure_ascii=False, indent=2)[:3000])
    break
PY
```

Find users where memory ranking differed from no-memory ranking:

```bash
python3 - <<'PY'
import json
from pathlib import Path
trace = Path('/path/to/trace_dir')
for line in (trace / 'no_harm_arbitration.jsonl').open():
    obj = json.loads(line)
    if obj.get('memory_rank') != obj.get('no_memory_rank'):
        print(json.dumps(obj, ensure_ascii=False, indent=2)[:3000])
PY
```
