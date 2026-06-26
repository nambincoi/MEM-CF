# MEMCF Ablation Protocol

This protocol isolates whether failure-derived memories help beyond a no-memory LLM reranker under the same candidate set.

The terminology is intentionally MEMCF-specific:

- **failure lesson**: a compact correction distilled from a wrong-vs-correct training failure.
- **failure graph**: typed links between users, history items, candidate items, and failure lessons.
- **applicability gate**: filtering that keeps a lesson only when it is supported by the current user/candidate context.
- **no-harm arbitration**: optional safety step that falls back to no-memory ranking when selected memories are weakly supported.

## Core Variants

| Variant | Purpose |
| --- | --- |
| `A0_no_memory` | LLM ranking from user history + candidate item metadata only. No memory training and no memory facts in ranking. |
| `A1_same_user_only` | Uses only failure lessons created from the target user. Tests personalized memory. |
| `A2_candidate_item_only` | Uses lessons connected to current candidate items. Tests candidate-specific corrective memory. |
| `A3_neighbor_user_only` | Uses lessons from users connected through shared-history items. Tests collaborative transfer. |
| `A4_full_graph` | Uses all supported graph paths: same user, candidate item, history item, and neighbor user. |
| `A5_full_graph_noharm` | Full graph plus no-harm arbitration. This is the recommended main MEMCF setting if it beats A4. |
| `A6_random_memory` | Random memory control. If this is close to full graph, memory content is not well targeted. |
| `A7_shuffled_memory` | Shuffled-memory control with real memory text but mismatched to users/candidates. Tests whether retrieval path matters. |
| `A8_profile_only` | Initializes/loads user profiles but disables graph-memory retrieval. Tests whether profile/context alone explains gains. |

## Main-Paper Ablation Set

Use this compact set when comparing against MemRec-style coarse module ablations:

| Variant | Role |
| --- | --- |
| `A0_no_memory` | Vanilla LLM reranker over user history and candidates. |
| `A8_profile_only` | User profile context without graph failure memories. |
| `A1_same_user_only` | Personalized failure memory only. |
| `A4_full_graph` | Main MEMCF graph retrieval. |
| `A5_full_graph_noharm` | Main graph retrieval with no-harm arbitration. |
| `A7_shuffled_memory` | Wrong-context real-memory control. |

Keep these for appendix/stress tests:

| Variant | Role |
| --- | --- |
| `A2_candidate_item_only` | Candidate-item path contribution. |
| `A3_neighbor_user_only` | Cross-user/neighbor path contribution. |
| `A6_random_memory` | Any-text random memory control. |

Backward-compatible variant names are still accepted:

| Legacy name | New equivalent |
| --- | --- |
| `A1_safe_graph_no_noharm` | `A4_full_graph` |
| `A2_safe_graph_noharm` | `A5_full_graph_noharm` |
| `A3_safe_graph_k5_noharm` | `A5_full_graph_noharm` with `graph_memory_k=5` |

## Standard 100-User Settings

```text
number_of_users = 100
max_positive_interactions = 5
max_negative_candidates = 19
max_iterations = 1
candidate_negative_mode = candidate_hard
graph_memory_k = 3
neighbor_k = 10
min_evidence_terms = 1
max_memory_facts = 3
max_memory_fact_words = 55
memory_token_budget = 420
ranking_prompt_style = compact_score
```

## Run One Dataset

```bash
bash scripts/run_ablation_one_dataset.sh Video_Game A0_no_memory
bash scripts/run_ablation_one_dataset.sh Video_Game A1_same_user_only
bash scripts/run_ablation_one_dataset.sh Video_Game A2_candidate_item_only
bash scripts/run_ablation_one_dataset.sh Video_Game A3_neighbor_user_only
bash scripts/run_ablation_one_dataset.sh Video_Game A4_full_graph
bash scripts/run_ablation_one_dataset.sh Video_Game A5_full_graph_noharm
bash scripts/run_ablation_one_dataset.sh Video_Game A6_random_memory
bash scripts/run_ablation_one_dataset.sh Video_Game A7_shuffled_memory
```

## Run 4 Datasets x 4 Variants

`run_16_ablation_jobs.sh` keeps the old 16-job workflow but now defaults to a compact paper set:

```bash
bash scripts/run_16_ablation_jobs.sh
```

Default datasets:

```text
Video_Game Digital_Music_1000u All_Beauty_1000u CDs_and_Vinyl_1000u
```

Default variants:

```text
A0_no_memory A1_same_user_only A4_full_graph A5_full_graph_noharm
```

Override datasets or variants:

```bash
DATASETS="Video_Game Digital_Music_1000u" \
VARIANTS="A0_no_memory A1_same_user_only A2_candidate_item_only A3_neighbor_user_only A4_full_graph A5_full_graph_noharm A6_random_memory A7_shuffled_memory" \
bash scripts/run_ablation_jobs.sh
```

## Train Once, Evaluate Main-Paper Ablations

The main memory variants share the same offline memory construction. To avoid
training the same failure graph repeatedly, train one memory file per dataset
and reuse it for all evaluation ablations.

Train shared memory files:

```bash
N_USERS=100 \
MEMCF_EVAL_ROOT=/path/to/eval_root \
MEMCF_MEMORY_ROOT=/path/to/memory_root \
bash scripts/run_mainpaper_train_memory_jobs.sh
```

Evaluate the compact main-paper ablation set from the saved memory:

```bash
N_USERS=100 \
MEMCF_EVAL_ROOT=/path/to/eval_root \
MEMCF_MEMORY_ROOT=/path/to/memory_root \
bash scripts/run_mainpaper_eval_jobs.sh
```

The default evaluated variants are:

```text
A0_no_memory A8_profile_only A1_same_user_only A4_full_graph A5_full_graph_noharm A7_shuffled_memory
```

If the memory file already exists, `run_mainpaper_train_memory_jobs.sh` skips
training by default. Set `FORCE_RETRAIN=1` to rebuild it.

## 1K-User Sharded Workflow

Train one shard:

```bash
N_USERS=1000 NUM_USER_SHARDS=10 USER_SHARD_ID=0 PHASE=train_only \
bash scripts/run_ablation_one_dataset.sh Video_Game A4_full_graph
```

After all shards finish, merge memories:

```bash
python scripts/merge_memory_shards.py \
  --output /tmp/video_game_memcf_merged.memory.json \
  /path/to/shard*/memcf_graph_*.memory.json
```

Evaluate a shard using the merged memory:

```bash
N_USERS=1000 NUM_USER_SHARDS=10 USER_SHARD_ID=0 PHASE=eval_only \
MEMORY_FILE=/tmp/video_game_memcf_merged.memory.json \
bash scripts/run_ablation_one_dataset.sh Video_Game A5_full_graph_noharm
```

This keeps training independent across shards. It is safe for current MEMCF because memory evolution is not a sequential dependency in the main protocol.

## Reporting Checklist

For each paper table, report:

- users requested and users evaluated
- candidate count
- Recall@5, Recall@10, Recall@20
- NDCG@5, NDCG@10, NDCG@20
- graph retrieval scope
- no-harm enabled or disabled
- memory pool size
- retrieved memories/user and kept memory facts/user
- LLM calls/user, total calls, tokens/user, total tokens
- runtime/user and total runtime
- invalid JSON/ranking fallback rate
- random-memory and shuffled-memory controls
- paired memory-vs-no-memory significance when comparing ranking JSON files

Aggregate summaries:

```bash
python scripts/aggregate_paper_tables.py \
  --root "$MEMCF_EVAL_ROOT" \
  --csv "$MEMCF_EVAL_ROOT/paper_table.csv" \
  --markdown "$MEMCF_EVAL_ROOT/paper_table.md"
```
