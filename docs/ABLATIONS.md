# MEMCF Ablation Protocol

The default ablation set isolates the contribution of graph memory.

## Variants

| Variant | Command behavior |
| --- | --- |
| `A0_no_memory` | Evaluation only. No memory training and no memory facts in ranking. |
| `A1_safe_graph_no_noharm` | Train graph memory and inject retrieved facts into ranking. |
| `A2_safe_graph_noharm` | Same as A1, but choose no-memory ranking if memory facts are not graph-supported. |
| `A3_safe_graph_k5_noharm` | Same as A2, but retrieve up to 5 memory facts. |

## Standard Settings

```text
number_of_users = 100
max_positive_interactions = 5
max_negative_candidates = 19
max_iterations = 1
graph_memory_k = 3 or 5
neighbor_k = 10
min_evidence_terms = 1
```

## Run One Dataset

```bash
bash scripts/run_ablation_one_dataset.sh Video_Game A0_no_memory
bash scripts/run_ablation_one_dataset.sh Video_Game A1_safe_graph_no_noharm
bash scripts/run_ablation_one_dataset.sh Video_Game A2_safe_graph_noharm
bash scripts/run_ablation_one_dataset.sh Video_Game A3_safe_graph_k5_noharm
```

## Run Four Datasets In Parallel

```bash
bash scripts/run_16_ablation_jobs.sh
```

## Reporting Checklist

For each table, report:

- users requested and users evaluated
- candidate count
- Recall@5, Recall@10, Recall@20
- NDCG@5, NDCG@10, NDCG@20
- trace directory
- local model name
- whether no-harm arbitration was enabled
- memory counts and retrieval diagnostics
