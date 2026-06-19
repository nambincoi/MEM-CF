# MEMCF Data Format

MEMCF expects preprocessed runtime datasets under:

```text
$MEMCF_DATA_ROOT/<dataset_name>/
```

Required files:

```text
items.json
user_sequences_10.json
user_negatives_10.json
```

## `items.json`

A JSON object keyed by item id.

```json
{
  "123": {
    "title": "Example item title",
    "description": "Short item description",
    "category": "Example category"
  }
}
```

Accepted metadata fields include `title`, `description`, `category`, `brand`, and `feature`. Missing fields are replaced by safe defaults during prompt construction.

## `user_sequences_10.json`

A JSON object keyed by user id. Each value is a user sequence with train/validation/test positives.

Common accepted shapes:

```json
{
  "USER_A": {
    "train": ["10", "11", "12"],
    "valid": ["13"],
    "test": ["14"]
  }
}
```

or list-style histories that the loader can normalize.

## `user_negatives_10.json`

A JSON object keyed by user id. Each value contains negative candidate ids for validation/test.

```json
{
  "USER_A": {
    "valid": ["30", "31", "32"],
    "test": ["40", "41", "42"]
  }
}
```

## Candidate Protocol

MEMCF evaluates each user with one ground-truth item plus sampled negatives. The standard setting used in the included scripts is:

```text
max_positive_interactions = 5
max_negative_candidates = 19
candidate count = 20
```

Keep the same candidate protocol across no-memory and memory variants.
