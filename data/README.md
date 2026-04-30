# Data README

This project uses two public text datasets and should load them through their public dataset APIs.

## Reddit MBTI Dataset

| Item | Value |
|---|---|
| Source | Kaggle `minhaozhang1/reddit-mbti-dataset` |
| Kaggle page | <https://www.kaggle.com/datasets/minhaozhang1/reddit-mbti-dataset/data> |
| File inside dataset | `reddit_post.csv` |
| Columns used | `author`, `body`, `mbti` |
| Access method | KaggleHub API |

Important distinction: KaggleHub is an API-backed download/cache workflow, not the same as pandas reading a remote `hf://` path. KaggleHub resolves the Kaggle dataset and returns a cached file path, then notebooks read that path with pandas.

### API Access Check

This lightweight API read worked on 2026-04-30:

```python
import kagglehub
from kagglehub import KaggleDatasetAdapter

df_head = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "minhaozhang1/reddit-mbti-dataset",
    "reddit_post.csv",
    pandas_kwargs={"nrows": 3},
)

print(df_head)
```

### Full Dataset Load

For full-dataset work, use KaggleHub to resolve the dataset path and pandas to read only the needed columns:

```python
from pathlib import Path

import kagglehub
import pandas as pd

reddit_path = Path(
    kagglehub.dataset_download(
        "minhaozhang1/reddit-mbti-dataset",
        path="reddit_post.csv",
    )
)

reddit_raw = pd.read_csv(reddit_path, usecols=["author", "body", "mbti"])
```

Project notebooks report the expected scale as 13,028,455 usable rows after removing 180 blank posts, with 11,773 unique authors.

## Hugging Face Emotion Dataset

| Item | Value |
|---|---|
| Source | Hugging Face `AdamCodd/emotion-balanced` |
| Public page | <https://huggingface.co/datasets/AdamCodd/emotion-balanced> |
| Files | `data/train.jsonl`, `data/validation.jsonl`, `data/test.jsonl` |
| Labels | sadness, joy, love, anger, fear, surprise |

### Option A: `datasets`

```python
from datasets import load_dataset

ds = load_dataset("AdamCodd/emotion-balanced")
train = ds["train"]
validation = ds["validation"]
test = ds["test"]
```

### Option B: pandas with `hf://`

```python
import pandas as pd

splits = {
    "train": "data/train.jsonl",
    "validation": "data/validation.jsonl",
    "test": "data/test.jsonl",
}

train_df = pd.read_json(
    "hf://datasets/AdamCodd/emotion-balanced/" + splits["train"],
    lines=True,
)
```

### Option C: `hf_hub_download`

```python
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

train_path = Path(
    hf_hub_download(
        repo_id="AdamCodd/emotion-balanced",
        repo_type="dataset",
        filename="data/train.jsonl",
    )
)

train_df = pd.read_json(train_path, lines=True)
```
