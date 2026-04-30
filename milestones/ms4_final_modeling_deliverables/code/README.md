# MS4 Code Workspace

Use this folder for the final main notebook and supporting scripts.

Recommended main notebook name:

```text
cs1090b_ms4_main_group66.ipynb
```

Minimum sections:

1. Project metadata and dependency notes.
2. Data paths and loading.
3. Preprocessing and author-level split.
4. Baseline reproduction or loaded MS3 baseline reference.
5. Final model implementation.
6. Training and validation.
7. Author-level aggregation and threshold tuning.
8. Test evaluation.
9. Ablations.
10. Interpretation and reproducibility notes.

Use KaggleHub for the Reddit dataset:

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
