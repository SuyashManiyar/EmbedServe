"""Load and encode the Criteo dataset into compact numeric arrays.

Input is the reczoo/Criteo_x1 mirror of the Kaggle Criteo data: a CSV with header
`label,I1..I13,C1..C26`. That mirror is already preprocessed (AFN split):
- I1..I13 are min-max scaled to [0, 1], so they are used as-is.
- C1..C26 are integer IDs in one ID space shared across fields.

Encoding: a per-field vocabulary keeps the IDs seen at least `min_count` times in
the training rows and renumbers them 1..V. Row 0 of each table is a shared row
for rare and unseen IDs. This keeps tables small and gives every table its own
0-based row numbering, which is what the embedding tables and TierManager expect.

Memory: rows are read in chunks into preallocated arrays; 5M rows take about
800 MB (labels, dense features and int32 IDs).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

N_DENSE = 13
N_SPARSE = 26
LABEL_COL = "label"
DENSE_COLS = [f"I{i}" for i in range(1, N_DENSE + 1)]
SPARSE_COLS = [f"C{i}" for i in range(1, N_SPARSE + 1)]
COLUMNS = [LABEL_COL, *DENSE_COLS, *SPARSE_COLS]


@dataclass
class CriteoArrays:
    labels: np.ndarray  # (n,) float32
    dense: np.ndarray  # (n, 13) float32
    sparse: np.ndarray  # (n, 26) int32; raw IDs from read_csv, row IDs after encode

    def __len__(self) -> int:
        return len(self.labels)

    def save(self, path: str | Path) -> None:
        np.savez(path, labels=self.labels, dense=self.dense, sparse=self.sparse)

    @classmethod
    def load(cls, path: str | Path) -> CriteoArrays:
        with np.load(path) as f:
            return cls(f["labels"], f["dense"], f["sparse"])


@dataclass
class Vocab:
    keys: list[np.ndarray]  # per field: sorted raw IDs kept; row ID = index + 1
    min_count: int

    @property
    def sizes(self) -> list[int]:
        """Rows per embedding table, including row 0 for rare/unseen IDs."""
        return [len(k) + 1 for k in self.keys]

    def save(self, path: str | Path) -> None:
        arrays = {f"keys_{j}": k for j, k in enumerate(self.keys)}
        np.savez(path, min_count=self.min_count, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> Vocab:
        with np.load(path) as f:
            n = sum(1 for name in f.files if name.startswith("keys_"))
            return cls([f[f"keys_{j}"] for j in range(n)], int(f["min_count"]))


def _count_rows(path: Path) -> int:
    """Data rows in a CSV with one header line."""
    with open(path, "rb") as f:
        n = sum(buf.count(b"\n") for buf in iter(lambda: f.read(1 << 24), b""))
        f.seek(-1, 2)
        if f.read(1) != b"\n":
            n += 1  # last line without a trailing newline
    return n - 1


def read_csv(path: str | Path, nrows: int | None = None, chunksize: int = 500_000) -> CriteoArrays:
    """Read the first `nrows` rows (all if None). `sparse` holds the raw IDs."""
    path = Path(path)
    with open(path) as f:
        header = f.readline().strip().split(",")
    if header != COLUMNS:
        raise ValueError(f"{path}: expected header {','.join(COLUMNS)}")
    total = _count_rows(path)
    n = total if nrows is None else min(nrows, total)

    out = CriteoArrays(
        labels=np.empty(n, dtype=np.float32),
        dense=np.empty((n, N_DENSE), dtype=np.float32),
        sparse=np.empty((n, N_SPARSE), dtype=np.int32),
    )
    dtypes = {LABEL_COL: np.float32, **{c: np.float32 for c in DENSE_COLS}, **{c: np.int32 for c in SPARSE_COLS}}
    start = 0
    for chunk in pd.read_csv(path, nrows=n, chunksize=chunksize, dtype=dtypes, engine="c"):
        end = start + len(chunk)
        out.labels[start:end] = chunk[LABEL_COL].to_numpy()
        out.dense[start:end] = chunk[DENSE_COLS].to_numpy()
        out.sparse[start:end] = chunk[SPARSE_COLS].to_numpy()
        start = end
    if start != n:
        raise ValueError(f"{path}: expected {n} rows, read {start}")
    return out


def build_vocab(raw_ids: np.ndarray, min_count: int = 4) -> Vocab:
    keys = []
    for j in range(raw_ids.shape[1]):
        uniq, counts = np.unique(raw_ids[:, j], return_counts=True)
        keys.append(uniq[counts >= min_count])
    return Vocab(keys, min_count)


def encode(vocab: Vocab, raw_ids: np.ndarray) -> np.ndarray:
    """Map raw IDs to per-table row IDs; IDs not in the vocabulary map to 0."""
    out = np.zeros(raw_ids.shape, dtype=np.int32)
    for j, keys in enumerate(vocab.keys):
        if len(keys) == 0:
            continue
        col = raw_ids[:, j]
        pos = np.minimum(np.searchsorted(keys, col), len(keys) - 1)
        out[:, j] = np.where(keys[pos] == col, pos + 1, 0)
    return out


def apply_vocab(raw: CriteoArrays, vocab: Vocab) -> CriteoArrays:
    return CriteoArrays(raw.labels, raw.dense, encode(vocab, raw.sparse))


def access_skew(sparse: np.ndarray, sizes: list[int], fractions=(0.001, 0.01, 0.05, 0.1, 0.2)) -> dict:
    """How concentrated lookups are, which is what a hot tier relies on.

    Treats all tables as one row space, like the shared TierManager hot tier. For
    each fraction f, `coverage[f]` is the share of all lookups that go to the most
    accessed f of embedding rows. That is the hit rate of an ideal hot tier holding
    that many rows (it always holds the most frequent rows): an upper bound for LRU.
    """
    counts = np.concatenate([np.bincount(sparse[:, j], minlength=s) for j, s in enumerate(sizes)])
    cum = np.cumsum(np.sort(counts)[::-1], dtype=np.int64)
    total = int(cum[-1])
    coverage = {f: float(cum[max(1, round(f * len(counts))) - 1] / total) for f in fractions}
    return {
        "total_rows": len(counts),
        "lookups": total,
        "rows_never_accessed": int((counts == 0).sum()),
        "rare_bucket_share": float((sparse == 0).mean()),
        "coverage": coverage,
    }
