# EmbedServe

A two-tier, quantized embedding store for serving a DLRM-style recommender.

In a DLRM (Deep Learning Recommendation Model), the MLPs are small and the embedding tables are
huge, so serving is limited by memory, not compute. Access is highly skewed: a small share of IDs
gets most lookups. So a small in-memory hot tier can serve most requests while the long tail lives
on disk.

**Question:** how much memory and latency do a two-tier store and INT8 embeddings save, and how
much accuracy (AUC) does each cost?

AUC is only a correctness check. Results are systems numbers: hit rate, p99 latency, memory.

## Design

```
batch of IDs ─► TierManager
                 ├─ hot tier: in-memory LRU, small
                 └─ cold tier: RocksDB on disk, one batched multi-get per lookup for all misses
```

## Scope

- Train a small DLRM once on a Criteo subset (CPU), then keep it fixed
- Two-tier embedding store: in-memory LRU hot tier + RocksDB cold tier
- INT8 quantization of embedding rows with a per-row scale

Course topics covered: data locality and memory, key-value stores, quantization.

## Experiments

Dataset: [Criteo Display Advertising Challenge](https://www.kaggle.com/c/criteo-display-ad-challenge),
starting on a 1–5M row subset.

1. **Baseline:** FP32 tables fully in RAM
2. **INT8 tables:** memory saved vs AUC change
3. **Two tiers:** hit rate and p50/p99 lookup latency across hot-tier sizes

Key plots: hit rate vs hot-tier size, memory vs AUC (FP32 vs INT8).

## Future work

Moved out of the current scope; we attempt these only after the core is done and measured.

- **Dynamic batching** in a serving API (model serving)
- **C++ hot tier** with pybind11, and a Python-vs-C++ comparison
- **Eviction policy comparison:** LRU vs frequency-based
- **Store INT8 rows in the tiers** directly instead of dequantizing offline
- GPU memory as a third tier; INT4 / mixed precision; FBGEMM / TorchRec comparison; Triton kernel; pruning cold rows

## Layout

```
embedserve/
  tiers.py      Hot tier, cold tiers (in-memory and RocksDB), TierManager
  quantize.py   FP32 → INT8 quantizer with a per-row scale
  data.py       Criteo CSV loader, per-table vocabulary, access-skew stats
  model.py      Small DLRM; can run from embeddings fetched by any store
scripts/
  prepare_data.py   Encode Criteo subsets into data/processed/*.npz
  train.py          Train the DLRM once; saves weights + FP32 tables to data/model/
tests/          pytest suite
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

For model training, also install the `model` extra: `pip install -e ".[dev,model]"`.

## Timeline

- **Oct 28:** proposal due
- **Nov 18:** milestone: two-tier store and INT8 measured against the baseline
- After that: write-up, then future work if time allows
