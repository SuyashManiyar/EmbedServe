# EmbedServe

A two-tier, quantized embedding store for serving a DLRM-style recommender.

In a DLRM (Deep Learning Recommendation Model), the MLPs are small and the embedding tables are
huge, so serving is limited by memory, not compute. Access is highly skewed: a small share of IDs
gets most lookups. So a small in-memory hot tier can serve most requests while the long tail lives
on disk.

**Question:** how much memory and latency do a two-tier store and INT8 embeddings save, and how
much accuracy (AUC) does each cost?

AUC is only a correctness check. Results are systems numbers: hit rate, p99 latency, memory.

## Status

| # | Phase | Status |
|---|---|---|
| 0 | Core store: LRU hot tier, RocksDB cold tier, `TierManager`, INT8 quantizer | ✅ Done |
| 1 | Data: Criteo subset → encoded arrays + access-skew stats | ✅ Done |
| 2 | Model: train a DLRM once on CPU, save weights and FP32 tables | ✅ Done (AUC 0.7996) |
| 3 | Evaluation harness: serve the fixed model from FP32 / INT8 / two-tier store; measure AUC, memory, hit rate, p50/p99 latency | ⏳ Next |
| 4 | Experiments + plots (see [Experiments](#experiments)) | Not started |
| 5 | Write-up for the Nov 18 milestone | Not started |

## Getting started

### 1. Set up the environment

Python 3.10+ (developed on 3.14, macOS, CPU only).

```bash
git clone https://github.com/SuyashManiyar/EmbedServe.git && cd EmbedServe
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,model]"
pytest
```

`dev` adds pytest; `model` adds PyTorch and scikit-learn (only needed for training and evaluation).

### 2. Get the data

Download the two CSV files from the shared Google Drive folder
**[Embedserve_Data](https://drive.google.com/drive/folders/1GHk95OoNSIRCiF8uMW0TlA80NW3CiPhf?usp=sharing)**
and put them in `data/` at the repo root:

| File | Rows | Size | Use |
|---|---|---|---|
| `data/train_5M.csv` | 5,000,000 | 1.2 GB | training, vocabulary, skew stats |
| `data/valid_1M.csv` | 1,000,000 | 237 MB | AUC / held-out evaluation |

Steps (no Google sign-in needed):

1. Open the folder link above and click **Download all** (top right). Drive packs both files into
   one zip; for files this large it may say it can't scan them for viruses. Click
   **Download anyway**.
2. Unzip it and move the two CSVs into `data/`, keeping the file names exactly as they are:

   ```bash
   mkdir -p data
   unzip ~/Downloads/Embedserve_Data-*.zip -d data/
   ```

   If the zip unpacks into a subfolder, move the files up so they sit directly in `data/`.
   (Downloading each file on its own also works; then just move both into `data/`.)
3. Check that both files are complete (each count includes the header line):

   ```bash
   wc -l data/train_5M.csv data/valid_1M.csv
   ```

   Expected: `5000001 data/train_5M.csv` and `1000001 data/valid_1M.csv`.

`data/` is gitignored. **Never commit data or model files.**

<details>
<summary>Where these files come from (to rebuild them without Drive)</summary>

They are the first rows of the train and validation splits of
[reczoo/Criteo_x1](https://huggingface.co/datasets/reczoo/Criteo_x1), a mirror of the
[Kaggle Criteo Display Advertising Challenge](https://www.kaggle.com/c/criteo-display-ad-challenge)
data (45.8M rows, already shuffled and split 7:2:1). Needs ~3 GB of temporary disk:

```bash
mkdir -p data && curl -L -C - -o data/Criteo_x1.zip https://huggingface.co/datasets/reczoo/Criteo_x1/resolve/main/Criteo_x1.zip
unzip -p data/Criteo_x1.zip train.csv | head -n 5000001 > data/train_5M.csv
unzip -p data/Criteo_x1.zip valid.csv | head -n 1000001 > data/valid_1M.csv
rm data/Criteo_x1.zip
```
</details>

**Data format.** CSV with header `label,I1..I13,C1..C26`. This mirror is already preprocessed:
the 13 dense features `I*` are min-max scaled to [0, 1], and the 26 categorical features `C*` are
integer IDs in one ID space shared across fields. Click rate is about 25%.

### 3. Encode the data (~40 s)

```bash
python scripts/prepare_data.py --train data/train_5M.csv --valid data/valid_1M.csv --min-count 1 --dim 64
```

Writes `data/processed/{train,valid,vocab}.npz`. Each of the 26 fields gets its own table with
rows numbered 1..V; row 0 is shared by rare/unseen IDs (`--min-count 1` keeps every ID seen in
training). It also prints how skewed the lookups are.

### 4. Train the model (~5 min on an M1)

```bash
python scripts/train.py
```

Writes `data/model/dlrm.pt` (weights), `data/model/embeddings.npz` (one FP32 array per table,
the input to the serving stores) and `data/model/metrics.json`. For a quick check, add
`--train-rows 200000 --valid-rows 100000` (~10 s, AUC ≈ 0.77).

Everyone should get the same encoded data. Training is seeded (`--seed 0`) but floating-point
results can differ slightly across machines, so for experiments we compare runs from **one**
trained model.

## Results so far

All on the 5M-row train / 1M-row validation subset.

**Tables:** 1,257,358 embedding rows over 26 tables (largest ~224K rows, smallest 4).
At dim 64 that is **307 MB in FP32** (~77 MB in INT8).

**Access skew** (train lookups; all tables treated as one row space, like the shared hot tier).
This is the hit rate of an ideal hot tier holding the most frequent rows, so an upper bound for LRU:

| Hottest share of rows | Rows | Share of lookups |
|---|---|---|
| 0.1% | 1,257 | 72.3% |
| 1% | 12,574 | 88.4% |
| 5% | 62,868 | 95.4% |
| 10% | 125,736 | 97.1% |

**Baseline model** (FP32 tables fully in RAM): validation **AUC 0.7996**, log loss 0.4473.
One epoch, batch 2048, Adam / SparseAdam lr 1e-3; 318 s on an M1 (4 threads), peak RAM ~1.7 GB.

## Design

```
batch of IDs ─► TierManager
                 ├─ hot tier: in-memory LRU, small
                 └─ cold tier: RocksDB on disk, one batched multi-get per lookup for all misses
```

**Model:** a standard DLRM. Bottom MLP 13→512→256→64 on the dense features; 26 embedding tables
of dim 64; pairwise dot products of all 27 vectors (351) plus the bottom output feed the top MLP
415→512→256→1. `DLRM.forward_from_embeddings` takes already-fetched embedding rows, so the same
trained MLPs can be served from any store (FP32 in RAM, INT8, or the two-tier store).

## Scope

- Train a small DLRM once on a Criteo subset (CPU), then keep it fixed
- Two-tier embedding store: in-memory LRU hot tier + RocksDB cold tier
- INT8 quantization of embedding rows with a per-row scale

Course topics covered: data locality and memory, key-value stores, quantization.

## Experiments

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
tests/          pytest suite (runs on synthetic data, no download needed)
data/           gitignored: CSVs, processed arrays, trained model
```

## Contributing

- Work on a branch (`git checkout -b <your-name>/<topic>`) and open a pull request into `main`;
  don't push to `main` directly.
- Run `pytest` before opening a PR, and add tests for new code. Tests use synthetic data, so they
  don't need the dataset.
- Keep data, models and results out of git (`data/` and `experiments/results/` are ignored).
- Scripts take paths and sizes as flags; put new entry points in `scripts/` and reusable code in
  `embedserve/`.
- When you report a number, say which data subset, model and settings produced it.

## Timeline

- **Oct 28:** proposal due
- **Nov 18:** milestone: two-tier store and INT8 measured against the baseline
- After that: write-up, then future work if time allows
