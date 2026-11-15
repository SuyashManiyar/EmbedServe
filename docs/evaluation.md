# Evaluation

The evaluation harness serves one saved DLRM checkpoint through FP32 arrays,
INT8 arrays, or an FP32 LRU cache backed by RocksDB. Each run uses the same MLP
weights and validation rows. The checkpoint contains the embedding weights,
so `embeddings.npz` is not required.

The environment requires the `dev` and `model` extras from `pyproject.toml`.
The default inputs are `data/model/dlrm.pt` and `data/processed/valid.npz`.

## Commands

Evaluate all validation rows with each store:

```bash
python scripts/evaluate.py
```

Evaluate the first 10,000 rows with two cache capacities:

```bash
python scripts/evaluate.py --rows 10000 --hot-capacities 0 12574 \
  --out experiments/results/evaluation-smoke.json
```

Compare cache capacities with a fixed batch size:

```bash
python scripts/evaluate.py --modes two-tier \
  --hot-capacities 0 1257 12574 62868 125736 \
  --batch-size 2048 --threads 4 \
  --out experiments/results/cache-capacities.json
```

`--model` and `--data` select the checkpoint and encoded evaluation file.
`--rows` selects a prefix in file order. If the requested count exceeds the
file length, the run uses all available rows and reports the actual count.
Evaluation requires nonempty data with both binary label classes.

## Stores

| Mode | Resident embedding payload | Lookup output |
|---|---|---|
| `fp32` | FP32 table arrays | FP32 rows |
| `int8` | INT8 arrays and one FP32 scale per row | Requested rows dequantized to FP32 |
| `two-tier` | FP32 rows currently in the LRU cache | FP32 rows from the cache or RocksDB |

For dimension D and N embedding rows, FP32 arrays contain `4 * N * D` bytes.
INT8 arrays and scales contain `N * (D + 4)` bytes. At dimension 64, the INT8
payload is 26.5625% of the FP32 payload.

The inference model retains its MLP weights without an internal copy of the
embedding tables. INT8 and two-tier runs release their source FP32 arrays
before measurement. Conversion and database population occur before timing.
Each two-tier run has a separate database under `--db-root`, which defaults to
`data/evaluation-db`. The database closes and its temporary directory is
removed when the run exits, including on evaluation errors.

## Hot-tier backends

`--hot-backends` selects one or both two-tier hot tier implementations. Each gets its own
run at every capacity:

- **`python`** (the default): an `OrderedDict` LRU, stepping through each ID in Python.
- **`cpp`**: the compiled `embedserve._hot` extension, which runs the per-ID pass of the
  whole batch in C++.

Both backends share the cold-tier fetch and the step that fills fetched rows into the
output. They give identical rows, hit and miss counts and LRU contents (checked in
`tests/test_hot_cpp.py`), so only latency differs. Each run reports its backend as
`hot_backend`.

## Repeats

`--repeats N` runs every configuration N times and records each as a separate run with a
`repeat` index. The order is interleaved: all configurations run once, then all run again.
`scripts/plot_results.py` merges repeats, reporting the median of each latency statistic
and its range.

## Row formats

`--row-formats` selects what the two-tier store keeps in both tiers: `fp32` rows (4 × dim
bytes) or `int8` rows (dim INT8 values plus one FP32 scale). INT8 rows are dequantized on
read and give exactly the output of `INT8Store`. Each run reports `row_format`.

## Exported stores

`scripts/export_store.py` writes a serving directory, `data/store/` by default. With
`--store`, `evaluate.py` opens stores from it instead of loading the checkpoint, and two-tier
runs open the exported RocksDB database read-only instead of building one. An opened database
is reused across runs, so the OS file cache may already hold it.

## Process memory

`scripts/measure_memory.py` serves each store in a fresh process that loads only the MLP
weights and that store, then records two values:

- **Peak footprint**: memory the process owns and the OS cannot drop. On macOS this is the
  kernel's lifetime peak physical footprint; elsewhere it is the peak unique set size.
- **Peak RSS**: resident memory, which also counts file-backed pages such as SST files read
  through mmap.

A run with no store gives the baseline for Python, PyTorch, the MLPs and the input data.
`--no-mmap` adds two-tier runs that read through RocksDB's block cache instead of mmap.

## Request serving

`scripts/benchmark_serving.py` measures the dynamic batcher (`embedserve/batching.py`). One
generator thread submits single requests at a target rate with Poisson arrivals and never
waits for results. Latency runs from submit to result, so it includes time queued.

- **Achieved rate** is completed requests divided by the time from the first send to the
  last result. A point is marked saturated when it is below 95% of the offered rate.
- **Mean batch** is requests served divided by batches dispatched.
- **Send lag** is how late the generator sent requests against its schedule. Large values
  mean the generator, not only the server, was overloaded.

`--max-batch-sizes 1` serves every request alone. The generator shares the server's process
and GIL, so the highest achieved rate is a lower bound on server capacity.

## Measurements

AUC and log loss cover every selected row once, including the final partial
batch. FP32 and two-tier predictions should agree for the same checkpoint and
batch settings. INT8 predictions include per-row quantization error.

`lookup_batch_latency` measures the complete lookup across all fields. It
includes ID validation, row assembly and INT8 dequantization where applicable.
The two-tier store calls `TierManager.lookup_fields` once per batch. IDs are
visited row by row, and the misses of every field are fetched with one cold-tier
multi-get, then promoted into the LRU after the whole batch has been looked up.

`inference_batch_latency` includes lookup, conversion of retrieved rows to a
tensor, the DLRM forward pass and sigmoid. It excludes input file loading,
store construction, metric calculation and JSON serialization. Both latency
objects report p50, p99, mean milliseconds and total seconds. These are batch
latencies. The final partial batch contributes one sample. The output includes
the number of batches and the size of the last batch, since both affect the
interpretation of the percentiles.

`embedding_payload_bytes` is the final resident embedding array payload.
`peak_embedding_payload_bytes` is the largest payload observed before or after
a measured batch. These values exclude Python objects, allocator overhead,
MLP weights, input arrays, temporary batch buffers, RocksDB caches and the OS
page cache. They are not process RSS measurements. A zero-capacity LRU reports
zero embedding payload while RocksDB still consumes memory.

`hits` and `misses` follow the existing `TierManager` definitions. Repeated
missing IDs within one field batch count as hits after the first occurrence.
The hit rate therefore includes batch deduplication and can be nonzero at zero
cache capacity. FP32 and INT8 modes report null cache statistics.

## Warmup and cache state

The default `--warmup-batches 0` starts measurement with an empty LRU cache.
RocksDB and OS caches can already contain data from database population. This
is not a cold-disk benchmark.

A positive `--warmup-batches` replays that many prefix batches through lookup
and inference before measurement. The cache contents remain, while the hit
counters reset. The measured pass then starts at the first validation row.
Warmup does not contribute to accuracy or latency samples. The count cannot
exceed the number of evaluation batches.

Each cache capacity starts a separate run with a fresh LRU. Comparisons require
the same row selection, batch size, warmup count, thread count and checkpoint.

## Output

The default output is `experiments/results/evaluation.json`. The report contains
SHA-256 hashes and paths for the checkpoint and encoded dataset, the source row
count, Python and library versions, platform details, thread count and per-run
settings and metrics. `table_sizes` and `embedding_dim` identify the embedding
layout. `schema_version` identifies the JSON format.

The dataset hash covers the complete encoded file even when `--rows` selects a
prefix. Each run records its actual evaluated row count. Training settings and
training subset size are available in the training script's `metrics.json`
next to the checkpoint. Reports and databases belong in the ignored
`experiments/results/` and `data/` directories.
