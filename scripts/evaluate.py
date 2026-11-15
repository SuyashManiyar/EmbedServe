"""Evaluate FP32, INT8 and two-tier stores against one saved DLRM checkpoint.

By default each run loads the checkpoint and two-tier runs build a temporary
RocksDB database. With --store, runs open a directory written by
scripts/export_store.py instead, which skips the rebuild.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import torch

from embedserve.data import CriteoArrays
from embedserve.evaluation import evaluate, load_checkpoint
from embedserve.export import load_mlp, open_store
from embedserve.serving import FP32Store, INT8Store, TwoTierStore


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run(args, data: CriteoArrays, mode: str, capacity: int | None, backend: str | None,
        row_format: str | None) -> dict:
    if args.store:
        model = load_mlp(args.store)
        store = open_store(args.store, mode, capacity or 0, backend or "python", row_format or "fp32")
    else:
        model, tables = load_checkpoint(args.model)
        if mode == "fp32":
            store = FP32Store(tables)
        elif mode == "int8":
            store = INT8Store(tables)
        else:
            store = TwoTierStore(tables, capacity, args.db_root, backend, row_format)
        del tables
    with store:
        return {
            "mode": mode,
            "table_sizes": store.table_sizes,
            "embedding_dim": store.dim,
            **evaluate(model, store, data, args.batch_size, args.warmup_batches),
        }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=Path("data/model/dlrm.pt"))
    p.add_argument("--store", type=Path,
                   help="exported serving directory to open instead of --model")
    p.add_argument("--data", type=Path, default=Path("data/processed/valid.npz"))
    p.add_argument("--rows", type=int, help="number of validation rows from the start of the file")
    p.add_argument("--modes", nargs="+", choices=["fp32", "int8", "two-tier"],
                   default=["fp32", "int8", "two-tier"])
    p.add_argument("--hot-capacities", nargs="+", type=int, default=[12574],
                   help="LRU row capacities for separate two-tier runs")
    p.add_argument("--hot-backends", nargs="+", choices=["python", "cpp"], default=["python"],
                   help="hot-tier implementations for separate two-tier runs")
    p.add_argument("--row-formats", nargs="+", choices=["fp32", "int8"], default=["fp32"],
                   help="row formats stored in the tiers for separate two-tier runs")
    p.add_argument("--repeats", type=int, default=1,
                   help="times to repeat every configuration; repeats are interleaved")
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--warmup-batches", type=int, default=0,
                   help="prefix batches replayed before measurement")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--db-root", type=Path, default=Path("data/evaluation-db"))
    p.add_argument("--out", type=Path, default=Path("experiments/results/evaluation.json"))
    args = p.parse_args()
    if args.rows is not None and args.rows <= 0:
        p.error("--rows must be positive")
    if args.batch_size <= 0 or args.threads <= 0 or args.warmup_batches < 0 or args.repeats <= 0:
        p.error("batch size, threads and repeats must be positive and warmup must be nonnegative")
    if any(capacity < 0 for capacity in args.hot_capacities):
        p.error("hot capacities must be nonnegative")
    if args.store:
        args.model = args.store / "model.pt"
    if args.out.resolve() in (args.model.resolve(), args.data.resolve()):
        p.error("output must not overwrite the checkpoint or evaluation data")
    torch.set_num_threads(args.threads)
    data = CriteoArrays.load(args.data)
    source_rows = len(data)
    if args.rows is not None:
        data = CriteoArrays(data.labels[:args.rows], data.dense[:args.rows], data.sparse[:args.rows])
    report = {
        "schema_version": 1,
        "model": {"path": str(args.model), "sha256": sha256(args.model)},
        "data": {"path": str(args.data), "sha256": sha256(args.data),
                 "source_rows": source_rows, "selection": "prefix in file order"},
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "numpy": np.__version__, "torch": torch.__version__,
                        "threads": torch.get_num_threads()},
        "memory_scope": "embedding array payload only, excluding model, batch buffers, Python overhead, RocksDB caches and OS page cache",
        "latency_scope": "one batch across all fields, including the final partial batch",
        "hit_rate_scope": "TierManager hits include duplicate IDs within each field batch",
        "runs": [],
    }
    configs = []
    for mode in args.modes:
        if mode == "two-tier":
            configs += [(mode, c, b, f) for c in args.hot_capacities
                        for b in args.hot_backends for f in args.row_formats]
        else:
            configs.append((mode, None, None, None))
    for repeat in range(args.repeats):
        for mode, capacity, backend, row_format in configs:
            result = {"repeat": repeat, **run(args, data, mode, capacity, backend, row_format)}
            report["runs"].append(result)
            print(f"{mode} capacity={capacity} backend={backend} format={row_format} repeat={repeat} "
                  f"rows={result['rows']:,} AUC={result['auc']:.6f} "
                  f"lookup p50={result['lookup_batch_latency']['p50_ms']:.2f} ms "
                  f"payload={result['embedding_payload_bytes']:,} bytes", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"results: {args.out}")


if __name__ == "__main__":
    main()
