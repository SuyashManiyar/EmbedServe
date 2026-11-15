"""Measure the process memory of serving from each embedding store.

Each configuration runs in a fresh process that loads only the MLP weights and
the store being measured from an exported directory (scripts/export_store.py),
then serves a prefix of the validation data. The "none" configuration serves
zero embeddings with no store and gives the baseline for everything else in
the process: Python, PyTorch, the MLPs and the evaluation data.

Two numbers are reported per run:

- footprint: memory the process owns and the OS cannot drop (on macOS the
  kernel's peak physical footprint, which is what Activity Monitor shows; on
  other systems the peak unique set size). It excludes clean file-backed pages.
- rss: peak resident memory, which also counts file-backed pages such as SST
  files read through mmap. The OS can reclaim those pages under pressure.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def lifetime_peak_footprint() -> int | None:
    """Peak physical footprint of this process in bytes (macOS only)."""
    if sys.platform != "darwin":
        return None

    class RusageInfoV4(ctypes.Structure):
        # <sys/resource.h> rusage_info_v4: a 16-byte UUID, then 35 uint64 counters.
        _fields_ = [("uuid", ctypes.c_uint8 * 16), ("counters", ctypes.c_uint64 * 35)]

    info = RusageInfoV4()
    libproc = ctypes.CDLL(ctypes.util.find_library("proc"), use_errno=True)
    if libproc.proc_pid_rusage(os.getpid(), 4, ctypes.byref(info)) != 0:
        return None
    return int(info.counters[28])  # ri_lifetime_max_phys_footprint


def child(config: dict) -> dict:
    import psutil
    import torch

    from embedserve.data import CriteoArrays
    from embedserve.export import load_meta, load_mlp, open_store

    torch.set_num_threads(config["threads"])
    process = psutil.Process()

    def memory() -> dict:
        info = process.memory_full_info()
        return {"rss": info.rss, "uss": info.uss}

    root = Path(config["store"])
    data = CriteoArrays.load(config["data"])
    model = load_mlp(root)
    meta = load_meta(root)
    batch = config["batch_size"]
    store = None
    if config["mode"] != "none":
        store = open_store(root, config["mode"], config["hot_capacity"] or 0, config["hot_backend"],
                           config["row_format"] or "fp32", config["mmap_reads"])
    peak = memory()
    with torch.inference_mode():
        for n, start in enumerate(range(0, len(data), batch)):
            ids = data.sparse[start : start + batch]
            if store is None:
                rows = np.zeros((len(ids), len(meta["table_sizes"]), meta["dim"]), dtype=np.float32)
            else:
                rows = store.lookup(ids)
            dense = torch.from_numpy(data.dense[start : start + batch])
            torch.sigmoid(model.forward_from_embeddings(dense, torch.from_numpy(rows)))
            if n % 10 == 9:
                now = memory()
                peak = {k: max(peak[k], now[k]) for k in peak}
    final = memory()
    result = {
        **config,
        "rows": len(data),
        "peak_footprint_bytes": lifetime_peak_footprint() or max(peak["uss"], final["uss"]),
        "peak_rss_bytes": max(peak["rss"], final["rss"]),
        "embedding_payload_bytes": store.payload_bytes if store else 0,
        "hit_rate": store.stats()["hit_rate"] if store else None,
    }
    if store:
        store.close()
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--child", help=argparse.SUPPRESS)
    p.add_argument("--store", type=Path, default=Path("data/store"))
    p.add_argument("--data", type=Path, default=Path("data/processed/valid.npz"))
    p.add_argument("--rows", type=int, default=409600, help="validation rows served per run")
    p.add_argument("--hot-capacities", nargs="+", type=int, default=[12574, 62868, 251472])
    p.add_argument("--row-formats", nargs="+", choices=["fp32", "int8"], default=["fp32", "int8"])
    p.add_argument("--hot-backend", choices=["python", "cpp"], default="cpp")
    p.add_argument("--no-mmap", action="store_true",
                   help="also measure two-tier stores reading through the block cache instead of mmap")
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--out", type=Path, default=Path("experiments/results/memory.json"))
    args = p.parse_args()
    if args.child:
        print(json.dumps(child(json.loads(args.child))))
        return

    from embedserve.data import CriteoArrays

    base = {"store": str(args.store), "batch_size": args.batch_size, "threads": args.threads,
            "hot_capacity": None, "hot_backend": None, "row_format": None, "mmap_reads": None}
    configs = [{**base, "mode": mode} for mode in ("none", "fp32", "int8")]
    for mmap_reads in ([True, False] if args.no_mmap else [True]):
        for row_format in args.row_formats:
            for capacity in args.hot_capacities:
                configs.append({**base, "mode": "two-tier", "hot_capacity": capacity,
                                "hot_backend": args.hot_backend, "row_format": row_format,
                                "mmap_reads": mmap_reads})
    runs = []
    with tempfile.TemporaryDirectory() as tmp:
        data = CriteoArrays.load(args.data)
        sample = Path(tmp) / "sample.npz"
        CriteoArrays(data.labels[:args.rows], data.dense[:args.rows], data.sparse[:args.rows]).save(sample)
        del data
        for repeat in range(args.repeats):
            for config in configs:
                payload = json.dumps({**config, "data": str(sample)})
                done = subprocess.run([sys.executable, __file__, "--child", payload],
                                      check=True, capture_output=True, text=True)
                run = {"repeat": repeat, **json.loads(done.stdout.strip().splitlines()[-1])}
                del run["data"]
                runs.append(run)
                print(f"{run['mode']:8s} format={run['row_format']} capacity={run['hot_capacity']} "
                      f"mmap={run['mmap_reads']} footprint={run['peak_footprint_bytes'] / 1e6:.0f} MB "
                      f"rss={run['peak_rss_bytes'] / 1e6:.0f} MB "
                      f"payload={run['embedding_payload_bytes'] / 1e6:.1f} MB", flush=True)
    report = {
        "schema_version": 1,
        "memory_scope": "whole serving process at its peak; footprint is memory the OS cannot "
                        "drop, rss also counts file-backed pages such as mmapped SST files",
        "runs": runs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"results: {args.out}")


if __name__ == "__main__":
    main()
