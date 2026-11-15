"""Measure request throughput and latency with and without dynamic batching.

An open-loop load generator submits single requests at a fixed target rate
with exponentially distributed gaps (Poisson arrivals). Requests are sent on
schedule whether or not earlier ones have finished. Submitting never blocks,
so a server that falls behind builds a queue, and that queueing delay is part
of each request's latency (measured from the moment it is submitted to its
result). The generator runs in the same Python process as the server; its own
lateness against the schedule is reported as send lag.

Each (store, max batch size, rate) point runs for --duration seconds against a
store opened from an exported directory (scripts/export_store.py). The hot
tier is filled first by serving --warmup-rows validation rows.
"""

from __future__ import annotations

import argparse
import json
import platform
import threading
import time
from pathlib import Path

import numpy as np
import torch

from embedserve.batching import DynamicBatcher
from embedserve.data import CriteoArrays
from embedserve.export import load_mlp, open_store


def run_point(model, store, data: CriteoArrays, first_row: int, rate: float, duration: float,
              max_batch_size: int, max_wait_ms: float, seed: int) -> dict:
    n = int(rate * duration)
    rng = np.random.default_rng(seed)
    schedule = np.cumsum(rng.exponential(1.0 / rate, size=n))
    sent_at = np.zeros(n)
    done = np.full(n, np.nan)
    remaining = threading.Semaphore(0)

    def finished(index: int):
        def callback(future):
            done[index] = time.perf_counter() if future.exception() is None else np.nan
            remaining.release()
        return callback

    rows = (first_row + np.arange(n)) % len(data)
    with DynamicBatcher(model, store, max_batch_size, max_wait_ms) as batcher:
        start = time.perf_counter()
        for i in range(n):
            delay = start + schedule[i] - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            row = rows[i]
            sent_at[i] = time.perf_counter()
            batcher.submit(data.dense[row], data.sparse[row]).add_done_callback(finished(i))
        sent = time.perf_counter() - start
        for _ in range(n):
            remaining.acquire()
        elapsed = time.perf_counter() - start
        mean_batch = batcher.mean_batch_size
    latency_ms = (done - sent_at) * 1000
    ok = latency_ms[~np.isnan(latency_ms)]
    lag_ms = (sent_at - (start + schedule)) * 1000
    return {
        "target_rate": rate,
        "max_batch_size": max_batch_size,
        "max_wait_ms": max_wait_ms,
        "requests": n,
        "failed": int(n - len(ok)),
        "send_seconds": sent,
        "achieved_rate": len(ok) / elapsed,
        "mean_batch_size": mean_batch,
        "p50_ms": float(np.percentile(ok, 50)),
        "p99_ms": float(np.percentile(ok, 99)),
        "mean_ms": float(ok.mean()),
        "max_ms": float(ok.max()),
        "send_lag_p99_ms": float(np.percentile(lag_ms, 99)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store", type=Path, default=Path("data/store"))
    p.add_argument("--data", type=Path, default=Path("data/processed/valid.npz"))
    p.add_argument("--mode", choices=["fp32", "int8", "two-tier"], default="two-tier")
    p.add_argument("--hot-capacity", type=int, default=251472)
    p.add_argument("--hot-backend", choices=["python", "cpp"], default="cpp")
    p.add_argument("--row-format", choices=["fp32", "int8"], default="int8")
    p.add_argument("--rates", nargs="+", type=float, default=[1000, 2000, 4000, 8000, 16000])
    p.add_argument("--max-batch-sizes", nargs="+", type=int, default=[1, 256])
    p.add_argument("--max-wait-ms", type=float, default=1.0)
    p.add_argument("--duration", type=float, default=10.0, help="seconds of load per point")
    p.add_argument("--warmup-rows", type=int, default=200000)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--repeats", type=int, default=1,
                   help="times to repeat every point; repeats are interleaved")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("experiments/results/serving.json"))
    args = p.parse_args()
    if min(args.rates) <= 0 or min(args.max_batch_sizes) <= 0 or args.duration <= 0:
        p.error("rates, batch sizes and duration must be positive")
    torch.set_num_threads(args.threads)
    data = CriteoArrays.load(args.data)
    if not 0 <= args.warmup_rows < len(data):
        p.error("warmup rows must be fewer than the rows in --data")
    model = load_mlp(args.store)
    store = open_store(args.store, args.mode, args.hot_capacity, args.hot_backend, args.row_format)
    two_tier = args.mode == "two-tier"
    report = {
        "schema_version": 1,
        "store": {"path": str(args.store), "mode": args.mode,
                  "hot_capacity": args.hot_capacity if two_tier else None,
                  "hot_backend": args.hot_backend if two_tier else None,
                  "row_format": args.row_format if two_tier else None},
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "numpy": np.__version__, "torch": torch.__version__,
                        "threads": torch.get_num_threads()},
        "load": "open loop, Poisson arrivals, one generator thread in the server process; "
                "latency from submit to result",
        "warmup_rows": args.warmup_rows,
        "duration_seconds": args.duration,
        "runs": [],
    }
    with store:
        for start in range(0, args.warmup_rows, 2048):
            store.lookup(data.sparse[start : min(start + 2048, args.warmup_rows)])
        first_row = args.warmup_rows
        for repeat in range(args.repeats):
            for rate in args.rates:
                for max_batch_size in args.max_batch_sizes:
                    store.reset_stats()
                    result = run_point(model, store, data, first_row, rate, args.duration,
                                       max_batch_size, args.max_wait_ms, args.seed + repeat)
                    result["repeat"] = repeat
                    result["hit_rate"] = store.stats()["hit_rate"]
                    first_row += result["requests"]
                    report["runs"].append(result)
                    print(f"repeat={repeat} max_batch={max_batch_size:4d} target={rate:7.0f}/s "
                          f"achieved={result['achieved_rate']:7.0f}/s "
                          f"batch={result['mean_batch_size']:6.1f} p50={result['p50_ms']:8.2f} ms "
                          f"p99={result['p99_ms']:8.2f} ms lag p99={result['send_lag_p99_ms']:.2f} ms",
                          flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"results: {args.out}")


if __name__ == "__main__":
    main()
