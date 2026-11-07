"""Encode Criteo CSV subsets into compact arrays and report access skew.

Example:
    python scripts/prepare_data.py --train data/train_5M.csv --valid data/valid_1M.csv \
        --train-rows 800000 --valid-rows 200000

Writes <out>/train.npz, <out>/valid.npz and <out>/vocab.npz. The vocabulary is built
from the training rows only, so validation measures how unseen IDs are handled.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from embedserve.data import access_skew, apply_vocab, build_vocab, read_csv


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", required=True, type=Path)
    p.add_argument("--valid", type=Path)
    p.add_argument("--train-rows", type=int, help="use only the first N training rows")
    p.add_argument("--valid-rows", type=int, help="use only the first N validation rows")
    p.add_argument("--min-count", type=int, default=4, help="IDs seen fewer times map to row 0")
    p.add_argument("--dim", type=int, default=16, help="embedding dim, for the memory estimate only")
    p.add_argument("--out", type=Path, default=Path("data/processed"))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    t = time.perf_counter()
    raw = read_csv(args.train, args.train_rows)
    print(f"read {len(raw):,} train rows in {time.perf_counter() - t:.1f}s, CTR {raw.labels.mean():.4f}")

    vocab = build_vocab(raw.sparse, args.min_count)
    train = apply_vocab(raw, vocab)
    del raw
    train.save(args.out / "train.npz")
    vocab.save(args.out / "vocab.npz")

    sizes = vocab.sizes
    total_rows = sum(sizes)
    mb = total_rows * args.dim * 4 / 2**20
    print(f"\nvocab (min_count={args.min_count}): {total_rows:,} embedding rows, "
          f"{mb:,.1f} MB at dim {args.dim} FP32 (~{mb / 4:,.1f} MB INT8)")
    print("rows per table:", ", ".join(f"{s:,}" for s in sizes))

    skew = access_skew(train.sparse, sizes)
    print(f"\naccess skew over {skew['lookups']:,} train lookups:")
    print(f"  rows never accessed: {skew['rows_never_accessed']:,}")
    print(f"  rare/unseen row (row 0) share of lookups: {skew['rare_bucket_share']:.1%}")
    for f, cov in skew["coverage"].items():
        print(f"  hottest {f:>6.1%} of rows ({round(f * total_rows):>9,}) serve {cov:6.1%} of lookups")

    if args.valid:
        t = time.perf_counter()
        valid = apply_vocab(read_csv(args.valid, args.valid_rows), vocab)
        valid.save(args.out / "valid.npz")
        print(f"\nread {len(valid):,} valid rows in {time.perf_counter() - t:.1f}s, "
              f"CTR {valid.labels.mean():.4f}, rare/unseen share {(valid.sparse == 0).mean():.1%}")

    print(f"\nwrote {args.out}/")


if __name__ == "__main__":
    main()
