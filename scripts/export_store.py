"""Export a checkpoint into a serving directory (MLP weights, FP32/INT8 arrays, RocksDB tiers)."""

from __future__ import annotations

import argparse
from pathlib import Path

from embedserve.export import export_store


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=Path("data/model/dlrm.pt"))
    p.add_argument("--out", type=Path, default=Path("data/store"))
    args = p.parse_args()
    meta = export_store(args.model, args.out)
    size = sum(f.stat().st_size for f in args.out.rglob("*") if f.is_file())
    print(f"exported {sum(meta['table_sizes']):,} rows of dim {meta['dim']} "
          f"to {args.out} ({size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
