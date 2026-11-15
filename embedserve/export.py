"""Export a trained checkpoint into a directory that can be served without the FP32 tables.

    model.pt          MLP weights and model config (no embedding tables)
    meta.json         table sizes and embedding dimension
    fp32.npy          all FP32 rows, tables concatenated      (for FP32Store)
    int8.npy          all INT8 rows, tables concatenated      (for INT8Store)
    int8_scale.npy    one FP32 scale per INT8 row
    tiers-fp32/       RocksDB database of FP32 rows           (for TwoTierStore)
    tiers-int8/       RocksDB database of INT8 rows and scales

A serving process opens only what its store needs, so its memory reflects that
store and not the full tables.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import torch

from embedserve.model import DLRM
from embedserve.quantize import quantize_int8
from embedserve.serving import EmbeddingStore, FP32Store, INT8Store, TwoTierStore
from embedserve.tiers import ROW_FORMATS, RocksDBColdTier, TierManager

STORE_MODES = ("fp32", "int8", "two-tier")


def export_store(checkpoint: Path, out: Path) -> dict:
    """Write every serving artifact for one checkpoint. Replaces `out` if it exists."""
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = saved["state_dict"]
    config = dict(saved["config"])
    n_tables = len(config["table_sizes"])
    tables = [state[f"embeddings.{j}.weight"].numpy() for j in range(n_tables)]
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    mlp_state = {k: v for k, v in state.items() if not k.startswith("embeddings.")}
    torch.save({"config": config, "state_dict": mlp_state}, out / "model.pt")
    meta = {"table_sizes": [len(t) for t in tables], "dim": int(tables[0].shape[1])}
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    np.save(out / "fp32.npy", np.concatenate(tables))
    quantized = [quantize_int8(t) for t in tables]
    np.save(out / "int8.npy", np.concatenate([q for q, _ in quantized]))
    np.save(out / "int8_scale.npy", np.concatenate([scale for _, scale in quantized]))

    for row_format in ROW_FORMATS:
        cold = RocksDBColdTier(str(out / f"tiers-{row_format}"), meta["dim"], row_format)
        try:
            manager = TierManager(meta["dim"], 0, cold)
            for j, table in enumerate(tables):
                manager.load(j, np.arange(len(table)), table)
            cold.finalize()
        finally:
            cold.close()
    return meta


def load_meta(root: Path) -> dict:
    return json.loads((root / "meta.json").read_text())


def load_mlp(root: Path) -> DLRM:
    """The model's MLPs, with no embedding tables allocated."""
    saved = torch.load(root / "model.pt", map_location="cpu", weights_only=True)
    model = DLRM(**saved["config"], embeddings=False)
    model.load_state_dict(saved["state_dict"])
    return model.eval()


def _split(rows: np.ndarray, table_sizes: list[int]) -> list[np.ndarray]:
    return np.split(rows, np.cumsum(table_sizes)[:-1])


def open_store(root: Path, mode: str, hot_capacity: int = 0, hot_backend: str = "python",
               row_format: str = "fp32", mmap_reads: bool = True) -> EmbeddingStore:
    """Open one store from an exported directory."""
    meta = load_meta(root)
    sizes = meta["table_sizes"]
    if mode == "fp32":
        return FP32Store(_split(np.load(root / "fp32.npy"), sizes))
    if mode == "int8":
        q, scale = np.load(root / "int8.npy"), np.load(root / "int8_scale.npy")
        return INT8Store.from_quantized(list(zip(_split(q, sizes), _split(scale, sizes))))
    if mode == "two-tier":
        return TwoTierStore.open(root / f"tiers-{row_format}", sizes, meta["dim"], hot_capacity,
                                 hot_backend, row_format, mmap_reads)
    raise ValueError(f"mode must be one of {STORE_MODES}")
