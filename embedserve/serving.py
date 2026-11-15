"""Embedding stores with a common batch lookup interface."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from embedserve.quantize import quantize_int8
from embedserve.tiers import RocksDBColdTier, TierManager


class EmbeddingStore:
    """FP32 output rows with shape (batch, fields, dim)."""

    def __init__(self, tables: list[np.ndarray]):
        if not tables or any(t.ndim != 2 or t.shape[0] == 0 for t in tables):
            raise ValueError("tables must be nonempty two-dimensional arrays")
        self.dim = tables[0].shape[1]
        if self.dim == 0 or any(t.shape[1] != self.dim for t in tables):
            raise ValueError("tables must have the same positive embedding dimension")
        if any(t.dtype != np.float32 or not np.isfinite(t).all() for t in tables):
            raise ValueError("tables must contain finite float32 values")
        self.table_sizes = [len(t) for t in tables]

    def validate_ids(self, ids: np.ndarray) -> None:
        if ids.ndim != 2 or ids.shape[1] != len(self.table_sizes):
            raise ValueError("IDs must have shape (batch, fields)")
        if not np.issubdtype(ids.dtype, np.integer):
            raise ValueError("IDs must be integers")
        for j, size in enumerate(self.table_sizes):
            if np.any(ids[:, j] < 0) or np.any(ids[:, j] >= size):
                raise ValueError(f"row ID outside table {j} with {size} rows")

    def lookup(self, ids: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @property
    def payload_bytes(self) -> int:
        """Resident embedding array bytes, excluding allocator and cache overhead."""
        raise NotImplementedError

    def reset_stats(self) -> None:
        pass

    def stats(self) -> dict:
        return {"hits": None, "misses": None, "hit_rate": None}

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class FP32Store(EmbeddingStore):
    def __init__(self, tables: list[np.ndarray]):
        super().__init__(tables)
        self.tables = tables

    def lookup(self, ids: np.ndarray) -> np.ndarray:
        self.validate_ids(ids)
        return np.stack([t[ids[:, j]] for j, t in enumerate(self.tables)], axis=1)

    @property
    def payload_bytes(self) -> int:
        return sum(t.nbytes for t in self.tables)


class INT8Store(EmbeddingStore):
    """INT8 tables with FP32 row scales and dequantization of requested rows."""

    def __init__(self, tables: list[np.ndarray]):
        super().__init__(tables)
        self.tables = [quantize_int8(t) for t in tables]

    def lookup(self, ids: np.ndarray) -> np.ndarray:
        self.validate_ids(ids)
        return np.stack([
            q[ids[:, j]].astype(np.float32) * scale[ids[:, j], None]
            for j, (q, scale) in enumerate(self.tables)
        ], axis=1)

    @property
    def payload_bytes(self) -> int:
        return sum(q.nbytes + scale.nbytes for q, scale in self.tables)


class TwoTierStore(EmbeddingStore):
    """FP32 LRU rows backed by a temporary RocksDB database.

    One TierManager pass covers every field of the batch, with a single
    cold-tier fetch for all misses. Hits include duplicate IDs within a field
    batch, consistent with TierManager statistics.
    """

    def __init__(self, tables: list[np.ndarray], hot_capacity: int, db_root: Path,
                 hot_backend: str = "python"):
        super().__init__(tables)
        if hot_capacity < 0:
            raise ValueError("hot capacity must be nonnegative")
        db_root.mkdir(parents=True, exist_ok=True)
        self._directory = TemporaryDirectory(prefix="eval-", dir=db_root)
        self.cold = None
        try:
            self.cold = RocksDBColdTier(self._directory.name, self.dim)
            self.manager = TierManager(self.dim, hot_capacity, self.cold, hot_backend)
            for j, table in enumerate(tables):
                self.manager.load(j, np.arange(len(table)), table)
        except BaseException:
            self.close()
            raise

    def lookup(self, ids: np.ndarray) -> np.ndarray:
        self.validate_ids(ids)
        out = self.manager.lookup_fields(ids)
        if self.manager.not_found:
            raise RuntimeError("embedding row missing from the cold tier")
        return out

    @property
    def payload_bytes(self) -> int:
        return self.manager.hot.nbytes

    def reset_stats(self) -> None:
        self.manager.reset_stats()

    def stats(self) -> dict:
        return {
            "hits": self.manager.hits,
            "misses": self.manager.misses,
            "hit_rate": self.manager.hit_rate,
            "not_found": self.manager.not_found,
            "hot_rows": len(self.manager.hot),
            "hot_capacity": self.manager.hot.capacity,
            "hot_backend": self.manager.hot_backend,
        }

    def close(self) -> None:
        try:
            if self.cold is not None:
                self.cold.close()
                self.cold = None
        finally:
            self._directory.cleanup()
