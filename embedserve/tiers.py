"""Two-tier embedding store: a small in-memory hot tier in front of a cold tier.

A lookup takes a whole batch of IDs. Pass 1 serves hot-tier hits; each distinct
missing ID is fetched once, and repeats of it in the same batch count as hits.
Pass 2 fetches all misses from the cold tier in one call and promotes them into
the hot tier.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
from rocksdict import Options, Rdict

Key = tuple[int, int]  # (table, row id)


class HotTier:
    """In-memory LRU cache of up to `capacity` rows."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._rows: OrderedDict[Key, np.ndarray] = OrderedDict()

    def get(self, key: Key) -> np.ndarray | None:
        row = self._rows.get(key)
        if row is not None:
            self._rows.move_to_end(key)
        return row

    def put(self, key: Key, row: np.ndarray) -> None:
        if self.capacity == 0:
            return
        self._rows[key] = row
        self._rows.move_to_end(key)
        if len(self._rows) > self.capacity:
            self._rows.popitem(last=False)

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def nbytes(self) -> int:
        return sum(row.nbytes for row in self._rows.values())


class InMemoryColdTier:
    """Dict-backed cold tier, for tests and quick runs."""

    def __init__(self, dim: int):
        self.dim = dim
        self._rows: dict[Key, np.ndarray] = {}

    def put(self, key: Key, row: np.ndarray) -> None:
        self._rows[key] = np.asarray(row, dtype=np.float32).copy()

    def multi_get(self, keys: list[Key]) -> list[np.ndarray | None]:
        return [self._rows.get(k) for k in keys]


class RocksDBColdTier:
    """RocksDB-backed cold tier. Keys are 12 bytes: table (u32) + row id (u64), big-endian."""

    def __init__(self, path: str, dim: int):
        self.dim = dim
        options = Options(raw_mode=True)
        options.create_if_missing(True)
        self._db = Rdict(path, options=options)

    @staticmethod
    def _encode(key: Key) -> bytes:
        table, row = key
        return table.to_bytes(4, "big") + row.to_bytes(8, "big")

    def put(self, key: Key, row: np.ndarray) -> None:
        self._db[self._encode(key)] = np.asarray(row, dtype=np.float32).tobytes()

    def multi_get(self, keys: list[Key]) -> list[np.ndarray | None]:
        values = self._db[[self._encode(k) for k in keys]]
        return [None if v is None else np.frombuffer(v, dtype=np.float32) for v in values]

    def close(self) -> None:
        self._db.close()


class TierManager:
    def __init__(self, dim: int, hot_capacity: int, cold):
        if cold.dim != dim:
            raise ValueError("cold tier dim mismatch")
        self.dim = dim
        self.hot = HotTier(hot_capacity)
        self.cold = cold
        self.reset_stats()

    def reset_stats(self) -> None:
        self.hits = 0
        self.misses = 0
        self.not_found = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def load(self, table: int, ids, rows) -> None:
        """Write rows into the cold tier (offline table load)."""
        rows = np.asarray(rows, dtype=np.float32)
        ids = np.asarray(ids)
        if rows.shape != (len(ids), self.dim):
            raise ValueError(f"rows must have shape ({len(ids)}, {self.dim})")
        for i, row in zip(ids, rows):
            self.cold.put((table, int(i)), row)

    def lookup(self, table: int, ids) -> np.ndarray:
        """Return a (len(ids), dim) array. IDs missing from every tier give zero rows."""
        ids = np.asarray(ids, dtype=np.int64)
        out = np.zeros((len(ids), self.dim), dtype=np.float32)
        miss_slot: dict[Key, int] = {}
        fills: list[tuple[int, int]] = []  # (output position, miss slot)

        for n, i in enumerate(ids):
            key = (table, int(i))
            if key in miss_slot:
                fills.append((n, miss_slot[key]))
                self.hits += 1
                continue
            row = self.hot.get(key)
            if row is not None:
                out[n] = row
                self.hits += 1
            else:
                fills.append((n, len(miss_slot)))
                miss_slot[key] = len(miss_slot)
        self.misses += len(miss_slot)
        if not miss_slot:
            return out

        fetched = self.cold.multi_get(list(miss_slot))
        for key, row in zip(miss_slot, fetched):
            if row is None:
                self.not_found += 1
            else:
                self.hot.put(key, row)
        for n, slot in fills:
            row = fetched[slot]
            if row is not None:
                out[n] = row
        return out
