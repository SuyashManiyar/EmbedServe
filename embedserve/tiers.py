"""Two-tier embedding store: a small in-memory hot tier in front of a cold tier.

A lookup takes a whole batch of IDs. Pass 1 serves hot-tier hits; each distinct
missing ID is fetched once, and repeats of it in the same batch count as hits.
Pass 2 fetches all misses from the cold tier in one call, promotes them into
the hot tier and fills the output with one array operation.

`TierManager.lookup` handles one table. `TierManager.lookup_fields` handles a
(batch, fields) ID matrix in a single pass, with one cold-tier fetch for the
misses of every field.

Two hot-tier backends give identical results and statistics: "python" (an
OrderedDict LRU, one Python step per ID) and "cpp" (embedserve._hot, which runs
pass 1 for the whole batch in C++). Everything else is shared, so a latency
difference between them comes from the hot tier alone.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
from rocksdict import DBCompressionType, Options, Rdict

try:
    from embedserve import _hot
except ImportError:  # extension not built
    _hot = None

Key = tuple[int, int]  # (table, row id)
HOT_BACKENDS = ("python", "cpp")


def cpp_available() -> bool:
    return _hot is not None


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


def _rows_from_values(values: list[np.ndarray | None], dim: int) -> tuple[np.ndarray, np.ndarray]:
    rows = np.zeros((len(values), dim), dtype=np.float32)
    found = np.array([v is not None for v in values], dtype=bool)
    for k, v in enumerate(values):
        if v is not None:
            rows[k] = v
    return rows, found


class InMemoryColdTier:
    """Dict-backed cold tier, for tests and quick runs."""

    def __init__(self, dim: int):
        self.dim = dim
        self._rows: dict[Key, np.ndarray] = {}

    def put(self, key: Key, row: np.ndarray) -> None:
        self._rows[key] = np.asarray(row, dtype=np.float32).copy()

    def multi_get(self, keys: list[Key]) -> list[np.ndarray | None]:
        return [self._rows.get(k) for k in keys]

    def multi_get_rows(self, tables: np.ndarray, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Fetch rows as a (len(ids), dim) array plus a found mask; missing rows are zero."""
        keys = list(zip(tables.tolist(), ids.tolist()))
        return _rows_from_values(self.multi_get(keys), self.dim)


class RocksDBColdTier:
    """RocksDB-backed cold tier. Keys are 12 bytes: table (u32) + row id (u64), big-endian.

    Compression is off because float rows don't compress, and SST files are read
    through mmap. Together these roughly halve the cost of a cold read compared
    with RocksDB's defaults.
    """

    def __init__(self, path: str, dim: int):
        self.dim = dim
        options = Options(raw_mode=True)
        options.create_if_missing(True)
        options.set_compression_type(DBCompressionType.none())
        options.set_allow_mmap_reads(True)
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

    def multi_get_rows(self, tables: np.ndarray, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Fetch rows as a (len(ids), dim) array plus a found mask; missing rows are zero."""
        encoded = np.empty(len(ids), dtype=[("table", ">u4"), ("row", ">u8")])
        encoded["table"] = tables
        encoded["row"] = ids
        buf = encoded.tobytes()
        values = self._db[[buf[i : i + 12] for i in range(0, len(buf), 12)]]
        if None not in values:
            rows = np.frombuffer(b"".join(values), dtype=np.float32).reshape(len(ids), self.dim)
            return rows, np.ones(len(ids), dtype=bool)
        return _rows_from_values(
            [None if v is None else np.frombuffer(v, dtype=np.float32) for v in values], self.dim)

    def close(self) -> None:
        self._db.close()


class TierManager:
    def __init__(self, dim: int, hot_capacity: int, cold, hot_backend: str = "python"):
        if cold.dim != dim:
            raise ValueError("cold tier dim mismatch")
        if hot_backend not in HOT_BACKENDS:
            raise ValueError(f"hot_backend must be one of {HOT_BACKENDS}")
        self.dim = dim
        self.hot_backend = hot_backend
        if hot_backend == "cpp":
            if _hot is None:
                raise RuntimeError("C++ hot tier not built; run `pip install -e .`")
            self.hot = _hot.HotTier(hot_capacity, dim)
        else:
            self.hot = HotTier(hot_capacity)
        self.cold = cold
        self._field_tables: dict[tuple[int, int], np.ndarray] = {}
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
        return self._lookup_flat(np.full(len(ids), table, dtype=np.int64), ids)

    def lookup_fields(self, ids) -> np.ndarray:
        """Return (batch, fields, dim) rows for a (batch, fields) ID matrix.

        Column j holds row IDs of table j. IDs are visited row by row, and all
        misses of the batch are fetched from the cold tier in one call.
        """
        ids = np.asarray(ids, dtype=np.int64)
        if ids.ndim != 2:
            raise ValueError("ids must have shape (batch, fields)")
        tables = self._field_tables.get(ids.shape)
        if tables is None:
            tables = np.tile(np.arange(ids.shape[1], dtype=np.int64), ids.shape[0])
            self._field_tables = {ids.shape: tables}  # keep only the latest shape
        return self._lookup_flat(tables, ids.reshape(-1)).reshape(*ids.shape, self.dim)

    def _lookup_flat(self, tables: np.ndarray, ids: np.ndarray) -> np.ndarray:
        """Rows for (tables[k], ids[k]) pairs, visited in order."""
        if self.hot_backend == "cpp":
            out, hits, miss_tables, miss_ids, fill_pos, fill_slot = self.hot.lookup(tables, ids)
        else:
            out, hits, miss_tables, miss_ids, fill_pos, fill_slot = self._pass1_python(tables, ids)
        self.hits += hits
        self.misses += len(miss_ids)
        if len(fill_pos) == 0:
            return out
        rows, found = self.cold.multi_get_rows(miss_tables, miss_ids)
        self.not_found += len(found) - int(np.count_nonzero(found))
        if self.hot_backend == "cpp":
            self.hot.put_batch(miss_tables, miss_ids, rows, found)
        elif self.hot.capacity:
            for table, i, row, ok in zip(miss_tables.tolist(), miss_ids.tolist(), rows, found.tolist()):
                if ok:
                    self.hot.put((table, i), row.copy())
        out[fill_pos] = rows[fill_slot]
        return out

    def _pass1_python(self, tables: np.ndarray, ids: np.ndarray):
        out = np.empty((len(ids), self.dim), dtype=np.float32)
        hits = 0
        miss_slot: dict[Key, int] = {}
        fill_pos: list[int] = []
        fill_slot: list[int] = []
        get = self.hot.get
        for n, key in enumerate(zip(tables.tolist(), ids.tolist())):
            row = get(key)
            if row is not None:
                out[n] = row
                hits += 1
                continue
            slot = miss_slot.get(key)
            if slot is None:
                slot = miss_slot[key] = len(miss_slot)
            else:
                hits += 1  # repeat of a key that already missed in this batch
            fill_pos.append(n)
            fill_slot.append(slot)
        miss = np.array(list(miss_slot), dtype=np.int64).reshape(-1, 2)
        return (out, hits, miss[:, 0], miss[:, 1],
                np.array(fill_pos, dtype=np.int64), np.array(fill_slot, dtype=np.int64))
