"""Two-tier embedding store: a small in-memory hot tier in front of a cold tier.

A lookup takes a whole batch of IDs. Pass 1 serves hot-tier hits; each distinct
missing ID is fetched once, and repeats of it in the same batch count as hits.
Pass 2 fetches all misses from the cold tier in one call and promotes them into
the hot tier. The Python backend then fills the output with one array operation.
The C++ backend decodes those records straight into the output buffer and copies
each found row into the hot tier once.

`TierManager.lookup` handles one table. `TierManager.lookup_fields` handles a
(batch, fields) ID matrix in a single pass, with one cold-tier fetch for the
misses of every field.

Both tiers hold rows as fixed-width records in one of two formats, chosen by
the cold tier: "fp32" (dim float32 values) or "int8" (dim int8 values followed
by one float32 scale, from `quantize_int8`). Lookups always return FP32 rows;
INT8 records are dequantized as they are read.

Two hot-tier backends give identical results and statistics: "python" (an
OrderedDict LRU, one Python step per ID) and "cpp" (embedserve._hot, which runs
both passes for the whole batch in C++). The cold-tier fetch is shared.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
from rocksdict import AccessType, DBCompressionType, Options, Rdict

from embedserve.quantize import quantize_int8

try:
    from embedserve import _hot
except ImportError:  # extension not built
    _hot = None

Key = tuple[int, int]  # (table, row id)
HOT_BACKENDS = ("python", "cpp")
ROW_FORMATS = ("fp32", "int8")


def cpp_available() -> bool:
    return _hot is not None


def record_width(dim: int, row_format: str) -> int:
    """Bytes per stored row."""
    if row_format not in ROW_FORMATS:
        raise ValueError(f"row_format must be one of {ROW_FORMATS}")
    return 4 * dim if row_format == "fp32" else dim + 4


def encode_rows(rows: np.ndarray, row_format: str) -> np.ndarray:
    """FP32 rows (n, dim) to uint8 records (n, width)."""
    rows = np.ascontiguousarray(rows, dtype=np.float32)
    n, dim = rows.shape
    if row_format == "fp32":
        return rows.view(np.uint8).reshape(n, 4 * dim)
    q, scale = quantize_int8(rows)
    records = np.empty((n, dim + 4), dtype=np.uint8)
    records[:, :dim] = q.view(np.uint8)
    records[:, dim:] = scale.view(np.uint8).reshape(n, 4)
    return records


def decode_records(records: np.ndarray, dim: int, row_format: str) -> np.ndarray:
    """uint8 records (n, width) to FP32 rows (n, dim). All-zero records give zero rows."""
    if row_format == "fp32":
        return np.ascontiguousarray(records).view(np.float32)
    scale = np.ascontiguousarray(records[:, dim:]).view(np.float32)
    return records[:, :dim].view(np.int8).astype(np.float32) * scale


def _records_from_values(values: list[bytes | None], width: int) -> tuple[np.ndarray, np.ndarray]:
    if None not in values:
        records = np.frombuffer(b"".join(values), dtype=np.uint8).reshape(len(values), width)
        return records, np.ones(len(values), dtype=bool)
    records = np.zeros((len(values), width), dtype=np.uint8)
    found = np.array([v is not None for v in values], dtype=bool)
    for k, v in enumerate(values):
        if v is not None:
            records[k] = np.frombuffer(v, dtype=np.uint8)
    return records, found


class HotTier:
    """In-memory LRU cache of up to `capacity` records."""

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

    def __init__(self, dim: int, row_format: str = "fp32"):
        self.dim = dim
        self.row_format = row_format
        self.width = record_width(dim, row_format)
        self._rows: dict[Key, bytes] = {}

    def put_records(self, table: int, ids: np.ndarray, records: np.ndarray) -> None:
        for i, record in zip(ids.tolist(), records):
            self._rows[(table, i)] = record.tobytes()

    def get_values(self, tables: np.ndarray, ids: np.ndarray) -> list[bytes | None]:
        """Raw record for each key, or None when the key is absent."""
        return [self._rows.get(k) for k in zip(tables.tolist(), ids.tolist())]

    def multi_get_records(self, tables: np.ndarray, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Fetch (len(ids), width) uint8 records plus a found mask; missing records are zero."""
        return _records_from_values(self.get_values(tables, ids), self.width)


class RocksDBColdTier:
    """RocksDB-backed cold tier. Keys are 12 bytes: table (u32) + row id (u64), big-endian.

    Compression is off because embedding rows don't compress. With mmap_reads
    (the default) SST files are read through mmap, which makes cold reads cheaper
    but maps file pages into the process; without it, reads go through RocksDB's
    block cache. read_only opens an existing database without writing to it.
    """

    def __init__(self, path: str, dim: int, row_format: str = "fp32",
                 read_only: bool = False, mmap_reads: bool = True):
        self.dim = dim
        self.row_format = row_format
        self.width = record_width(dim, row_format)
        options = Options(raw_mode=True)
        options.create_if_missing(not read_only)
        options.set_compression_type(DBCompressionType.none())
        options.set_allow_mmap_reads(mmap_reads)
        access = AccessType.read_only() if read_only else AccessType.read_write()
        self._db = Rdict(path, options=options, access_type=access)

    @staticmethod
    def _encode(key: Key) -> bytes:
        table, row = key
        return table.to_bytes(4, "big") + row.to_bytes(8, "big")

    @staticmethod
    def _encode_many(tables, ids: np.ndarray) -> list[bytes]:
        encoded = np.empty(len(ids), dtype=[("table", ">u4"), ("row", ">u8")])
        encoded["table"] = tables
        encoded["row"] = ids
        buf = encoded.tobytes()
        return [buf[i : i + 12] for i in range(0, len(buf), 12)]

    def put_records(self, table: int, ids: np.ndarray, records: np.ndarray) -> None:
        for key, record in zip(self._encode_many(table, ids), records):
            self._db[key] = record.tobytes()

    def get_values(self, tables: np.ndarray, ids: np.ndarray) -> list[bytes | None]:
        """Raw record for each key, or None when the key is absent. One multi-get."""
        return self._db[self._encode_many(tables, ids)]

    def multi_get_records(self, tables: np.ndarray, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Fetch (len(ids), width) uint8 records plus a found mask; missing records are zero."""
        return _records_from_values(self.get_values(tables, ids), self.width)

    def finalize(self) -> None:
        """Flush and compact after an offline load, so a later open reads only SST files."""
        self._db.flush()
        self._db.compact_range(None, None)

    def close(self) -> None:
        self._db.close()


class TierManager:
    """LRU hot tier over a cold tier. The row format is taken from the cold tier."""

    def __init__(self, dim: int, hot_capacity: int, cold, hot_backend: str = "python"):
        if cold.dim != dim:
            raise ValueError("cold tier dim mismatch")
        if hot_backend not in HOT_BACKENDS:
            raise ValueError(f"hot_backend must be one of {HOT_BACKENDS}")
        self.dim = dim
        self.hot_backend = hot_backend
        self.row_format = cold.row_format
        if hot_backend == "cpp":
            if _hot is None:
                raise RuntimeError("C++ hot tier not built; run `pip install -e .`")
            self.hot = _hot.HotTier(hot_capacity, dim, self.row_format == "int8")
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
        """Write FP32 rows into the cold tier in its row format (offline table load)."""
        rows = np.asarray(rows, dtype=np.float32)
        ids = np.asarray(ids, dtype=np.int64)
        if rows.shape != (len(ids), self.dim):
            raise ValueError(f"rows must have shape ({len(ids)}, {self.dim})")
        self.cold.put_records(table, ids, encode_rows(rows, self.row_format))

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
        if self.hot_backend == "cpp":
            # Cold records are bytes from the fetch. C++ decodes each one into
            # the output rows that need it and copies it into the hot tier.
            values = self.cold.get_values(miss_tables, miss_ids)
            self.not_found += self.hot.complete(
                out, miss_tables, miss_ids, values, fill_pos, fill_slot)
            return out
        records, found = self.cold.multi_get_records(miss_tables, miss_ids)
        self.not_found += len(found) - int(np.count_nonzero(found))
        if self.hot.capacity:
            for table, i, record, ok in zip(
                    miss_tables.tolist(), miss_ids.tolist(), records, found.tolist()):
                if ok:
                    self.hot.put((table, i), record.copy())
        out[fill_pos] = decode_records(records, self.dim, self.row_format)[fill_slot]
        return out

    def _pass1_python(self, tables: np.ndarray, ids: np.ndarray):
        out = np.empty((len(ids), self.dim), dtype=np.float32)
        hits = 0
        miss_slot: dict[Key, int] = {}
        fill_pos: list[int] = []
        fill_slot: list[int] = []
        get, dim, quantized = self.hot.get, self.dim, self.row_format == "int8"
        for n, key in enumerate(zip(tables.tolist(), ids.tolist())):
            record = get(key)
            if record is not None:
                if quantized:
                    scale = record[dim:].view(np.float32)[0]
                    out[n] = record[:dim].view(np.int8).astype(np.float32) * scale
                else:
                    out[n] = record.view(np.float32)
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
