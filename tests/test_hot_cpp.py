"""The C++ hot tier must match the Python hot tier exactly: rows, stats and LRU state."""

import numpy as np
import pytest

from embedserve import InMemoryColdTier, RocksDBColdTier, TierManager, dequantize_int8, quantize_int8
from embedserve.tiers import cpp_available

pytestmark = pytest.mark.skipif(not cpp_available(), reason="C++ extension not built")

DIM = 8


def loaded(cold_factory, capacity, backend, tables=3, n_rows=200):
    rng = np.random.default_rng(0)
    data = rng.normal(size=(tables, n_rows, DIM)).astype(np.float32)
    tm = TierManager(DIM, capacity, cold_factory(), backend)
    for t in range(tables):
        tm.load(t, np.arange(n_rows), data[t])
    return tm, data


def state(tm):
    return tm.hits, tm.misses, tm.not_found, len(tm.hot), tm.hot.nbytes


@pytest.mark.parametrize("capacity", [0, 1, 7, 50, 1000])
def test_random_workload_matches_python(capacity):
    py, data = loaded(lambda: InMemoryColdTier(DIM), capacity, "python")
    cc, _ = loaded(lambda: InMemoryColdTier(DIM), capacity, "cpp")
    rng = np.random.default_rng(capacity)
    for _ in range(60):
        table = int(rng.integers(3))
        # Zipf-ish IDs with duplicates, plus some IDs past the end that are missing.
        ids = np.minimum(rng.zipf(1.3, size=int(rng.integers(0, 120))) - 1, 220)
        a, b = py.lookup(table, ids), cc.lookup(table, ids)
        np.testing.assert_array_equal(a, b)
        found = ids < 200
        np.testing.assert_array_equal(b[found], data[table][ids[found]])
        assert np.all(b[~found] == 0)
        assert state(py) == state(cc)


def test_rocksdb_cold_tier(tmp_path):
    cold = RocksDBColdTier(str(tmp_path / "db"), DIM)
    try:
        tm, data = loaded(lambda: cold, 5, "cpp", tables=2, n_rows=20)
        ids = np.array([3, 3, 25, 4, 3])
        out = tm.lookup(1, ids)
        np.testing.assert_array_equal(out[[0, 1, 3, 4]], data[1][[3, 3, 4, 3]])
        assert np.all(out[2] == 0)
        assert (tm.hits, tm.misses, tm.not_found) == (2, 3, 1)
        assert len(tm.hot) == 2  # missing rows are not cached
    finally:
        cold.close()


def test_batch_larger_than_capacity_keeps_most_recent_misses():
    tm, data = loaded(lambda: InMemoryColdTier(DIM), 2, "cpp")
    tm.lookup(0, [1, 2, 3, 4])
    tm.reset_stats()
    tm.lookup(0, [3, 4])
    assert (tm.hits, tm.misses) == (2, 0)


@pytest.mark.parametrize("capacity", [0, 1, 7, 50, 1000])
@pytest.mark.parametrize("row_format", ["fp32", "int8"])
def test_lookup_fields_matches_python(capacity, row_format):
    py, data = loaded(lambda: InMemoryColdTier(DIM, row_format), capacity, "python")
    cc, _ = loaded(lambda: InMemoryColdTier(DIM, row_format), capacity, "cpp")
    if row_format == "int8":
        data = np.stack([dequantize_int8(*quantize_int8(t)) for t in data])
    rng = np.random.default_rng(100 + capacity)
    for _ in range(40):
        ids = np.minimum(rng.zipf(1.3, size=(int(rng.integers(0, 60)), 3)) - 1, 220)
        a, b = py.lookup_fields(ids), cc.lookup_fields(ids)
        assert b.shape == (*ids.shape, DIM) and b.dtype == np.float32
        np.testing.assert_array_equal(a, b)
        for j in range(3):
            found = ids[:, j] < 200
            np.testing.assert_array_equal(b[found, j], data[j][ids[found, j]])
            assert np.all(b[~found, j] == 0)
        assert state(py) == state(cc)


def test_heavy_eviction_keeps_index_consistent():
    """Many inserts and evictions over a small table exercise hash-map deletion."""
    py, data = loaded(lambda: InMemoryColdTier(DIM), 37, "python", tables=1, n_rows=5000)
    cc, _ = loaded(lambda: InMemoryColdTier(DIM), 37, "cpp", tables=1, n_rows=5000)
    rng = np.random.default_rng(3)
    for _ in range(300):
        ids = rng.integers(0, 5000, size=64)
        np.testing.assert_array_equal(cc.lookup(0, ids), data[0][ids])
        py.lookup(0, ids)
        assert state(py) == state(cc)


def test_rejects_bad_arguments():
    from embedserve import _hot

    with pytest.raises(ValueError):
        _hot.HotTier(-1, DIM)
    hot = _hot.HotTier(4, DIM)
    zero = np.zeros(1, dtype=np.int64)
    with pytest.raises(IndexError):
        hot.lookup(zero, np.array([-1]))
    with pytest.raises(IndexError):
        hot.lookup(np.array([1 << 23]), zero)
    with pytest.raises(ValueError):
        hot.lookup(zero, np.zeros(2, dtype=np.int64))
    with pytest.raises(ValueError):
        hot.put_batch(zero, zero, np.zeros((1, DIM + 1), np.float32), np.array([True]))
