import numpy as np
import pytest

from embedserve import InMemoryColdTier, RocksDBColdTier, TierManager

DIM = 8


@pytest.fixture(params=["memory", "rocksdb"])
def cold(request, tmp_path):
    if request.param == "memory":
        yield InMemoryColdTier(DIM)
    else:
        tier = RocksDBColdTier(str(tmp_path / "db"), DIM)
        yield tier
        tier.close()


def make(cold, hot_capacity, n_rows=50):
    rows = np.random.default_rng(0).normal(size=(n_rows, DIM)).astype(np.float32)
    tm = TierManager(DIM, hot_capacity, cold)
    tm.load(0, np.arange(n_rows), rows)
    return tm, rows


def test_lookup_returns_rows_and_counts_hits(cold):
    tm, rows = make(cold, hot_capacity=10)
    out = tm.lookup(0, [3, 4, 3])  # second 3 reuses the first fetch
    np.testing.assert_array_equal(out, rows[[3, 4, 3]])
    assert (tm.hits, tm.misses) == (1, 2)
    tm.lookup(0, [3, 4])
    assert (tm.hits, tm.misses) == (3, 2)


def test_missing_rows_are_zero(cold):
    tm, _ = make(cold, hot_capacity=10)
    assert np.all(tm.lookup(0, [999]) == 0) and tm.not_found == 1


def test_tables_are_separate(cold):
    tm, _ = make(cold, hot_capacity=10)
    assert np.all(tm.lookup(1, [3]) == 0)


def test_lru_eviction(cold):
    tm, _ = make(cold, hot_capacity=2)
    tm.lookup(0, [1, 2])
    tm.lookup(0, [1])  # 1 is now most recent
    tm.lookup(0, [3])  # evicts 2
    tm.reset_stats()
    tm.lookup(0, [1, 2])
    assert (tm.hits, tm.misses) == (1, 1)
    assert len(tm.hot) == 2


def test_skewed_access_has_high_hit_rate(cold):
    tm, rows = make(cold, hot_capacity=10, n_rows=1000)
    ids = np.random.default_rng(1).zipf(1.5, size=5000) % 1000
    out = tm.lookup(0, ids)
    np.testing.assert_array_equal(out, rows[ids])
    assert tm.hit_rate > 0.7


def test_load_shape_check(cold):
    tm = TierManager(DIM, 4, cold)
    with pytest.raises(ValueError):
        tm.load(0, [1, 2], np.zeros((1, DIM), dtype=np.float32))
