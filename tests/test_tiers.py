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


@pytest.fixture(params=["python", "cpp"])
def backend(request):
    return request.param


def make(cold, hot_capacity, n_rows=50, backend="python"):
    rows = np.random.default_rng(0).normal(size=(n_rows, DIM)).astype(np.float32)
    tm = TierManager(DIM, hot_capacity, cold, backend)
    tm.load(0, np.arange(n_rows), rows)
    return tm, rows


def test_lookup_returns_rows_and_counts_hits(cold, backend):
    tm, rows = make(cold, backend=backend, hot_capacity=10)
    out = tm.lookup(0, [3, 4, 3])  # second 3 reuses the first fetch
    np.testing.assert_array_equal(out, rows[[3, 4, 3]])
    assert (tm.hits, tm.misses) == (1, 2)
    tm.lookup(0, [3, 4])
    assert (tm.hits, tm.misses) == (3, 2)


def test_missing_rows_are_zero(cold, backend):
    tm, _ = make(cold, backend=backend, hot_capacity=10)
    assert np.all(tm.lookup(0, [999]) == 0) and tm.not_found == 1


def test_tables_are_separate(cold, backend):
    tm, _ = make(cold, backend=backend, hot_capacity=10)
    assert np.all(tm.lookup(1, [3]) == 0)


def test_lru_eviction(cold, backend):
    tm, _ = make(cold, backend=backend, hot_capacity=2)
    tm.lookup(0, [1, 2])
    tm.lookup(0, [1])  # 1 is now most recent
    tm.lookup(0, [3])  # evicts 2
    tm.reset_stats()
    tm.lookup(0, [1, 2])
    assert (tm.hits, tm.misses) == (1, 1)
    assert len(tm.hot) == 2


def test_skewed_access_has_high_hit_rate(cold, backend):
    tm, rows = make(cold, backend=backend, hot_capacity=10, n_rows=1000)
    ids = np.random.default_rng(1).zipf(1.5, size=5000) % 1000
    out = tm.lookup(0, ids)
    np.testing.assert_array_equal(out, rows[ids])
    assert tm.hit_rate > 0.7


def test_load_shape_check(cold, backend):
    tm = TierManager(DIM, 4, cold, backend)
    with pytest.raises(ValueError):
        tm.load(0, [1, 2], np.zeros((1, DIM), dtype=np.float32))


def test_lookup_fields_uses_one_table_per_column(cold, backend):
    rng = np.random.default_rng(2)
    data = rng.normal(size=(3, 20, DIM)).astype(np.float32)
    tm = TierManager(DIM, 5, cold, backend)
    for t in range(3):
        tm.load(t, np.arange(20), data[t])
    ids = np.array([[1, 1, 7], [1, 4, 7], [2, 1, 30]])
    out = tm.lookup_fields(ids)
    assert out.shape == (3, 3, DIM)
    for j in range(3):
        ok = ids[:, j] < 20
        np.testing.assert_array_equal(out[ok, j], data[j][ids[ok, j]])
    assert np.all(out[2, 2] == 0)
    # distinct keys: (0,1) (0,2) (1,1) (1,4) (2,7) (2,30); three repeats count as hits
    assert (tm.hits, tm.misses, tm.not_found) == (3, 6, 1)
    assert len(tm.hot) == 5
    with pytest.raises(ValueError):
        tm.lookup_fields(np.array([1, 2]))


@pytest.mark.parametrize("row_format,width", [("fp32", 4 * DIM), ("int8", DIM + 4)])
def test_record_round_trip(row_format, width):
    from embedserve.quantize import dequantize_int8, quantize_int8
    from embedserve.tiers import decode_records, encode_rows, record_width

    rows = np.random.default_rng(5).normal(size=(6, DIM)).astype(np.float32)
    rows[2] = 0  # an all-zero row keeps scale 1 and decodes to zeros
    records = encode_rows(rows, row_format)
    assert records.shape == (6, width) and records.dtype == np.uint8
    assert record_width(DIM, row_format) == width
    expected = rows if row_format == "fp32" else dequantize_int8(*quantize_int8(rows))
    np.testing.assert_array_equal(decode_records(records, DIM, row_format), expected)
    assert np.all(decode_records(np.zeros((2, width), np.uint8), DIM, row_format) == 0)
