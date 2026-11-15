import numpy as np
import pytest

from embedserve.serving import FP32Store, INT8Store, TwoTierStore


@pytest.fixture
def tables():
    rng = np.random.default_rng(7)
    return [rng.normal(size=(n, 8)).astype(np.float32) for n in (5, 7)]


@pytest.fixture(params=["fp32", "int8", "two-tier", "two-tier-cpp"])
def store(request, tables, tmp_path):
    if request.param == "fp32":
        obj = FP32Store(tables)
    elif request.param == "int8":
        obj = INT8Store(tables)
    elif request.param == "two-tier":
        obj = TwoTierStore(tables, 3, tmp_path)
    else:
        obj = TwoTierStore(tables, 3, tmp_path, hot_backend="cpp")
    with obj:
        yield obj


def test_lookup_preserves_field_order_and_duplicate_ids(store, tables):
    ids = np.array([[0, 6], [4, 0], [0, 6]])
    expected = np.stack([table[ids[:, j]] for j, table in enumerate(tables)], axis=1)
    actual = store.lookup(ids)
    assert actual.dtype == np.float32
    if isinstance(store, INT8Store):
        bounds = np.stack([np.abs(t[ids[:, j]]).max(axis=1) / 254 for j, t in enumerate(tables)], axis=1)
        assert np.all(np.abs(actual - expected) <= bounds[:, :, None] + 1e-6)
    else:
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("ids", [np.array([[-1, 0]]), np.array([[5, 0]]),
                                  np.array([[0, 7]]), np.array([[0.0, 0.0]]),
                                  np.array([[0]]), np.array([0, 0])])
def test_invalid_ids_fail(store, ids):
    with pytest.raises(ValueError):
        store.lookup(ids)


def test_empty_batch(store):
    assert store.lookup(np.empty((0, 2), dtype=np.int64)).shape == (0, 2, 8)


def test_payload_includes_quantization_scales(tables):
    assert FP32Store(tables).payload_bytes == 12 * 8 * 4
    assert INT8Store(tables).payload_bytes == 12 * (8 + 4)


def test_tier_statistics_reset_without_clearing_cache(tables, tmp_path):
    with TwoTierStore(tables, 12, tmp_path) as store:
        ids = np.array([[0, 0], [0, 0]])
        store.lookup(ids)
        assert store.stats()["hits"] == 2
        assert store.stats()["misses"] == 2
        assert store.payload_bytes == 2 * 8 * 4
        store.reset_stats()
        store.lookup(ids)
        assert store.stats()["hits"] == 4
        assert store.stats()["misses"] == 0
    assert list(tmp_path.iterdir()) == []


def test_zero_capacity_still_deduplicates_batch(tables, tmp_path):
    with TwoTierStore(tables, 0, tmp_path) as store:
        store.lookup(np.array([[0, 0], [0, 0]]))
        assert store.payload_bytes == 0
        assert store.stats()["hit_rate"] == 0.5


def test_missing_cold_row_fails(tables, tmp_path):
    with TwoTierStore(tables, 0, tmp_path) as store:
        del store.cold._db[store.cold._encode((0, 0))]
        with pytest.raises(RuntimeError, match="missing"):
            store.lookup(np.array([[0, 0]]))


def test_negative_capacity_fails(tables, tmp_path):
    with pytest.raises(ValueError, match="nonnegative"):
        TwoTierStore(tables, -1, tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("backend", ["python", "cpp"])
def test_int8_tiers_match_int8_store(tables, tmp_path, backend):
    ids = np.array([[0, 6], [4, 0], [0, 6], [3, 2]])
    with INT8Store(tables) as reference, \
            TwoTierStore(tables, 3, tmp_path, backend, row_format="int8") as store:
        for _ in range(3):  # cold reads, then hot hits, both dequantized the same way
            np.testing.assert_array_equal(store.lookup(ids), reference.lookup(ids))
        assert store.stats()["row_format"] == "int8"
        assert store.payload_bytes == 3 * (8 + 4)


def test_unknown_row_format_fails(tables, tmp_path):
    with pytest.raises(ValueError, match="row_format"):
        TwoTierStore(tables, 3, tmp_path, row_format="int4")
    assert list(tmp_path.iterdir()) == []
