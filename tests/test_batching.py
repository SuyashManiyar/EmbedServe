import threading
import time

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from embedserve.batching import DynamicBatcher
from embedserve.model import DLRM
from embedserve.serving import FP32Store


@pytest.fixture
def setup():
    torch.manual_seed(5)
    model = DLRM([5, 7], dim=4, bottom=(8,), top=(8,)).eval()
    rng = np.random.default_rng(5)
    dense = rng.random((40, 13), dtype=np.float32)
    sparse = np.stack([rng.integers(0, 5, 40), rng.integers(0, 7, 40)], axis=1)
    store = FP32Store([e.weight.detach().numpy() for e in model.embeddings])
    with torch.inference_mode():
        expected = torch.sigmoid(model(torch.from_numpy(dense), torch.from_numpy(sparse))).numpy()
    return model, store, dense, sparse, expected


class CountingStore(FP32Store):
    """Records the batch size of every lookup, optionally blocking the first one."""

    def __init__(self, tables, gate: threading.Event | None = None):
        super().__init__(tables)
        self.sizes: list[int] = []
        self.gate = gate

    def lookup(self, ids):
        self.sizes.append(len(ids))
        if self.gate is not None and len(self.sizes) == 1:
            self.gate.wait(5)
        return super().lookup(ids)


@pytest.mark.parametrize("max_batch_size", [1, 8, 64])
def test_results_match_direct_inference(setup, max_batch_size):
    model, store, dense, sparse, expected = setup
    with DynamicBatcher(model, store, max_batch_size, max_wait_ms=5) as batcher:
        futures = [batcher.submit(d, s) for d, s in zip(dense, sparse)]
        actual = np.array([f.result(5) for f in futures], dtype=np.float32)
        assert batcher.requests == 40
    np.testing.assert_allclose(actual, expected, rtol=1e-5)
    assert model.training is False


def test_queued_requests_are_batched_up_to_the_cap(setup):
    model, store, dense, sparse, _ = setup
    gate = threading.Event()
    counting = CountingStore(store.tables, gate)
    with DynamicBatcher(model, counting, max_batch_size=8, max_wait_ms=0) as batcher:
        first = batcher.submit(dense[0], sparse[0])
        while not counting.sizes:  # the worker is now blocked inside the first lookup
            time.sleep(0.001)
        rest = [batcher.submit(d, s) for d, s in zip(dense[1:21], sparse[1:21])]
        gate.set()
        for future in [first, *rest]:
            future.result(5)
    assert counting.sizes == [1, 8, 8, 4]
    assert batcher.mean_batch_size == pytest.approx(21 / 4)


def test_batch_size_one_never_batches(setup):
    model, store, dense, sparse, _ = setup
    counting = CountingStore(store.tables)
    with DynamicBatcher(model, counting, max_batch_size=1, max_wait_ms=50) as batcher:
        for future in [batcher.submit(d, s) for d, s in zip(dense[:10], sparse[:10])]:
            future.result(5)
    assert counting.sizes == [1] * 10


def test_lone_request_is_served_after_max_wait(setup):
    model, store, dense, sparse, expected = setup
    with DynamicBatcher(model, store, max_batch_size=64, max_wait_ms=20) as batcher:
        start = time.perf_counter()
        assert batcher.predict(dense[0], sparse[0]) == pytest.approx(expected[0], rel=1e-5)
        assert time.perf_counter() - start < 2


def test_bad_request_fails_alone(setup):
    model, store, dense, sparse, expected = setup
    gate = threading.Event()
    counting = CountingStore(store.tables, gate)
    with DynamicBatcher(model, counting, max_batch_size=8, max_wait_ms=0) as batcher:
        first = batcher.submit(dense[0], sparse[0])
        while not counting.sizes:
            time.sleep(0.001)
        good = batcher.submit(dense[1], sparse[1])
        bad = batcher.submit(dense[2], np.array([99, 0]))
        also_good = batcher.submit(dense[3], sparse[3])
        gate.set()
        first.result(5)
        assert good.result(5) == pytest.approx(expected[1], rel=1e-5)
        assert also_good.result(5) == pytest.approx(expected[3], rel=1e-5)
        with pytest.raises(ValueError, match="row ID outside"):
            bad.result(5)


def test_close_serves_queued_requests_and_rejects_new_ones(setup):
    model, store, dense, sparse, _ = setup
    batcher = DynamicBatcher(model, store, max_batch_size=4, max_wait_ms=1)
    futures = [batcher.submit(d, s) for d, s in zip(dense, sparse)]
    batcher.close()
    assert all(f.done() and f.exception() is None for f in futures)
    with pytest.raises(RuntimeError, match="closed"):
        batcher.submit(dense[0], sparse[0])
    batcher.close()  # idempotent


@pytest.mark.parametrize("size,wait", [(0, 1), (-1, 1), (4, -1)])
def test_invalid_settings(setup, size, wait):
    model, store, *_ = setup
    with pytest.raises(ValueError):
        DynamicBatcher(model, store, size, wait)
