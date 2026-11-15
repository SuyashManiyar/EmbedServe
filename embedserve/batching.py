"""Dynamic request batching in front of an embedding store and a DLRM.

Callers submit one request at a time. A single worker thread groups requests
that arrive close together and serves each group with one store lookup and one
model forward pass. A batch is dispatched when it reaches `max_batch_size`, or
`max_wait_ms` after its first request was taken from the queue, whichever comes
first. With max_batch_size=1 every request is served on its own.
"""

from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import Future

import numpy as np
import torch

from embedserve.model import DLRM
from embedserve.serving import EmbeddingStore

_STOP = object()


class DynamicBatcher:
    def __init__(self, model: DLRM, store: EmbeddingStore, max_batch_size: int = 256,
                 max_wait_ms: float = 2.0):
        if max_batch_size <= 0 or max_wait_ms < 0:
            raise ValueError("max batch size must be positive and max wait nonnegative")
        self.model = model
        self.store = store
        self.max_batch_size = max_batch_size
        self.max_wait = max_wait_ms / 1000
        self.batches = 0
        self.requests = 0
        self._queue: queue.SimpleQueue = queue.SimpleQueue()
        self._closed = False
        self._worker = threading.Thread(target=self._run, name="embedserve-batcher", daemon=True)
        self._worker.start()

    def submit(self, dense: np.ndarray, sparse: np.ndarray) -> Future:
        """Queue one request: dense (n_dense,) float32 and sparse (fields,) row IDs.

        The returned future resolves to the click probability as a float.
        """
        if self._closed:
            raise RuntimeError("batcher is closed")
        future: Future = Future()
        self._queue.put((dense, sparse, future))
        return future

    def predict(self, dense: np.ndarray, sparse: np.ndarray) -> float:
        return self.submit(dense, sparse).result()

    @property
    def mean_batch_size(self) -> float:
        return self.requests / self.batches if self.batches else 0.0

    def close(self) -> None:
        """Serve everything already queued, then stop the worker."""
        if not self._closed:
            self._closed = True
            self._queue.put(_STOP)
            self._worker.join()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _run(self) -> None:
        with torch.inference_mode():
            stopping = False
            while not stopping:
                item = self._queue.get()
                if item is _STOP:
                    break
                batch = [item]
                deadline = time.perf_counter() + self.max_wait
                while len(batch) < self.max_batch_size:
                    try:
                        # Take what is already queued first; wait only for the remainder.
                        item = self._queue.get_nowait()
                    except queue.Empty:
                        remaining = deadline - time.perf_counter()
                        if remaining <= 0:
                            break
                        try:
                            item = self._queue.get(timeout=remaining)
                        except queue.Empty:
                            break
                    if item is _STOP:
                        stopping = True
                        break
                    batch.append(item)
                self._serve(batch)

    def _infer(self, batch: list) -> np.ndarray:
        dense = np.stack([b[0] for b in batch]).astype(np.float32, copy=False)
        sparse = np.stack([b[1] for b in batch])
        rows = self.store.lookup(sparse)
        logits = self.model.forward_from_embeddings(torch.from_numpy(dense), torch.from_numpy(rows))
        return torch.sigmoid(logits).numpy()

    def _serve(self, batch: list) -> None:
        self.batches += 1
        self.requests += len(batch)
        try:
            probabilities = self._infer(batch)
        except Exception as error:
            if len(batch) == 1:
                batch[0][2].set_exception(error)
                return
            # One bad request must not fail its neighbours: retry each on its own.
            for item in batch:
                try:
                    item[2].set_result(float(self._infer([item])[0]))
                except Exception as single:
                    item[2].set_exception(single)
            return
        for item, p in zip(batch, probabilities.tolist()):
            item[2].set_result(p)
