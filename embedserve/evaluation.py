"""Accuracy and batch latency measurements for fixed DLRM weights."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import log_loss, roc_auc_score

from embedserve.data import CriteoArrays
from embedserve.model import DLRM
from embedserve.serving import EmbeddingStore


def load_checkpoint(path: Path) -> tuple[DLRM, list[np.ndarray]]:
    """Load CPU weights and detach embedding storage from the inference model."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = DLRM(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    tables = [emb.weight.detach().numpy() for emb in model.embeddings]
    model.embeddings = torch.nn.ModuleList()
    return model.eval(), tables


def validate_data(model: DLRM, store: EmbeddingStore, data: CriteoArrays) -> None:
    n = len(data)
    if n == 0:
        raise ValueError("evaluation data must contain rows")
    if data.labels.shape != (n,) or not np.array_equal(np.unique(data.labels), [0, 1]):
        raise ValueError("AUC requires binary labels with both classes present")
    if data.dense.shape != (n, model.bottom[0].in_features):
        raise ValueError("dense feature shape does not match the model")
    if data.dense.dtype != np.float32 or not np.isfinite(data.dense).all():
        raise ValueError("dense features must contain finite float32 values")
    if store.table_sizes != model.table_sizes or store.dim != model.dim:
        raise ValueError("store dimensions do not match the model")
    if len(data.sparse) != n:
        raise ValueError("sparse feature row count does not match labels")
    store.validate_ids(data.sparse)


def latency_summary(samples_ns: list[int]) -> dict:
    samples = np.asarray(samples_ns, dtype=np.float64) / 1e6
    return {
        "p50_ms": float(np.percentile(samples, 50)),
        "p99_ms": float(np.percentile(samples, 99)),
        "mean_ms": float(samples.mean()),
        "total_seconds": float(samples.sum() / 1000),
    }


@torch.inference_mode()
def evaluate(
    model: DLRM,
    store: EmbeddingStore,
    data: CriteoArrays,
    batch_size: int = 2048,
    warmup_batches: int = 0,
) -> dict:
    """Measure all rows in order after optional replay of prefix batches.

    Lookup latency includes ID checks, row assembly and INT8 dequantization.
    Inference latency includes lookup, tensor conversion, MLPs and sigmoid.
    Warmup batches do not contribute to timings, accuracy or hit counters.
    """
    if batch_size <= 0 or warmup_batches < 0:
        raise ValueError("batch size must be positive and warmup must be nonnegative")
    validate_data(model, store, data)
    n_batches = (len(data) + batch_size - 1) // batch_size
    if warmup_batches > n_batches:
        raise ValueError("warmup exceeds the number of evaluation batches")
    prior_training = model.training
    model.eval()
    try:
        for start in range(0, min(len(data), warmup_batches * batch_size), batch_size):
            dense = torch.from_numpy(data.dense[start : start + batch_size])
            rows = store.lookup(data.sparse[start : start + batch_size])
            torch.sigmoid(model.forward_from_embeddings(dense, torch.from_numpy(rows)))
        store.reset_stats()
        probabilities = np.empty(len(data), dtype=np.float32)
        lookup_ns, inference_ns = [], []
        peak_payload = store.payload_bytes
        for start in range(0, len(data), batch_size):
            dense = torch.from_numpy(data.dense[start : start + batch_size])
            ids = data.sparse[start : start + batch_size]
            t0 = time.perf_counter_ns()
            rows = store.lookup(ids)
            t1 = time.perf_counter_ns()
            logits = model.forward_from_embeddings(dense, torch.from_numpy(rows))
            p = torch.sigmoid(logits).numpy()
            t2 = time.perf_counter_ns()
            probabilities[start : start + len(p)] = p
            lookup_ns.append(t1 - t0)
            inference_ns.append(t2 - t0)
            peak_payload = max(peak_payload, store.payload_bytes)
        if not np.isfinite(probabilities).all():
            raise ValueError("model predictions contain nonfinite values")
        return {
            "rows": len(data),
            "batches": n_batches,
            "batch_size": batch_size,
            "last_batch_rows": len(data) - (n_batches - 1) * batch_size,
            "warmup_batches": warmup_batches,
            "auc": float(roc_auc_score(data.labels, probabilities)),
            "logloss": float(log_loss(data.labels, probabilities.astype(np.float64))),
            "embedding_payload_bytes": store.payload_bytes,
            "peak_embedding_payload_bytes": peak_payload,
            "lookup_batch_latency": latency_summary(lookup_ns),
            "inference_batch_latency": latency_summary(inference_ns),
            **store.stats(),
        }
    finally:
        model.train(prior_training)
