"""Offline FP32 -> INT8 quantizer for embedding tables, one scale per row."""

from __future__ import annotations

import numpy as np


def quantize_int8(table: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric per-row INT8 quantization.

    Returns (q, scale) where q has dtype int8 and the same shape as `table`,
    and scale has shape (rows,) such that table ~= q * scale[:, None].
    """
    table = np.asarray(table, dtype=np.float32)
    if table.ndim != 2:
        raise ValueError(f"expected a 2-D table, got shape {table.shape}")
    max_abs = np.abs(table).max(axis=1)
    scale = np.where(max_abs > 0, max_abs / 127.0, 1.0).astype(np.float32)
    q = np.clip(np.rint(table / scale[:, None]), -127, 127).astype(np.int8)
    return q, scale


def dequantize_int8(q: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return q.astype(np.float32) * scale[:, None]
