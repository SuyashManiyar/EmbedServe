import numpy as np

from embedserve.quantize import dequantize_int8, quantize_int8


def test_roundtrip_error_within_half_step():
    rng = np.random.default_rng(0)
    table = rng.normal(size=(100, 16)).astype(np.float32)
    q, scale = quantize_int8(table)
    assert q.dtype == np.int8 and scale.shape == (100,)
    err = np.abs(dequantize_int8(q, scale) - table)
    assert np.all(err <= scale[:, None] / 2 + 1e-6)


def test_zero_row():
    q, scale = quantize_int8(np.zeros((1, 4), dtype=np.float32))
    assert np.all(q == 0) and scale[0] == 1.0
