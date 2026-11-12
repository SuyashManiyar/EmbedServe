import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")
from sklearn.metrics import log_loss, roc_auc_score

from embedserve.data import CriteoArrays
from embedserve.evaluation import evaluate, load_checkpoint
from embedserve.model import DLRM
from embedserve.serving import FP32Store, INT8Store, TwoTierStore


@pytest.fixture
def sample():
    torch.manual_seed(3)
    model = DLRM([5, 7], dim=4, bottom=(8,), top=(8,))
    rng = np.random.default_rng(3)
    data = CriteoArrays(
        np.array([0, 1, 1, 0, 1], dtype=np.float32),
        rng.random((5, 13), dtype=np.float32),
        np.array([[0, 1], [4, 6], [2, 3], [0, 0], [4, 6]], dtype=np.int32),
    )
    return model, data


def tables(model):
    return [e.weight.detach().numpy() for e in model.embeddings]


@pytest.mark.parametrize("mode", ["fp32", "int8", "two-tier"])
def test_metrics_match_direct_model_with_same_tables(sample, tmp_path, mode):
    model, data = sample
    source = tables(model)
    if mode == "int8":
        store = INT8Store(source)
        reference = DLRM([5, 7], dim=4, bottom=(8,), top=(8,))
        reference.load_state_dict(model.state_dict())
        with torch.no_grad():
            for emb, (q, scale) in zip(reference.embeddings, store.tables):
                emb.weight.copy_(torch.from_numpy(q.astype(np.float32) * scale[:, None]))
    else:
        store = FP32Store(source) if mode == "fp32" else TwoTierStore(source, 3, tmp_path)
        reference = model
    with store, torch.inference_mode():
        p = torch.sigmoid(reference(torch.from_numpy(data.dense), torch.from_numpy(data.sparse))).numpy()
        result = evaluate(model, store, data, batch_size=2)
    assert result["rows"] == 5
    assert result["batches"] == 3
    assert result["last_batch_rows"] == 1
    assert result["auc"] == pytest.approx(roc_auc_score(data.labels, p))
    assert result["logloss"] == pytest.approx(log_loss(data.labels, p.astype(np.float64)))
    assert model.training
    if mode == "two-tier":
        assert result["hits"] + result["misses"] == 10
    else:
        assert result["hit_rate"] is None


def test_warmup_excluded_from_counters_and_timings(sample, tmp_path, monkeypatch):
    model, data = sample
    ticks = iter([0, 2_000_000, 5_000_000, 10_000_000, 14_000_000, 19_000_000,
                  20_000_000, 26_000_000, 33_000_000])
    monkeypatch.setattr("embedserve.evaluation.time.perf_counter_ns", lambda: next(ticks))
    with TwoTierStore(tables(model), 12, tmp_path) as store:
        result = evaluate(model, store, data, batch_size=2, warmup_batches=3)
    assert result["hits"] == 10
    assert result["misses"] == 0
    assert result["lookup_batch_latency"]["p50_ms"] == 4
    assert result["lookup_batch_latency"]["p99_ms"] == pytest.approx(5.96)
    assert result["inference_batch_latency"]["p50_ms"] == 9
    assert result["inference_batch_latency"]["total_seconds"] == pytest.approx(0.027)


@pytest.mark.parametrize("batch,warmup", [(0, 0), (-1, 0), (2, -1), (2, 4)])
def test_invalid_settings(sample, batch, warmup):
    model, data = sample
    with pytest.raises(ValueError):
        evaluate(model, FP32Store(tables(model)), data, batch, warmup)


@pytest.mark.parametrize("kind", ["empty", "one-class", "nan", "bad-ids", "dense-shape"])
def test_invalid_data(sample, kind):
    model, data = sample
    if kind == "empty":
        data = CriteoArrays(data.labels[:0], data.dense[:0], data.sparse[:0])
    elif kind == "one-class":
        data.labels[:] = 0
    elif kind == "nan":
        data.dense[0, 0] = np.nan
    elif kind == "bad-ids":
        data.sparse[0, 0] = -1
    else:
        data.dense = data.dense[:, :1]
    with pytest.raises(ValueError):
        evaluate(model, FP32Store(tables(model)), data)


def save_sample(sample, tmp_path):
    model, data = sample
    checkpoint = tmp_path / "dlrm.pt"
    torch.save({"config": {"table_sizes": [5, 7], "dim": 4, "bottom": (8,), "top": (8,)},
                "state_dict": model.state_dict()}, checkpoint)
    data.save(tmp_path / "valid.npz")
    return checkpoint


def test_checkpoint_releases_model_embeddings(sample, tmp_path):
    checkpoint = save_sample(sample, tmp_path)
    loaded, source = load_checkpoint(checkpoint)
    assert len(loaded.embeddings) == 0
    assert not loaded.training
    model, data = sample
    with torch.inference_mode():
        expected = model(torch.from_numpy(data.dense), torch.from_numpy(data.sparse))
        actual = loaded.forward_from_embeddings(
            torch.from_numpy(data.dense), torch.from_numpy(FP32Store(source).lookup(data.sparse)))
    torch.testing.assert_close(actual, expected)


def test_cli_outputs_all_modes_and_cleans_databases(sample, tmp_path):
    checkpoint = save_sample(sample, tmp_path)
    script = Path(__file__).resolve().parents[1] / "scripts" / "evaluate.py"
    out = tmp_path / "results" / "evaluation.json"
    root = tmp_path / "databases"
    command = [sys.executable, str(script), "--model", str(checkpoint), "--data",
               str(tmp_path / "valid.npz"), "--out", str(out), "--db-root", str(root),
               "--hot-capacities", "0", "12", "--batch-size", "2", "--threads", "1"]
    subprocess.run(command, check=True, capture_output=True, text=True)
    report = json.loads(out.read_text())
    assert [r["mode"] for r in report["runs"]] == ["fp32", "int8", "two-tier", "two-tier"]
    assert len(report["model"]["sha256"]) == 64
    assert len(report["data"]["sha256"]) == 64
    assert report["data"]["source_rows"] == 5
    assert report["environment"]["threads"] == 1
    assert report["runs"][0]["auc"] == report["runs"][2]["auc"]
    assert list(root.iterdir()) == []
    rejected = subprocess.run(command + ["--rows", "0"], capture_output=True, text=True)
    assert rejected.returncode != 0
    assert "--rows must be positive" in rejected.stderr
