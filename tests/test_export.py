import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from embedserve.export import export_store, load_meta, load_mlp, open_store
from embedserve.model import DLRM
from embedserve.serving import FP32Store, INT8Store

SIZES = [5, 7]
IDS = np.array([[0, 6], [4, 0], [0, 6], [3, 2]])


@pytest.fixture
def exported(tmp_path):
    torch.manual_seed(4)
    model = DLRM(SIZES, dim=4, bottom=(8,), top=(8,))
    checkpoint = tmp_path / "dlrm.pt"
    torch.save({"config": {"table_sizes": SIZES, "dim": 4, "bottom": (8,), "top": (8,)},
                "state_dict": model.state_dict()}, checkpoint)
    root = tmp_path / "store"
    assert export_store(checkpoint, root) == {"table_sizes": SIZES, "dim": 4}
    return model, root


def test_mlp_loads_without_embedding_tables(exported):
    model, root = exported
    mlp = load_mlp(root)
    assert len(mlp.embeddings) == 0 and not mlp.training
    assert load_meta(root) == {"table_sizes": SIZES, "dim": 4}
    dense = torch.rand(4, 13)
    with torch.inference_mode():
        rows = torch.from_numpy(open_store(root, "fp32").lookup(IDS))
        torch.testing.assert_close(mlp.forward_from_embeddings(dense, rows),
                                   model(dense, torch.from_numpy(IDS)))


@pytest.mark.parametrize("backend", ["python", "cpp"])
def test_opened_stores_match_in_memory_stores(exported, backend):
    model, root = exported
    tables = [e.weight.detach().numpy() for e in model.embeddings]
    fp32, int8 = FP32Store(tables).lookup(IDS), INT8Store(tables).lookup(IDS)
    np.testing.assert_array_equal(open_store(root, "fp32").lookup(IDS), fp32)
    np.testing.assert_array_equal(open_store(root, "int8").lookup(IDS), int8)
    for row_format, expected in (("fp32", fp32), ("int8", int8)):
        for mmap_reads in (True, False):
            with open_store(root, "two-tier", 3, backend, row_format, mmap_reads) as store:
                np.testing.assert_array_equal(store.lookup(IDS), expected)
                np.testing.assert_array_equal(store.lookup(IDS), expected)
                assert store.stats()["hot_rows"] == 3
    assert (root / "tiers-fp32").is_dir()  # opened stores leave the export in place


def test_open_rejects_bad_arguments(exported):
    _, root = exported
    with pytest.raises(ValueError):
        open_store(root, "int4")
    with pytest.raises(ValueError):
        open_store(root, "two-tier", -1)


def test_memory_cli(exported, tmp_path):
    from embedserve.data import CriteoArrays

    _, root = exported
    rng = np.random.default_rng(0)
    CriteoArrays(np.array([0, 1, 1, 0], dtype=np.float32), rng.random((4, 13), dtype=np.float32),
                 IDS.astype(np.int32)).save(tmp_path / "valid.npz")
    script = Path(__file__).resolve().parents[1] / "scripts" / "measure_memory.py"
    out = tmp_path / "memory.json"
    subprocess.run([sys.executable, str(script), "--store", str(root), "--data",
                    str(tmp_path / "valid.npz"), "--hot-capacities", "3", "--row-formats", "int8",
                    "--batch-size", "2", "--threads", "1", "--out", str(out)],
                   check=True, capture_output=True, text=True)
    runs = json.loads(out.read_text())["runs"]
    assert [r["mode"] for r in runs] == ["none", "fp32", "int8", "two-tier"]
    assert all(r["peak_footprint_bytes"] > 0 and r["peak_rss_bytes"] > 0 for r in runs)
    assert runs[3]["embedding_payload_bytes"] == 3 * (4 + 4)
    assert "data" not in runs[0]
