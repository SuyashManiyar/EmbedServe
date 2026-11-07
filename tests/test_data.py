import numpy as np
import pytest

from embedserve.data import (
    COLUMNS,
    N_DENSE,
    N_SPARSE,
    CriteoArrays,
    Vocab,
    access_skew,
    apply_vocab,
    build_vocab,
    encode,
    read_csv,
)


def write_csv(path, rows):
    lines = [",".join(COLUMNS)] + [",".join(str(v) for v in r) for r in rows]
    path.write_text("\n".join(lines) + "\n")


def make_rows(n, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        dense = np.round(rng.random(N_DENSE), 4).tolist()
        # field j uses IDs from its own range, like the shared ID space in Criteo_x1
        sparse = [1000 * j + int(rng.zipf(2.0)) % 50 for j in range(N_SPARSE)]
        rows.append([int(rng.random() < 0.25), *dense, *sparse])
    return rows


def test_read_csv_parses_columns(tmp_path):
    rows = make_rows(7)
    write_csv(tmp_path / "a.csv", rows)
    data = read_csv(tmp_path / "a.csv", chunksize=3)  # several chunks
    assert len(data) == 7
    assert data.dense.shape == (7, N_DENSE) and data.sparse.shape == (7, N_SPARSE)
    np.testing.assert_array_equal(data.labels, [r[0] for r in rows])
    np.testing.assert_allclose(data.dense[2], rows[2][1 : 1 + N_DENSE], rtol=1e-6)
    np.testing.assert_array_equal(data.sparse[4], rows[4][1 + N_DENSE :])


def test_read_csv_nrows_and_no_trailing_newline(tmp_path):
    rows = make_rows(5)
    path = tmp_path / "a.csv"
    write_csv(path, rows)
    path.write_text(path.read_text().rstrip("\n"))
    assert len(read_csv(path)) == 5
    assert len(read_csv(path, nrows=2)) == 2


def test_read_csv_rejects_wrong_header(tmp_path):
    path = tmp_path / "a.csv"
    path.write_text("0,1,2\n")
    with pytest.raises(ValueError):
        read_csv(path)


def test_vocab_drops_rare_ids_and_renumbers():
    raw = np.array([[10], [10], [10], [20], [20], [30]], dtype=np.int32)
    vocab = build_vocab(raw, min_count=2)
    assert vocab.sizes == [3]  # row 0 + IDs 10 and 20
    np.testing.assert_array_equal(encode(vocab, raw)[:, 0], [1, 1, 1, 2, 2, 0])
    # unseen IDs, including ones below and above every kept key, go to row 0
    np.testing.assert_array_equal(encode(vocab, np.array([[5], [20], [15], [99]]))[:, 0], [0, 2, 0, 0])


def test_vocab_with_empty_field():
    raw = np.array([[1, 7], [1, 8]], dtype=np.int32)
    vocab = build_vocab(raw, min_count=2)
    assert vocab.sizes == [2, 1]
    np.testing.assert_array_equal(encode(vocab, raw), [[1, 0], [1, 0]])


def test_save_and_load_roundtrip(tmp_path):
    write_csv(tmp_path / "a.csv", make_rows(20))
    raw = read_csv(tmp_path / "a.csv")
    vocab = build_vocab(raw.sparse, min_count=2)
    data = apply_vocab(raw, vocab)
    data.save(tmp_path / "d.npz")
    vocab.save(tmp_path / "v.npz")
    loaded, lvocab = CriteoArrays.load(tmp_path / "d.npz"), Vocab.load(tmp_path / "v.npz")
    np.testing.assert_array_equal(loaded.sparse, data.sparse)
    np.testing.assert_array_equal(loaded.dense, data.dense)
    assert lvocab.sizes == vocab.sizes and lvocab.min_count == 2
    assert np.all(data.sparse.max(axis=0) < np.array(vocab.sizes))


def test_access_skew():
    # table 0 has 4 rows: row 1 gets 7 of 10 lookups; table 1 has 2 rows, all on row 0
    sparse = np.array([[1, 0]] * 7 + [[2, 0], [3, 0], [0, 0]], dtype=np.int32)
    skew = access_skew(sparse, sizes=[4, 2], fractions=(1 / 6, 2 / 6, 1.0))
    assert skew["total_rows"] == 6 and skew["lookups"] == 20
    assert skew["rows_never_accessed"] == 1  # table 1, row 1
    assert skew["rare_bucket_share"] == pytest.approx(11 / 20)
    assert skew["coverage"][1 / 6] == pytest.approx(10 / 20)  # table 1 row 0
    assert skew["coverage"][2 / 6] == pytest.approx(17 / 20)
    assert skew["coverage"][1.0] == pytest.approx(1.0)
