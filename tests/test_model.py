import pytest

torch = pytest.importorskip("torch")

from embedserve.model import DLRM  # noqa: E402

SIZES = [5, 7, 3]


def batch(n=4):
    g = torch.Generator().manual_seed(0)
    dense = torch.rand(n, 13, generator=g)
    sparse = torch.stack([torch.randint(0, s, (n,), generator=g) for s in SIZES], dim=1)
    return dense, sparse


def test_output_shape():
    model = DLRM(SIZES, dim=8, bottom=(16,), top=(16,))
    dense, sparse = batch()
    assert model(dense, sparse).shape == (4,)


def test_forward_from_embeddings_matches_forward():
    model = DLRM(SIZES, dim=8, bottom=(16,), top=(16,))
    dense, sparse = batch()
    embs = model.lookup(sparse)
    assert embs.shape == (4, 3, 8)
    torch.testing.assert_close(model.forward_from_embeddings(dense, embs), model(dense, sparse))


def test_training_step_reduces_loss():
    torch.manual_seed(0)
    model = DLRM(SIZES, dim=8, bottom=(16,), top=(16,))
    dense, sparse = batch(64)
    labels = (sparse[:, 0] % 2).float()  # learnable from the first table alone
    opts = [torch.optim.Adam(model.dense_parameters(), lr=1e-2),
            torch.optim.SparseAdam(model.embedding_parameters(), lr=1e-2)]
    loss_fn = torch.nn.BCEWithLogitsLoss()
    first = loss_fn(model(dense, sparse), labels).item()
    for _ in range(100):
        loss = loss_fn(model(dense, sparse), labels)
        for o in opts:
            o.zero_grad()
        loss.backward()
        for o in opts:
            o.step()
    assert loss.item() < first / 2
