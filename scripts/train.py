"""Train the DLRM once on the encoded Criteo subset and save it.

Example:
    python scripts/train.py                      # all of data/processed/train.npz
    python scripts/train.py --train-rows 100000  # quick smoke run

Writes <out>/dlrm.pt (config + weights), <out>/embeddings.npz (one FP32 array per
table, the input to the serving stores) and <out>/metrics.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import log_loss, roc_auc_score
from torch import nn

from embedserve.data import CriteoArrays, Vocab
from embedserve.model import DLRM


@torch.no_grad()
def predict(model: DLRM, data: CriteoArrays, batch_size: int = 16384) -> np.ndarray:
    model.eval()
    dense, sparse = torch.from_numpy(data.dense), torch.from_numpy(data.sparse)
    out = [torch.sigmoid(model(dense[i : i + batch_size], sparse[i : i + batch_size]))
           for i in range(0, len(data), batch_size)]
    model.train()
    return torch.cat(out).numpy()


def evaluate(model: DLRM, data: CriteoArrays) -> dict:
    p = predict(model, data)
    return {"auc": float(roc_auc_score(data.labels, p)), "logloss": float(log_loss(data.labels, p.astype(np.float64)))}


def subset(data: CriteoArrays, n: int | None) -> CriteoArrays:
    if n is None or n >= len(data):
        return data
    return CriteoArrays(data.labels[:n], data.dense[:n], data.sparse[:n])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=Path("data/processed"))
    p.add_argument("--out", type=Path, default=Path("data/model"))
    p.add_argument("--train-rows", type=int)
    p.add_argument("--valid-rows", type=int)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--eval-every", type=int, default=500, help="steps between progress lines")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    vocab = Vocab.load(args.data / "vocab.npz")
    train = subset(CriteoArrays.load(args.data / "train.npz"), args.train_rows)
    valid = subset(CriteoArrays.load(args.data / "valid.npz"), args.valid_rows)
    print(f"train {len(train):,} rows, valid {len(valid):,} rows, {sum(vocab.sizes):,} embedding rows, "
          f"{torch.get_num_threads()} threads")

    model = DLRM(vocab.sizes, dim=args.dim)
    dense_opt = torch.optim.Adam(model.dense_parameters(), lr=args.lr)
    sparse_opt = torch.optim.SparseAdam(model.embedding_parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()
    labels, dense, sparse = (torch.from_numpy(a) for a in (train.labels, train.dense, train.sparse))

    t0 = time.perf_counter()
    step, running = 0, 0.0
    for epoch in range(args.epochs):
        order = torch.from_numpy(rng.permutation(len(train)))
        for i in range(0, len(train), args.batch_size):
            idx = order[i : i + args.batch_size]
            loss = loss_fn(model(dense[idx], sparse[idx]), labels[idx])
            dense_opt.zero_grad()
            sparse_opt.zero_grad()
            loss.backward()
            dense_opt.step()
            sparse_opt.step()
            step += 1
            running += loss.item()
            if step % args.eval_every == 0:
                rows_done = min(i + args.batch_size, len(train))
                rate = (epoch * len(train) + rows_done) / (time.perf_counter() - t0)
                print(f"epoch {epoch} step {step:>6} rows {rows_done:>9,} "
                      f"loss {running / args.eval_every:.4f}  {rate:,.0f} rows/s", flush=True)
                running = 0.0
        metrics = evaluate(model, valid)
        print(f"epoch {epoch} done in {time.perf_counter() - t0:.0f}s: "
              f"valid AUC {metrics['auc']:.4f}, logloss {metrics['logloss']:.4f}", flush=True)

    train_seconds = time.perf_counter() - t0
    config = {"table_sizes": vocab.sizes, "dim": args.dim}
    torch.save({"config": config, "state_dict": model.state_dict()}, args.out / "dlrm.pt")
    np.savez(args.out / "embeddings.npz",
             **{f"table_{j}": emb.weight.detach().numpy() for j, emb in enumerate(model.embeddings)})
    settings = {k: v for k, v in vars(args).items() if k not in ("data", "out", "train_rows", "valid_rows")}
    summary = {**metrics, "train_seconds": train_seconds, "train_rows": len(train),
               "valid_rows": len(valid), **settings}
    (args.out / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {args.out}/")


if __name__ == "__main__":
    main()
