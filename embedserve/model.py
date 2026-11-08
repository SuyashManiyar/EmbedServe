"""A small DLRM (Naumov et al., 2019) for Criteo.

    dense (13) ─► bottom MLP ─► x (dim)
    sparse (26 IDs) ─► 26 embedding tables ─► e_1..e_26 (dim each)
    pairwise dot products of [x, e_1..e_26] (351) ++ x ─► top MLP ─► logit

`forward_from_embeddings` takes the looked-up rows directly, so serving can fetch
them from any store (FP32 in RAM, INT8, or the two-tier store) and reuse the
same trained MLPs.
"""

from __future__ import annotations

import torch
from torch import nn


def mlp(sizes: list[int], last_activation: bool) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or last_activation:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class DLRM(nn.Module):
    def __init__(
        self,
        table_sizes: list[int],
        dim: int = 64,
        n_dense: int = 13,
        bottom: tuple[int, ...] = (512, 256),
        top: tuple[int, ...] = (512, 256),
    ):
        super().__init__()
        self.table_sizes = list(table_sizes)
        self.dim = dim
        self.embeddings = nn.ModuleList(nn.Embedding(n, dim, sparse=True) for n in table_sizes)
        for emb in self.embeddings:
            # DLRM reference init: U(-1/sqrt(n), 1/sqrt(n))
            bound = 1.0 / emb.num_embeddings**0.5
            nn.init.uniform_(emb.weight, -bound, bound)
        self.bottom = mlp([n_dense, *bottom, dim], last_activation=True)
        n_vec = len(table_sizes) + 1
        n_pairs = n_vec * (n_vec - 1) // 2
        self.top = mlp([dim + n_pairs, *top, 1], last_activation=False)
        rows, cols = torch.triu_indices(n_vec, n_vec, offset=1)
        self.register_buffer("_rows", rows, persistent=False)
        self.register_buffer("_cols", cols, persistent=False)

    def lookup(self, sparse: torch.Tensor) -> torch.Tensor:
        """(batch, 26) row IDs -> (batch, 26, dim) embedding rows."""
        return torch.stack([emb(sparse[:, j]) for j, emb in enumerate(self.embeddings)], dim=1)

    def forward_from_embeddings(self, dense: torch.Tensor, embs: torch.Tensor) -> torch.Tensor:
        """dense (batch, 13), embs (batch, 26, dim) -> logits (batch,)."""
        x = self.bottom(dense)
        vecs = torch.cat([x.unsqueeze(1), embs], dim=1)  # (batch, 27, dim)
        dots = torch.bmm(vecs, vecs.transpose(1, 2))[:, self._rows, self._cols]
        return self.top(torch.cat([x, dots], dim=1)).squeeze(1)

    def forward(self, dense: torch.Tensor, sparse: torch.Tensor) -> torch.Tensor:
        return self.forward_from_embeddings(dense, self.lookup(sparse))

    def dense_parameters(self):
        return [*self.bottom.parameters(), *self.top.parameters()]

    def embedding_parameters(self):
        return list(self.embeddings.parameters())
