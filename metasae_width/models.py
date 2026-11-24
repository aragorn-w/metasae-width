"""The BatchTopK sparse autoencoder this study trains, with its AuxK term and threshold fit.
W_enc is (dim, width) and W_dec is (width, dim)."""

from collections.abc import Iterable
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn


class Forward(NamedTuple):
    recon: torch.Tensor
    latents: torch.Tensor
    pre: torch.Tensor


class BatchTopKSAE(nn.Module):
    """Encoder relu((x - b_dec) W_enc + b_enc), decoder z W_dec + b_dec; mode ``batch_topk`` keeps the k times
    batch-size largest activations, ``threshold`` those above the fitted threshold."""

    threshold: torch.Tensor

    def __init__(self, dim: int, width: int, k: int) -> None:
        super().__init__()
        self.dim, self.width, self.k = dim, width, k
        self.W_enc = nn.Parameter(torch.zeros(dim, width))
        self.b_enc = nn.Parameter(torch.zeros(width))
        self.W_dec = nn.Parameter(torch.zeros(width, dim))
        self.b_dec = nn.Parameter(torch.zeros(dim))
        self.register_buffer("threshold", torch.zeros(width))

    def preact(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu((x - self.b_dec) @ self.W_enc + self.b_enc)

    def select(self, pre: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "batch_topk":
            return batch_topk(pre, self.k)
        return torch.where(pre > self.threshold, pre, torch.zeros_like(pre))

    def forward(self, x: torch.Tensor, mode: str) -> Forward:
        pre = self.preact(x)
        latents = self.select(pre, mode)
        return Forward(recon=latents @ self.W_dec + self.b_dec, latents=latents, pre=pre)


def batch_topk(pre: torch.Tensor, k: int) -> torch.Tensor:
    """Keep the k times batch-size largest entries of a (rows, width) batch, counted over the whole batch."""
    rows, width = pre.shape
    flat = pre.flatten()
    keep = torch.zeros(flat.shape, dtype=torch.bool, device=flat.device)
    keep[torch.topk(flat, k * rows, sorted=False).indices] = True
    return torch.where(keep, flat, torch.zeros_like(flat)).reshape(rows, width)


def auxk_loss(
    pre: torch.Tensor, residual: torch.Tensor, W_dec: torch.Tensor, dead: torch.Tensor, *, k_aux: int, weight: float
) -> torch.Tensor:
    """AuxK: the top min(k_aux, #dead) activations of dead latents reconstruct the residual; zero if none is dead."""
    n_dead = int(dead.sum().item())
    if n_dead == 0:
        return pre.new_zeros(())
    acts = pre[:, dead]
    top = torch.topk(acts, min(k_aux, n_dead), dim=1)
    sparse = torch.zeros_like(acts).scatter(1, top.indices, top.values)
    return weight * (sparse @ W_dec[dead] - residual).pow(2).mean()


@torch.no_grad()
def fit_threshold_(sae: BatchTopKSAE, batches: Iterable[torch.Tensor]) -> float:
    """Set the shared threshold to the mean over batches of the smallest positive batch-top-k activation (Bussmann,
    Leask and Nanda, arXiv:2412.06410)."""
    minima = []
    for x in batches:
        selected = batch_topk(sae.preact(x), sae.k)
        positive = selected[selected > 0]
        if positive.numel() > 0:
            minima.append(positive.min())
    theta = torch.stack(minima).mean()
    sae.threshold.fill_(theta.item())
    return float(theta.item())
