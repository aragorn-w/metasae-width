"""Measurements: decoder agreement between dictionaries, and variance explained."""

from typing import NamedTuple

import torch
from scipy.optimize import linear_sum_assignment

from metasae_width.data import unit_rows


class Match(NamedTuple):
    """The mean cosine over a one-to-one matching, and the fraction of matched pairs at or above 0.7."""

    mean: float
    fraction: float


def one_to_one(dec_a: torch.Tensor, dec_b: torch.Tensor) -> Match:
    """Match the unit rows of A and B one to one (Hungarian, maximizing cosine) and score the matched pairs."""
    sim = unit_rows(dec_a) @ unit_rows(dec_b).T
    rows, cols = linear_sum_assignment(-sim.numpy())
    matched = sim[torch.as_tensor(rows, dtype=torch.long), torch.as_tensor(cols, dtype=torch.long)]
    n_close = int((matched >= torch.tensor(0.7, dtype=torch.float32)).sum().item())
    return Match(float(matched.mean().item()), n_close / matched.numel())


def many_to_one(dec_a: torch.Tensor, dec_b: torch.Tensor) -> float:
    """Each latent's best cosine in the other dictionary, averaged over A, over B, and then the two averages."""
    sim = unit_rows(dec_a) @ unit_rows(dec_b).T
    return (sim.max(dim=1).values.mean().item() + sim.max(dim=0).values.mean().item()) / 2


def variance_explained(recon: torch.Tensor, target: torch.Tensor) -> float:
    """One minus the mean squared residual over the mean squared deviation of the target from its column mean."""
    r = recon.detach().to(torch.float32, copy=False).cpu()
    t = target.detach().to(torch.float32, copy=False).cpu()
    num = (t - r).pow(2).mean()
    den = (t - t.mean(dim=0, keepdim=True)).pow(2).mean()
    return float(1.0 - (num / den).item())
