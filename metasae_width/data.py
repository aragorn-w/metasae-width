"""Rows: unit rows, the 80/10/10 split, the matched noise twin, and the rows each arm trains and scores on.

A real run trains on its parent's unit rows; its twin on Gaussian draws with their mean and covariance."""

import json
from pathlib import Path
from typing import Any, NamedTuple

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from metasae_width.seeds import derived_seed


def unit_rows(x: torch.Tensor) -> torch.Tensor:
    """Each row divided by its L2 norm, detached, in float32 on the CPU."""
    return F.normalize(x.detach().to(torch.float32, copy=False).cpu(), dim=1)


def split_rows(n: int, key: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Training, validation and test indices of n rows: one randperm seeded by the key, cut 80/10/10."""
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(derived_seed(key)))
    n_train, n_val = 80 * n // 100, 10 * n // 100
    return perm[:n_train], perm[n_train : n_train + n_val], perm[n_train + n_val :]


def _moments(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = x.mean(0)
    centered = x - mean
    return mean, centered.T @ centered / x.shape[0]


def row_moments(rows: torch.Tensor, top: int = 10) -> dict[str, Any]:
    """Norm of the mean, covariance trace and top eigenvalues of the rows (float64, divisor n)."""
    mean, cov = _moments(rows.detach().cpu().double())
    eigenvalues = torch.linalg.eigvalsh(cov).flip(0)[:top]
    return {"mean_norm": float(mean.norm()), "trace": float(cov.trace()), "top_eigenvalues": eigenvalues.tolist()}


def noise_rows(parent_rows: torch.Tensor, seed: int) -> torch.Tensor:
    """n unit rows drawn from the Gaussian with the mean and covariance of the parent's unit rows."""
    unit = unit_rows(parent_rows).double()
    mean, cov = _moments(unit)
    lam, vecs = torch.linalg.eigh(cov)
    z = torch.randn(*unit.shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    return F.normalize(mean + (z * lam.clamp_min(0).sqrt()) @ vecs.T, dim=1).float()


def save_noise(rows: torch.Tensor, folder: Path, key: str, parent: torch.Tensor) -> None:
    """Write the noise rows and a sidecar comparing their moments with the parent's."""
    train = split_rows(len(rows), f"{key}/split")[0]
    sidecar = {"key": key, "parent": row_moments(parent), "noise_train": row_moments(rows[train])}
    save_file({"rows": rows.contiguous()}, folder / "noise.safetensors")
    (folder / "noise.json").write_text(json.dumps(sidecar, indent=2) + "\n")


class Arm(NamedTuple):
    """The rows one run trains on and scores on."""

    train: torch.Tensor
    val: torch.Tensor
    test: torch.Tensor


def real_arm(parent: torch.Tensor, size: str, *, quarter: int) -> Arm:
    """The real run's rows: all of the parent split by ``<size>/split`` (quarter 0), or for quarter draw d a random
    n // 4 of them (``<size>/subset``, ``<size>/subset2``, ...) split by ``<that key>/split``."""
    if quarter == 0:
        train, val, test = split_rows(len(parent), f"{size}/split")
        return Arm(parent[train], parent[val], parent[test])
    key = f"{size}/subset" if quarter == 1 else f"{size}/subset{quarter}"
    perm = torch.randperm(len(parent), generator=torch.Generator().manual_seed(derived_seed(key)))
    kept = parent[perm[: len(parent) // 4]]
    train, val, test = split_rows(len(kept), f"{key}/split")
    return Arm(kept[train], kept[val], kept[test])


def twin_arm(parent: torch.Tensor, noise: torch.Tensor, size: str, *, quarter: int) -> Arm:
    """The twin trains on the training part of the noise rows (split by ``<size>/null/split``; for quarter draw d
    the d-th n // 4 rows of the noise, each draw its own) and scores on the real run's validation and test rows."""
    real = real_arm(parent, size, quarter=quarter)
    q = len(noise) // 4
    source = noise[(quarter - 1) * q : quarter * q] if quarter else noise
    return Arm(source[split_rows(len(source), f"{size}/null/split")[0]], real.val, real.test)
