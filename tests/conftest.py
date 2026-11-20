"""Fixtures: a tiny stand-in for a Gemma Scope 2 SAE (64 latents of dimension 6) so everything runs on the CPU."""

from pathlib import Path

import pytest
import torch

from metasae_width import runs, train
from metasae_width.data import real_arm, unit_rows
from metasae_width.runs import train_run
from metasae_width.train import train_on_rows


def decoder() -> torch.Tensor:
    rows = torch.randn(64, 6, generator=torch.Generator().manual_seed(0))
    return rows * torch.linspace(0.5, 3.0, 6)  # an anisotropic dictionary


@pytest.fixture
def arm():
    return real_arm(unit_rows(decoder()), "4b", quarter=0)


@pytest.fixture
def fit(arm):
    """Train the tiny dictionary (width 8) on the CPU; extra keyword arguments go to ``train_on_rows``."""

    def run(epochs: int = 4, **kwargs):
        kwargs = {"run_id": "t.s0", "width": 8, "lr": 3e-3, "device": "cpu"} | kwargs
        return train_on_rows(arm.train, arm.val, arm.test, epochs=epochs, **kwargs)

    return run


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A root whose only SAE, 4b, is the tiny one, with its noise twin drawn."""
    monkeypatch.setattr(runs, "load_decoder", lambda size, root=None: decoder())
    (tmp_path / "sae" / "4b").mkdir(parents=True)
    runs.check_sae("4b", tmp_path)
    return tmp_path


@pytest.fixture
def grid(monkeypatch: pytest.MonkeyPatch) -> None:
    """Snapshots every two epochs through epoch 4, then every four."""
    for name, value in (("SNAPSHOT_EARLY_EVERY", 2), ("SNAPSHOT_EARLY_UNTIL", 4), ("SNAPSHOT_EVERY", 4)):
        monkeypatch.setattr(train, name, value)


@pytest.fixture
def trained(root: Path, grid: None) -> Path:
    """The root with seeds 0 to 2 of the real and twin 4b width-8 runs trained for 8 epochs, one step each, with
    snapshots at epochs 2 and 4."""
    for role in ("real", "twin"):
        for seed in (0, 1, 2):
            train_run(f"4b.w8.{role}.s{seed}", 3e-3, epochs=8, root=root, device="cpu")
    return root
