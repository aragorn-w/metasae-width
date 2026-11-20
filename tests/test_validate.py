import json

import pytest
import torch
from safetensors.torch import load_file

from metasae_width import runs, train, validate
from metasae_width.data import unit_rows
from metasae_width.models import BatchTopKSAE
from metasae_width.score import load_sae
from metasae_width.validate import LR, score_validate, validate_run
from tests.conftest import decoder


@pytest.fixture
def gpt2(tmp_path, monkeypatch, grid):
    """A root whose GPT-2 parent is a tiny decoder (64 latents of dimension 6) with a width-8 meta-SAE, snapshots every
    two epochs until 4 and then every four."""
    monkeypatch.setattr(validate, "_parent_decoder", decoder)
    monkeypatch.setattr(validate, "load_anchor", lambda: BatchTopKSAE(6, 8, 4))
    return tmp_path


def noise_of(root):
    return load_file(root / "external" / "gpt2" / "noise.safetensors")["rows"]


def test_the_validate_twin_trains_on_all_the_noise_rows_and_is_tested_on_all_the_parents_rows(gpt2, monkeypatch):
    calls, real = [], runs.train_on_rows
    monkeypatch.setattr(runs, "train_on_rows", lambda *args, **kwargs: calls.append(args) or real(*args, **kwargs))
    validate_run(0, 4, gpt2, "cpu", twin=True)
    rows, noise = unit_rows(validate._parent_decoder()), noise_of(gpt2)
    assert len(noise) == 64 and not torch.equal(noise, rows) and (gpt2 / "external" / "gpt2" / "noise.json").exists()
    assert torch.equal(calls[0][0], noise) and torch.equal(calls[0][1], rows) and torch.equal(calls[0][2], rows)
    record = json.loads((gpt2 / "runs" / "validate.twin.s0" / "record.json").read_text())
    assert record["width"] == 8 and record["settings"]["lr"] == LR
    assert "validation" not in record


def test_score_validate_fits_each_role_to_its_own_rows(gpt2):
    for seed in (0, 1, 2):
        validate_run(seed, 8, gpt2, "cpu")
        validate_run(seed, 8, gpt2, "cpu", twin=True)
    tsv = score_validate([0, 1, 2], gpt2)
    assert tsv.name == "validate.s0-1-2.e8.tsv" and len(tsv.read_text().splitlines()) == 1 + 4 * 3 * 2 + 4 * 4
    fits = json.loads(tsv.with_suffix(".json").read_text())["fit"]
    rows, noise = unit_rows(validate._parent_decoder()), noise_of(gpt2)
    twin, real = load_sae(gpt2 / "runs" / "validate.twin.s0"), load_sae(gpt2 / "runs" / "validate.s1")
    assert fits["twin"]["0"]["train_mse"] == pytest.approx(train._score(twin, noise)[0], rel=1e-5)
    assert fits["twin"]["0"]["train_mse"] != pytest.approx(train._score(twin, rows)[0], rel=1e-3)
    assert fits["real"]["1"]["train_mse"] == pytest.approx(train._score(real, rows)[0], rel=1e-5)
