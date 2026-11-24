import pytest
import torch

from metasae_width import runs
from metasae_width import spectrum as spectrum_module
from metasae_width.gemma_scope2 import SAES
from metasae_width.spectrum import effective_ranks, spectrum, spectrum_row


def test_effective_ranks_and_the_spectrum_row_on_a_known_diagonal_covariance():
    assert effective_ranks(torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0], dtype=torch.float64)) == (4.0, 4.0)
    participation, entropy = effective_ranks(torch.tensor([2.0, 1.0, 1.0], dtype=torch.float64))
    assert participation == pytest.approx(16 / 6) and entropy == pytest.approx(2**1.5)
    rows = torch.tensor([[1.0, 0, 0], [-1.0, 0, 0], [0, 1.0, 0], [0, -1.0, 0]])  # covariance diag(1/2, 1/2, 0)
    row = spectrum_row("x", "parent", rows)
    assert row[:4] == ["x", "parent", 4, 3] and row[4] == pytest.approx(4 / 3)
    assert row[5:] == pytest.approx([1.0, 0.5, 2.0, 2.0, 2 / 3, 2 / 3])


def test_the_spectrum_command_writes_a_row_per_source_for_each_parent_and_for_gpt2(root, monkeypatch):
    monkeypatch.setattr(spectrum_module, "SAES", {"4b": SAES["4b"]})
    monkeypatch.setattr(spectrum_module, "load_decoder", runs.load_decoder)  # the fixture's tiny 4b
    monkeypatch.setattr(spectrum_module, "_parent_decoder", lambda: runs.load_decoder("4b"))  # also its GPT-2
    spectrum(root)
    lines = (root / "summary" / "spectrum.tsv").read_text().splitlines()
    assert [line.split("\t")[:3] for line in lines[1:]] == [
        ["4b", "parent", "64"],
        ["4b", "noise", "64"],
        ["gpt2", "parent", "64"],
        ["gpt2", "noise", "64"],
    ]
