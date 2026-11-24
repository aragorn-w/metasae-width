"""The covariance spectrum of each parent's unit rows (the SAEs' and GPT-2's) and of its noise twin's rows, in
``summary/spectrum.tsv``."""

import math
from pathlib import Path

import torch
from safetensors.torch import load_file

from metasae_width.data import _moments, unit_rows
from metasae_width.gemma_scope2 import ROOT, SAES, load_decoder, sae_dir
from metasae_width.validate import _parent_decoder, gpt2_noise

COLUMNS = (
    "parent source latents hidden latents_per_dim trace top_share participation_rank entropy_rank "
    "participation_fraction entropy_fraction"
).split()


def effective_ranks(eigenvalues: torch.Tensor) -> tuple[float, float]:
    """The participation ratio, (sum of eigenvalues) squared over the sum of their squares, and the exponential of the
    Shannon entropy of the eigenvalues over their sum (zeros dropped)."""
    p = eigenvalues.clamp_min(0) / eigenvalues.clamp_min(0).sum()
    p_nonzero = p[p > 0]
    return 1 / float((p**2).sum()), math.exp(-float((p_nonzero * p_nonzero.log()).sum()))


def spectrum_row(parent: str, source: str, rows: torch.Tensor) -> list[object]:
    """One table row for a set of unit rows: the covariance (float64, divisor n) trace, the top eigenvalue's share of
    it and the two effective ranks, which are also given as fractions of the hidden size."""
    latents, hidden = rows.shape
    cov = _moments(rows.double())[1]
    eigenvalues = torch.linalg.eigvalsh(cov)
    trace = float(cov.trace())
    participation, entropy = effective_ranks(eigenvalues)
    share = float(eigenvalues[-1]) / trace
    return [parent, source, latents, hidden, latents / hidden, trace, share, participation, entropy] + [
        participation / hidden,
        entropy / hidden,
    ]


def spectrum(root: Path = ROOT) -> str:
    """Write ``<root>/summary/spectrum.tsv``: a parent row and a noise row for every SAE in SAES and for GPT-2."""
    rows = []
    for name in SAES:
        noise = load_file(sae_dir(name, root) / "noise.safetensors")["rows"]
        rows += [spectrum_row(name, "parent", unit_rows(load_decoder(name, root))), spectrum_row(name, "noise", noise)]
    decoder = _parent_decoder()
    rows += [
        spectrum_row("gpt2", "parent", unit_rows(decoder)),
        spectrum_row("gpt2", "noise", gpt2_noise(root, decoder)),
    ]
    path = root / "summary" / "spectrum.tsv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(["\t".join(COLUMNS), *("\t".join(map(str, row)) for row in rows)]) + "\n")
    return f"wrote {path}"
