"""Validate the trainer's recipe on the released Leask et al. meta-SAE of GPT-2 small and its parent, and train and
score the noise twin of that parent for the same configuration."""

import json
from fractions import Fraction
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from metasae_width.checks import dec
from metasae_width.data import Arm, noise_rows, save_noise, unit_rows
from metasae_width.gemma_scope2 import ROOT
from metasae_width.metrics import variance_explained
from metasae_width.models import BatchTopKSAE
from metasae_width.runs import save_run, train_into
from metasae_width.score import score_runs, source
from metasae_width.seeds import derived_seed
from metasae_width.train import EPOCHS

EXTERNAL_CACHE = ROOT / "external" / "hf"
TOLERANCE = Fraction("0.03")
LR = 1e-3  # the rate in the released meta-SAE's name and config
# (repository, file, revision) of the released meta-SAE and of its parent.
ANCHOR = (
    "pleask/gpt2-small_blocks.8.hook_resid_pre_2304_topk_4_0.001_2000",
    "93a3e3d19e5031673b3699e188c64bcc92234f86",
)
PARENT = ("jbloom/GPT2-Small-Feature-Splitting-Experiment-Layer-8", "174d298f1633aab71e440685c3292ce8108a5a73")
PARENT_FOLDER = "blocks.8.hook_resid_pre_49152"


def _fetch(repo_revision: tuple[str, str], filename: str) -> Path:
    repo, revision = repo_revision
    return Path(hf_hub_download(repo, filename, revision=revision, cache_dir=EXTERNAL_CACHE))


def _parent_decoder() -> torch.Tensor:
    """The parent's decoder, one row per latent, as released."""
    return load_file(_fetch(PARENT, f"{PARENT_FOLDER}/sae_weights.safetensors"))["W_dec"]


def load_anchor() -> BatchTopKSAE:
    """The released meta-SAE: its encoder is relu((x - b_dec) W_enc) with no bias, k is its ``top_k``."""
    cfg = json.loads(_fetch(ANCHOR, "Config.json").read_text())
    sae = BatchTopKSAE(cfg["act_size"], cfg["dict_size"], cfg["top_k"])
    state = torch.load(_fetch(ANCHOR, "SAE.pt"), map_location="cpu", weights_only=True)
    with torch.no_grad():
        for name in ("W_enc", "b_enc", "W_dec", "b_dec"):
            getattr(sae, name).copy_(state[name])
    return sae


def validate_run(
    seed: int, epochs: int = EPOCHS, root: Path = ROOT, device: str = "cuda", *, twin: bool = False
) -> bool:
    """Train a meta-SAE at the anchor's width and rate into ``<root>/runs/validate.s<seed>``, on the anchor's parent's
    unit rows. It passes if its variance explained on them is within 0.03 of the anchor's own. With ``twin``, train it
    on all the parent's noise rows instead, validated and tested on the parent's unit rows, into
    ``<root>/runs/validate.twin.s<seed>``; that run is not compared and always passes."""
    run_id = f"validate.twin.s{seed}" if twin else f"validate.s{seed}"
    folder = root / "runs" / run_id
    decoder = _parent_decoder()
    rows, anchor = unit_rows(decoder), load_anchor()
    train = gpt2_noise(root, decoder) if twin else rows
    arm = Arm(train, rows, rows)
    trained, record = train_into(folder, run_id, arm, width=anchor.width, lr=LR, epochs=epochs, device=device)
    passed = True
    if not twin:
        ve = trained.record["final"]["val_ve"]
        with torch.no_grad():
            anchor_ve = variance_explained(anchor.to(device)(rows.to(device), "batch_topk").recon, rows)
        passed = abs(dec(ve) - dec(anchor_ve)) <= TOLERANCE
        record["validation"] = {"anchor_ve": anchor_ve, "ve": ve, "tolerance": float(TOLERANCE), "passed": passed}
        verdict, dead = "PASS" if passed else "FAIL", trained.record["dead_fraction"]
        print(
            f"{verdict}: variance explained {ve:.6f}, anchor {anchor_ve:.6f}, tolerance 0.03, dead fraction {dead:.4f}"
        )
    save_run(folder, trained, record)
    return passed


def gpt2_noise(root: Path, decoder: torch.Tensor) -> torch.Tensor:
    """The noise twin of the parent's decoder rows, from ``<root>/external/gpt2/noise.safetensors``; drawn and saved
    there first if absent."""
    folder = root / "external" / "gpt2"
    if not (folder / "noise.safetensors").exists():
        folder.mkdir(parents=True, exist_ok=True)
        save_noise(noise_rows(decoder, derived_seed("gpt2/null")), folder, "gpt2/null", unit_rows(decoder))
    return load_file(folder / "noise.safetensors")["rows"]


def score_validate(seeds: list[int], root: Path = ROOT, *, workers: int = 1) -> Path:
    """Score the ``validate.s<seed>`` runs against their ``validate.twin.s<seed>`` runs as ``score`` scores an arm,
    with the fit on the parent's unit rows (real) and the noise rows (twin). Writes
    ``<root>/scores/validate.s<seeds>.e<epochs>.tsv`` and ``.json``."""
    names = {"real": "validate", "twin": "validate.twin"}
    sources = {role: [source(root / "runs" / f"{name}.s{s}", None) for s in seeds] for role, name in names.items()}
    decoder = _parent_decoder()
    rows = {"real": unit_rows(decoder), "twin": gpt2_noise(root, decoder)}
    return score_runs("validate", seeds, sources, lambda role: rows[role], root, workers)
