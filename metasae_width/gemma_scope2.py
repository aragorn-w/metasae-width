"""The Gemma Scope 2 residual-stream SAEs (medium L0): one 65,536-latent SAE per model size and a 16,384-latent SAE at
the 4B and at the 270M, at the revisions the study used."""

from pathlib import Path
from typing import NamedTuple

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

ROOT = Path("/mnt/fast/metasae-width")


class Sae(NamedTuple):
    """A released SAE: its model size, layer, repository revision, latent count and width tag in its folder name."""

    model: str
    layer: int
    revision: str
    latents: int
    width_tag: str


SAES: dict[str, Sae] = {
    "270m": Sae("270m", 9, "b218cd5d69dc2fa71cff448b68d625e6c9702d49", 65536, "65k"),
    "1b": Sae("1b", 13, "b738dc06961818c011fb2e44a316352ca0f4e873", 65536, "65k"),
    "4b": Sae("4b", 17, "a0ffd6132a985bc84077a66d1a1033e10b604fa8", 65536, "65k"),
    "12b": Sae("12b", 24, "bbabd1e4a3964914f5bf0f5f99b56c2a2da09802", 65536, "65k"),
    "27b": Sae("27b", 31, "5bf702afc144f3ba57b65bb3f76b12616385a213", 65536, "65k"),
    "4b16k": Sae("4b", 17, "a0ffd6132a985bc84077a66d1a1033e10b604fa8", 16384, "16k"),
    "270m16k": Sae("270m", 9, "b218cd5d69dc2fa71cff448b68d625e6c9702d49", 16384, "16k"),
}
SIZES = ("270m", "1b", "4b", "12b", "27b")  # the model sizes the width sweep runs over


def sae_dir(size: str, root: Path = ROOT) -> Path:
    return root / "sae" / size


def load_decoder(size: str, root: Path = ROOT) -> torch.Tensor:
    """The (latents, hidden) decoder, one row per latent (files cached under root/sae)."""
    model, layer, revision, _, width_tag = SAES[size]
    repo, folder = f"google/gemma-scope-2-{model}-pt", f"resid_post/layer_{layer}_width_{width_tag}_l0_medium"
    path = hf_hub_download(
        repo, f"{folder}/params.safetensors", revision=revision, cache_dir=sae_dir(size, root) / "hf"
    )
    with safe_open(path, framework="pt") as f:
        return f.get_tensor("w_dec").contiguous()
