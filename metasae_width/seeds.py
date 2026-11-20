"""Seeds from named keys, and the numeric settings training needs."""

import hashlib

import torch


def derived_seed(key: str) -> int:
    """The low 31 bits of the SHA-256 of the key, so every random draw has its own named seed."""
    return int.from_bytes(hashlib.sha256(key.encode()).digest(), "big") & (2**31 - 1)


def set_numeric_flags() -> None:
    """Deterministic algorithms (warn only), and a refusal unless float32 matrix multiplies are full float32."""
    torch.use_deterministic_algorithms(True, warn_only=True)
    precisions = {
        "generic": torch.backends.fp32_precision,  # type: ignore[attr-defined]
        "cuda": torch.backends.cuda.matmul.fp32_precision,
        "mkldnn": torch.backends.mkldnn.matmul.fp32_precision,  # type: ignore[attr-defined]
    }
    if any(value not in ("none", "ieee") for value in precisions.values()):
        raise RuntimeError(f"float32 matrix multiplies are not full float32: {precisions}")
