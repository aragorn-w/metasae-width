"""The two checks on a width: the shared learning rate (tune-pick) and the length rule (extend-check)."""

import itertools
from fractions import Fraction
from pathlib import Path

import numpy as np
from safetensors.torch import load_file
from torch import Tensor

from metasae_width.gemma_scope2 import ROOT
from metasae_width.metrics import one_to_one
from metasae_width.runs import TUNING_SEED, RunId, lr_tag, parse_arm, read_log, read_record, run_folder

# One learning rate for every size, width, seed and role; the tuning runs show how far it is from the best of three.
SHARED_LR = 3e-3
CHECK_LRS = (1e-3, SHARED_LR, 1e-2)
MSE_FALL = Fraction("0.01")
AGREEMENT_RISE = Fraction("0.005")


def dec(x: float) -> Fraction:
    """The exact decimal value of a float's shortest round-trip string, so thresholds are decimal rules."""
    return Fraction(repr(float(x)))


def tune_pick(arm: str, denominator: int, root: Path = ROOT) -> tuple[dict[float, float], str]:
    """Each rate's validation MSE at the last step of the shortest finished tuning run (seed 100, rates 1e-3, 3e-3,
    1e-2), and how far the shared rate is above the best."""
    size, quarter = parse_arm(arm)
    folders = {lr: run_folder(root, RunId(size, quarter, denominator, "real", TUNING_SEED), lr) for lr in CHECK_LRS}
    for folder in folders.values():
        read_record(folder)  # a run still training has no record yet
    logs = {lr: read_log(folder / "log.tsv") for lr, folder in folders.items()}
    step = min(log[-1].step for log in logs.values())
    mse = {lr: {row.step: row.val_mse for row in log}[step] for lr, log in logs.items()}
    best = min(mse, key=lambda lr: (mse[lr], lr))
    shown = ", ".join(f"{lr_tag(lr)} {value:.6g}" for lr, value in mse.items())
    above = f"{lr_tag(SHARED_LR)} is {mse[SHARED_LR] / mse[best] - 1:.2%} above the best ({lr_tag(best)})"
    return mse, f"validation MSE at step {step}: {shown}; {above}"


def _mean_agreement(decoders: list[Tensor]) -> float:
    pairs = itertools.combinations(range(len(decoders)), 2)
    return float(np.mean([one_to_one(decoders[i], decoders[j]).mean for i, j in pairs]))


def extend_check(arm: str, denominator: int, seeds: list[int] | None = None, root: Path = ROOT) -> tuple[str, str]:
    """The length rule: extend if the tuning run's validation MSE fell over 1% in the last tenth of its log, or the
    seeds' agreement rose over 0.005 from half-way to the end; after one extension, report ``not_converged``."""
    size, quarter = parse_arm(arm)
    tuning = run_folder(root, RunId(size, quarter, denominator, "real", TUNING_SEED), SHARED_LR)
    log = read_log(tuning / "log.tsv")
    back = max(1, len(log) // 10)
    start, end = log[max(0, len(log) - 1 - back)].val_mse, log[-1].val_mse
    fall = (start - end) / start
    exceeds = dec(start) - dec(end) > MSE_FALL * dec(start)
    records = [read_record(tuning)]
    rise = rise_exceeds = None
    if seeds is not None:
        folders = [run_folder(root, RunId(size, quarter, denominator, "real", s)) for s in seeds]
        records += [read_record(folder) for folder in folders]
        if len({record["settings"]["epochs"] for record in records}) > 1:
            raise ValueError("the tuning run and the seeds differ in their epochs, so they cannot be checked together")
        halfway = f"step_{records[1]['trainer']['schedule']['steps'] // 2}.safetensors"
        half = _mean_agreement([load_file(f / "snapshots" / halfway)["W_dec"] for f in folders])
        final = _mean_agreement([load_file(f / "dictionary.safetensors")["W_dec"] for f in folders])
        rise = final - half
        rise_exceeds = dec(final) - dec(half) > AGREEMENT_RISE
    extensions = max(len(record["extensions"]) for record in records)
    if not (exceeds or rise_exceeds):
        verdict, advice = "converged", "do not extend"
    elif extensions >= 1:
        verdict, advice = "not_converged", f"not converged after {extensions} extension(s): no further extension"
    else:
        verdict, advice = "extend", "extend"
    message = f"tuning run lr {lr_tag(SHARED_LR)}: validation MSE fell {fall:.4%} over the last tenth of its log"
    message += "" if rise is None else f"; agreement rose {rise:.5f} from the half-way snapshot"
    return verdict, f"{message}; {advice}"
