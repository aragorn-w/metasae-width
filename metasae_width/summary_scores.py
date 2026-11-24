"""Score files: the study's conditions, a score file loaded as a ``Score``, and the near-best sets."""

import json
import re
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from metasae_width.gemma_scope2 import SAES, SIZES
from metasae_width.jackknife import Estimate, difference, estimate
from metasae_width.runs import parse_arm
from metasae_width.train import EPOCHS

SEEDS = tuple(range(8))
SWEEP = (4, 8, 16, 32, 64, 128)
ARMS = {
    **{size: SWEEP for size in SIZES},
    "270m": (*SWEEP, 256),
    "4b": (*SWEEP, 256),
    "4bq": (4, 16, 64),
    "4bq2": (4, 16, 64),
    "4bq3": (4, 16, 64),
    "4b16k": (4, 16, 64),
    "270m16k": (4, 16, 64),
}
# The three views of a width's agreement, by column suffix: (measure, kind).
VIEWS = {"": ("matched", "excess"), "_live": ("live_matched", "excess"), "_corrected": ("matched", "corrected")}
_FILE = re.compile(r"(?P<arm>[0-9a-z]+)\.w(?P<den>\d+)\.s(?P<seeds>[\d-]+)\.e(?P<epochs>\d+)\.tsv")
Key = tuple[str, int, int]


def pair_matrix(values: dict[str, float]) -> np.ndarray:
    """The symmetric seed-by-seed matrix of the ``"<a>-<b>"`` pair values."""
    matrix = np.zeros((len(SEEDS), len(SEEDS)))
    for key, value in values.items():
        a, b = (SEEDS.index(int(label)) for label in key.split("-"))
        matrix[a, b] = matrix[b, a] = value
    return matrix


class Score(NamedTuple):
    """One score file: its TSV rows, and the pair values and fit from the JSON beside it."""

    arm: str
    den: int
    epochs: int
    rows: list[dict[str, str]]
    pairs: dict[str, Any]
    fit: dict[str, Any]

    @property
    def name(self) -> str:
        return f"{self.arm}.w{self.den}.e{self.epochs}"

    @property
    def latents(self) -> int:
        """The meta-SAE's width: the parent's latents (a quarter for a quarter-draw arm) over the denominator."""
        size, quarter = parse_arm(self.arm)
        return SAES[size].latents // (4 if quarter else 1) // self.den

    def est(self, measure: str = "matched", kind: str = "excess") -> Estimate:
        row = next(r for r in self.rows if (r["stage"], r["measure"], r["kind"]) == ("final", measure, kind))
        return Estimate(float(row["point"]), float(row["lo"]), float(row["hi"]))

    def matrices(self, measure: str = "matched") -> tuple[np.ndarray, np.ndarray]:
        """The final stage's (real, twin) seed-by-seed matrices of a measure."""
        by_role = self.pairs["final"][measure]
        return pair_matrix(by_role["real"]), pair_matrix(by_role["twin"])


def load_scores(root: Path) -> dict[Key, Score]:
    """Every score file of seeds 0 to 7 for a condition in ARMS under ``<root>/scores``, by (arm, den, epochs)."""
    scores = {}
    for path in sorted((root / "scores").glob("*.tsv")):
        found = _FILE.fullmatch(path.name)
        if (
            found is None
            or found["seeds"] != "-".join(map(str, SEEDS))
            or int(found["den"]) not in ARMS.get(found["arm"], ())
        ):
            continue
        lines = [line.split("\t") for line in path.read_text().splitlines()]
        data = json.loads(path.with_suffix(".json").read_text())
        rows = [dict(zip(lines[0], line, strict=True)) for line in lines[1:]]
        score = Score(found["arm"], int(found["den"]), int(found["epochs"]), rows, data["pairs"], data["fit"])
        scores[(score.arm, score.den, score.epochs)] = score
    return scores


def check_pairs(score: Score) -> None:
    """Raise if the matched excess recomputed from the pairs JSON differs from the TSV's."""
    got, want = estimate("excess", *score.matrices()), score.est()
    if max(abs(a - b) for a, b in zip(got, want, strict=True)) > 1e-12:
        raise ValueError(f"{score.name}: the pairs JSON gives excess {tuple(got)}, the TSV {tuple(want)}")


def order(key: Key) -> tuple[int, int, int]:
    return list(ARMS).index(key[0]), key[1], key[2]


def pick(scores: dict[Key, Score], basis: str, widths: tuple[int, ...] = SWEEP) -> dict[tuple[str, int], Score]:
    """The score file each width (of the sweep by default) contributes to a basis."""
    chosen = {}
    for key in sorted(scores, key=order):
        if key[0] in SIZES and key[1] in widths and (basis == "final" or key[2] == EPOCHS):
            chosen[key[:2]] = scores[key]
    return chosen


def gap(a: Score, b: Score, measure: str, kind: str) -> Estimate:
    """Condition ``a`` minus ``b`` in a measure's kind with its jackknife interval."""
    return difference(kind, a.matrices(measure), b.matrices(measure))


def near_best(widths: dict[int, Score], measure: str, kind: str) -> tuple[int, dict[int, bool]]:
    """The top width (largest point) and, per width, whether the 90% interval on top minus it contains 0."""
    top = max(widths, key=lambda den: widths[den].est(measure, kind).point)
    gaps = {den: gap(widths[top], s, measure, kind) for den, s in widths.items()}
    return top, {den: lo <= 0 <= hi for den, (_, lo, hi) in gaps.items()}
