"""Leave-one-seed-out jackknife intervals for statistics of seed-by-seed matrices that share their seed labels."""

from collections.abc import Callable
from typing import NamedTuple

import numpy as np
from scipy.stats import t as student_t


class Estimate(NamedTuple):
    point: float
    lo: float
    hi: float


# A kind turns the real and twin mean agreements into one number.
KINDS: dict[str, Callable[[float, float], float]] = {
    "real": lambda real, twin: real,
    "twin": lambda real, twin: twin,
    "excess": lambda real, twin: real - twin,
    "corrected": lambda real, twin: (real - twin) / (1 - twin),
}


def pair_mean(w: np.ndarray, V: np.ndarray) -> float:
    """The mean of V[i, j] over pairs i < j weighted by w_i w_j, for w a (1, n) row of ones with a 0 for each seed
    left out. The same sums, in the same order, as the bootstrap this replaced."""
    use = np.triu(np.ones(V.shape, dtype=bool), 1)
    weights, pairs = use.astype(np.float64), np.where(use, V, 0.0)
    return float((np.einsum("ri,ij,rj->r", w, pairs, w) / np.einsum("ri,ij,rj->r", w, weights, w))[0])


def jackknife(statistic: Callable[..., float], *matrices: np.ndarray) -> Estimate:
    """``statistic(w, *matrices)`` with its 90% interval: point +/- t * se, where se is the jackknife standard error
    over the n leave-one-seed-out values and t is Student's t at 0.95 with n - 1 degrees of freedom. The point has
    w all ones; leaving seed k out puts a 0 at k in w, for every matrix at once."""
    n = len(matrices[0])
    point = statistic(np.ones((1, n)), *matrices)
    leave = np.array([statistic(np.where(np.arange(n) == k, 0.0, 1.0)[None], *matrices) for k in range(n)])
    half = float(student_t.ppf(0.95, n - 1)) * float(np.sqrt((n - 1) / n * ((leave - leave.mean()) ** 2).sum()))
    return Estimate(point, point - half, point + half)


def _kind(kind: str, w: np.ndarray, real: np.ndarray, twin: np.ndarray) -> float:
    return KINDS[kind](pair_mean(w, real), pair_mean(w, twin))


def estimate(kind: str, real: np.ndarray, twin: np.ndarray) -> Estimate:
    """One kind of the real and twin pairwise matrices of the same seeds."""
    return jackknife(lambda w, r, t: _kind(kind, w, r, t), real, twin)


def difference(kind: str, a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]) -> Estimate:
    """The kind of condition ``a`` minus that of ``b``, each given as its (real, twin) matrices: four matrices over
    the same seed labels, so seed k leaves all four."""
    return jackknife(lambda w, ar, at, br, bt: _kind(kind, w, ar, at) - _kind(kind, w, br, bt), *a, *b)
