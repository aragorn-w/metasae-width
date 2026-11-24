import math

import numpy as np
import pytest

from metasae_width.jackknife import difference, estimate

T3 = 2.353363  # Student's t, 0.95 quantile at 3 degrees of freedom


def matrix(pairs: list[float]) -> np.ndarray:
    """Four seeds; the values are those of pairs (0,1) (0,2) (0,3) (1,2) (1,3) (2,3)."""
    m = np.zeros((4, 4))
    m[np.triu_indices(4, 1)] = pairs
    return m + m.T


V = matrix([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
# Leaving out seed 0, 1, 2, 3 leaves the means 0.5, 1.1 / 3, 0.3, 0.7 / 3 around the point 0.35.
SE = math.sqrt(3 / 4 * (0.15**2 + (1 / 60) ** 2 + 0.05**2 + (7 / 60) ** 2))


def test_the_interval_is_the_point_plus_or_minus_t_times_the_jackknife_se():
    est = estimate("real", V, np.zeros((4, 4)))
    assert est.point == pytest.approx(0.35)
    assert est.lo == pytest.approx(0.35 - T3 * SE, abs=1e-5) and est.hi == pytest.approx(0.35 + T3 * SE, abs=1e-5)


def test_excess_subtracts_the_twin_and_corrected_divides_by_one_minus_it():
    twin = matrix([0.5] * 6)
    excess, corrected = estimate("excess", V, twin), estimate("corrected", V, twin)
    assert excess.point == pytest.approx(-0.15) and excess.hi - excess.point == pytest.approx(T3 * SE, abs=1e-5)
    assert corrected.point == pytest.approx(-0.3) and corrected.hi - corrected.point == pytest.approx(
        2 * T3 * SE, abs=1e-5
    )


def test_a_difference_removes_each_seed_from_all_four_matrices():
    swapped = V[np.ix_([3, 1, 2, 0], [3, 1, 2, 0])]  # the same pairs with seeds 0 and 3 exchanged
    zero = np.zeros((4, 4))
    est = difference("real", (V, zero), (swapped, zero))
    # Both means are 0.35; leaving out seeds 0, 1, 2, 3 gives the differences 4/15, 0, 0, -4/15.
    assert est.point == pytest.approx(0.0, abs=1e-12)
    assert est.hi == pytest.approx(T3 * math.sqrt(3 / 4 * 2 * (4 / 15) ** 2), abs=1e-5) and est.lo == pytest.approx(
        -est.hi
    )
