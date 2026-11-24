import itertools

import pytest
import torch

from metasae_width.data import unit_rows
from metasae_width.metrics import many_to_one, one_to_one, variance_explained


def test_one_to_one_matches_a_brute_force_search():
    g = torch.Generator().manual_seed(0)
    a, b = torch.randn(5, 4, generator=g), torch.randn(6, 4, generator=g)
    sim = unit_rows(a) @ unit_rows(b).T
    best = max(itertools.permutations(range(6), 5), key=lambda cols: sim[range(5), list(cols)].sum().item())
    matched = sim[range(5), list(best)]
    result = one_to_one(a, b)
    assert result.mean == pytest.approx(matched.mean().item(), abs=1e-6)
    assert result.fraction == pytest.approx((matched >= 0.7).float().mean().item())


def test_many_to_one_is_the_mean_of_the_two_best_match_averages():
    g = torch.Generator().manual_seed(1)
    a, b = torch.randn(7, 4, generator=g), torch.randn(5, 4, generator=g)
    sim = unit_rows(a) @ unit_rows(b).T
    assert many_to_one(a, b) == pytest.approx((sim.max(1).values.mean() + sim.max(0).values.mean()) / 2)


def test_variance_explained_is_one_for_a_perfect_fit_and_zero_for_the_column_mean():
    target = torch.randn(30, 4, generator=torch.Generator().manual_seed(2))
    assert variance_explained(target, target) == 1.0
    assert variance_explained(target.mean(0, keepdim=True).expand_as(target), target) == pytest.approx(0.0, abs=1e-6)
