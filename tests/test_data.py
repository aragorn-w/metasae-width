import torch

from metasae_width.data import noise_rows, real_arm, row_moments, split_rows, twin_arm, unit_rows


def test_split_sizes_are_80_10_10_and_disjoint():
    train, val, test = split_rows(103, "k")
    assert (len(train), len(val), len(test)) == (82, 10, 11)
    assert sorted(torch.cat([train, val, test]).tolist()) == list(range(103))


def test_the_noise_twin_matches_the_parent_mean_and_covariance():
    g = torch.Generator().manual_seed(0)
    parent = unit_rows(torch.randn(20000, 64, generator=g) * torch.linspace(0.3, 2.0, 64) + 0.3)
    noise = noise_rows(parent, seed=1)
    assert noise.shape == parent.shape and torch.allclose(noise.norm(dim=1), torch.ones(len(noise)), atol=1e-5)
    assert (noise.mean(0) - parent.mean(0)).norm() < 0.02 < parent.mean(0).norm()
    assert torch.allclose(noise.T.cov(), parent.T.cov(), atol=0.005)
    assert abs(row_moments(noise)["trace"] - row_moments(parent)["trace"]) < 1e-3


def test_arms_split_the_parent_the_twin_scores_on_the_real_rows_and_each_quarter_draw_has_its_own_rows():
    parent = unit_rows(torch.randn(80, 5, generator=torch.Generator().manual_seed(0)))
    noise = noise_rows(parent, seed=2)
    for quarter, n in ((0, 80), (1, 20), (2, 20)):
        real, twin = real_arm(parent, "4b", quarter=quarter), twin_arm(parent, noise, "4b", quarter=quarter)
        assert len(real.train) == 80 * n // 100 and len(real.val) + len(real.test) + len(real.train) == n
        assert (
            torch.equal(twin.val, real.val) and torch.equal(twin.test, real.test) and len(twin.train) == len(real.train)
        )
    first, second = real_arm(parent, "4b", quarter=1), real_arm(parent, "4b", quarter=2)
    assert not torch.equal(first.train, second.train)
    for draw, own in ((1, noise[:20]), (2, noise[20:40])):
        train = twin_arm(parent, noise, "4b", quarter=draw).train
        assert all((own == row).all(1).any() for row in train)
