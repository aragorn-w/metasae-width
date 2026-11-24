import torch

from metasae_width import train
from metasae_width.models import auxk_loss, batch_topk
from metasae_width.train import update_dead_


def same(a, b) -> bool:
    return all(torch.equal(a.sae.state_dict()[k], b.sae.state_dict()[k]) for k in a.sae.state_dict())


def test_extending_equals_training_to_the_total_and_snapshots_the_grid(fit, grid, monkeypatch):
    monkeypatch.setattr(train, "LOG_EVERY", 2)
    # Grid epochs 2, 4, 8, 12, 16, ...; one step per epoch, so a snapshot's step is its epoch.
    direct_log, first_log, extra_log, direct_snaps, first_snaps, extra_snaps = [], [], [], {}, {}, {}
    direct = fit(epochs=16, on_log=direct_log.append, on_snapshot=direct_snaps.__setitem__)
    half = fit(epochs=8, on_log=first_log.append, on_snapshot=first_snaps.__setitem__)
    more = fit(epochs=16, state=half.state, on_log=extra_log.append, on_snapshot=extra_snaps.__setitem__)  # starts on 8
    assert same(direct, more) and torch.equal(direct.state["counters"], more.state["counters"])
    assert direct_log == first_log + extra_log
    assert list(direct_snaps) == [2, 4, 8, 12] and list(first_snaps) == [2, 4] and list(extra_snaps) == [8, 12]
    assert direct.record["snapshot_steps"] == [2, 4, 8, 12] and more.record["snapshot_steps"] == [8, 12]
    shared = {**first_snaps, **extra_snaps}
    assert all(torch.equal(w[k], direct_snaps[step][k]) for step, w in shared.items() for k in train.PARAMS)
    for weights in direct_snaps.values():
        assert list(weights) == list(train.PARAMS) and {w.dtype for w in weights.values()} == {torch.float32}


def test_batch_topk_keeps_k_times_the_batch():
    pre = torch.rand(5, 20) + 0.1
    kept = batch_topk(pre, 3)
    assert (kept > 0).sum() == 15 and kept[kept > 0].min() >= pre[kept == 0].max()


def test_auxk_is_zero_without_dead_latents_and_matches_the_formula_with_two():
    g = torch.Generator().manual_seed(0)
    pre, residual, w_dec = torch.rand(4, 6, generator=g), torch.randn(4, 3, generator=g), torch.randn(6, 3, generator=g)
    assert auxk_loss(pre, residual, w_dec, torch.zeros(6, dtype=torch.bool), k_aux=1, weight=0.5) == 0
    dead = torch.tensor([False, True, False, False, True, False])
    top = torch.zeros(4, 2).scatter(1, pre[:, dead].argmax(1, keepdim=True), pre[:, dead].max(1, keepdim=True).values)
    expected = 0.5 * (top @ w_dec[dead] - residual).pow(2).mean()
    assert torch.allclose(auxk_loss(pre, residual, w_dec, dead, k_aux=1, weight=0.5), expected)


def test_dead_counters_reset_on_firing_and_grow_by_the_batch():
    counters = torch.tensor([7, 7, 0])
    latents = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    dead = update_dead_(counters, latents, dead_after=9)
    assert counters.tolist() == [0, 9, 2] and dead.tolist() == [False, True, False]
