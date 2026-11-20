import json

import pytest

from metasae_width import train
from metasae_width.runs import extend_run
from metasae_width.score import score


def read(path):
    return path.read_text(), path.with_suffix(".json").read_text()


def test_workers_give_the_serial_scores(trained):
    serial = read(score("4b", 8, [0, 1, 2], trained))
    assert read(score("4b", 8, [0, 1, 2], trained, workers=2)) == serial


def test_the_fit_is_the_training_score_on_the_runs_own_training_rows(trained, fit, arm):
    fits = json.loads(score("4b", 8, [0, 1, 2], trained).with_suffix(".json").read_text())["fit"]
    assert {role: set(by_seed) for role, by_seed in fits.items()} == {"real": {"0", "1", "2"}, "twin": {"0", "1", "2"}}
    mse, ve = train._score(fit(epochs=8, run_id="4b.w8.real.s0").sae, arm.train)
    assert fits["real"]["0"] == {"train_mse": pytest.approx(mse, rel=1e-5), "train_ve": pytest.approx(ve, rel=1e-5)}


def test_runs_that_differ_in_their_stage_are_refused_by_name(trained):
    extend_run("4b.w8.real.s2", 16, root=trained, device="cpu")
    with pytest.raises(ValueError, match=r"4b\.w8\.real\.s2 differs from 4b\.w8\.real\.s0"):
        score("4b", 8, [0, 1, 2], trained)
