import json
from pathlib import Path

import pytest

from metasae_width.checks import extend_check, tune_pick
from metasae_width.runs import TUNING_SEED, RunId, run_folder


def fake_tuning_run(root: Path, lr: float, val_mses: list[float], extensions: int = 0) -> None:
    folder = run_folder(root, RunId("4b", 0, 8, "real", TUNING_SEED), lr)
    folder.mkdir(parents=True)
    record = {"status": "done", "settings": {"lr": lr, "epochs": 100}, "extensions": [{}] * extensions}
    (folder / "record.json").write_text(json.dumps(record))
    header = "step\tepoch\ttrain_mse\ttrain_aux\tval_mse\tval_ve\tdead\tgrad_norm"
    rows = [f"{100 * (i + 1)}\t{i}\t0.1\t0.0\t{mse!r}\t0.5\t0\t0.1" for i, mse in enumerate(val_mses)]
    (folder / "log.tsv").write_text("\n".join([header, *rows]) + "\n")


def test_the_tuning_table_gives_each_rates_mse_and_the_shared_rates_gap_to_the_best(tmp_path):
    for lr, final in ((1e-3, 1.0), (3e-3, 1.005), (1e-2, 1.1)):
        fake_tuning_run(tmp_path, lr, [2.0, 1.5, final])
    table, message = tune_pick("4b", 8, tmp_path)
    assert table == {1e-3: 1.0, 3e-3: 1.005, 1e-2: 1.1} and "0.50% above" in message


@pytest.mark.parametrize(
    ("last_mses", "extensions", "verdict"),
    [([1.0, 0.98, 0.9], 0, "extend"), ([1.0, 0.98, 0.9], 1, "not_converged"), ([1.0, 0.999, 0.998], 0, "converged")],
)
def test_the_length_rule_extends_once(tmp_path, last_mses, extensions, verdict):
    fake_tuning_run(tmp_path, 3e-3, last_mses, extensions)
    assert extend_check("4b", 8, None, tmp_path)[0] == verdict
