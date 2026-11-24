import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from metasae_width.score import COLUMNS, stage_lines
from metasae_width.summary import summarize
from metasae_width.summary_figures import read_tsv
from metasae_width.summary_scores import pair_matrix

PAIRS = [f"{i}-{j}" for i, j in itertools.combinations(range(8), 2)]


def write_scores(root: Path, arm: str, den: int, real: float, *, epochs: int = 2000, fit: float = 0.7) -> None:
    """A final-stage score file at seeds 0 to 7: real pairs near ``real``, twin pairs near 0.2; ``fit`` is the
    training-row variance explained recorded for every run."""
    rng = np.random.default_rng([den, epochs, len(arm)])
    pairs = {
        role: {pair: level + 0.01 * rng.standard_normal() for pair in PAIRS}
        for role, level in (("real", real), ("twin", 0.2))
    }
    matrices = {m: {role: pair_matrix(values) for role, values in pairs.items()} for m in ("matched", "live_matched")}
    stem = f"{root}/scores/{arm}.w{den}.s0-1-2-3-4-5-6-7.e{epochs}"
    (root / "scores").mkdir(parents=True, exist_ok=True)
    Path(f"{stem}.tsv").write_text("\n".join(["\t".join(COLUMNS), *stage_lines("final", 26000, matrices)]) + "\n")
    fits = {role: {str(s): {"train_mse": 0.1, "train_ve": fit} for s in range(8)} for role in pairs}
    data = {"seeds": list(range(8)), "pairs": {"final": {m: pairs for m in matrices}}, "fit": fits}
    Path(f"{stem}.json").write_text(json.dumps(data))


def test_near_best_keeps_the_top_and_the_widths_its_gap_interval_does_not_separate(tmp_path):
    for den, level in ((4, 0.3), (8, 0.6), (16, 0.598)):
        write_scores(tmp_path, "4b", den, level)
    for den, level in ((4, 0.5), (8, 0.3)):
        write_scores(tmp_path, "1b", den, level)
    summarize(tmp_path)
    best = {
        (r["arm"], r["denominator"]): r for r in read_tsv(tmp_path / "summary" / "best.tsv") if r["basis"] == "e2000"
    }
    flags = {
        key: (r["is_top"], r["near_best"], r["near_best_live"], r["near_best_corrected"]) for key, r in best.items()
    }
    assert flags[("4b", "4")] == ("0", "0", "0", "0")
    assert flags[("4b", "8")][1:] == ("1", "1", "1") and flags[("4b", "16")][1:] == ("1", "1", "1")
    assert flags[("1b", "4")] == ("1", "1", "1", "1") and flags[("1b", "8")] == ("0", "0", "0", "0")
    gaps = {r["arm"]: r for r in read_tsv(tmp_path / "summary" / "best_minus_quarter.tsv") if r["basis"] == "e2000"}
    assert 0.25 < float(gaps["4b"]["lo"]) < float(gaps["4b"]["point"]) < float(gaps["4b"]["hi"])
    assert [gaps["1b"][c] for c in ("point", "lo", "hi")] == ["0.0", "0.0", "0.0"]


def test_pairs_that_disagree_with_their_tsv_are_refused(tmp_path):
    write_scores(tmp_path, "4b", 4, 0.3)
    path = next((tmp_path / "scores").glob("*.json"))
    data = json.loads(path.read_text())
    data["pairs"]["final"]["matched"]["real"]["0-1"] += 0.05
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=r"4b\.w4\.e2000"):
        summarize(tmp_path)


def test_rows_pair_each_extra_arm_with_its_base_arm_at_the_same_rows_latents_or_width_and_skip_missing_files(tmp_path):
    for arm, den, epochs, level in (
        ("4b", 4, 2000, 0.3),
        ("4b", 16, 2000, 0.5),
        ("4bq", 4, 2000, 0.4),
        ("4bq", 4, 6500, 0.6),
        ("270m", 4, 2000, 0.2),
        ("270m16k", 4, 2000, 0.25),
        ("270m16k", 4, 6500, 0.35),
    ):
        write_scores(tmp_path, arm, den, level, epochs=epochs)
    summarize(tmp_path)
    rows = read_tsv(tmp_path / "summary" / "rows.tsv")
    assert [
        (r["comparison"], r["first_arm"], r["first_epochs"], r["second_arm"], r["second_denominator"]) for r in rows
    ] == [
        ("same_rows_per_latent", "4bq", "2000", "4b", "4"),
        ("same_rows_per_latent", "270m16k", "2000", "270m", "4"),
        ("same_latents_equal_steps", "4bq", "6500", "4b", "16"),
        ("same_width_equal_steps", "4bq", "6500", "4b", "4"),
        ("same_width_equal_steps", "270m16k", "6500", "270m", "4"),
    ]
    for r in rows:
        assert float(r["difference_point"]) == pytest.approx(
            float(r["first_excess_point"]) - float(r["second_excess_point"])
        )
        assert float(r["difference_lo"]) < float(r["difference_point"]) < float(r["difference_hi"])


def test_best_4b256_adds_the_256_width_to_the_4b_near_best_sets_and_leaves_the_sweep_tables_alone(tmp_path):
    for den, level in ((4, 0.3), (128, 0.5), (256, 0.6)):
        write_scores(tmp_path, "4b", den, level)
    summarize(tmp_path)
    rows = {(r["basis"], r["denominator"]): r for r in read_tsv(tmp_path / "summary" / "best_4b256.tsv")}
    assert set(rows) == {(basis, den) for basis in ("e2000", "final") for den in ("4", "128", "256")}
    flags = {
        den: [rows["e2000", den][f"matched_excess_{c}"] for c in ("near_best", "is_top")] for den in ("4", "128", "256")
    }
    assert flags == {"4": ["0", "0"], "128": ["0", "0"], "256": ["1", "1"]}
    assert [rows["e2000", "256"][f"matched_excess_gap_{c}"] for c in ("point", "lo", "hi")] == ["0.0", "0.0", "0.0"]
    assert (
        0.25
        < float(rows["e2000", "4"]["matched_excess_gap_lo"])
        < float(rows["e2000", "4"]["matched_excess_gap_point"])
    )
    best = {(r["arm"], r["denominator"]): r["is_top"] for r in read_tsv(tmp_path / "summary" / "best.tsv")}
    assert best == {("4b", "4"): "0", ("4b", "128"): "1"}


def test_the_validation_score_file_gives_validate_tsv(tmp_path):
    rng = np.random.default_rng(0)
    matrices = {
        m: {
            role: pair_matrix(dict(zip(PAIRS, level + 0.01 * rng.standard_normal(28), strict=True)))
            for role, level in (("real", 0.5), ("twin", 0.2))
        }
        for m in ("matched", "fraction_matched", "many_to_one", "live_matched")
    }
    path = tmp_path / "scores" / "validate.s0-1-2-3-4-5-6-7.e2000.tsv"
    path.parent.mkdir()
    path.write_text("\n".join(["\t".join(COLUMNS), *stage_lines("final", 24000, matrices)]) + "\n")
    summarize(tmp_path)
    rows = read_tsv(tmp_path / "summary" / "validate.tsv")
    assert [(r["measure"], r["kind"]) for r in rows] == [
        (m, k) for m in ("matched", "fraction_matched", "live_matched") for k in ("real", "twin", "excess", "corrected")
    ]
    excess = next(r for r in rows if (r["measure"], r["kind"]) == ("matched", "excess"))
    assert 0.2 < float(excess["lo"]) < float(excess["point"]) < float(excess["hi"]) < 0.4


def test_runs_report_the_mean_training_fit_and_leave_it_empty_without_a_score_file(tmp_path):
    write_scores(tmp_path, "4b", 4, 0.3, fit=0.7)
    record = {
        "status": "done",
        "settings": {"epochs": 2000},
        "trainer": {"dead_fraction": 0.1, "final": {"test_ve": 0.6}},
    }
    for den, role, seed in itertools.product((4, 16), ("real", "twin"), range(8)):
        folder = tmp_path / "runs" / f"4b.w{den}.{role}.s{seed}"
        folder.mkdir(parents=True)
        (folder / "record.json").write_text(json.dumps(record))
    summarize(tmp_path)
    fits = {(r["denominator"], r["role"]): r["train_ve"] for r in read_tsv(tmp_path / "summary" / "runs.tsv")}
    assert fits == {("4", "real"): "0.7", ("4", "twin"): "0.7", ("16", "real"): "", ("16", "twin"): ""}


def test_the_rate_check_pairs_each_27b_and_270m_width_at_1e_3_with_the_main_run_and_skips_missing_files(tmp_path):
    root = tmp_path / "metasae-width"
    checked = tmp_path / "metasae-width-lr1e-3"
    for size, den in (("27b", 32), ("27b", 64), ("270m", 64), ("270m", 128)):
        write_scores(root, size, den, 0.5)
    for size, den in (("27b", 32), ("270m", 64)):
        write_scores(checked, size, den, 0.55)
    summarize(root)
    rows = read_tsv(root / "summary" / "rate.tsv")
    assert [(r["arm"], r["denominator"]) for r in rows] == [("27b", "32"), ("270m", "64")]
    for row in rows:
        point = float(row["difference_point"])
        assert point == pytest.approx(float(row["lr1e-3_excess_point"]) - float(row["lr3e-3_excess_point"]))
        assert float(row["difference_lo"]) < point < float(row["difference_hi"])
