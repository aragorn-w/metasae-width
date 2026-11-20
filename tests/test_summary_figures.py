import csv
from collections.abc import Mapping, Sequence
from pathlib import Path

from metasae_width.summary_figures import arm_curves, length_curves, near_widths, read_tsv, width_curves


def table(path: Path, rows: Sequence[Mapping[str, object]]) -> list[dict[str, str]]:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, rows[0], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return read_tsv(path)


def excess_row(arm, den, epochs, point, *, stage="final", measure="matched", kind="excess"):
    return dict(
        arm=arm,
        denominator=den,
        latents=65536 // den,
        epochs=epochs,
        stage=stage,
        measure=measure,
        kind=kind,
        point=point,
        lo=point - 0.01,
        hi=point + 0.01,
    )


def test_width_figure_draws_matched_final_2000_epoch_rows_at_the_main_widths_and_only_the_4b_at_1_256(tmp_path):
    rows = [
        excess_row("4b", 4, 2000, 0.1),
        excess_row("4b", 256, 2000, 0.2),
        excess_row("270m", 256, 2000, 0.3),
        excess_row("4b", 8, 4000, 0.4),
        excess_row("4b", 8, 2000, 0.5, measure="live_matched"),
        excess_row("4b", 8, 2000, 0.6, stage="step_325"),
        excess_row("4b", 8, 2000, 0.7, kind="corrected"),
        excess_row("4bq", 4, 2000, 0.8),
    ]
    curves = width_curves(table(tmp_path / "excess.tsv", rows))
    assert {den: est[0] for den, est in curves["4b"]["excess"].items()} == {4: 0.1, 256: 0.2}
    assert curves["270m"]["excess"] == {} and curves["4b"]["real"] == {}
    best = [
        dict(arm="4b", basis=basis, denominator=den, near_best=flag)
        for basis, den, flag in (("e2000", 64, 1), ("e2000", 128, 0), ("final", 128, 1), ("e2000", 32, 1))
    ]
    assert near_widths(table(tmp_path / "best.tsv", best))["4b"] == {32, 64}


def test_controls_figure_reads_the_4b_at_2000_epochs_and_each_control_at_6500_in_order_of_meta_latent_count(tmp_path):
    rows = [
        *(excess_row("4b", den, 2000, 0.1 * den) for den in (4, 64, 256)),
        excess_row("4b", 4, 4000, 0.9),
        *(excess_row("4bq", den, 6500, 0.2 * den) for den in (64, 4, 16, 8)),
        excess_row("4bq", 4, 2000, 0.9),
    ]
    curves = arm_curves(table(tmp_path / "excess.tsv", rows))
    assert [n for n, _ in curves["4b"]["excess"]] == [256, 1024, 16384]
    assert [(n, est[0]) for n, est in curves["4bq"]["excess"]] == [(1024, 12.8), (4096, 3.2), (16384, 0.8)]
    assert curves["4bq2"]["excess"] == []


def test_length_figure_takes_every_row_of_the_longest_file_of_each_main_width(tmp_path):
    def length_row(arm, den, run_epochs, epochs, point):
        return dict(arm=arm, denominator=den, run_epochs=run_epochs, epochs=epochs, point=point, lo=point, hi=point)

    rows = [
        length_row("4b", 4, 2000, 25, 0.1),
        length_row("4b", 4, 2000, 2000, 0.2),
        length_row("4b", 4, 4000, 25, 0.3),
        length_row("4b", 4, 4000, 4000, 0.4),
        length_row("1b", 4, 2000, 2000, 0.5),
        length_row("4b", 256, 2000, 2000, 0.6),
        length_row("4bq", 4, 6500, 2000, 0.7),
    ]
    curves = length_curves(table(tmp_path / "length.tsv", rows))
    assert [(n, est[0]) for n, est in curves["4b"][4]] == [(25, 0.3), (4000, 0.4)]
    assert [(n, est[0]) for n, est in curves["1b"][4]] == [(2000, 0.5)]
    assert 256 not in curves["4b"] and set(curves) == {"270m", "1b", "4b", "12b", "27b"}
