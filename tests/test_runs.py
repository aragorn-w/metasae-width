import json

import torch
from safetensors.torch import load_file

from metasae_width import train
from metasae_width.runs import RunId, extend_run, parse_arm, parse_run_id, run_folder
from metasae_width.score import score


def test_run_ids_and_folder_names_read_back_what_the_old_code_wrote(tmp_path):
    rid = parse_run_id("27bq.w16.twin.s3")
    assert rid == RunId("27b", 1, 16, "twin", 3) and str(rid) == "27bq.w16.twin.s3"
    assert run_folder(tmp_path, rid) == tmp_path / "runs" / "27bq.w16.twin.s3"
    third = parse_run_id("4bq3.w64.real.s7")
    assert third == RunId("4b", 3, 64, "real", 7) and str(third) == "4bq3.w64.real.s7"
    assert parse_arm("4bq2") == ("4b", 2)
    assert parse_run_id("4b16k.w4.twin.s2").size == "4b16k" and parse_arm("270m16k") == ("270m16k", 0)
    assert run_folder(tmp_path, parse_run_id("12b.w4.real.s100"), 1e-3).name == "12b.w4.real.s100.lr1e-3"


def test_train_extend_and_score_work_end_to_end_and_an_earlier_stage_scores_as_before(trained):
    folder = trained / "runs" / "4b.w8.real.s0"
    record = json.loads((folder / "record.json").read_text())
    assert record["status"] == "done" and record["width"] == 8 and record["settings"]["lr"] == 3e-3
    assert {p.name for p in (folder / "snapshots").iterdir()} == {f"step_{s}.safetensors" for s in (2, 4)}
    assert record["trainer"]["snapshot_steps"] == [2, 4]
    assert set(load_file(folder / "snapshots" / "step_2.safetensors")) == set(train.PARAMS)
    before = load_file(folder / "dictionary.safetensors")["W_dec"]
    early = score("4b", 8, [0, 1, 2], trained)
    stages = {line.split("\t")[0] for line in early.read_text().splitlines()[1:]}
    assert early.name == "4b.w8.s0-1-2.e8.tsv" and stages == {"step_2", "step_4", "final"}
    texts = early.read_text(), early.with_suffix(".json").read_text()

    for role in ("real", "twin"):
        for seed in (0, 1, 2):
            extend_run(f"4b.w8.{role}.s{seed}", 16, root=trained, device="cpu")
    record = json.loads((folder / "record.json").read_text())
    assert record["settings"]["epochs"] == 16 and record["extensions"][0]["from_epochs"] == 8
    assert record["trainer"]["snapshot_steps"] == [2, 4, 8, 12]
    assert {p.name for p in (folder / "snapshots").iterdir()} == {f"step_{s}.safetensors" for s in (2, 4, 8, 12)}
    assert torch.equal(load_file(folder / "epochs_8" / "dictionary.safetensors")["W_dec"], before)

    assert score("4b", 8, [0, 1, 2], trained, epochs=8) == early
    assert (early.read_text(), early.with_suffix(".json").read_text()) == texts
    tsv = score("4b", 8, [0, 1, 2], trained)
    lines = tsv.read_text().splitlines()
    assert tsv.name == "4b.w8.s0-1-2.e16.tsv" and lines[0].startswith("stage\tstep\tmeasure\tkind")
    assert len(lines) == 1 + 4 * 3 * 4 + 4 * 4  # four snapshots with three measures, the end with four; four kinds
