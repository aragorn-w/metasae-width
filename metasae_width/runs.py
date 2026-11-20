"""Run ids and folders, the files of a run, and the train, extend and check-sae commands."""

import json
import os
import re
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

import torch
from safetensors.torch import load_file, save_file

from metasae_width.data import Arm, noise_rows, real_arm, save_noise, twin_arm, unit_rows
from metasae_width.gemma_scope2 import ROOT, SAES, load_decoder, sae_dir
from metasae_width.seeds import derived_seed
from metasae_width.train import EPOCHS, LogRow, Trained, schedule, train_on_rows

TUNING_SEED = 100
DRAWS = ("", "q", "q2", "q3")  # the arm suffix of each quarter draw, 0 for all the rows
_ARM = rf"(?P<size>{'|'.join(SAES)})(?P<q>(?:q[23]?)?)"
_RUN_ID = re.compile(rf"{_ARM}\.w(?P<den>[1-9]\d*)\.(?P<role>real|twin)\.s(?P<seed>0|[1-9]\d*)")


class RunId(NamedTuple):
    """``<size>[q|q2|q3].w<denominator>.<real|twin>.s<seed>``: the model size (or 16k parent: ``4b16k``, ``270m16k``),
    which random quarter of its rows the arm trains on (0 for all of them, 1 to 3 for the draws ``q``, ``q2``,
    ``q3``), the dictionary width's denominator, the role and the seed. Seed 100 is the tuning seed."""

    size: str
    quarter: int
    denominator: int
    role: str
    seed: int

    def __str__(self) -> str:
        return f"{self.size}{DRAWS[self.quarter]}.w{self.denominator}.{self.role}.s{self.seed}"


def parse_run_id(text: str) -> RunId:
    found = _RUN_ID.fullmatch(text)
    if found is None:
        raise ValueError(f"{text!r} is not <size>[q|q2|q3].w<denominator>.<real|twin>.s<seed>")
    return RunId(found["size"], DRAWS.index(found["q"]), int(found["den"]), found["role"], int(found["seed"]))


def parse_arm(text: str) -> tuple[str, int]:
    """A model size or 16k parent with an optional ``q``, ``q2`` or ``q3`` for a random-quarter arm: (size, quarter
    draw)."""
    found = re.fullmatch(_ARM, text)
    if found is None:
        raise ValueError(f"{text!r} is not <size>[q|q2|q3]")
    return found["size"], DRAWS.index(found["q"])


def lr_tag(lr: float) -> str:
    """A learning rate in short scientific form: 3e-5, 1e-4, 5e-2."""
    mantissa, exponent = f"{lr:e}".split("e")
    return f"{mantissa.rstrip('0').rstrip('.')}e{int(exponent)}"


def run_folder(root: Path, rid: RunId, lr: float = 1.0) -> Path:
    """``<root>/runs/<run id>``; a tuning run trains at several learning rates, so its folder ends in ``.lr<tag>``."""
    return root / "runs" / (f"{rid}.lr{lr_tag(lr)}" if rid.seed == TUNING_SEED else str(rid))


def read_record(folder: Path) -> dict[str, Any]:
    record: dict[str, Any] = json.loads((folder / "record.json").read_text())
    if record["status"] != "done":
        raise ValueError(f"{folder}: the run's status is {record['status']!r}, not 'done'")
    return record


def read_log(path: Path) -> list[LogRow]:
    rows = [line.split("\t") for line in path.read_text().splitlines()[1:]]
    return [
        LogRow(int(c[0]), int(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5]), int(c[6]), float(c[7]))
        for c in rows
    ]


def log_to(path: Path, steps: int) -> Callable[[LogRow], None]:
    """A logger that appends each row to the tab-separated log (with its header first) and prints it."""

    def write(row: LogRow) -> None:
        with path.open("a") as handle:
            if handle.tell() == 0:
                handle.write("\t".join(LogRow._fields) + "\n")
            handle.write("\t".join(str(v) if isinstance(v, int) else repr(float(v)) for v in row) + "\n")
        print(f"step {row.step} of {steps}\ttrain_mse {row.train_mse:.6g}\tval_mse {row.val_mse:.6g}\tdead {row.dead}")

    return write


def snapshot_writer(folder: Path) -> Callable[[int, dict[str, torch.Tensor]], None]:
    """An ``on_snapshot`` that saves each snapshot to ``<folder>/snapshots/step_<step>.safetensors`` as it is taken."""

    def write(step: int, weights: dict[str, torch.Tensor]) -> None:
        (folder / "snapshots").mkdir(exist_ok=True)
        save_file(
            {name: w.contiguous() for name, w in weights.items()}, folder / "snapshots" / f"step_{step}.safetensors"
        )

    return write


def save_run(folder: Path, trained: Trained, record: dict[str, Any]) -> None:
    """Write a stage's files: the dictionary, the state, the live mask, and last the record."""
    sae = trained.sae
    weights = {name: t.detach().contiguous() for name, t in sae.state_dict().items()}
    save_file(weights, folder / "dictionary.safetensors")
    lr, epochs = record["settings"]["lr"], record["settings"]["epochs"]
    sidecar = {"dim": sae.dim, "width": sae.width, "k": sae.k, "lr": lr, "epochs": epochs}
    (folder / "dictionary.json").write_text(json.dumps(sidecar) + "\n")
    torch.save(trained.state, folder / "state.pt")
    save_file({"live": trained.live.contiguous()}, folder / "live.safetensors")
    (folder / "record.json").write_text(json.dumps(record, indent=2) + "\n")


def load_arm(rid: RunId, root: Path) -> tuple[Arm, int]:
    """The rows a run trains and scores on, and the dictionary width: the source's row count over the denominator."""
    parent = unit_rows(load_decoder(rid.size, root))
    if rid.role == "real":
        arm = real_arm(parent, rid.size, quarter=rid.quarter)
    else:
        noise = load_file(sae_dir(rid.size, root) / "noise.safetensors")["rows"]
        arm = twin_arm(parent, noise, rid.size, quarter=rid.quarter)
    return arm, (len(parent) // 4 if rid.quarter else len(parent)) // rid.denominator


def train_into(
    folder: Path, run_id: str, arm: Arm, *, width: int, lr: float, epochs: int, device: str
) -> tuple[Trained, dict[str, Any]]:
    """Train on ``arm`` into a new folder (an existing one is an error), writing the log and snapshots; return the
    result and the record of the finished run."""
    folder.mkdir(parents=True)
    settings = {"lr": lr, "epochs": epochs, "device": device}
    on_log = log_to(folder / "log.tsv", schedule(len(arm.train), width, epochs).steps)
    on_snapshot = snapshot_writer(folder)
    trained = train_on_rows(*arm, run_id=run_id, width=width, **settings, on_log=on_log, on_snapshot=on_snapshot)
    record = {"status": "done", "run_id": run_id, "width": width, "settings": settings, "trainer": trained.record}
    return trained, record | {"extensions": []}


def train_run(run_id: str, lr: float, *, epochs: int = EPOCHS, root: Path = ROOT, device: str = "cuda") -> Path:
    """Train one run into a new folder (an existing one is an error) and return it."""
    rid = parse_run_id(run_id)
    folder = run_folder(root, rid, lr)
    arm, width = load_arm(rid, root)
    save_run(folder, *train_into(folder, run_id, arm, width=width, lr=lr, epochs=epochs, device=device))
    return folder


def extend_run(run_id: str, epochs: int, *, lr: float | None = None, root: Path = ROOT, device: str = "cuda") -> Path:
    """Continue a finished run to ``epochs`` in all. The weights equal those of training to that total directly.
    The earlier stage's record, dictionary, state and live mask move to ``epochs_<earlier epochs>/``."""
    rid = parse_run_id(run_id)
    folder = run_folder(root, rid, lr or 1.0)
    old = read_record(folder)
    done = old["settings"]["epochs"]
    if epochs <= done:
        raise ValueError(f"{run_id} has {done} epochs; extending needs more, got {epochs}")
    arm, _ = load_arm(rid, root)
    state = torch.load(folder / "state.pt", weights_only=True, map_location="cpu")
    pending = folder / "log.extend.tsv"
    shutil.copyfile(folder / "log.tsv", pending)
    steps = epochs * old["trainer"]["schedule"]["per_epoch"]
    run = {"run_id": state["run_id"], "width": state["width"], "lr": state["lr"], "epochs": epochs, "state": state}
    on_log, on_snapshot = log_to(pending, steps), snapshot_writer(folder)
    trained = train_on_rows(arm.train, arm.val, arm.test, **run, device=device, on_log=on_log, on_snapshot=on_snapshot)
    stage = folder / f"epochs_{done}"
    stage.mkdir()
    for name in ("record.json", "dictionary.json", "dictionary.safetensors", "state.pt", "live.safetensors"):
        os.replace(folder / name, stage / name)
    os.replace(pending, folder / "log.tsv")
    written = sorted(set(old["trainer"]["snapshot_steps"]) | set(trained.record["snapshot_steps"]))
    trainer = {**trained.record, "snapshot_steps": written}
    record = {**old, "trainer": trainer, "settings": {**old["settings"], "epochs": epochs}}
    record["extensions"] = [*old["extensions"], {"from_epochs": done, "to_epochs": epochs}]
    save_run(folder, trained, record)
    return folder


def check_sae(size: str, root: Path = ROOT) -> None:
    """Download a size's SAE, draw its noise twin if there is none, and print how closely the noise's mean and
    covariance match the SAE's decoder rows."""
    folder = sae_dir(size, root)
    rows = load_decoder(size, root)
    parent = unit_rows(rows)
    key = f"{size}/null"
    if not (folder / "noise.safetensors").exists():
        save_noise(noise_rows(rows, derived_seed(key)), folder, key, parent)
    print(f"{size}: {len(rows)} latents of dimension {rows.shape[1]}")
    print((folder / "noise.json").read_text())
