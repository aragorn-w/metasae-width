"""Agreement between seeds for real dictionaries and their noise twins, with jackknife intervals."""

import itertools
import json
from collections.abc import Callable
from concurrent.futures import Executor, ProcessPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file

from metasae_width import train
from metasae_width.gemma_scope2 import ROOT
from metasae_width.jackknife import KINDS, estimate
from metasae_width.metrics import many_to_one, one_to_one
from metasae_width.models import BatchTopKSAE
from metasae_width.runs import RunId, load_arm, parse_arm, read_record, run_folder

ROLES = ("real", "twin")
COLUMNS = ("stage", "step", "measure", "kind", "n_seeds", "n_pairs", "point", "lo", "hi")


class Source(NamedTuple):
    """A run as scored: its folder (with the snapshots), the folder of the scored stage's files, that stage's record,
    and the snapshot steps before the stage."""

    folder: Path
    stage: Path
    record: dict[str, Any]
    steps: list[int]


class Stage(NamedTuple):
    """A scored stage: its name and step, and per role and seed the decoder file and the live-mask file (or None)."""

    name: str
    step: int
    files: dict[str, list[tuple[Path, Path | None]]]


def source(folder: Path, epochs: int | None) -> Source:
    """The run in ``folder`` at ``epochs`` (its current epochs by default); an earlier stage of an extended run comes
    from its ``epochs_<n>`` folder, and the snapshots before it from that stage's own record."""
    current = read_record(folder)
    have = current["settings"]["epochs"]
    want = have if epochs is None else epochs
    stage = folder if want == have else folder / f"epochs_{want}"
    record = current if stage == folder else read_record(stage)
    return Source(folder, stage, record, record["trainer"]["snapshot_steps"])


def common_setup(sources: dict[str, list[Source]]) -> Source:
    """The first run's source, after checking that every run has its width, rate, schedule and snapshot steps."""
    runs = [s for role in ROLES for s in sources[role]]

    def setup(s: Source) -> tuple[Any, ...]:
        return s.record["width"], s.record["settings"]["lr"], s.record["trainer"]["schedule"], s.steps

    for s in runs[1:]:
        if setup(s) != setup(runs[0]):
            names = f"{s.folder.name} differs from {runs[0].folder.name}"
            raise ValueError(f"{names} in its width, rate, schedule or snapshot steps")
    return runs[0]


def stages(sources: dict[str, list[Source]], first: Source) -> list[Stage]:
    """Every snapshot step, then the final stage."""

    def stage(name: str, step: int, files: Callable[[Source], tuple[Path, Path | None]]) -> Stage:
        return Stage(name, step, {role: [files(s) for s in ss] for role, ss in sources.items()})

    def snapshot(step: int) -> Stage:
        return stage(f"step_{step}", step, lambda s: (s.folder / "snapshots" / f"step_{step}.safetensors", None))

    steps = first.record["trainer"]["schedule"]["steps"]
    final = stage("final", steps, lambda s: (s.stage / "dictionary.safetensors", s.stage / "live.safetensors"))
    return [*map(snapshot, first.steps), final]


def _tensor(path: Path, name: str) -> torch.Tensor:
    with safe_open(path, framework="pt") as f:
        return f.get_tensor(name)


def pair_values(dec_a: Path, dec_b: Path, live_a: Path | None = None, live_b: Path | None = None) -> dict[str, float]:
    """Every measure between two dictionaries' decoders, read from their files; the live-latent measure when both
    live-mask files are given."""
    a, b = _tensor(dec_a, "W_dec"), _tensor(dec_b, "W_dec")
    match = one_to_one(a, b)
    values = {"matched": match.mean, "fraction_matched": match.fraction, "many_to_one": many_to_one(a, b)}
    if live_a is not None and live_b is not None:
        values["live_matched"] = one_to_one(a[_tensor(live_a, "live")], b[_tensor(live_b, "live")]).mean
    return values


def stage_matrices(stage: Stage, pool: Executor | None) -> dict[str, dict[str, np.ndarray]]:
    """Every measure's symmetric seed-by-seed matrix for real and twin, one task per role and pair."""
    n = len(stage.files["real"])
    jobs = [(role, i, j) for role in stage.files for i, j in itertools.combinations(range(n), 2)]
    args = [
        (*(stage.files[role][k][0] for k in (i, j)), *(stage.files[role][k][1] for k in (i, j))) for role, i, j in jobs
    ]
    results = list(
        itertools.starmap(pair_values, args) if pool is None else pool.map(pair_values, *zip(*args, strict=True))
    )
    matrices = {m: {role: np.zeros((n, n)) for role in stage.files} for m in results[0]}
    for (role, i, j), values in zip(jobs, results, strict=True):
        for measure, value in values.items():
            matrices[measure][role][i, j] = matrices[measure][role][j, i] = value
    return matrices


def stage_lines(name: str, step: int, matrices: dict[str, dict[str, np.ndarray]]) -> list[str]:
    """The TSV rows of one stage: each measure's real, twin, excess and corrected estimates."""
    n = len(matrices["matched"]["real"])
    lines = []
    for measure, by_role in matrices.items():
        for kind in KINDS:
            cell = (name, step, measure, kind, n, n * (n - 1) // 2, *estimate(kind, by_role["real"], by_role["twin"]))
            lines.append("\t".join(str(c) if isinstance(c, (int, str)) else repr(float(c)) for c in cell))
    return lines


def load_sae(folder: Path) -> BatchTopKSAE:
    side = json.loads((folder / "dictionary.json").read_text())
    sae = BatchTopKSAE(side["dim"], side["width"], side["k"])
    sae.load_state_dict(load_file(folder / "dictionary.safetensors"))
    return sae


def training_fit(
    seeds: list[int], sources: dict[str, list[Source]], train_rows: Callable[[str], torch.Tensor]
) -> dict[str, dict[str, dict[str, float]]]:
    """Each run's squared error and variance explained on its own training rows (``train_rows(role)``), all in one
    batch as in training. One run at a time on the CPU: the widest width needs about 16 GiB."""
    fit: dict[str, dict[str, dict[str, float]]] = {}
    for role, role_sources in sources.items():
        rows = train_rows(role)
        fit[role] = {}
        for seed, s in zip(seeds, role_sources, strict=True):
            mse, ve = train._score(load_sae(s.stage), rows)
            fit[role][str(seed)] = {"train_mse": mse, "train_ve": ve}
            print(f"fit {s.folder.name}")
    return fit


def score_runs(
    prefix: str,
    seeds: list[int],
    sources: dict[str, list[Source]],
    train_rows: Callable[[str], torch.Tensor],
    root: Path,
    workers: int,
) -> Path:
    """Pairwise agreement among the seeds of the real runs and their twins (``sources`` by role, in seed order) at each
    snapshot and at the end, with 90% jackknife intervals for real, twin, excess and corrected, and each run's fit to
    its training rows. ``workers`` computes the pairs in that many processes. Writes
    ``<root>/scores/<prefix>.s<seeds>.e<epochs>.tsv`` and the same stem ``.json`` with the seed labels, every seed
    pair's value per stage, measure and role, and the fit."""
    if len(seeds) < 3:
        raise ValueError("scoring needs at least three seeds")
    first = common_setup(sources)
    lines, pairs, index = ["\t".join(COLUMNS)], {}, list(itertools.combinations(range(len(seeds)), 2))
    executor = ProcessPoolExecutor(workers, initializer=torch.set_num_threads, initargs=(1,)) if workers > 1 else None
    with executor or nullcontext() as pool:
        for stage in stages(sources, first):
            matrices = stage_matrices(stage, pool)
            lines += stage_lines(stage.name, stage.step, matrices)
            pairs[stage.name] = {
                measure: {
                    role: {f"{seeds[i]}-{seeds[j]}": float(m[i, j]) for i, j in index} for role, m in by_role.items()
                }
                for measure, by_role in matrices.items()
            }
            print(f"scored {stage.name}")
    fit = training_fit(seeds, sources, train_rows)
    stem = f"{prefix}.s{'-'.join(map(str, seeds))}.e{first.record['settings']['epochs']}"
    path = root / "scores" / f"{stem}.tsv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    path.with_suffix(".json").write_text(json.dumps({"seeds": seeds, "pairs": pairs, "fit": fit}, indent=2) + "\n")
    return path


def score(
    arm: str, denominator: int, seeds: list[int], root: Path = ROOT, *, epochs: int | None = None, workers: int = 1
) -> Path:
    """Score an arm's real runs against their twins (see ``score_runs``). ``epochs`` scores an extended run at an
    earlier stage. Writes ``<root>/scores/<arm>.w<den>.s<seeds>.e<epochs>.tsv`` and ``.json``."""
    size, quarter = parse_arm(arm)
    ids = {role: [RunId(size, quarter, denominator, role, s) for s in seeds] for role in ROLES}
    sources = {role: [source(run_folder(root, rid), epochs) for rid in role_ids] for role, role_ids in ids.items()}
    return score_runs(
        f"{arm}.w{denominator}", seeds, sources, lambda role: load_arm(ids[role][0], root)[0].train, root, workers
    )
