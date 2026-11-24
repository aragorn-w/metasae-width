"""Tables from the score files, under ``<root>/summary/``. The width sweep is read on two bases: ``e2000``
(every width at 2,000 epochs) and ``final`` (each width at the longest length it has a score file for)."""

import csv
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

from metasae_width.gemma_scope2 import ROOT, SIZES
from metasae_width.jackknife import KINDS
from metasae_width.runs import RunId, parse_arm, read_record, run_folder
from metasae_width.score import COLUMNS, ROLES
from metasae_width.summary_scores import (
    ARMS,
    SEEDS,
    SWEEP,
    VIEWS,
    Key,
    Score,
    check_pairs,
    gap,
    load_scores,
    near_best,
    order,
    pick,
)
from metasae_width.train import EPOCHS

# Each extra arm and the arm it is compared with: three random quarters of the 4b's rows, and the 16k parents.
BASE = {"4bq": "4b", "4bq2": "4b", "4bq3": "4b", "4b16k": "4b", "270m16k": "270m"}
BASES = ("e2000", "final")
LONG = 6500  # the epochs at which the extra arms have the base arm's steps at 2,000
# Rows tables: (comparison, epochs of the extra arm's files, factor from its denominator to the base arm's at EPOCHS).
COMPARISONS = (
    ("same_rows_per_latent", EPOCHS, 1),
    ("same_latents_equal_steps", LONG, 4),
    ("same_width_equal_steps", LONG, 1),
)
# The rate check: (size, widths) trained again at 1e-3, the main runs' ids under <root>-lr1e-3.
RATE_CHECK = (("27b", (4, 32, 64, 128)), ("270m", (64, 128)))
INTERVAL = ("point", "lo", "hi")
VIEW_NAMES = {"": "matched_excess", "_live": "live_matched_excess", "_corrected": "corrected_matched_excess"}
BEST_4B256_COLUMNS = [
    *("basis", "denominator", "latents", "epochs"),
    *(
        f"{view}_{c}"
        for view in VIEW_NAMES.values()
        for c in (*INTERVAL, *(f"gap_{i}" for i in INTERVAL), "near_best", "is_top")
    ),
]
VALIDATE_MEASURES = ("matched", "fraction_matched", "live_matched")
VALIDATE_COLUMNS = ["measure", "kind", "n_seeds", "n_pairs", *INTERVAL]


def best_tables(scores: dict[Key, Score]) -> tuple[list[list[Any]], list[list[Any]], list[str]]:
    """The rows of best.tsv and best_minus_quarter.tsv, and notes on what could not be compared."""
    best, diff, notes = [], [], []
    for basis in BASES:
        chosen = pick(scores, basis)
        for size in SIZES:
            widths = {den: s for (arm, den), s in chosen.items() if arm == size}
            if not widths:
                continue
            views = {suffix: near_best(widths, *view) for suffix, view in VIEWS.items()}
            top = views[""][0]
            for den, s in widths.items():
                cells = [*s.est("matched", "excess"), *s.est("live_matched", "excess"), *s.est("matched", "corrected")]
                shared = [s.est("matched", "real").point, s.est("matched", "twin").point, den == top]
                best.append(
                    [size, basis, den, s.latents, s.epochs, *cells, *shared, *(v[1][den] for v in views.values())]
                )
            if 4 not in widths:
                notes.append(f"{size} {basis}: the 1/4 width is not scored, so no best_minus_quarter")
            else:
                diff.append(
                    [size, basis, top, widths[top].epochs, widths[4].epochs, *gap(widths[top], widths[4], *VIEWS[""])]
                )
    return best, diff, notes


def best_4b256(scores: dict[Key, Score]) -> tuple[list[list[Any]], list[str]]:
    """The 4b's widths over the sweep and 1/256 on each basis: per view the matched excess, the gap top minus the
    width (the top is the width with the largest point), whether the width is near-best, and whether it is the top;
    and notes on bases without a 1/256 score file, which get no rows."""
    out, notes = [], []
    for basis in BASES:
        widths = {den: s for (arm, den), s in pick(scores, basis, (*SWEEP, 256)).items() if arm == "4b"}
        if 256 not in widths:
            notes.append(f"4b {basis}: the 1/256 width is not scored, so no best_4b256 rows")
            continue
        views = {suffix: near_best(widths, *view) for suffix, view in VIEWS.items()}
        for den, s in widths.items():
            cells = []
            for suffix, view in VIEWS.items():
                top, flags = views[suffix]
                cells += [*s.est(*view), *gap(widths[top], s, *view), flags[den], den == top]
            out.append([basis, den, s.latents, s.epochs, *cells])
    return out, notes


def length_rows(scores: dict[Key, Score]) -> list[list[Any]]:
    """The matched excess at every stage of every score file; epochs = step / (final step / the file's epochs)."""
    rows = []
    for key in sorted(scores, key=order):
        s = scores[key]
        final = next(int(r["step"]) for r in s.rows if r["stage"] == "final")
        for r in s.rows:
            if (r["measure"], r["kind"]) == ("matched", "excess"):
                epochs = int(r["step"]) * s.epochs // final
                rows.append(
                    [s.arm, s.den, s.latents, s.epochs, r["stage"], r["step"], epochs, r["point"], r["lo"], r["hi"]]
                )
    return rows


def rows_table(scores: dict[Key, Score]) -> tuple[list[list[Any]], list[str]]:
    """Each extra arm's widths' excess beside its base arm's (the 4b's, or the 270m's for 270m16k) with the same rows
    per latent, with the same latents at equal steps and with the same width at equal steps, each with the difference
    (first minus second) and its jackknife interval; and the score files that are missing, whose rows are skipped."""
    out, missing = [], []
    for comparison, epochs, factor in COMPARISONS:
        for arm, den in ((a, d) for a in BASE for d in ARMS[a]):
            first_key, second_key = (arm, den, epochs), (BASE[arm], factor * den, EPOCHS)
            first, second = scores.get(first_key), scores.get(second_key)
            if first is None or second is None:
                missing += [
                    f"{k[0]}.w{k[1]}.e{k[2]}" for k, s in ((first_key, first), (second_key, second)) if s is None
                ]
                continue
            sides = [[s.arm, s.den, s.latents, s.epochs, *s.est()] for s in (first, second)]
            out.append([comparison, *sides[0], *sides[1], *gap(first, second, "matched", "excess")])
    return out, sorted(set(missing))


def rate_table(scores: dict[Key, Score], checked: dict[Key, Score]) -> tuple[list[list[Any]], list[str]]:
    """The rate check's widths at 2,000 epochs: the matched excess at 3e-3 (``scores``) and at 1e-3 (``checked``),
    with the difference (1e-3 minus 3e-3) and its jackknife interval; and the score files that are missing."""
    out, missing = [], []
    for size, dens in RATE_CHECK:
        for den in dens:
            shared, check = scores.get((size, den, EPOCHS)), checked.get((size, den, EPOCHS))
            if shared is None or check is None:
                missing += [
                    f"{tag}{size}.w{den}.e{EPOCHS}" for tag, s in (("", shared), ("lr1e-3 ", check)) if s is None
                ]
                continue
            cells = [size, den, shared.latents, EPOCHS, *shared.est(), *check.est(), *gap(check, shared, *VIEWS[""])]
            out.append(cells)
    return out, missing


def validate_rows(root: Path) -> tuple[list[list[str]], str]:
    """The final stage of the validation score file (seeds 0 to 7 at EPOCHS) for the three measures and four kinds, and
    the report line saying whether the file was found."""
    path = root / "scores" / f"validate.s{'-'.join(map(str, SEEDS))}.e{EPOCHS}.tsv"
    if not path.exists():
        return [], f"no validation score file {path.name}, so validate.tsv is empty"
    with path.open(newline="") as handle:
        found = {(r["measure"], r["kind"]): r for r in csv.DictReader(handle, delimiter="\t") if r["stage"] == "final"}
    rows = [[m, k, *(found[m, k][c] for c in VALIDATE_COLUMNS[2:])] for m in VALIDATE_MEASURES for k in KINDS]
    return rows, f"validation score file {path.name} found, so validate.tsv has its final stage"


def run_stages(root: Path, rid: RunId) -> dict[int, dict[str, Any]]:
    """A run's trainer section by epochs: the latest from record.json, earlier stages from epochs_<n>/record.json."""
    folder = run_folder(root, rid)
    try:
        latest = read_record(folder)
    except OSError, ValueError:
        return {}
    stages = {latest["settings"]["epochs"]: latest["trainer"]}
    for path in folder.glob("epochs_*/record.json"):
        old = json.loads(path.read_text())
        stages[old["settings"]["epochs"]] = old["trainer"]
    return stages


def runs_table(root: Path, scores: dict[Key, Score]) -> tuple[list[list[Any]], set[tuple[str, int]]]:
    """Mean over seeds 0 to 7 of dead fraction, training-row and test variance explained, per arm, denominator, epochs
    and role; and the (arm, denominator) pairs where a role lacks a run at an epoch count that either role reached."""
    rows, incomplete = [], set()
    for arm, dens in ARMS.items():
        size, quarter = parse_arm(arm)
        for den in dens:
            stages = {r: [run_stages(root, RunId(size, quarter, den, r, seed)) for seed in SEEDS] for r in ROLES}
            epochs = sorted(set().union(*(st for seeds in stages.values() for st in seeds)))
            if not epochs:
                incomplete.add((arm, den))
            for n, role in itertools.product(epochs, ROLES):
                trainers = [st[n] for st in stages[role] if n in st]
                if len(trainers) < len(SEEDS):
                    incomplete.add((arm, den))
                    continue
                dead = float(np.mean([t["dead_fraction"] for t in trainers]))
                test_ve = float(np.mean([t["final"]["test_ve"] for t in trainers]))
                s = scores.get((arm, den, n))
                train_ve = None if s is None else float(np.mean([v["train_ve"] for v in s.fit[role].values()]))
                rows.append([arm, den, n, role, len(trainers), dead, train_ve, test_ve])
    return rows, incomplete


def write_tsv(path: Path, columns: list[str], rows: list[list[Any]]) -> None:
    """Floats in full precision, True and False as 1 and 0, None as an empty cell."""

    def cell(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, bool):
            return str(int(value))
        return repr(float(value)) if isinstance(value, float) else str(value)

    lines = ["\t".join(columns), *("\t".join(cell(v) for v in row) for row in rows)]
    path.write_text("\n".join(lines) + "\n")


def _compact(missing: dict[str, list[int]]) -> str:
    """Arms with the widths listed against them, or ``all`` when every width of the arm is."""
    parts = [
        f"{arm} {'all' if dens == list(ARMS[arm]) else ' '.join(f'w{d}' for d in dens)}"
        for arm, dens in missing.items()
    ]
    return "; ".join(parts) or "none"


def report(
    scores: dict[Key, Score],
    notes: list[str],
    incomplete: set[tuple[str, int]],
    skipped: list[str],
    rate_skipped: list[str],
    out: Path,
) -> str:
    """The printout: the score files and runs that are missing, the notes on what could not be compared, and the
    folder written."""
    no_score = {a: [d for d in dens if (a, d, EPOCHS) not in scores] for a, dens in ARMS.items()}
    no_runs = {a: [d for d in dens if (a, d) in incomplete] for a, dens in ARMS.items()}
    text = ["no score file at 2,000 epochs: " + _compact({a: d for a, d in no_score.items() if d})]
    text += ["missing score files, so rows.tsv skips their rows: " + (", ".join(skipped) or "none")]
    text += ["missing score files, so rate.tsv skips their rows: " + (", ".join(rate_skipped) or "none")]
    text += ["runs missing or incomplete (runs.tsv): " + _compact({a: d for a, d in no_runs.items() if d})]
    return "\n".join([*text, *notes, "", f"wrote {out}"])


def summarize(root: Path = ROOT) -> str:
    """Write the tables under ``<root>/summary/`` and return the printout. Missing data is skipped."""
    scores, checked = load_scores(root), load_scores(Path(f"{root.resolve()}-lr1e-3"))
    for s in [*scores.values(), *checked.values()]:
        check_pairs(s)
    out = root / "summary"
    out.mkdir(parents=True, exist_ok=True)
    head = ["arm", "denominator", "latents", "epochs"]
    final = [
        [s.arm, s.den, s.latents, s.epochs, *(r[c] for c in COLUMNS)]
        for s in (scores[k] for k in sorted(scores, key=order))
        for r in s.rows
        if r["stage"] == "final"
    ]
    best, diff, notes = best_tables(scores)
    lengths, (runs, incomplete), (rows, skipped) = length_rows(scores), runs_table(root, scores), rows_table(scores)
    rate, rate_skipped = rate_table(scores, checked)
    best256, notes256 = best_4b256(scores)
    validate, validate_line = validate_rows(root)
    side = ["arm", "denominator", "latents", "epochs", *(f"excess_{c}" for c in INTERVAL)]
    tables = {
        "excess": ([*head, *COLUMNS], final),
        "best": (
            [
                *["arm", "basis", *head[1:]],
                *(f"{v}_{c}" for v in VIEW_NAMES.values() for c in INTERVAL),
                *["real_matched", "twin_matched", "is_top", "near_best", "near_best_live", "near_best_corrected"],
            ],
            best,
        ),
        "best_minus_quarter": (["arm", "basis", "top_denominator", "top_epochs", "quarter_epochs", *INTERVAL], diff),
        "best_4b256": (BEST_4B256_COLUMNS, best256),
        "length": ([*head[:3], "run_epochs", "stage", "step", "epochs", *INTERVAL], lengths),
        "runs": (["arm", "denominator", "epochs", "role", "n_seeds", "dead_fraction", "train_ve", "test_ve"], runs),
        "rows": (
            [
                "comparison",
                *(f"first_{c}" for c in side),
                *(f"second_{c}" for c in side),
                *(f"difference_{c}" for c in INTERVAL),
            ],
            rows,
        ),
        "rate": (
            [
                *head,
                *(f"lr{tag}_excess_{c}" for tag in ("3e-3", "1e-3") for c in INTERVAL),
                *(f"difference_{c}" for c in INTERVAL),
            ],
            rate,
        ),
        "validate": (VALIDATE_COLUMNS, validate),
    }
    for name, (columns, table) in tables.items():
        write_tsv(out / f"{name}.tsv", columns, table)
    return report(scores, [*notes, *notes256, validate_line], incomplete, skipped, rate_skipped, out)
