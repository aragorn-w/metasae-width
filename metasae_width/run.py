"""The command line: ``python -m metasae_width.run <command> ...``.

Files go under ``/mnt/fast/metasae-width`` (``--root``): ``sae/<size>/``, ``runs/``, ``scores/`` and ``summary/``."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from metasae_width.checks import extend_check, tune_pick
from metasae_width.examples import examples
from metasae_width.gemma_scope2 import ROOT
from metasae_width.runs import check_sae, extend_run, lr_tag, train_run
from metasae_width.score import score
from metasae_width.spectrum import spectrum
from metasae_width.summary import summarize
from metasae_width.summary_figures import FIGURES, figures
from metasae_width.train import EPOCHS
from metasae_width.validate import score_validate, validate_run


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m metasae_width.run")
    commands = parser.add_subparsers(dest="command", required=True)

    def command(name: str, help: str) -> argparse.ArgumentParser:
        sub = commands.add_parser(name, help=help)
        sub.add_argument("--root", type=Path, default=ROOT, help="the output root (default /mnt/fast/metasae-width)")
        return sub

    command("check-sae", "download a size's SAE and draw its noise twin").add_argument("size")
    train = command("train", "train one run for 2,000 epochs on the card CUDA_VISIBLE_DEVICES names")
    train.add_argument("run_id")
    train.add_argument("--lr", type=float, required=True, help="3e-3 for the main runs, 1e-3 for the rate check")
    ext = command("extend", "continue a run to more epochs")
    ext.add_argument("run_id")
    ext.add_argument("--epochs", type=int, required=True)
    ext.add_argument("--lr", type=float, help="required for a tuning run (seed 100)")
    for name, help in (
        ("tune-pick", "the shared learning rate's validation loss beside the other two tuned rates"),
        ("extend-check", "the length rule"),
        ("score", "pairwise agreement for real and twin runs, with jackknife intervals"),
    ):
        sub = command(name, help)
        sub.add_argument(
            "size", help="<size> (270m, 1b, 4b, 12b, 27b, 4b16k or 270m16k), or <size>q, q2 or q3 for a random quarter"
        )
        sub.add_argument("denominator", type=int)
        if name != "tune-pick":
            sub.add_argument("--seeds", type=int, nargs="+", required=name == "score")
        if name == "score":
            sub.add_argument("--epochs", type=int, help="score a run extended past this many epochs at that stage")
            sub.add_argument("--workers", type=int, default=1, help="processes computing the seed pairs")
    command("summarize", "tables from the score files, under <root>/summary/")
    command("figures", "the paper's three figures from the summary tables").add_argument(
        "--out",
        type=Path,
        default=FIGURES,
        help="the folder for width.pdf, arms.pdf and length.pdf (default paper/figures)",
    )
    ex = command("examples", "the example meta-latent figure; it fetches Neuronpedia labels")
    ex.add_argument(
        "--out",
        type=Path,
        default=FIGURES,
        help="the folder for examples.json and examples.pdf (default paper/figures)",
    )
    ex.add_argument(
        "--redraw", action="store_true", help="draw from the saved examples.json without recomputing or fetching"
    )
    command("spectrum", "the covariance spectrum of each parent's rows and its noise twin's, in <root>/summary/")
    val = command("validate", "train on the released meta-SAE's parent and compare, or on its noise twin")
    val.add_argument("--seed", type=int, required=True)
    val.add_argument("--epochs", type=int, default=EPOCHS)
    val.add_argument("--twin", action="store_true", help="train on the noise twin of the parent, with no comparison")
    sv = command("score-validate", "agreement between the validate runs and their twins, with jackknife intervals")
    sv.add_argument("--seeds", type=int, nargs="+", required=True)
    sv.add_argument("--workers", type=int, default=1, help="processes computing the seed pairs")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command. Return 1 on a failed validation, else 0; any error raises."""
    args = _parser().parse_args(argv)
    if args.command == "check-sae":
        check_sae(args.size, args.root)
    elif args.command == "train":
        train_run(args.run_id, args.lr, root=args.root)
    elif args.command == "extend":
        extend_run(args.run_id, args.epochs, lr=args.lr, root=args.root)
    elif args.command == "tune-pick":
        table, message = tune_pick(args.size, args.denominator, args.root)
        print(json.dumps({lr_tag(lr): mse for lr, mse in table.items()}))
        print(message)
    elif args.command == "extend-check":
        verdict, message = extend_check(args.size, args.denominator, args.seeds, args.root)
        print(json.dumps({"verdict": verdict}))
        print(message)
    elif args.command == "score":
        print(score(args.size, args.denominator, args.seeds, args.root, epochs=args.epochs, workers=args.workers))
    elif args.command == "summarize":
        print(summarize(args.root))
    elif args.command == "figures":
        print(figures(args.root, args.out))
    elif args.command == "examples":
        print(examples(args.root, args.out, redraw=args.redraw))
    elif args.command == "spectrum":
        print(spectrum(args.root))
    elif args.command == "validate":
        return 0 if validate_run(args.seed, args.epochs, args.root, twin=args.twin) else 1
    elif args.command == "score-validate":
        print(score_validate(args.seeds, args.root, workers=args.workers))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
