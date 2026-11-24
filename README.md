# metasae-width

Code for the paper in `paper/`, which measures how reproducible meta-SAEs are across seeds. A meta-SAE here is a BatchTopK SAE trained on a parent SAE's decoder rows, scaled to unit length. Each setup trains eight seeds, matches their decoder rows one to one, and reports excess agreement: the real runs' mean matched cosine minus that of a noise twin trained on unit-length draws from a Gaussian with the mean and covariance of the parent's unit rows, with 90% jackknife intervals. The sweep covers widths 1/4 to 1/128 on five Gemma Scope 2 parents (Gemma 3 270M to 27B), plus a GPT-2 small reference.

## Requirements

- Python 3.15 and [uv](https://docs.astral.sh/uv/). Dependencies are pinned in `uv.lock`.
- A CUDA GPU for `train`, `extend` and `validate` (one card per process, chosen with `CUDA_VISIBLE_DEVICES`). Everything else runs on the CPU; `score` needs about 16 GiB at the widest width.
- Access to the Hugging Face Hub for `check-sae`, `validate`, `score-validate` and `spectrum`, which download the SAEs at pinned revisions.
- A TeX installation: `figures` and `examples` set their text with LaTeX, and the paper builds with `pdflatex` and `bibtex`.

## Setup

```
uv sync
```

## Rebuild the tables and figures

With the score files in `<root>/scores/` (`<root>` is the data folder, see Layout):

```
uv run pytest
uv run python -m metasae_width.run summarize --root <root>
uv run python -m metasae_width.run figures --root <root>
uv run python -m metasae_width.run examples --root <root>
```

`summarize` writes the tables to `<root>/summary/`. It also reads the run records in `<root>/runs/` and the learning-rate check in `<root>-lr1e-3`, and reports missing score files and runs. `figures` writes `width.pdf`, `arms.pdf` and `length.pdf` to `paper/figures` (or to `--out DIR`). `examples` reads the runs in `<root>/runs/` and fetches Neuronpedia labels (it needs network access), then writes `examples.json` and `examples.pdf` to `paper/figures` (or `--out DIR`); `--redraw` draws the figure from the saved `examples.json` without recomputing or fetching.

## Full pipeline

```
mw() { uv run python -m metasae_width.run "$@"; }

mw check-sae 4b --root <root>                       # size: 270m, 1b, 4b, 12b, 27b, 4b16k or 270m16k
mw train 4b.w16.real.s0 --lr 3e-3 --root <root>     # run id: <size>[q|q2|q3].w<denominator>.<real|twin>.s<seed>
mw train 4b.w16.real.s100 --lr 1e-3 --root <root>   # tuning seed 100, at 1e-3, 3e-3 and 1e-2
mw tune-pick 4b 16 --root <root>                    # validation MSE of the three tuned rates
mw extend-check 4b 16 --seeds 0 1 2 3 4 5 6 7 --root <root>   # length rule: converged, extend or not_converged
mw extend 4b.w16.real.s0 --epochs 4000 --root <root>          # --lr too for a seed-100 run
mw score 4b 16 --seeds 0 1 2 3 4 5 6 7 --root <root>          # also [--epochs N] [--workers N]
mw validate --seed 0 --root <root>                  # GPT-2 reference vs. the released meta-SAE; [--epochs N]
mw validate --seed 0 --twin --root <root>           # the same on the parent's noise twin
mw score-validate --seeds 0 1 2 3 4 5 6 7 --root <root>       # also [--workers N]
mw summarize --root <root>
mw figures --root <root>
mw spectrum --root <root>                           # covariance spectra of parents and twins
```

`train` runs 2,000 epochs. Train seeds 0 to 7 of both the real and the twin runs of a setup (`w16` is width 1/16) before scoring. The learning-rate check repeats the same steps in `<root>-lr1e-3` with `--lr 1e-3`. Always pass `--root`; the GPT-2 downloads are cached under the default root (`EXTERNAL_CACHE` in `metasae_width/validate.py`).

## Layout

```
<root>/
  sae/<size>/      downloaded SAE (hf/), and the noise twin (noise.safetensors, noise.json)
  runs/<run id>/   record.json, log.tsv, dictionary.safetensors, live.safetensors, snapshots/
  scores/          <arm>.w<den>.s<seeds>.e<epochs>.tsv and .json (every seed pair's value)
  summary/         the tables from summarize and spectrum
  external/gpt2/   the noise twin of the GPT-2 parent
```

## Building the paper

```
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```
