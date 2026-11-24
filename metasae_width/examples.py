"""The example meta-latent figure: per size, a seed-0 meta-latent that recurs in all other seeds and one in none."""

import json
import math
import textwrap
import time
import urllib.request
from pathlib import Path
from typing import Any

import circlify
import matplotlib
import numpy as np
import torch
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from scipy.optimize import linear_sum_assignment

from metasae_width.data import unit_rows
from metasae_width.gemma_scope2 import ROOT, SAES, load_decoder
from metasae_width.score import load_sae
from metasae_width.summary_figures import FIGURES, STYLE, TEXT_WIDTH

PANELS = (("270m", 4, "270M"), ("4b", 128, "4B"), ("27b", 32, "27B"))  # size, width denominator, label
SEEDS = range(8)
OTHERS = len(SEEDS) - 1
COSINE = np.float32(0.7)  # a matched pair at or above this cosine recurs
TOP = 7  # parent latents per example meta-latent
UNIT = 0.095  # inches: radius of a parent bubble fired on in 1 seed; 8 seeds gives sqrt(8) times this
CAP = 0.18  # inches: the outer bubble is this much wider than the packed parents, all of it above them
INSIDE = 0.18  # inches: parent bubbles at least this radius carry their label inside
HEIGHT = 4.8  # inches
ROWS = (3.3, 1.35)  # inches: the bottom of the recurring and of the non-recurring bubble
# Descriptions from three AI readers who were blind to the row and the model size, shown only when all three agreed
# (else "no clear theme").
DESCRIPTIONS = {
    ("270m", 12414): "Title Case and capitalized text",
    ("270m", 15): "no clear theme",
    ("4b", 451): "old newspaper and archival text",
    ("4b", 361): "no clear theme",
    ("27b", 203): "the word ``at'' (and ``@'')",
    ("27b", 1193): "no clear theme",
}
# LaTeX escapes for label text, applied in one pass.
TEX = str.maketrans(
    {c: "\\" + c for c in "&%$#_{}"}
    | {"\\": r"\textbackslash{}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
)


def choose(recur: np.ndarray, fires: np.ndarray) -> list[tuple[int, int]]:
    """Of live meta-latents recurring in all seven other seeds, then in none: the median by firing count, and size."""
    live = fires > 0
    groups = np.where((recur == OTHERS) & live)[0], np.where((recur == 0) & live)[0]
    return [(int(g[np.argsort(fires[g], kind="stable")][len(g) // 2]), len(g)) for g in groups]


@torch.no_grad()
def _panel(size: str, den: int, root: Path) -> dict[str, Any]:
    """Seed 0's two example meta-latents at one width (a run's top-level dictionary is its longest training length)."""
    saes = [load_sae(root / "runs" / f"{size}.w{den}.real.s{seed}") for seed in SEEDS]
    decs = [unit_rows(sae.W_dec) for sae in saes]
    recur = np.zeros(len(decs[0]), dtype=int)
    partners = []
    for dec in decs[1:]:
        sim = (decs[0] @ dec.T).numpy()
        rows, cols = linear_sum_assignment(-sim)
        partners.append(cols)
        recur += sim[rows, cols] >= COSINE
    parent = unit_rows(load_decoder(size, root))
    acts = torch.cat([saes[0](x, "threshold").latents for x in parent.split(4096)])
    fires = (acts > 0).sum(0).numpy()
    chosen = choose(recur, fires)
    return {
        "denominator": den,
        "recurring": chosen[0][1],
        "nonrecurring": chosen[1][1],
        "picks": [_pick(meta, int(recur[meta]), int(fires[meta]), acts, parent, saes, partners) for meta, _ in chosen],
    }


def _pick(
    meta: int,
    recur: int,
    fires: int,
    acts: torch.Tensor,
    parent: torch.Tensor,
    saes: list[Any],
    partners: list[np.ndarray],
) -> dict[str, Any]:
    """A meta-latent's top parent latents, and per parent the seeds (of 8) whose matched meta-latent fires on it."""
    top = torch.topk(acts[:, meta], TOP)
    parents = [(int(i), round(float(v), 4)) for i, v in zip(top.indices, top.values, strict=True) if v > 0]
    rows = parent[[i for i, _ in parents]]
    seeds = [1] * len(parents)
    for sae, partner in zip(saes[1:], partners, strict=True):
        z = sae(rows, "threshold").latents[:, int(partner[meta])]
        seeds = [n + int(v > 0) for n, v in zip(seeds, z.tolist(), strict=True)]
    return {
        "meta_latent": meta,
        "recurs_in_other_seeds": recur,
        "fires_on": fires,
        "parents": [{"id": i, "activation": a, "seeds": n} for (i, a), n in zip(parents, seeds, strict=True)],
    }


def _neuronpedia(size: str, feature: int) -> dict[str, Any]:
    """Neuronpedia's first explanation of a parent latent, its HF source, and its six top activating texts."""
    sae = SAES[size]
    url = f"https://www.neuronpedia.org/api/feature/gemma-3-{sae.model}/{sae.layer}-gemmascope-2-res-{sae.width_tag}/{feature}"
    with urllib.request.urlopen(
        urllib.request.Request(url, headers={"User-Agent": "research-script/0.1"}), timeout=60
    ) as r:
        data = json.load(r)
    time.sleep(0.4)
    first = (data.get("explanations") or [{}])[0]
    source = data.get("source") or {}
    top = sorted(data.get("activations", []), key=lambda a: -a.get("maxValue", 0))[:6]
    return {
        "label": first.get("description"),
        "label_model": first.get("explanationModelName"),
        "hf": [source.get("hfRepoId"), source.get("hfFolderId")],
        "contexts": [_context(a["tokens"], a["maxValueTokenIndex"]) for a in top],
    }


def _context(tokens: list[str], k: int) -> str:
    """The max token in double brackets, with six tokens before it and two after."""
    before, after = "".join(tokens[max(0, k - 6) : k]), "".join(tokens[k + 1 : k + 3])
    return f"{before} [[{tokens[k]}]] {after}".replace("\n", "\\n")


def _tex(s: str) -> str:
    return s.translate(TEX)


def _label(parent: dict[str, Any], width: int, lines: int) -> str:
    text = "(no label)" if parent["label"] is None else parent["label"]
    text = textwrap.shorten(text, width * lines, placeholder="...")
    wrapped = textwrap.wrap(text, width, break_long_words=False, max_lines=lines, placeholder="...")
    return "\n".join(_tex(w) for w in wrapped)


def _fill(n: int) -> Any:
    return matplotlib.colormaps["Blues"](0.12 + 0.43 * n / 8)


def _layout(counts: list[int]) -> tuple[float, list[tuple[float, float, float]]]:
    """Pack circles of area proportional to counts in a unit circle; return the enclosing radius in inches and each
    child's (x, y, r) in inches relative to the enclosure's centre, in input order."""
    circles = circlify.circlify([{"id": k, "datum": c} for k, c in enumerate(counts)])
    circles.sort(key=lambda c: c.ex["id"])
    scale = UNIT * math.sqrt(max(counts)) / circles[counts.index(max(counts))].r  # inches per unit of the enclosure
    return scale, [(c.x * scale, c.y * scale, c.r * scale) for c in circles]


def _side_labels(
    ax: Any, cx: float, cy: float, R: float, outside: list[tuple[float, float, float, dict[str, Any]]]
) -> None:
    """Labels of the parent bubbles too small to carry their own, on the left and right of the outer bubble."""
    for side in (-1, 1):
        group = [t for t in outside if (t[0] >= 0) == (side > 0)]
        want = sorted(
            ((min(max(cy + t[1] * 1.15, cy - R + 0.05), cy + R - 0.12), t) for t in group), key=lambda w: -w[0]
        )
        placed, last = [], math.inf
        for y_want, t in want:
            last = min(y_want, last - 0.2)
            placed.append((last, t))
        shift = max(0, cy - R - 0.1 - last)
        tx = cx + side * (R + 0.1)
        for y, (x, yy, r, p) in placed:
            y += shift
            a = math.atan2(yy, x)
            ax.plot(
                [cx + x + r * math.cos(a), tx - side * 0.025],
                [cy + yy + r * math.sin(a), y],
                color="#9a9a9a",
                lw=0.4,
                zorder=1,
            )
            ax.text(
                tx,
                y,
                _label(p, 15, 2),
                ha="left" if side > 0 else "right",
                va="center",
                fontsize=5.3,
                linespacing=1.0,
            )


def _bubble(ax: Any, cx: float, cy: float, size: str, pick: dict[str, Any]) -> None:
    parents = pick["parents"]
    R, kids = _layout([p["seeds"] for p in parents])
    ax.add_patch(Circle((cx, cy + CAP), R + CAP, facecolor="#f4f4f4", edgecolor="#9a9a9a", linewidth=0.6, zorder=1))
    text = DESCRIPTIONS.get((size, pick["meta_latent"]))
    if text:
        ax.text(
            cx,
            cy + R + 0.17,
            "\n".join(rf"\textit{{{_tex(w)}}}" for w in textwrap.wrap(text, 34)),
            ha="center",
            va="center",
            fontsize=6.3,
            linespacing=1.0,
            zorder=3,
        )
    ax.text(
        cx,
        cy + R + 2 * CAP + 0.05,
        f"meta-latent {pick['meta_latent']}: recurs in {pick['recurs_in_other_seeds']} of 7 other seeds",
        ha="center",
        va="bottom",
        fontsize=6.5,
        color="#333333",
    )
    outside = []
    for (x, y, r), p in zip(kids, parents, strict=True):
        ax.add_patch(
            Circle((cx + x, cy + y), r, facecolor=_fill(p["seeds"]), edgecolor="white", linewidth=0.6, zorder=2)
        )
        if r >= INSIDE:
            ax.text(cx + x, cy + y, _label(p, 13, 3), ha="center", va="center", fontsize=5.3, zorder=3, linespacing=1.0)
        else:
            outside.append((x, y, r, p))
    _side_labels(ax, cx, cy, R, outside)


def _key(ax: Any) -> None:
    x, ky = 0.35, 0.06
    for n in (1, 2, 4, 8):
        r = UNIT * math.sqrt(n)
        ax.add_patch(Circle((x + r, ky + r), r, facecolor=_fill(n), edgecolor="white", linewidth=0.6))
        ax.text(x + r, ky + r, str(n), ha="center", va="center", fontsize=6)
        x += 2 * r + 0.08
    ax.text(
        x + 0.05,
        ky + 0.15,
        "inner bubble area: seeds (of 8) whose matched meta-latent fires on that parent "
        "latent; outer bubble: one meta-latent of seed 0",
        fontsize=6,
        va="center",
    )


def draw(data: dict[str, Any], path: Path) -> None:
    """One column per size: the recurring example above the non-recurring one."""
    with matplotlib.rc_context(STYLE):
        fig = Figure(figsize=(TEXT_WIDTH, HEIGHT))
        ax = fig.add_axes((0, 0, 1, 1))
        ax.set_xlim(0, TEXT_WIDTH)
        ax.set_ylim(0, HEIGHT)
        ax.set_aspect("equal")
        ax.axis("off")
        for c, (size, den, name) in enumerate(PANELS):
            cx = TEXT_WIDTH / 3 * (c + 0.5)
            ax.text(
                cx, HEIGHT - 0.04, rf"\textbf{{Gemma 3 {name}}}, width 1/{den}", ha="center", va="top", fontsize=7.5
            )
            for pick, cy in zip(data[size]["picks"], ROWS, strict=True):
                _bubble(ax, cx, cy, size, pick)
        _key(ax)
        fig.savefig(path)


def examples(root: Path = ROOT, out: Path = FIGURES, redraw: bool = False) -> str:
    """Write examples.json and examples.pdf to ``out``; with ``redraw``, only draw the saved examples.json."""
    out.mkdir(parents=True, exist_ok=True)
    path = out / "examples.json"
    if not redraw:
        data = {size: _panel(size, den, root) for size, den, _ in PANELS}
        for size, panel in data.items():
            for pick in panel["picks"]:
                for parent in pick["parents"]:
                    parent.update(_neuronpedia(size, parent["id"]))
        path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    data = json.loads(path.read_text(encoding="utf-8"))
    draw(data, out / "examples.pdf")
    return f"wrote {out}/examples.pdf"
