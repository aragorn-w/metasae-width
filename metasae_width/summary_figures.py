"""The paper's three figures (``width.pdf``, ``arms.pdf``, ``length.pdf``), drawn from ``excess.tsv``, ``best.tsv`` and
``length.tsv`` under ``<root>/summary/``, so every plotted value is a row the paper's tables also read."""

import csv
from pathlib import Path
from typing import Any, NamedTuple

import matplotlib
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

from metasae_width.gemma_scope2 import ROOT, SIZES

FIGURES = Path(__file__).resolve().parents[1] / "paper" / "figures"
INK, GRID, SURFACE = "black", "#d9d9d9", "white"
# Okabe-Ito blue, orange and bluish green: real, twin, excess.
SERIES = {"real": "#0072B2", "twin": "#E69F00", "excess": "#009E73"}
MARKERS = {"real": "o", "twin": "s", "excess": "D"}
GREENS = ["#6CC6AD", SERIES["excess"], "#004F39"]  # a ramp through the excess green, light to dark
# The six main widths, blue (1/4) to red (1/128), lightest in the middle. Unlike a sweep from blue through purple to
# red, it keeps neighbouring widths apart under simulated red-green colour blindness.
WIDTH_COLORS = ("#084491", "#296EB4", "#5A98CB", "#D67D5E", "#C74D36", "#A92321")
STYLE: dict[Any, Any] = {
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.ymargin": 0.1,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.2,
    "lines.markersize": 3.5,
    "lines.markeredgewidth": 0.4,
    "lines.markeredgecolor": SURFACE,
    "legend.frameon": False,
    # LaTeX text in the paper's fonts, embedded as Type 1: Times (Nimbus Roman) for text, as times.sty and mathptmx set
    # it, though maths symbols embed as Computer Modern. Naming all three families keeps matplotlib's default lists,
    # which name Computer Modern and so need the cm-super package (not installed), out of the preamble.
    "text.usetex": True,
    "font.family": "times",
    "font.sans-serif": ["helvetica"],
    "font.monospace": ["courier"],
    "font.size": 7.5,
    "axes.titlesize": 7.5,
    "figure.labelsize": 7.5,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 7,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
}
WIDTHS = (4, 8, 16, 32, 64, 128)  # the six main widths, as denominators
NEAR = "near-best width"
# Offset in points, horizontal and vertical alignment of a size's label on the sixth panel, against its curve's
# narrowest point: by default left of it, moved where the default would sit on a neighbouring label or line.
LABELS = {"270m": ((-3, 1), "right", "bottom"), "1b": ((-3, -1), "right", "top"), "12b": ((0, -3), "center", "top")}

Row = dict[str, str]
Est = tuple[float, float, float]  # point, lo, hi


class Entity(NamedTuple):
    """An arm of the controls figure: its legend label, the epochs and denominators of the score files it draws (equal
    steps, 26,000), and its line style, marker and whether the marker is filled."""

    label: str
    epochs: int
    dens: tuple[int, ...]
    line: str
    marker: str
    filled: bool


ENTITIES = {
    "4b": Entity("4B 65k", 2000, (*WIDTHS, 256), "-", "o", True),
    "4bq": Entity("quarter draw 1", 6500, (4, 16, 64), ":", "^", False),
    "4bq2": Entity("quarter draw 2", 6500, (4, 16, 64), ":", "v", False),
    "4bq3": Entity("quarter draw 3", 6500, (4, 16, 64), ":", "D", False),
    "4b16k": Entity("4B 16k", 6500, (4, 16, 64), "--", "s", True),
}


def read_tsv(path: Path) -> list[Row]:
    with path.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _est(row: Row) -> Est:
    return float(row["point"]), float(row["lo"]), float(row["hi"])


def width_curves(excess: list[Row]) -> dict[str, dict[str, dict[int, Est]]]:
    """Size, kind and denominator to the matched agreement at the final stage of 2,000 epochs, at the six main widths
    and, for the 4B only, 1/256."""
    curves: dict[str, dict[str, dict[int, Est]]] = {size: {kind: {} for kind in SERIES} for size in SIZES}
    for r in excess:
        arm, den = r["arm"], int(r["denominator"])
        if (r["epochs"], r["stage"], r["measure"]) == ("2000", "final", "matched") and r["kind"] in SERIES:
            if arm in SIZES and (den in WIDTHS or (arm, den) == ("4b", 256)):
                curves[arm][r["kind"]][den] = _est(r)
    return curves


def near_widths(best: list[Row]) -> dict[str, set[int]]:
    """The denominators each size's matched excess flags near-best among the six main widths at 2,000 epochs."""
    near: dict[str, set[int]] = {size: set() for size in SIZES}
    for r in best:
        if r["basis"] == "e2000" and r["near_best"] == "1":
            near[r["arm"]].add(int(r["denominator"]))
    return near


def arm_curves(excess: list[Row]) -> dict[str, dict[str, list[tuple[int, Est]]]]:
    """Arm and kind to (meta-latent count, matched agreement) in order of count, at each arm's equal-steps files."""
    curves: dict[str, dict[str, list[tuple[int, Est]]]] = {arm: {kind: [] for kind in SERIES} for arm in ENTITIES}
    for r in excess:
        entity = ENTITIES.get(r["arm"])
        if entity and r["kind"] in SERIES and (r["stage"], r["measure"]) == ("final", "matched"):
            if int(r["epochs"]) == entity.epochs and int(r["denominator"]) in entity.dens:
                curves[r["arm"]][r["kind"]].append((int(r["latents"]), _est(r)))
    return {arm: {kind: sorted(points) for kind, points in by_kind.items()} for arm, by_kind in curves.items()}


def length_curves(length: list[Row]) -> dict[str, dict[int, list[tuple[int, Est]]]]:
    """Size and denominator to (epochs, matched excess) in order of epochs, from every row of the longest score file
    of that setup (the 4,000-epoch file repeats the 2,000-epoch file up to epoch 2,000)."""
    longest: dict[tuple[str, int], int] = {}
    for r in length:
        key = r["arm"], int(r["denominator"])
        longest[key] = max(longest.get(key, 0), int(r["run_epochs"]))
    curves: dict[str, dict[int, list[tuple[int, Est]]]] = {size: {den: [] for den in WIDTHS} for size in SIZES}
    for r in length:
        arm, den = r["arm"], int(r["denominator"])
        if arm in SIZES and den in WIDTHS and int(r["run_epochs"]) == longest[arm, den]:
            curves[arm][den].append((int(r["epochs"]), _est(r)))
    return {size: {den: sorted(points) for den, points in by_den.items()} for size, by_den in curves.items()}


def _split(points: list[tuple[float, Est]]) -> tuple[list[float], list[float], list[float], list[float]]:
    """(x, (point, lo, hi)) pairs to the lists xs, points, los and his."""
    xs, ests = zip(*points, strict=True)
    return list(xs), *(list(col) for col in zip(*ests, strict=True))


def _by_width(curve: dict[int, Est]) -> list[tuple[float, Est]]:
    """(width as a fraction of the parent's latents, estimate) from the narrowest width to the widest."""
    return sorted((1 / den, est) for den, est in curve.items())


def _log2_axis(ax: Axes, ticks: list[float], labels: list[str]) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(ticks, labels)
    ax.minorticks_off()


def _figure(
    size: tuple[float, float], shape: tuple[int, int], footer: float, **share: bool
) -> tuple[Figure, list[Axes]]:
    """A figure with ``footer`` inches below the axes, left out of the layout for the shared x label and the legend."""
    fig = Figure(figsize=size, layout="constrained")
    fig.get_layout_engine().set(rect=(0, footer / size[1], 1, 1 - footer / size[1]))
    return fig, fig.subplots(*shape, **share).ravel().tolist()


LEGEND_ROW = 0.17  # inches between stacked legend rows in a footer
TEXT_WIDTH = 6.5  # inches, the text width of the TMLR style


def _footer(fig: Figure, label: str, *rows: tuple[list[Any], list[str]]) -> None:
    """The shared x label, and below it one legend row per (handles, names) pair, top to bottom."""
    height = fig.get_figheight()
    fig.text(0.5, (0.34 + LEGEND_ROW * (len(rows) - 1)) / height, label, ha="center", va="center")
    for i, (handles, names) in enumerate(rows):
        lift = LEGEND_ROW * (len(rows) - 1 - i) / height
        fig.legend(handles, names, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, lift))


def plot_width(curves: dict[str, dict[str, dict[int, Est]]], near: dict[str, set[int]], path: Path) -> None:
    """Five panels, one per size: real, twin and excess matched agreement against width (fraction of the parent's
    latents, log2) with 90% intervals as bands, the near-best widths ringed on the excess line; the sixth panel draws
    the five excess curves together, each labelled at its narrowest width."""
    with matplotlib.rc_context(STYLE):
        fig, axes = _figure((TEXT_WIDTH, 3.25), (2, 3), 0.5, sharex=True, sharey=True)
        for ax, size in zip(axes, SIZES, strict=False):
            for kind, color in SERIES.items():
                xs, ys, los, his = _split(_by_width(curves[size][kind]))
                ax.plot(xs, ys, color=color, marker=MARKERS[kind], label=kind)
                ax.fill_between(xs, los, his, color=color, alpha=0.25, linewidth=0)
            ax.plot(
                [1 / d for d in sorted(near[size])],
                [curves[size]["excess"][d][0] for d in sorted(near[size])],
                "o",
                ms=8,
                mfc="none",
                mec=INK,
                mew=0.8,
                label=NEAR,
            )
            ax.set_title(size.upper(), loc="left")
        greens = LinearSegmentedColormap.from_list("green", GREENS)
        for i, size in enumerate(SIZES):
            color = greens(i / (len(SIZES) - 1))
            xs, ys, _, _ = _split(_by_width(curves[size]["excess"]))
            axes[5].plot(xs, ys, color=color, marker=MARKERS["excess"])
            offset, ha, va = LABELS.get(size, ((-3, 0), "right", "center"))
            axes[5].annotate(size.upper(), (xs[0], ys[0]), offset, textcoords="offset points", ha=ha, va=va)
        axes[5].set_title("Excess, all sizes", loc="left")
        dens = (4, 8, 16, 32, 64, 128, 256)
        _log2_axis(axes[0], [1 / d for d in dens], [f"1/{d}" for d in dens])
        axes[0].set_xlim(1 / 512, 1 / 4 * 2**0.4)  # an octave left of 1/256 holds the sixth panel's labels
        for ax in axes:
            # Every panel labels its ticks, not only the bottom row; at 6.5 pt "1/256" and "1/128" touch.
            ax.tick_params(axis="x", labelbottom=True, labelsize=5.5)
        axes[0].yaxis.set_major_locator(MultipleLocator(0.1))  # ticks and grid lines every 0.1, on every panel
        fig.supylabel("agreement (mean matched cosine)")
        _footer(fig, "meta-SAE width (fraction of parent latents)", axes[0].get_legend_handles_labels())
        fig.savefig(path)


def plot_arms(curves: dict[str, dict[str, list[tuple[int, Est]]]], path: Path) -> None:
    """Real and twin (left) and excess (right) matched agreement of the controls against meta-latent count (log2), with
    90% intervals as error bars; colour is the kind as in the width figure, line and marker style is the arm."""
    with matplotlib.rc_context(STYLE):
        fig, (left, right) = _figure((TEXT_WIDTH, 2.2 + LEGEND_ROW), (1, 2), 0.5 + LEGEND_ROW, sharex=True)
        for ax, kinds in ((left, ("real", "twin")), (right, ("excess",))):
            for arm, e in ENTITIES.items():
                for kind in kinds:
                    xs, ys, los, his = _split(curves[arm][kind])
                    color = SERIES[kind]
                    errors = [
                        [y - lo for y, lo in zip(ys, los, strict=True)],
                        [hi - y for y, hi in zip(ys, his, strict=True)],
                    ]
                    ax.errorbar(
                        xs,
                        ys,
                        errors,
                        color=color,
                        ls=e.line,
                        marker=e.marker,
                        mec=color,
                        elinewidth=0.6,
                        capsize=1.2,
                        capthick=0.6,
                        mfc=color if e.filled else "none",
                    )
        left.set_title("Real and twin", loc="left")
        right.set_title("Excess", loc="left")
        left.set_ylabel("agreement")
        right.set_ylabel("excess agreement")
        _log2_axis(left, [256 * 2**i for i in range(7)], ["256", "512", "1k", "2k", "4k", "8k", "16k"])
        left.set_xlim(256 / 2**0.4, 16384 * 2**0.4)
        for ax in (left, right):
            ax.yaxis.set_major_locator(MultipleLocator(0.05))  # ticks and grid lines every 0.05
        arms = [
            Line2D([], [], color=INK, ls=e.line, marker=e.marker, mfc=INK if e.filled else "none", mec=INK)
            for e in ENTITIES.values()
        ]
        kinds = [Line2D([], [], color=color, lw=2) for color in SERIES.values()]  # no marker: markers are the arm
        _footer(fig, "meta-latent count", (kinds, list(SERIES)), (arms, [e.label for e in ENTITIES.values()]))
        fig.savefig(path)


def plot_length(curves: dict[str, dict[int, list[tuple[int, Est]]]], path: Path) -> None:
    """Five panels, one per size: matched excess against training epochs (log2), one line per main width with its 90%
    interval as a band, coloured blue (widest) to red (narrowest); the sixth cell holds the width legend."""
    colors = dict(zip(WIDTHS, WIDTH_COLORS, strict=True))
    with matplotlib.rc_context(STYLE):
        fig, axes = _figure((TEXT_WIDTH, 3.3), (2, 3), 0, sharex=True, sharey=True)
        for ax, size in zip(axes, SIZES, strict=False):
            for den in WIDTHS:
                xs, ys, los, his = _split(curves[size][den])
                ax.plot(xs, ys, color=colors[den], label=f"1/{den}")
                ax.fill_between(xs, los, his, color=colors[den], alpha=0.25, linewidth=0)
            ax.set_title(size.upper(), loc="left")
        _log2_axis(axes[0], [25, 100, 250, 500, 1000, 2000, 4000], ["25", "100", "250", "500", "1k", "2k", "4k"])
        for ax in axes[:5]:
            ax.tick_params(labelbottom=True, labelleft=True)  # every panel labels its ticks, not only the outer ones
        axes[5].axis("off")
        axes[5].legend(*axes[0].get_legend_handles_labels(), title="meta-SAE width", loc="center", ncol=2)
        fig.supylabel("excess agreement")
        fig.supxlabel("training epochs")
        fig.savefig(path)


def figures(root: Path = ROOT, out: Path = FIGURES) -> str:
    """Write the three figures to ``out`` from the tables under ``<root>/summary/``."""
    summary = root / "summary"
    excess, best, length = (read_tsv(summary / f"{name}.tsv") for name in ("excess", "best", "length"))
    out.mkdir(parents=True, exist_ok=True)
    plot_width(width_curves(excess), near_widths(best), out / "width.pdf")
    plot_arms(arm_curves(excess), out / "arms.pdf")
    plot_length(length_curves(length), out / "length.pdf")
    return f"wrote {out}"
