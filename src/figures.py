"""
Figures of the paper: the optimal weights of Section 5.1.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.ticker import ScalarFormatter

# The figures are drawn at the size at which they are printed
# (``\includegraphics[width=192.5mm]``), so that a font size is the printed one
FIGURE_WIDTH = 192.5 / 25.4  # inches; wider than the 160 mm text width of rQUF2e
FONT_SIZE = 9
DPI = 500
# Value labels of the bars: small, so that those of adjacent bars do not overlap
BAR_LABEL = {"fmt": "%.1f", "padding": 1.5, "fontsize": 5.5}
BAR_WIDTH = 0.32
# Distance between the centres of adjacent groups of bars and gap between the outer bars
# and the axes
GROUP_SPACING = 1.15
EDGE_GAP = 0.15

# Computer Modern, as the text of the paper
RCPARAMS = {
    "font.family": "serif",
    "font.serif": ["cmr10"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "font.size": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "axes.titlesize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE,
    "figure.titlesize": FONT_SIZE,
    "axes.linewidth": 0.6,
    "grid.linewidth": 0.4,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "savefig.dpi": DPI,
}

# Seaborn "deep", with colour-blind-safe contrasts (the baseline is grey, the two CFVaR
# objectives blue and orange; no red-green pair)
_DEEP = sns.color_palette("deep").as_hex()
COLORS = {"var": _DEEP[7], "cf2": _DEEP[0], "cf3": _DEEP[1]}
LABELS = {"var": "Variance", "cf2": r"CFVaR$_2$", "cf3": r"CFVaR$_3$"}


def _pct_change(w_new: np.ndarray, w_base: np.ndarray) -> np.ndarray:
    """
    Percentage change relative to ``|w_base|``, zero-safe.

    Parameters
    ----------
    w_new : np.ndarray
        New weights.
    w_base : np.ndarray
        Baseline weights.

    Returns
    -------
    np.ndarray
        ``100 * (w_new - w_base) / |w_base|``, and 0 where the baseline is 0.
    """
    return np.where(
        np.abs(w_base) < 1e-10, 0.0, (w_new - w_base) / np.abs(w_base) * 100
    )


def _grouped_bars(
    ax: Axes,
    x_pos: np.ndarray,
    series: list[np.ndarray],
    labels: list[str],
    colors: list[str],
    ylabel: str | None,
    tickers: list[str],
) -> None:
    """
    Draw grouped bars with value labels.

    Parameters
    ----------
    ax : Axes
        Axes to draw on.
    x_pos : np.ndarray
        Position of each group.
    series : list[np.ndarray]
        One array of values per bar of a group.
    labels : list[str]
        Legend entry of each series.
    colors : list[str]
        Colour of each series.
    ylabel : str or None
        Label of the vertical axis, none if ``None``.
    tickers : list[str]
        Label of each group.
    """
    offsets = (np.arange(len(series)) - (len(series) - 1) / 2) * BAR_WIDTH
    for values, label, color, offset in zip(series, labels, colors, offsets):
        bars = ax.bar(
            x_pos + offset, values, BAR_WIDTH, label=label, color=color, alpha=0.8
        )
        ax.bar_label(bars, **BAR_LABEL)
    reach = len(series) / 2 * BAR_WIDTH + EDGE_GAP
    ax.set_xticks(x_pos)
    ax.set_xlim(x_pos[0] - reach, x_pos[-1] + reach)
    ax.set_xticklabels(tickers)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5)


def _pad_for_bar_labels(axes: list[Axes], fraction: float = 0.15) -> None:
    """
    Make room for the vertical value labels of the bars.

    Extends the vertical limits beyond the bars, on the side of the positive bars and on
    that of the negative ones, by ``fraction`` of the data range (a linear axis) or by
    a factor of 5 (a symmetric log axis). The axes get the same limits, which are those
    of all of them.

    Parameters
    ----------
    axes : list[Axes]
        Axes with the bars.
    fraction : float, optional
        Room added on a linear axis, as a fraction of the range of the bars, by
        default 0.15.
    """
    low = min(0.0, *(ax.dataLim.y0 for ax in axes))
    high = max(0.0, *(ax.dataLim.y1 for ax in axes))
    if axes[0].get_yscale() == "symlog":
        low, high = low * 5.0, high * 5.0
    else:
        room = fraction * (high - low)
        low, high = (low - room if low < 0 else 0.0), (high + room if high > 0 else 0.0)
    for ax in axes:
        ax.set_ylim(low, high)


def weight_figures(
    weights: dict[str, pd.DataFrame], folder: Path, prefix: str = ""
) -> None:
    """
    Figures of the optimal weights of skew-t and Student's t portfolios.

    Writes ``{prefix}skew_vs_t_weights.png``, the weights of the variance, CFVaR2 and
    CFVaR3 optimal portfolios of each distribution, and
    ``{prefix}skew_vs_t_pct_change.png``, their percentage change from the variance
    weights.  Each has one row per option type, skew-t on the left.

    Parameters
    ----------
    weights : dict[str, pd.DataFrame]
        For each option type (in the order of the rows), the weights (fractions of the
        budget) with the columns ``"Skew-t var"``, ``"Skew-t cf2"``, ``"Skew-t cf3"``,
        ``"Student t var"``, ``"Student t cf2"`` and ``"Student t cf3"``, indexed by the
        assets.
    folder : Path
        Folder of the figures.
    prefix : str, optional
        Prefix of the file names, by default none.
    """
    sns.set_style("white")
    sns.set_palette("Set1")
    sns.set_context("paper")
    plt.rcParams.update(RCPARAMS)
    folder.mkdir(parents=True, exist_ok=True)
    kinds = list(weights)
    tickers = list(weights[kinds[0]].index)
    x_pos = np.arange(len(tickers)) * GROUP_SPACING
    cells = [
        (row, col, kind, dist)
        for row, kind in enumerate(kinds)
        for col, dist in enumerate(("Skew-t", "Student t"))
    ]

    fig, axes = plt.subplots(2, 2, figsize=(FIGURE_WIDTH, 3.9))
    for row, col, kind, dist in cells:
        _grouped_bars(
            axes[row][col],
            x_pos,
            [100 * weights[kind][f"{dist} {k}"].values for k in LABELS],
            list(LABELS.values()),
            list(COLORS.values()),
            "Portfolio Weight (%)",
            tickers,
        )
        _pad_for_bar_labels([axes[row][col]])
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.suptitle(r"Optimal Weights: Skew-$t$ and Student's $t$")
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(folder / f"{prefix}skew_vs_t_weights.png", dpi=DPI)
    plt.close(fig)

    # The panels of a row share the vertical axis, a symmetric log axis, so that the
    # small changes of the Student's t portfolios stay visible next to the large ones
    fig, axes = plt.subplots(2, 2, figsize=(FIGURE_WIDTH, 3.3), sharey="row")
    for row, col, kind, dist in cells:
        w = weights[kind]
        ax = axes[row][col]
        _grouped_bars(
            ax,
            x_pos,
            [
                _pct_change(w[f"{dist} {k}"].values, w[f"{dist} var"].values)
                for k in ("cf2", "cf3")
            ],
            [LABELS["cf2"] + r" % $\Delta$", LABELS["cf3"] + r" % $\Delta$"],
            [COLORS["cf2"], COLORS["cf3"]],
            "% Change" if col == 0 else None,
            tickers,
        )
        ax.set_yscale("symlog", linthresh=10)
        ax.set_yticks([-100, -10, 0, 10, 100])
        ax.yaxis.set_major_formatter(ScalarFormatter())
    for row_axes in axes:
        _pad_for_bar_labels(list(row_axes))
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.suptitle("% Change from Variance Weights")
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(folder / f"{prefix}skew_vs_t_pct_change.png", dpi=DPI)
    plt.close(fig)
