from __future__ import annotations

import math
from typing import Iterable

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

PRODUCT_COLORS = {
    "bivalent": "#2C7FB8",
    "quadrivalent": "#F28E2B",
    "nonavalent": "#D1495B",
    "domestic_bivalent": "#2C7FB8",
    "imported_bivalent": "#76B7E5",
    "imported_quadrivalent": "#F28E2B",
    "domestic_nonavalent": "#59A14F",
    "imported_nonavalent": "#D1495B",
}


def _scientific_ticklabel(value: float, _: float) -> str:
    if value == 0:
        return "0"
    abs_value = abs(value)
    exponent = int(math.floor(math.log10(abs_value)))
    mantissa = value / (10**exponent)
    mantissa_rounded = round(mantissa, 2)
    if math.isclose(mantissa_rounded, 1.0):
        return rf"$10^{{{exponent}}}$"
    if math.isclose(mantissa_rounded, round(mantissa_rounded)):
        mantissa_text = str(int(round(mantissa_rounded)))
    else:
        mantissa_text = f"{mantissa_rounded:.2f}".rstrip("0").rstrip(".")
    return rf"${mantissa_text}\times 10^{{{exponent}}}$"


def apply_nature_style(fig: Figure, axes: Axes | Iterable[Axes]) -> None:
    axis_list = [axes] if isinstance(axes, Axes) else list(axes)
    fig.patch.set_facecolor("white")
    for ax in axis_list:
        ax.set_facecolor("white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.0)
        ax.spines["bottom"].set_linewidth(1.0)
        ax.tick_params(axis="both", labelsize=10, width=1.0, length=4, color="#444444")
        ax.grid(axis="y", color="#D9DEE7", linewidth=0.8, alpha=0.8)
        ax.set_axisbelow(True)


def apply_scientific_format(ax: Axes, *, x: bool = False, y: bool = True) -> None:
    formatter = FuncFormatter(_scientific_ticklabel)
    if x:
        ax.xaxis.set_major_formatter(formatter)
    if y:
        ax.yaxis.set_major_formatter(formatter)
