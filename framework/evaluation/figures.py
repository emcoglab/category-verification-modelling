from pathlib import Path
from typing import Tuple, Optional

from matplotlib import pyplot
from matplotlib.colors import to_rgba
from pandas import DataFrame
from seaborn import heatmap


RGBA = Tuple[float, float, float, float]


def save_heatmap(hitrates: DataFrame, path: Path, value_col: str, vlims: Tuple[Optional[float], Optional[float]]):
    vmin, vmax = vlims
    pivot = hitrates.pivot(index="Decision threshold (no)", columns="Decision threshold (yes)", values=value_col)
    ax = heatmap(pivot,
                 annot=True, fmt=".02f",
                 vmin=vmin, vmax=vmax, cmap='Reds',
                 square = True,
                 )
    ax.figure.savefig(path)
    pyplot.close(ax.figure)


def opacity_for_overlap(desired_total_opacity: float, n_overlaps: int) -> float:
    """
    Given n overlapping patches which together have a desired total opacity, return the required opacity of the
    individual patches.
    """
    assert 0 <= desired_total_opacity <= 1
    if desired_total_opacity == 0: return 0
    if desired_total_opacity == 1: return 1

    desired_light = 1 - desired_total_opacity
    individual_light = desired_light ** (1 / n_overlaps)

    return 1 - individual_light


def named_colour(name: str, *, with_alpha: float = 1) -> RGBA:
    assert 0 <= with_alpha <= 1
    rgba = to_rgba(name)
    return rgba[0], rgba[1], rgba[2], with_alpha
