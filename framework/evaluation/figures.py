from pathlib import Path
from typing import Tuple, Optional

from matplotlib import pyplot
from pandas import DataFrame
from seaborn import heatmap


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