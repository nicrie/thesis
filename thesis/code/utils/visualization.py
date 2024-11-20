import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def set_style():
    """Sets the plotting style."""
    sns.set_context("paper")
    plt.style.use("utils/style/thesis.mplstyle")
    mpl.rcParams["font.size"] = 7
    mpl.rcParams["axes.linewidth"] = 0.5
    mpl.rcParams["xtick.major.width"] = 0.5
    mpl.rcParams["ytick.major.width"] = 0.5


def shift_cmap(cmap, frac):
    """Shifts a colormap by a certain fraction.

    Keyword arguments:
    cmap -- the colormap to be shifted. Can be a colormap name or a Colormap object
    frac -- the fraction of the colorbar by which to shift (must be between 0 and 1)
    """
    N = 256
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    n = cmap.name
    x = np.linspace(0, 1, N)
    out = np.roll(x, int(N * frac))
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(f"{n}_s", cmap(out))
    return new_cmap


def get_sequential_color_palette(as_cmap: bool = True, n_colors: int = 8, **kwargs):
    """Returns a sequential cubehelix color palette.

    Keyword arguments:
    as_cmap -- whether to return the color palette as a colormap (default True)
    n_colors -- the number of colors to be returned (default 8)
    """
    return sns.cubehelix_palette(
        start=0.5, rot=-0.8, as_cmap=as_cmap, n_colors=n_colors, **kwargs
    )
