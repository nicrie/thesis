import matplotlib
import matplotlib.pyplot as plt
import numpy as np


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
    new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(f"{n}_s", cmap(out))
    return new_cmap
