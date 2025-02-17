# %%
import cmocean.cm as cmo
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import utils.visualization as vis
import xarray as xr
from matplotlib.gridspec import GridSpec
from utils.tools import get_figure_path

vis.set_style()


def fmt(x):
    """Write a formatter that makes the labels convert to MB, GB, etc.
    so 1000 -> 1 KB, 1e6 -> 1 MB, 1e9 -> 1 GB, etc.
    and 1523 -> 1.52 KB, 1.52e6 -> 1.52 MB, etc.

    """
    if x < 1e3:
        return f"{x:.0f} B"
    elif x < 1e6:
        return f"{x / 1e3:.0f} KB"
    elif x < 1e9:
        return f"{x / 1e6:.0f} MB"
    elif x < 1e12:
        return f"{x / 1e9:.0f} GB"
    else:
        return f"{x / 1e12:.0f} TB"


# %%

data = xr.open_dataset("../data/chapter3/timings_xeofs_eofs.nc")
nbytes = data["nbytes"]
timings = data[["eofs", "eofs_dask", "xeofs", "xeofs_dask"]].to_array("solver")
timings = timings.min("run")
eofs = timings.sel(solver=["eofs", "eofs_dask"]).min("solver")
xeofs = timings.sel(solver=["xeofs", "xeofs_dask"]).min("solver")

speed_ratio = eofs / xeofs
lvls = np.logspace(-2, 2, 10)
speed_ratio.plot(xscale="log", yscale="log", cmap="RdBu_r")

# %%
# Figure
# =============================================================================
cmap = vis.get_sequential_color_palette()
cmap_div = cmo.curl_r

SOLVER = ["xeofs", "xeofs_dask", "eofs", "eofs_dask"]
colors = {
    "xeofs": cmap_div(0.8),
    "xeofs_dask": cmap_div(0.7),
    "eofs_dask": cmap_div(0.3),
    "eofs": cmap_div(0.2),
    "highlight": "#c33c54",
}

weights = {"xeofs": 800, "eofs": 800, "xeofs_dask": 400, "eofs_dask": 400}
linewidths = {"xeofs": 3, "eofs": 3, "xeofs_dask": 1.5, "eofs_dask": 1.5}


lvls_abs = [0.01, 0.1, 1, 10, 100]
lvls_ratio = [0.01, 0.02, 0.05, 0.2, 0.5, 2, 5, 20, 50, 100]

fig = plt.figure(figsize=(7, 8.0))
gs = GridSpec(2, 2, hspace=0.3, height_ratios=[1, 1.5])
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])
axes = [ax1, ax2, ax3, ax4]

# xeofs timings
xeofs.plot.contourf(
    ax=ax1,
    levels=lvls_abs,
    cmap=cmap,
    cbar_kwargs={"ticks": lvls_abs, "format": mticker.LogFormatterSciNotation()},
)
lvl_labels_ratio = ["100x", "50x", "20x", "5x", "2x", "2x", "5x", "20x", "50x", "100x"]

# speed-up xeofs vs. eofs
speed_ratio.plot.contourf(
    ax=ax2,
    levels=lvls_ratio,
    cmap=cmap_div,
    cbar_kwargs={
        "ticks": mticker.FixedLocator(lvls_ratio),
        "format": mticker.FixedFormatter(lvl_labels_ratio),
        "label": "",
        "extend": "both",
    },
)
cp = nbytes.plot.contour(
    ax=ax2,
    levels=[1e3, 1e6, 1e9],
    colors=".3",
    linestyles=":",
    linewidths=0.7,
    locator=mticker.LogLocator(),
)


ax2.clabel(cp, cp.levels, fmt=fmt, inline=True, inline_spacing=5, zorder=50)


# timings vs. number of features
for solver in SOLVER:
    timings.sel(solver=solver, n_samples=10).plot(
        ax=ax3, color=colors[solver], lw=linewidths[solver]
    )

    timings.sel(solver=solver, n_samples=10_000).plot(
        ax=ax4, color=colors[solver], lw=linewidths[solver]
    )


# Draw rectangles to denote the range of n_samples
rect_low = mpatches.Rectangle(
    (10.5, 10),
    9e4,
    6,
    linewidth=1,
    edgecolor=".3",
    facecolor="none",
)
rect_high = mpatches.Rectangle(
    (10.5, 6e3), 9e4, 3900, linewidth=1, edgecolor=".3", facecolor="none"
)

ax1.add_patch(rect_low)
ax1.add_patch(rect_high)

ax1.text(20, 12.5, "Panel (C)", ha="left", va="center", color=".3")
ax1.text(20, 7.5e3, "Panel (D)", ha="left", va="center", color=".3")

# Set scales and labels
for ax in axes:
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel("Number of features")


for ax in [ax1, ax2]:
    ax.set_xlim(1e1, 1e5)
    ax.set_xticks([1e1, 1e2, 1e3, 1e4, 1e5])

    ax.set_ylabel("Number of samples")

ax3.set_ylim(1e-3, 1e1)
ax4.set_ylim(1e-3, 1e4)
ax2.set_ylabel("")
ax3.set_ylabel("")
ax4.set_ylabel("")

# Add y grid lines
for ax in [ax3, ax4]:
    ax.yaxis.grid(True, which="major", linestyle="--", linewidth=0.5)

# Legends
for solver in SOLVER:
    y0 = timings.sel(solver=solver, n_samples=10, n_features=10).item()
    label = solver.replace("_", " ")
    ax3.text(10, y0 + y0 * 0.15, label, color=colors[solver], weight=weights[solver])

for solver in SOLVER:
    y0 = timings.sel(solver=solver, n_samples=10_000).max("n_features").item()
    label = solver.replace("_", " ")
    if solver == "xeofs_dask":
        y0 *= 1.8
    ax4.text(2e5, y0 + y0 * 0.15, label, color=colors[solver], weight=weights[solver])

# Titles
ax1.set_title("A | Timings of xeofs [in s]")
kws_ax2_tle = {"y": 1.04, "ha": "left", "va": "bottom", "transform": ax2.transAxes}
ax2.text(0.0, s="B | Speed-up", **kws_ax2_tle)
fig.text(0.39, s="xeofs", color=colors["xeofs"], weight="bold", **kws_ax2_tle)
fig.text(0.59, s="vs.", **kws_ax2_tle)
fig.text(0.70, s="eofs", color=colors["eofs"], weight="bold", **kws_ax2_tle)

ax3.set_title("C | Timings for $10$ samples [in s]")
ax4.set_title("D | Timings for $10,000$ samples [in s]")


# Save figure
path_vector = get_figure_path("chapter5", "vector", "performance_xeofs.svg")
path_raster = get_figure_path("chapter5", "raster", "performance_xeofs.png")
fig.savefig(path_raster, bbox_inches="tight", format="png")
fig.savefig(path_vector, bbox_inches="tight", format="svg")


# %%


def perf_svd(N, D, M):
    s2d = n_samples[:, None] * np.ones(n_features.size)
    f2d = np.ones(n_samples.size)[:, None] * n_features[None, :]
    return s2d * f2d * np.min(np.stack([s2d, f2d], axis=0))


def perf_rsvd(N, D, M):
    return N[:, None] * D[None, :] * np.log(M) + (N[:, None] + D[None, :]) * M**2


n_samples = np.logspace(0, 5, 50)
n_features = np.logspace(0, 5, 100)
n_components = 2

tt_svd = perf_svd(n_samples, n_features, 2)
tt_rsvd = perf_rsvd(n_samples, n_features, 2)

tt_svd = xr.DataArray(
    tt_svd,
    dims=["n_samples", "n_features"],
    coords={"n_samples": n_samples, "n_features": n_features},
)
tt_rsvd = xr.DataArray(
    tt_rsvd,
    dims=["n_samples", "n_features"],
    coords={"n_samples": n_samples, "n_features": n_features},
)
tt = xr.Dataset({"svd": tt_svd, "rsvd": tt_rsvd}).to_array("solver")

# %%
