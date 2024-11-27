# %%
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns
import xarray as xr
from cycler import cycler
from matplotlib.gridspec import GridSpec
from utils.tools import get_figure_path

# plt.style.use("style/tex.mplstyle")
plt.style.use("style/thesis.mplstyle")

clrs = sns.color_palette("tab20", n_colors=8, desat=0.9)

default_cycler = cycler(color=[clrs[0], clrs[1], clrs[6], clrs[7]])
plt.rc("axes", prop_cycle=default_cycler)


# mpl.rcParams["font.size"] = 5
# mpl.rcParams["axes.linewidth"] = 0.5
# mpl.rcParams["xtick.major.width"] = 0.5
# mpl.rcParams["ytick.major.width"] = 0.5


def fmt(x):
    """Write a formatter that makes the labels convert to MB, GB, etc.
    so 1000 -> 1 KB, 1e6 -> 1 MB, 1e9 -> 1 GB, etc.
    and 1523 -> 1.52 KB, 1.52e6 -> 1.52 MB, etc.

    """
    if x < 1e3:
        return f"{x:.0f} B"
    elif x < 1e6:
        return f"{x/1e3:.0f} KB"
    elif x < 1e9:
        return f"{x/1e6:.0f} MB"
    elif x < 1e12:
        return f"{x/1e9:.0f} GB"
    else:
        return f"{x/1e12:.0f} TB"


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


SOLVER = ["xeofs", "xeofs_dask", "eofs", "eofs_dask"]
COLOR = ["C0", "C1", "C2", "C3"]
lvls_abs = [0.01, 0.1, 1, 10, 100]
lvls_ratio = [0.01, 0.02, 0.05, 0.2, 0.5, 2, 5, 20, 50, 100]

fig = plt.figure(figsize=(7, 8.0))
gs = GridSpec(2, 2, hspace=0.3, height_ratios=[1, 1.5])
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, :])
axes = [ax1, ax2, ax3]

# xeofs timings
xeofs.plot.contourf(
    ax=ax1,
    levels=lvls_abs,
    cmap="cividis",
    cbar_kwargs={"ticks": lvls_abs, "format": mticker.LogFormatterSciNotation()},
)
lvl_labels_ratio = ["100x", "50x", "20x", "5x", "2x", "2x", "5x", "20x", "50x", "100x"]

# speed-up xeofs vs. eofs
speed_ratio.plot.contourf(
    ax=ax2,
    levels=lvls_ratio,
    cmap="RdBu",
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
for solver, color in zip(SOLVER, COLOR):
    timings.sel(solver=solver, n_samples=10).plot(lw=1, ax=ax3, color=color, ls="--")
    timings.sel(solver=solver, n_samples=10_000).plot(
        lw=1.5,
        ax=ax3,
        color=color,
    )


# Draw rectangles to denote the range of n_samples
rect_low = mpatches.Rectangle(
    (10.5, 10), 9e4, 6, linewidth=1, edgecolor=".8", facecolor="none", ls="--"
)
rect_high = mpatches.Rectangle(
    (10.5, 6e3), 9e4, 3900, linewidth=1, edgecolor=".8", facecolor="none"
)

ax1.add_patch(rect_low)
ax1.add_patch(rect_high)

ax1.text(20, 12.5, "$n=10$", ha="left", va="center", color=".8", style="italic")
ax1.text(20, 7.5e3, "$n=10^4$", ha="left", va="center", color=".8", style="italic")

# Set scales and labels
for ax in axes:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e1, 1e5)
    ax.set_xticks([1e1, 1e2, 1e3, 1e4, 1e5])

    ax.set_xlabel("Number of features")
    ax.set_ylabel("Number of samples")

ax2.set_ylabel("")
ax3.set_ylabel("")

# Legends
fsr = mlines.Line2D([], [], color=".3", lw=1.5, ls="--", label="$n=10$")
msr = mlines.Line2D([], [], color=".3", lw=1.5, label="$n=10^4$")
lg_samples = ax3.legend(
    handles=[fsr, msr],
    title="Samples",
    loc="upper left",
    bbox_to_anchor=(0, 0.7),
    alignment="center",
    frameon=False,
)
ax3.add_artist(lg_samples)

ax3.plot([], [], color="C0", lw=1.5, label="xeofs")
ax3.plot([], [], color="C1", lw=1.5, label="xeofs (dask)")
ax3.plot([], [], color="C2", lw=1.5, label="eofs")
ax3.plot([], [], color="C3", lw=1.5, label="eofs (dask)")
lg_solver = ax3.legend(
    title="Solver",
    loc="upper left",
    bbox_to_anchor=(0, 1),
    alignment="center",
    frameon=False,
)

# Titles
ax1.set_title("A | Timings of xeofs [in s]")
kws_ax2_tle = {"y": 1.04, "ha": "left", "va": "bottom", "transform": ax2.transAxes}
ax2.text(0.0, s="B | Speed-up", **kws_ax2_tle)
fig.text(0.39, s="xeofs", color="C0", weight="bold", **kws_ax2_tle)
fig.text(0.59, s="vs.", **kws_ax2_tle)
fig.text(0.70, s="eofs", color="C2", weight="bold", **kws_ax2_tle)

ax3.set_title("C | Timings vs. number of features [in s]")


# Save figure
path_fig = get_figure_path("chapter5", "vector", "performance_xeofs.svg")
fig.savefig(path_fig, bbox_inches="tight", format="svg")


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
