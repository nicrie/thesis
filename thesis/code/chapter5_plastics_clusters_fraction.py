# %%
from string import ascii_uppercase

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import utils.visualization as viz
import xarray as xr
from cartopy.crs import PlateCarree, TransverseMercator
from cartopy.feature import LAND, OCEAN, RIVERS
from matplotlib.gridspec import GridSpec
from utils.tools import get_figure_path

viz.set_style()


def trans_effect_size(s):
    return s * 2e3


def trans_prob(c):
    return c


def load_data(path, sign="+"):
    """Load PCA clustering results from a NetCDF file.

    Parameters
    ----------
    path : str
        Path to the NetCDF file.
    sign : str, optional
        Sign of the cluster, by default "+".

    """
    try:
        pca_result = xr.open_dataset(path + "pca_clustering.nc", engine="netcdf4")
    except FileNotFoundError:
        print(f"File not found: {path}")
        return None
    if sign == "+":
        pcs = pca_result.scores
        confidence = pca_result.confidence_pos
        effect_size = pca_result.effect_size_pos
        exp_var_ratio = (
            pca_result.expvar_ratio_pos.quantile([0.5, 0.025, 0.975], "n") * 100
        )
    elif sign == "-":
        pcs = -pca_result.scores
        confidence = pca_result.confidence_neg
        effect_size = pca_result.effect_size_neg
        exp_var_ratio = (
            pca_result.expvar_ratio_neg.quantile([0.5, 0.025, 0.975], "n") * 100
        )
    else:
        raise ValueError("Invalid sign")

    return xr.Dataset(
        {"s": effect_size, "c": confidence, "pcs": pcs, "quantiles": exp_var_ratio},
    )


# %%
# Load data
# =============================================================================
YEAR = 2001
QUANTITY = "fraction"
VARIABLE = ["LAND", "FISH", "AQUA"]
NAMES = ["Land", "Fishing", "Aquaculture"]
signs = {"LAND": "+", "FISH": "+", "AQUA": "+"}
path_project = "/home/nrieger/Projects/MINKE/seasonality_ospar/data/"
paths = {v: path_project + f"clustering/pca/{QUANTITY}/{v}/{YEAR}/" for v in VARIABLE}
ds = {v: load_data(p, signs[v]) for v, p in paths.items()}


# %%
# Figure - PCA eigenvectors and projections
# =============================================================================
seasons = ds["LAND"].season
season_labels = ["Win", "Spr", "Sum", "Aut"]
extent = [-15, 13, 34, 64]
proj = TransverseMercator(central_latitude=50)

colors = {
    "ocean": "0.9",
    "land": "0.75",
    "coastline": ".5",
    "rivers": ".6",
    "text": ".3",
    "beach_ec": "C1",
}

max_probability = 0.25
norm = mcolors.Normalize(vmin=0.0, vmax=max_probability)
cmap = viz.get_sequential_color_palette(as_cmap=True)
cmap_clrs = viz.get_sequential_color_palette(as_cmap=False, n_colors=4)
clr_highlight = cmap_clrs[2]

palettes = {
    "LAND": [clr_highlight, ".5", clr_highlight, ".5"],
    "FISH": [".5", ".5", ".5", clr_highlight],
    "AQUA": [clr_highlight, clr_highlight, ".5", ".5"],
}

fig = plt.figure(figsize=(7.2, 3.1))
gs = GridSpec(
    1,
    4,
    figure=fig,
    hspace=0.02,
    wspace=0.0,
    width_ratios=[1, 1, 1, 0.05],
)
ax = {v: fig.add_subplot(gs[0, i], projection=proj) for i, v in enumerate(VARIABLE)}
cax = fig.add_subplot(gs[0, 3])

for i, (v, a) in enumerate(ax.items()):
    # Map background
    a.set_extent(extent, crs=PlateCarree())
    a.add_feature(OCEAN, facecolor=colors["ocean"])
    a.add_feature(LAND, facecolor=colors["land"])
    a.add_feature(RIVERS.with_scale("10m"), edgecolor=colors["rivers"], lw=0.5)

    # Spatial distribution of clusters
    a.scatter(
        ds[v]["s"].lon,
        ds[v]["s"].lat,
        s=trans_effect_size(ds[v]["s"].sel(mode=1)),
        c=ds[v]["c"].sel(mode=1).values,
        norm=norm,
        cmap=cmap,
        transform=PlateCarree(),
        ec=colors["text"],
        lw=0.5,
        alpha=0.5,
    )
    # Cluster centroid in lower right corners
    xticks = np.arange(0.0, 3.5)
    axin = a.inset_axes(
        [0.55, 0.02, 0.43, 0.29],
        fc=colors["ocean"],
        frameon=True,
        transform=a.transAxes,
    )
    axin.patch.set_alpha(0.3)
    df_pcs = ds[v]["pcs"].sel(mode=1, drop=True).to_dataframe().reset_index()
    sns.barplot(
        df_pcs,
        x="season",
        y="pcs",
        hue=df_pcs["season"],
        palette=palettes[v],
        zorder=1,
        ax=axin,
        err_kws={"color": colors["text"]},
    )
    sns.despine(ax=axin, right=False, top=False)
    axin.set_title("PC scores", color=colors["text"], size=7, y=0.7)
    axin.set_xticks(xticks)
    axin.set_xticklabels(season_labels, color=colors["text"], size=5, y=0.2)
    axin.set_yticks([])
    axin.set_xlabel("")
    axin.set_ylabel("")
    axin.set_ylim(-0.3, 0.3)
    axin.tick_params(axis="x", length=0)

    # Title
    a.text(
        0.01,
        0.99,
        "{:} | {:}".format(ascii_uppercase[i], NAMES[i]),
        transform=a.transAxes,
        fontsize=10,
        fontweight="bold",
        color=colors["text"],
        va="top",
    )

    # Explained variance
    ax_expvar = a.inset_axes(
        [0.01, 0.8, 0.4, 0.08], facecolor=".1", frameon=False, transform=a.transAxes
    )
    mid, lower, upper = ds[v]["quantiles"].sel(mode=1)
    sns.barplot(
        x=[mid.item()],
        y=["Explained Variance"],
        ax=ax_expvar,
        color=clr_highlight,
        edgecolor=colors["text"],
        linewidth=1,
    )
    ax_expvar.errorbar(
        x=mid,
        y=["Explained Variance"],
        xerr=[[mid - lower], [upper - mid]],
        fmt="none",
        color=colors["text"],
        capsize=2,
    )

    ax_expvar.text(
        5,
        0,
        f"{mid.item():.1f}%",
        color=colors["ocean"],
        ha="left",
        va="center",
        size=6,
        weight="bold",
    )
    ax_expvar.text(
        0,
        0.7,
        "Explained variance",
        va="center",
        color=colors["text"],
        style="italic",
        size=6,
        transform=ax_expvar.transData,
    )

    ax_expvar.set_xlim([0, 50])
    ax_expvar.set_ylim([-0.5, 0.5])
    ax_expvar.set_xticks([])
    ax_expvar.set_yticks([])
    ax_expvar.set_xlabel("")
    ax_expvar.spines[["left", "right", "top", "bottom"]].set_visible(False)


# Add vertical colorbar to the right border
cbar = fig.colorbar(
    mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
    cax=cax,
    label="Confidence in Cluster Membership",
)
cticks = [0, 0.05, 0.1, 0.15, 0.2, 0.25]
cbar.set_ticks(cticks)
cbar.ax.set_yticklabels([f"{t:.0%}" for t in cticks])

# save_to_vector = get_figure_path(
#     "chapter5", "vector", "plastics_clusters_fraction_map.svg"
# )
save_to_raster = get_figure_path(
    "chapter5", "raster", "plastics_clusters_fraction_map.png"
)
# plt.savefig(save_to_vector, bbox_inches="tight", dpi=300)
plt.savefig(save_to_raster, bbox_inches="tight", dpi=300)
plt.show()


# %%
