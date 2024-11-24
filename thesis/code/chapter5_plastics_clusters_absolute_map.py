# %%
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import utils.visualization as viz
import xarray as xr
from cartopy.crs import PlateCarree, TransverseMercator
from cartopy.feature import LAND, OCEAN, RIVERS
from cycler import cycler
from matplotlib.gridspec import GridSpec
from utils.tools import get_figure_path

viz.set_style()
clrs = sns.color_palette("tab20", n_colors=8, desat=0.9)
default_cycler = cycler(color=[clrs[0], clrs[1], clrs[2], clrs[3], clrs[-1]])
plt.rc("axes", prop_cycle=default_cycler)


# %%
# Load data
# =============================================================================
YEAR = 2001
QUANTITY = "absolute"
VARIABLE = "Plastic"
base_path = f"/home/nrieger/projects/MINKE/seasonality_ospar/data/clustering/pca/{QUANTITY}/{VARIABLE}/{YEAR}/"

pca_result = xr.open_dataset(base_path + "pca_clustering.nc", engine="netcdf4")
components = pca_result.comps
pcs = pca_result.scores
confidence = pca_result.confidence_pos
effect_size = pca_result.effect_size_pos

exp_var_ratio = pca_result.expvar_ratio_pos.quantile([0.5, 0.025, 0.975], "n") * 100
mid1, low1, up1 = exp_var_ratio.sel(mode=1)
mid2, low2, up2 = exp_var_ratio.sel(mode=2)


def trans_effect_size(s):
    return s * 2e3


def trans_prob(c):
    return c


# %%
# Figure 3
# =============================================================================
seasons = ["Winter", "Spring", "Summer", "Autumn"]
extent = [-15, 13, 34, 64]
proj = TransverseMercator(central_latitude=50)

norm = mcolors.Normalize(vmin=0.0, vmax=0.8)

cmap = viz.get_sequential_color_palette(as_cmap=True)
cmap_clrs = viz.get_sequential_color_palette(as_cmap=False, n_colors=4)
clr_highlight = cmap_clrs[2]

fig = plt.figure(figsize=(7.2, 4.6))
gs = GridSpec(
    1,
    3,
    figure=fig,
    hspace=0.02,
    wspace=0.0,
    width_ratios=[1, 1, 0.05],
)
ax1 = fig.add_subplot(gs[0, 0], projection=proj)
ax2 = fig.add_subplot(gs[0, 1], projection=proj)
cax = fig.add_subplot(gs[0, 2])

for ax in [ax1, ax2]:
    ax.add_feature(OCEAN, color=".9")
    ax.add_feature(LAND, color=".75")
    ax.add_feature(RIVERS.with_scale("10m"), edgecolor=".6", lw=0.5)
    ax.set_extent(extent, crs=PlateCarree())

ax1.scatter(
    components.lon,
    components.lat,
    s=trans_effect_size(effect_size.sel(mode=1)),
    c=confidence.sel(mode=1).values,
    norm=norm,
    cmap=cmap,
    transform=PlateCarree(),
    ec=".3",
    lw=0.5,
    alpha=0.5,
)
ax2.scatter(
    components.lon,
    components.lat,
    s=trans_effect_size(effect_size.sel(mode=2)),
    c=confidence.sel(mode=2).values,
    norm=norm,
    cmap=cmap,
    transform=PlateCarree(),
    ec=".3",
    lw=0.5,
    alpha=0.5,
)

cbar3 = fig.colorbar(
    mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
    cax=cax,
    label="Confidence in Cluster Membership",
)
cbar3.set_ticks([0, 0.2, 0.4, 0.6, 0.8])
cbar3.ax.set_yticklabels(["0%", "20%", "40%", "60%", "80%"])

xticks = np.arange(0.0, 3.5)
modes = [1, 2]
titles = ["A | Cluster $C1^+$", "B | Cluster $C2^+$"]
axin_titles = ["PC$1^+$ scores", "PC$2^+$ scores"]
palettes = [
    [".5", clr_highlight, ".5", ".5"],
    [clr_highlight, clr_highlight, ".5", ".5"],
]
for ax, mode, palette, intitle, title in zip(
    [ax1, ax2], modes, palettes, axin_titles, titles
):
    axin = ax.inset_axes([0.53, 0.05, 0.45, 0.2], transform=ax.transAxes)
    axin.patch.set_alpha(0.3)
    df_pcs = pcs.sel(mode=mode, drop=True).to_dataframe().reset_index()
    sns.barplot(
        df_pcs,
        x="season",
        y="scores",
        hue="season",
        palette=palette,
        zorder=1,
        ax=axin,
        err_kws={"color": ".3"},
    )
    axin.set_title(intitle, color=".3", size=8)
    axin.set_xticks(xticks)
    axin.set_xticklabels(seasons, color=".3", size=5, y=0.2)
    axin.set_xlabel("")
    axin.set_yticks([])
    axin.set_ylim([-0.5, 0.7])
    axin.set_ylabel("")
    axin.tick_params(axis="x", length=0)

    sns.despine(ax=axin, left=False, bottom=False, right=False, top=False)

    ax.text(
        0.01,
        0.99,
        title,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        color=".3",
        va="top",
    )

# Explained variance
# -----------------------------------------------------------------------------
ax_expvar1 = ax1.inset_axes(
    [0.01, 0.8, 0.4, 0.08], facecolor=".1", frameon=False, transform=ax1.transAxes
)
ax_expvar2 = ax2.inset_axes(
    [0.01, 0.8, 0.4, 0.08], facecolor=".1", frameon=False, transform=ax2.transAxes
)

for ax, mid in zip([ax_expvar1, ax_expvar2], [mid1, mid2]):
    sns.barplot(
        x=[mid.item()],
        y=["Explained Variance"],
        ax=ax,
        color=clr_highlight,
        edgecolor=".9",
        linewidth=1,
    )
    ax.errorbar(
        x=mid,
        y=["Explained Variance"],
        xerr=[[mid1 - low1], [up1 - mid1]],
        fmt="none",
        color=".3",
        capsize=2,
    )

    ax.text(
        5,
        0,
        f"{mid.item():.1f}%",
        color=".9",
        ha="left",
        va="center",
        weight="bold",
    )
    ax.text(
        0,
        0.5,
        "Explained variance",
        color=".3",
        style="italic",
        size=8,
        transform=ax.transData,
    )
    ax.set_xlim([0, 50])
    ax.set_ylim([-0.5, 0.5])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)


save_to_vector = get_figure_path("chapter5", "vector/figure03.svg")
save_to_raster = get_figure_path("chapter5", "raster/figure03.png")
plt.savefig(save_to_vector, bbox_inches="tight", dpi=150)
plt.savefig(save_to_raster, bbox_inches="tight", dpi=150)

# %%
