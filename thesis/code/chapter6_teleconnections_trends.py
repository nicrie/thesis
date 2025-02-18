# %%
from string import ascii_uppercase as ABC

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean.cm as cmo
import matplotlib.pyplot as plt
import utils.visualization as viz
import xarray as xr
from matplotlib.gridspec import GridSpec
from utils.tools import get_figure_path

viz.set_style()

trends = xr.load_dataset("/home/nrieger/Projects/cpcca/tele/data/trends.nc")

# activate for quick testing
# trends = trends.coarsen({"lon": 10, "lat": 10}, boundary="trim").mean()

mean = trends[["sst_mean", "prcp_mean"]]
std = trends[["sst_std", "prcp_std"]]
slope = trends[["sst_slope", "prcp_slope"]]
pvals = trends[["sst_pvalues", "prcp_pvalues"]]

mean = mean.rename({"sst_mean": "sst", "prcp_mean": "prcp"})
std = std.rename({"sst_std": "sst", "prcp_std": "prcp"})
slope = slope.rename({"sst_slope": "sst", "prcp_slope": "prcp"})
pvals = pvals.rename({"sst_pvalues": "sst", "prcp_pvalues": "prcp"})

slope = slope * 10  # per decade
slope = slope.where(pvals < 0.05)


# %%

proj_map = {"sst": ccrs.EqualEarth(central_longitude=180), "prcp": ccrs.EqualEarth()}
proj_data = ccrs.PlateCarree()

cmap = xr.DataArray(
    [[cmo.thermal, cmo.rain], ["viridis", "viridis"], [cmo.balance, cmo.tarn]],
    dims=["quantity", "variable"],
    coords={"variable": ["sst", "prcp"], "quantity": ["mean", "std", "slope"]},
)
cmap = cmap.to_dataset("variable")

fig = plt.figure(figsize=(7.2, 6), dpi=300)
gs = GridSpec(
    3,
    5,
    figure=fig,
    width_ratios=[1, 0.02, 0.2, 1, 0.02],
    height_ratios=[1, 1, 1],
    wspace=0.1,
)
ax1 = fig.add_subplot(gs[0, 0], projection=proj_map["sst"])
ax2 = fig.add_subplot(gs[0, 3], projection=proj_map["prcp"])
cax1 = fig.add_subplot(gs[0, 1])
cax2 = fig.add_subplot(gs[0, 4])

ax3 = fig.add_subplot(gs[1, 0], projection=proj_map["sst"])
ax4 = fig.add_subplot(gs[1, 3], projection=proj_map["prcp"])
cax3 = fig.add_subplot(gs[1, 1])
cax4 = fig.add_subplot(gs[1, 4])

ax5 = fig.add_subplot(gs[2, 0], projection=proj_map["sst"])
ax6 = fig.add_subplot(gs[2, 3], projection=proj_map["prcp"])
cax5 = fig.add_subplot(gs[2, 1])
cax6 = fig.add_subplot(gs[2, 4])

ax_sst = [ax1, ax3, ax5]
ax_prcp = [ax2, ax4, ax6]

for ax in ax_sst:
    ax.add_feature(cfeature.LAND, facecolor=".9")
    ax.coastlines(color=".3", lw=0.2)

for ax in ax_prcp:
    ax.add_feature(cfeature.OCEAN, facecolor=".9")
    ax.coastlines(color=".3", lw=0.2)


# ------------------------------------------------
# MEAN STATES
mean["sst"].plot(
    ax=ax1,
    vmin=0,
    vmax=30,
    cmap=cmap["sst"].sel(quantity="mean").item(),
    transform=proj_data,
    cbar_ax=cax1,
    cbar_kwargs={"ticks": [0, 10, 20, 30]},
)
(365 * mean["prcp"]).plot(
    ax=ax2,
    vmin=0,
    vmax=3000,
    cmap=cmap["prcp"].sel(quantity="mean").item(),
    transform=proj_data,
    cbar_ax=cax2,
    cbar_kwargs={"ticks": [0, 1000, 2000, 3000]},
)

# ------------------------------------------------
# ANNUAL VARIABILITY
std["sst"].plot(
    ax=ax3,
    vmin=0.2,
    vmax=0.8,
    cmap=cmap["sst"].sel(quantity="std").item(),
    transform=proj_data,
    cbar_ax=cax3,
    cbar_kwargs={"ticks": [0.2, 0.4, 0.6, 0.8]},
)
(365 * std["prcp"]).plot(
    ax=ax4,
    vmin=10,
    vmax=300,
    cmap=cmap["prcp"].sel(quantity="std").item(),
    transform=proj_data,
    cbar_ax=cax4,
    cbar_kwargs={"ticks": [10, 100, 200, 300]},
)

# ------------------------------------------------
# LINEAR TRENDS
slope["sst"].plot(
    ax=ax5,
    vmin=-0.2,
    vmax=0.2,
    cmap=cmap["sst"].sel(quantity="slope").item(),
    transform=proj_data,
    cbar_ax=cax5,
    cbar_kwargs={"ticks": [-0.2, -0.1, 0, 0.1, 0.2]},
)
(365 * slope["prcp"]).plot(
    ax=ax6,
    vmin=-100,
    vmax=100,
    cmap=cmap["prcp"].sel(quantity="slope").item(),
    transform=proj_data,
    cbar_ax=cax6,
    cbar_kwargs={"ticks": [-100, -50, 0, 50, 100]},
)

# ------------------------------------------------
# Titles
ax1.set_title("Sea Surface Temperature", weight="bold", loc="center")
ax2.set_title("Precipitation", weight="bold", loc="center")

# Label each subplot
for ax, letter in zip([ax1, ax2, ax3, ax4, ax5, ax6], ABC):
    ax.text(
        0,
        1.05,
        f"({letter})",
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        weight="bold",
    )

ax1.text(
    -0.2,
    0.5,
    "Annual Mean",
    transform=ax1.transAxes,
    va="center",
    ha="center",
    rotation=90,
    weight="bold",
)
ax3.text(
    -0.2,
    0.5,
    "Interannual Variability",
    transform=ax3.transAxes,
    va="center",
    ha="center",
    rotation=90,
    weight="bold",
)
ax5.text(
    -0.2,
    0.5,
    "Linear Trend",
    transform=ax5.transAxes,
    va="center",
    ha="center",
    rotation=90,
    weight="bold",
)

# Add colorbar units
kws_ylabel = {"loc": "top", "labelpad": -30, "rotation": 0}
cax1.set_ylabel("[°C]", **kws_ylabel)
cax2.set_ylabel("[mm]", **kws_ylabel)
cax3.set_ylabel("[°C]", **kws_ylabel)
cax4.set_ylabel("[mm]", **kws_ylabel)
cax5.set_ylabel("[°C/dc]", **kws_ylabel)
cax6.set_ylabel("[mm/dc]", **kws_ylabel)

# Gridlines
kws_gl = dict(draw_labels=["left", "bottom"], linestyle=":", color=".3", linewidth=0.2)
for ax in ax_sst + ax_prcp:
    gl = ax.gridlines(**kws_gl)


# ------------------------------------------------
# Save figure
figname = "tele_trends"
path_to_pdf = get_figure_path("chapter6", f"pdf/{figname}.svg")
path_to_raster = get_figure_path("chapter6", f"raster/{figname}.png")

# plt.savefig(path_to_pdf, bbox_inches="tight")
plt.savefig(path_to_raster, bbox_inches="tight")

# %%
