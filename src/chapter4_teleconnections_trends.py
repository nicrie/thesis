# %%
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean.cm as cmo
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
from matplotlib.gridspec import GridSpec

from utils.tools import get_figure_path

sns.set_context("paper")
plt.style.use("style/thesis.mplstyle")


mpl.rcParams["font.size"] = 7
mpl.rcParams["axes.linewidth"] = 0.5
mpl.rcParams["xtick.major.width"] = 0.3
mpl.rcParams["ytick.major.width"] = 0.3

trends = xr.load_dataset("/home/nrieger/Projects/cpcca/tele/data/trends.nc")

slope_sst = trends["slope_sst"] * 10  # °C/decade
slope_prcp = trends["slope_prcp"] * 10  # mm/decade

pvals_sst = trends["pvalues_sst"]
pvals_prcp = trends["pvalues_prcp"]

slope_sst = slope_sst.where(pvals_sst < 0.05)
slope_prcp = slope_prcp.where(pvals_prcp < 0.05)
# %%

proj_map = {"sst": ccrs.EqualEarth(central_longitude=180), "prcp": ccrs.EqualEarth()}
proj_data = ccrs.PlateCarree()

cmap = {"sst": cmo.balance, "prcp": cmo.tarn}

fig = plt.figure(figsize=(7.2, 2), dpi=300)
gs = GridSpec(2, 2, figure=fig, width_ratios=[1, 1], height_ratios=[1, 0.08])
ax1 = fig.add_subplot(gs[0, 0], projection=proj_map["sst"])
ax2 = fig.add_subplot(gs[0, 1], projection=proj_map["prcp"])
cax1 = fig.add_subplot(gs[1, 0])
cax2 = fig.add_subplot(gs[1, 1])

ax1.add_feature(cfeature.LAND, facecolor=".9")
ax2.add_feature(cfeature.OCEAN, facecolor=".9")
ax1.coastlines(color="w", lw=0.3)
ax2.coastlines(color="w", lw=0.3)


# SST
slope_sst.plot(
    ax=ax1,
    vmin=-0.2,
    vmax=0.2,
    cmap=cmap["sst"],
    transform=proj_data,
    cbar_ax=cax1,
    cbar_kwargs={
        "label": "SST [°C/decade]",
        "orientation": "horizontal",
        "ticks": [-0.2, -0.1, 0, 0.1, 0.2],
    },
)

# Precipitation
slope_prcp.plot(
    ax=ax2,
    vmin=-0.2,
    vmax=0.2,
    cmap=cmap["prcp"],
    transform=proj_data,
    cbar_ax=cax2,
    cbar_kwargs={
        "label": "Precipitation [mm/decade]",
        "orientation": "horizontal",
        "ticks": [-0.2, -0.1, 0, 0.1, 0.2],
    },
)

kws_gl = dict(draw_labels=["left", "top"], linestyle=":", color=".3", linewidth=0.2)
gl1 = ax1.gridlines(**kws_gl)
gl2 = ax2.gridlines(**kws_gl)

ax1.text(0, 1.05, "(A)", transform=ax1.transAxes, va="bottom", ha="left", weight="bold")
ax2.text(0, 1.05, "(B)", transform=ax2.transAxes, va="bottom", ha="left", weight="bold")

figname = "tele_trends"
path_to_pdf = get_figure_path("chapter4", f"pdf/{figname}.pdf")
path_to_raster = get_figure_path("chapter4", f"raster/{figname}.png")

plt.savefig(path_to_pdf, bbox_inches="tight")
plt.savefig(path_to_raster, bbox_inches="tight")

# %%
