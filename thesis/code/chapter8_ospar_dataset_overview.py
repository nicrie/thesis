# %%
import matplotlib.pyplot as plt
import utils.visualization as viz
import xarray as xr
from cartopy.crs import PlateCarree, TransverseMercator
from cartopy.feature import LAND, OCEAN, RIVERS
from matplotlib.gridspec import GridSpec
from utils.tools import get_figure_path

viz.set_style()


# %%
SEASONS = ["DJF", "MAM", "JJA", "SON"]

project_path = "/home/nrieger/Projects/MINKE/seasonality_ospar/data"

ospar = xr.open_datatree(
    project_path + "/beach_litter/ospar/preprocessed.zarr", engine="zarr"
)

litter_o = ospar["preprocessed/absolute"].to_dataset()
litter_o = litter_o["Plastic"].dropna("beach_id", **{"how": "all"})

has_survey = litter_o.notnull()
n_survey = has_survey.sum("beach_id")

# %%
df1 = has_survey.sum("year").to_dataframe().reset_index()

df2 = n_survey.to_dataframe().reset_index()
df2 = df2.pivot(index="year", columns="season", values="Plastic").reset_index()
df2 = df2[["year"] + SEASONS]

# %%

import seaborn as sns

color_highlight = sns.color_palette("colorblind")[1]

cmap = viz.get_sequential_color_palette(as_cmap=True)
clr_beach = viz.get_sequential_color_palette(as_cmap=False)[-1]
palette = viz.get_sequential_color_palette(as_cmap=False, n_colors=9)[1:-1][::2]

proj = TransverseMercator(central_longitude=0.0, central_latitude=50.0)
extent = [-15, 13, 34, 64]

fig = plt.figure(figsize=(7.2, 4.5))
gs = GridSpec(
    2,
    4,
    figure=fig,
    hspace=0.05,
    wspace=0.05,
    # width_ratios=[1, 1],
    # height_ratios=[1, 1],
)
ax1 = fig.add_subplot(gs[:, :2], projection=proj)
ax2 = fig.add_subplot(gs[0, 2])
ax3 = fig.add_subplot(gs[0, 3])
ax4 = fig.add_subplot(gs[1, 2:])

ax1.add_feature(OCEAN.with_scale("50m"), facecolor=".9")
ax1.add_feature(LAND.with_scale("50m"), facecolor=".75")
ax1.add_feature(RIVERS.with_scale("10m"), edgecolor=".6", lw=0.5)
ax1.set_extent(extent, crs=PlateCarree())

ax1.scatter(litter_o.lon, litter_o.lat, s=0.5, color=clr_beach, transform=PlateCarree())

df2.plot(
    ax=ax2,
    x="year",
    kind="bar",
    stacked=True,
    color=palette,
    legend=False,
)
df2.cumsum().plot(
    ax=ax3,
    x="year",
    kind="bar",
    stacked=True,
    color=palette,
    legend=False,
)

sns.violinplot(
    df1,
    ax=ax4,
    x="season",
    y="Plastic",
    hue="season",
    width=0.8,
    linewidth=0.5,
    cut=0,
    inner="quart",
    palette=palette,
    inner_kws={"color": ".95", "linewidth": 1.0},
)

sns.stripplot(
    df1, ax=ax4, x="season", y="Plastic", legend=False, size=1.7, color=".1", alpha=0.75
)


# %%
save_to_raster = get_figure_path("chapter8", "raster", "ospar_dataset_overview.png")
save_to_vector = get_figure_path("chapter8", "vector", "ospar_dataset_overview.svg")
plt.savefig(save_to_raster, bbox_inches="tight", dpi=300)
plt.savefig(save_to_vector, bbox_inches="tight", dpi=300)
plt.show()

# %%
