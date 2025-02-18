# %%
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns
import utils.visualization as viz
import xarray as xr
from cartopy.crs import NearsidePerspective, PlateCarree, TransverseMercator
from cartopy.feature import LAND, OCEAN, RIVERS
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from utils.tools import get_figure_path

viz.set_style()


# %%
SEASONS = ["DJF", "MAM", "JJA", "SON"]
SEASON_NAMES = ["Winter", "Spring", "Summer", "Autumn"]
map_season_name = dict(zip(SEASONS, SEASON_NAMES))

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
df1["season"] = df1["season"].replace(map_season_name)


df2 = n_survey.to_dataframe().reset_index()
df2 = df2.pivot(index="year", columns="season", values="Plastic").reset_index()
df2 = df2[["year"] + SEASONS]
df2 = df2.rename(columns=map_season_name)

# %%


clr_highlight = sns.color_palette("colorblind")[1]

cmap = viz.get_sequential_color_palette(as_cmap=True)
clr_ec = viz.get_sequential_color_palette(as_cmap=False)[-1]
palette = viz.get_sequential_color_palette(as_cmap=False, n_colors=9)[1:-1][::2]

proj = TransverseMercator(central_longitude=0.0, central_latitude=50.0)
extent = [-15, 13, 34, 64]

fig = plt.figure(figsize=(7.2, 4.5))
gs = GridSpec(
    2,
    4,
    figure=fig,
    hspace=0.3,
    wspace=0.3,
    # width_ratios=[1, 1],
    # height_ratios=[1, 1],
)
ax1 = fig.add_subplot(gs[:, :2], projection=proj)
ax2 = fig.add_subplot(gs[0, 2])
ax3 = fig.add_subplot(gs[0, 3])
ax4 = fig.add_subplot(gs[1, 2:])

# Add map features
ax1.add_feature(OCEAN.with_scale("50m"), facecolor=".9")
ax1.add_feature(LAND.with_scale("50m"), facecolor=".75")
ax1.add_feature(RIVERS.with_scale("10m"), edgecolor=".6", lw=0.5)
ax1.set_extent(extent, crs=PlateCarree())

# Add gridlines
gl = ax1.gridlines(
    lw=0.3,
    color=".5",
    alpha=1,
    linestyle="--",
    x_inline=True,
    y_inline=True,
    draw_labels=True,
    rotate_labels=False,
)
gl.xlocator = mticker.FixedLocator([-10, 0, 10])
gl.ylocator = mticker.FixedLocator([40, 50, 60])
gl.top_labels = False
gl.right_labels = False

ax1.scatter(
    litter_o.lon,
    litter_o.lat,
    s=5,
    lw=0.4,
    color=clr_highlight,
    ec=clr_ec,
    transform=PlateCarree(),
)

# Add mini globe to upper left corner
proj_near_side = NearsidePerspective(
    central_longitude=-15.0, central_latitude=40.0, satellite_height=1e7
)
globe_ax = fig.add_axes([0.06, 0.7, 0.2, 0.2], projection=proj_near_side)
globe_ax.set_global()
globe_ax.add_feature(LAND, facecolor=".6")
globe_ax.add_feature(OCEAN, facecolor=".8")
rect = Rectangle(
    xy=(-15, 33),
    width=30,
    height=32,
    edgecolor=clr_highlight,
    facecolor="none",
    alpha=1,
    linewidth=1,
    transform=PlateCarree(),
)
globe_ax.add_patch(rect)
globe_ax.spines["geo"].set_edgecolor(".3")
globe_ax.spines["geo"].set_linewidth(0.3)

df2.plot(
    ax=ax2,
    x="year",
    kind="bar",
    stacked=True,
    color=palette,
    legend=False,
)
n_surveys_cumsum = df2.cumsum()
n_surveys_cumsum.plot(
    ax=ax3,
    x="year",
    kind="bar",
    stacked=True,
    color=palette,
    legend=False,
)
# Add label for each season to the last bar
total_surveys_last_year = n_surveys_cumsum.iloc[-1, 1:]
for i, season in enumerate(SEASON_NAMES):
    y = total_surveys_last_year.cumsum()[season]

    ax3.text(
        20.5,
        y - 0.5,
        season + f"\n({total_surveys_last_year[season]})",
        ha="left",
        va="top",
        color=palette[i],
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
    df1,
    ax=ax4,
    x="season",
    y="Plastic",
    legend=False,
    size=1.7,
    color=clr_highlight,
    edgecolor=clr_ec,
    linewidth=0.4,
    alpha=0.75,
)

# Set labels
ax1.set_title("A | Locations of beach litter surveys", loc="right")
ax2.set_title("B | Number of surveys over time")
ax4.set_title("C | Number of surveys per season")
ax2.text(0.05, 0.95, "Per year", transform=ax2.transAxes, ha="left", va="top")
ax3.text(0.05, 0.95, "Cumulative", transform=ax3.transAxes, ha="left", va="top")

ax2.set_xticks(np.arange(-1, 21, 5))
ax2.set_xlabel("")
ax2.set_xticklabels(np.arange(2000, 2021, 5), rotation=0)

ax3.set_xticks(np.arange(-1, 21, 5))
ax3.set_xticklabels(np.arange(2000, 2021, 5), rotation=0)
ax3.set_xlabel("")

ax4.set_yticks(np.arange(0, 21, 4))
ax4.set_xlabel("")
ax4.set_ylabel("")

# Label individual point as beach
ax4.annotate(
    "Each point\nrepresents a\nbeach in the\nOSPAR dataset",
    xy=(3.4, 4),
    xytext=(3.6, 18),
    ha="left",
    va="top",
    arrowprops=dict(facecolor=".3", arrowstyle="-", connectionstyle="arc3,rad=-0.2"),
)

# Remove spines
sns.despine(ax=ax2)
sns.despine(ax=ax3)
sns.despine(ax=ax4, bottom=True, left=True)

# Aesthetics
ax4.tick_params(axis="x", which="both", length=0)
ax4.tick_params(axis="y", which="both", length=0)
ax4.grid(axis="y", linestyle="-", alpha=0.5, linewidth=0.5)


save_to_raster = get_figure_path("chapter7", "raster", "ospar_dataset_overview.png")
save_to_vector = get_figure_path("chapter7", "vector", "ospar_dataset_overview.svg")
plt.savefig(save_to_raster, bbox_inches="tight", dpi=300)
plt.savefig(save_to_vector, bbox_inches="tight", dpi=300)
plt.show()

# %%
