# %%
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import utils.visualization as viz
import xarray as xr
from utils.tools import get_figure_path

viz.set_style()

project_path = "/home/nrieger/Projects/MINKE/seasonality_ospar/data/"
wave_clim = xr.open_dataset(project_path + "physical/wave_height/wave_climatology.nc")

wave_clim = wave_clim["wave_height"]

is_land = wave_clim.isnull().all("month")

max_month = wave_clim.argmax("month", skipna=False).where(~is_land)
max_diff = wave_clim.max("month") - wave_clim.min("month")


# %%
# Figure
# =============================================================================
cmap = {
    "height": viz.get_sequential_color_palette(as_cmap=True),
    "season": plt.get_cmap("twilight"),
}
norm = colors.BoundaryNorm(np.arange(-0.5, 12, 1), cmap["season"].N)

proj = ccrs.TransverseMercator(central_longitude=0.0, central_latitude=50.0)
extent = [-15, 15, 34, 64]

fig, ax = plt.subplots(
    ncols=2,
    figsize=(7.2, 3),
    subplot_kw={"projection": proj},
    gridspec_kw={"wspace": 0.1},
    dpi=300,
)


max_diff.plot(
    ax=ax[0],
    transform=ccrs.PlateCarree(),
    cbar_kwargs={"ticks": np.arange(0, 3.1, 0.5), "label": ""},
    vmin=0,
    vmax=3,
    cmap=cmap["height"],
    zorder=2,
)
max_month.plot(
    ax=ax[1],
    transform=ccrs.PlateCarree(),
    cmap=cmap["season"],
    norm=norm,
    zorder=2,
    cbar_kwargs={"ticks": np.arange(0, 11.5, 1), "label": ""},
)

# change colorbar labels from 0..11 to Jan...Dec
cbar = ax[1].collections[0].colorbar
# cbar.set_ticks(np.arange(0, 12, 1))
cbar.set_ticklabels(
    [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
)

for a in ax:
    a.set_extent(extent)
    a.add_feature(cfeature.OCEAN.with_scale("50m"), color=".9", zorder=1)
    a.add_feature(cfeature.LAND.with_scale("50m"), color=".75", zorder=3)
    a.coastlines(resolution="50m", color=".5", zorder=3, lw=0.5)
    gl = a.gridlines(
        lw=0.3,
        color=".5",
        alpha=1,
        linestyle="--",
        draw_labels=True,
        rotate_labels=False,
        zorder=10,
    )
    gl.xlocator = mticker.FixedLocator([-10, 0, 10])
    gl.ylocator = mticker.FixedLocator([40, 50, 60])
    gl.top_labels = False
    gl.right_labels = False

ax[0].set_title("A | Annual spread in wave height [in m]", loc="left")
ax[1].set_title("B | Month of maximum wave height", loc="left")

save_to_raster = get_figure_path("chapter7", "raster", "wave_height.png")
plt.savefig(save_to_raster, bbox_inches="tight", dpi=300)
plt.show()


# %%
