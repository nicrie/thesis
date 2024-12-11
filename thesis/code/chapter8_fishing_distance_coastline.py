# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import regionmask as rm
import seaborn as sns
import utils.visualization as viz
import xarray as xr
from sklearn.metrics.pairwise import haversine_distances
from utils.tools import get_figure_path

viz.set_style()

# Mask for North-East Atlantic region
boundaries_nea = np.array([[-20, 35], [-6, 35], [15, 60], [15, 65], [-20, 65]])
nea_mask = rm.Regions([boundaries_nea], names=["North East Atlantic"], abbrevs=["NEA"])

# from scipy.stats import cumfreq

path_project = "/home/nrieger/Projects/MINKE/seasonality_ospar/data/"

# %%
# Aquaculture
# -----------------------------------------------------------------------------
aquaculture = pd.read_csv(
    path_project + "economic/aquaculture/aquaculture_shellfish_finfish.csv"
)

# Consider only aquaculture at sea, ignore land-based farms
aquaculture = aquaculture[aquaculture["POSITION_COASTLINE"] == "At sea"]


# %%
# Coastline
# -----------------------------------------------------------------------------
# Land sea mask
landmask = rm.defined_regions.natural_earth_v5_0_0.land_10

lons = np.arange(-20, 15.1, 0.25)
lats = np.arange(35, 65.1, 0.25)
is_land = landmask.mask(lons, lats).notnull().astype(int)


def get_shore_nodes(landmask):
    """Function that detects the shore nodes, i.e. the land nodes directly
    next to the ocean. Computes the Laplacian of landmask.

    - landmask: the land mask built using `make_landmask`, where land cell = 1
                and ocean cell = 0.

    Output: 2D array array containing the shore nodes, the shore nodes are
            equal to one, and the rest is zero.
    """
    mask_lap = np.roll(landmask, -1, axis=0) + np.roll(landmask, 1, axis=0)
    mask_lap += np.roll(landmask, -1, axis=1) + np.roll(landmask, 1, axis=1)
    mask_lap -= 4 * landmask
    shore = np.ma.masked_array(landmask, mask_lap < 0)
    shore = shore.mask.astype("int")

    return shore


shore_nodes = get_shore_nodes(is_land)
shore_nodes = xr.DataArray(
    shore_nodes,
    dims=("lat", "lon"),
    coords={"lat": is_land.lat.values, "lon": is_land.lon.values},
)

is_coastline = shore_nodes.where(shore_nodes == 1)
is_coastline = is_coastline.where(nea_mask.mask(is_coastline).notnull())

coastline = is_coastline.stack(lonlat=("lon", "lat")).dropna("lonlat")


# %%
# Fishing intensity
# -----------------------------------------------------------------------------
wild_capture = xr.open_dataarray(
    path_project + "economic/fishing/wild_capture_fishing_intensity.nc"
)

wild_capture = wild_capture.sum("time")

# %%

capture_in_nea = nea_mask.mask(wild_capture).notnull()
wild_capture_nea = wild_capture.where(capture_in_nea)
wild_capture_nea = wild_capture_nea.stack(x=[...])
wild_capture_nea = wild_capture_nea.where(wild_capture_nea != 0.0, drop=True)
wild_capture_nea = wild_capture_nea.dropna("x")


# %%
# Compute distance to coast
# =============================================================================


def _np_dist_to_coast(lat1, lon1, lat2, lon2):
    coords1 = np.vstack([lat1, lon1]).T
    coords2 = np.vstack([lat2, lon2]).T
    coords1 = np.radians(coords1)
    coords2 = np.radians(coords2)
    return haversine_distances(coords1, coords2).min(axis=1) * 6371


def dist_to_coast(lat, lon, lat_coast, lon_coast):
    return xr.apply_ufunc(
        _np_dist_to_coast,
        lat,
        lon,
        input_core_dims=[["x"], ["x"]],
        output_core_dims=[["x"]],
        vectorize=False,
        dask="parallelized",
        output_dtypes=[np.float32],
        kwargs={"lat2": lat_coast.values, "lon2": lon_coast.values},
    )


dist_wild_capture = dist_to_coast(
    wild_capture_nea.lat, wild_capture_nea.lon, coastline.lat, coastline.lon
)
dist_wild_capture.name = "distance_to_coast"
dist_wild_capture = dist_wild_capture.to_dataframe()

dist_aqua = aquaculture.COAST_DIST_M * 1e-3
dist_aqua.name = "distance_to_coast"
dist_aqua = dist_aqua.to_frame()
# %%
# Histogram plot
# =============================================================================
palette = viz.get_sequential_color_palette(as_cmap=False, n_colors=4)

fig = plt.figure(figsize=(7.2, 3))
ax = fig.add_subplot(111)

ax_cum = ax.twinx()

sns.kdeplot(
    data=dist_aqua.dropna(),
    x="distance_to_coast",
    ax=ax,
    fill=True,
    color=palette[1],
    log_scale=True,
)
rng = np.random.default_rng(124)
jitter = rng.uniform(0.1, 30, size=len(dist_wild_capture))
jitter = pd.DataFrame(
    jitter, index=dist_wild_capture.index, columns=["distance_to_coast"]
)
sns.kdeplot(
    data=dist_wild_capture + jitter,
    x="distance_to_coast",
    ax=ax,
    weights=wild_capture_nea.values,
    fill=True,
    color=palette[2],
    log_scale=True,
)

# Cumulative distribution
sns.kdeplot(
    data=dist_aqua.dropna(),
    x="distance_to_coast",
    ax=ax_cum,
    fill=False,
    color=palette[1],
    log_scale=True,
    cumulative=True,
    linewidth=1.5,
)
sns.kdeplot(
    data=dist_wild_capture + jitter,
    x="distance_to_coast",
    ax=ax_cum,
    weights=wild_capture_nea.values,
    fill=False,
    color=palette[2],
    log_scale=True,
    cumulative=True,
)

ax.text(0.3, 0.2, "Aquaculture", color=palette[1], weight="bold", ha="center")
ax.text(9e1, 0.2, "Fishing", color=palette[2], weight="bold", ha="center")
ax_cum.text(8, 0.95, "cumulative", ha="center", style="italic", color=palette[1])
ax_cum.text(1e3, 0.95, "cumulative", ha="center", style="italic", color=palette[2])


ax.set_xlabel("Distance to coast [km]")
ax.set_ylabel("Probability density")
ax.set_xlim(1e-2, 4e3)
ax.set_ylim(0, 1.4)

ax_cum.set_ylabel("")
ax_cum.set_yticks([])
ax_cum.set_ylim(0, 1.01)

sns.despine(fig)


save_to_raster = get_figure_path("chapter8", "raster", "fishing_distance_coastline.png")
save_to_vector = get_figure_path("chapter8", "vector", "fishing_distance_coastline.svg")
plt.savefig(save_to_raster, bbox_inches="tight", dpi=300)
plt.savefig(save_to_vector, bbox_inches="tight", dpi=300)
plt.show()
# %%
