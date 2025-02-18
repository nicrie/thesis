# %%

import matplotlib.pyplot as plt
import seaborn as sns
import utils.visualization as viz
import xarray as xr
from matplotlib.gridspec import GridSpec
from utils.tools import get_figure_path

viz.set_style()


# %%
COLORS = viz.get_sequential_color_palette(as_cmap=False, n_colors=4)
SEASONS = ["DJF", "MAM", "JJA", "SON"]
VARIABLE = "absolute/Plastic"
YEAR = 2001

project_path = "/home/nrieger/Projects/MINKE/seasonality_ospar/data/"
base_path = project_path + f"clustering/pca/{VARIABLE}/{YEAR}/"

pca_result = xr.open_dataset(base_path + "pca_clustering.nc", engine="netcdf4")
# %%
expvar_pos = pca_result["expvar_ratio_pos"]
expvar_neg = pca_result["expvar_ratio_neg"]

expvar_pos = expvar_pos.assign_coords(
    mode=[str(m) + "+" for m in expvar_pos.mode.values]
)
expvar_neg = expvar_neg.assign_coords(
    mode=[str(m) + "-" for m in expvar_neg.mode.values]
)

expvar = xr.concat([expvar_pos, expvar_neg], dim="mode")

idx_sorted = expvar.median("n").to_series().sort_values(ascending=False).index
expvar = expvar.sel(mode=idx_sorted).isel(mode=slice(None, 5))
expvar.name = "explained_variance"

q025 = expvar.quantile(0.025, "n")
q975 = expvar.quantile(0.975, "n")
n_significant = (q025 > q975.shift({"mode": -1})).sum("mode").item()


# %%
palette_colors = [COLORS[0]] * 5
for i in range(n_significant):
    palette_colors[i] = COLORS[1]
palette = sns.color_palette(palette_colors, desat=0.8)

fig = plt.figure(figsize=(7.2, 5))
gs = GridSpec(1, 1, figure=fig)
ax = fig.add_subplot(gs[0, 0])
sns.violinplot(
    data=expvar.to_dataframe().reset_index(),
    x="mode",
    y="explained_variance",
    hue="mode",
    legend=False,
    ax=ax,
    density_norm="width",
    bw_adjust=1,
    cut=1,
    linewidth=1,
    palette=palette,
)
ax.set_ylabel("Explained variance (%)")
ax.set_xlabel("Cluster")
ax.set_ylim(-0.01, 0.5)
ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
ax.set_yticklabels([0, 10, 20, 30, 40, 50])
ax.grid(axis="y", linestyle="--", linewidth=1)
# add horizontal grid line
ax.grid(axis="y", linestyle="--", alpha=0.5)
# make length of ticks shorter
ax.tick_params(axis="both", length=0)
sns.despine(fig, trim=True, bottom=True, left=True)

save_to_raster = get_figure_path(
    "chapter7", "raster", "plastic_clustering_explained_variance.png"
)
save_to_vector = get_figure_path(
    "chapter7", "vector", "plastic_clustering_explained_variance.svg"
)
plt.savefig(save_to_raster, bbox_inches="tight", dpi=300)
plt.savefig(save_to_vector, bbox_inches="tight", dpi=300)
plt.show()
# %%
