# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import utils.visualization as viz
import xarray as xr
from utils.tools import get_figure_path

viz.set_style()

# %%
project_path = "/home/nrieger/Projects/MINKE/seasonality_ospar/data/"
ospar = xr.open_datatree(
    project_path + "/beach_litter/ospar/preprocessed.zarr", engine="zarr"
)
ospar = ospar["preprocessed/absolute"].to_dataset()
plastics = ospar["Plastic"]
# %%
SEASONS = ["DJF", "MAM", "JJA", "SON"]
COLORS = viz.get_sequential_color_palette(as_cmap=False, n_colors=5)[1:]
mu = plastics.mean("year")
var = plastics.var("year", ddof=1)
n_surveys = plastics.notnull().sum("year").fillna(0)

ds = xr.Dataset(
    {
        "mu": mu,
        "var": var,
        "n_surveys": n_surveys,
    }
)
ds = ds.where(ds.n_surveys >= 4, drop=True)

# %%
# Figure Overdisperion in Beach Litter
# =============================================================================
df = ds.to_dataframe().reset_index()
df.rename(columns={"season": "Season", "n_surveys": "Number of surveys"}, inplace=True)
# use seaborn darkgrid


def compute_var(mu, phi):
    return mu * (1 + mu / phi)


def compute_mu(var, phi):
    return -phi / 2 + np.sqrt(phi**2 / 4 + phi * var)


plt.figure(figsize=(7.2, 7.2), dpi=300)
plt.fill_between(
    [1e0, 1e10],
    [1e10, 1e10],
    [1e0, 1e10],
    color=".9",
    zorder=0,
)
plt.text(0.05, 0.95, "Overdispersion", transform=plt.gca().transAxes, color=".3")
sns.scatterplot(
    data=df,
    x="mu",
    y="var",
    hue="Season",
    legend="brief",
    palette=COLORS,
    size="Number of surveys",
    alpha=0.75,
    zorder=50,
)
plt.xscale("log")
plt.yscale("log")
# plt.grid()
plt.xlim(1e0, 1e10)
plt.ylim(1e0, 1e10)
plt.xlabel("Sample mean $\mu$")
plt.ylabel("Unbiased sample variance $\sigma^2$")
plt.plot([1e0, 1e10], [1e0, 1e10], ls="--", color=".3", lw=0.5)
sns.despine()
# add text "Poisson distribution" along the diagonal
plt.text(
    1e8,
    6e7,
    "Theoretical Poisson distribution",
    color=".3",
    rotation=45,
    va="center",
    ha="center",
)
x = np.logspace(0, 10)
phis = np.array([1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3])
for phi in phis:
    plt.plot(x, compute_var(x, phi), ls="--", color=".3", lw=0.5)
    plt.text(
        compute_mu(var=1e8, phi=phi),
        2e8,
        f"{phi:.0E}",
        rotation=60,
        color=".3",
        va="center",
        ha="center",
    )


plt.legend(loc="lower right", frameon=False)

save_to_raster = get_figure_path("chapter8", "raster", "plastic_overdispersion.png")
save_to_vector = get_figure_path("chapter8", "vector", "plastic_overdispersion.svg")
plt.savefig(save_to_raster, bbox_inches="tight", dpi=300)
plt.savefig(save_to_vector, bbox_inches="tight", dpi=300)
plt.show()

# %%
