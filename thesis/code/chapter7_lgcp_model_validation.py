# %%

import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
import utils.visualization as viz
import xarray as xr
from matplotlib.gridspec import GridSpec
from utils.statistics import yeojohnson_inv
from utils.tools import get_figure_path

viz.set_style()

COLORS = sns.color_palette("husl", 4, desat=0.8)
# COLORS = viz.get_sequential_color_palette(as_cmap=False, n_colors=6)[1:-1]
SEASONS = ["DJF", "MAM", "JJA", "SON"]
VARIABLE = "absolute/Plastic"
YEAR = 2001

project_path = "/home/nrieger/Projects/MINKE/seasonality_ospar/data/"
lgcp_path = project_path + f"gpr/{VARIABLE}/{YEAR}/"


# %%
ospar = xr.open_datatree(
    project_path + "beach_litter/ospar/preprocessed.zarr", engine="zarr"
)
litter_o = ospar["preprocessed/" + VARIABLE]
litter_o = litter_o.sel(year=slice(YEAR, 2020)).dropna("beach_id", **{"how": "all"})

model = xr.open_datatree(lgcp_path + "posterior_predictive.zarr", engine="zarr")
litter_m = model["posterior_predictive"][VARIABLE.split("/")[1]]

idata = {}
for s in SEASONS:
    idata[s] = az.from_netcdf(lgcp_path + f"idata_{s}.nc")


# %%
# Model convergence // trace plots
# =============================================================================
mu_o = litter_o.mean("year")
mu_m = litter_m.mean("n")

var_o = litter_o.var("year", ddof=1)
var_m = litter_m.var("n", ddof=1)
# %%
fig, ax = plt.subplots(1, 2, figsize=(7.2, 4))

for c, s in zip(COLORS, SEASONS):
    ax[0].scatter(
        mu_o.sel(season=s),
        mu_m.sel(season=s),
        s=10,
        ec="k",
        lw=0.2,
        marker="o",
        alpha=0.7,
        color=c,
        label=s,
    )
    ax[1].scatter(
        var_o.sel(season=s),
        var_m.sel(season=s),
        s=10,
        ec="k",
        lw=0.2,
        alpha=0.7,
        marker="o",
        color=c,
    )
for a in ax:
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlabel("Observed")
    a.set_ylabel("Modeled")
ax[0].set_xlim(1e0, 1e5)
ax[0].set_ylim(1e0, 1e5)
ax[1].set_xlim(1e0, 1e10)
ax[1].set_ylim(1e0, 1e10)
ax[0].plot([1e0, 1e5], [1e0, 1e5], "k--", lw=0.5)
ax[1].plot([1e0, 1e10], [1e0, 1e10], "k--", lw=0.5)
ax[0].set_title("A | Expected value E[Y] [in items/100m]", loc="left")
ax[1].set_title("B | Variance Var[Y] [in items$^2$/(100m)$^2$]", loc="left")
ax[0].legend(loc="upper left", frameon=False)
sns.despine(fig)

save_to_raster = get_figure_path("chapter7", "raster", "lgcp_model_evaluation.png")
save_to_vector = get_figure_path("chapter7", "vector", "lgcp_model_evaluation.svg")
plt.savefig(save_to_raster, bbox_inches="tight", dpi=300)
plt.savefig(save_to_vector, bbox_inches="tight", dpi=300)
plt.show()


# %%


# priors are the same for all season
lmbda = model["lambda_yeojohnson"]["lambda"]
prior = idata["DJF"].prior.squeeze()

fig = plt.figure(figsize=(7.2, 9))
gs = GridSpec(3, 2, figure=fig, hspace=0.2, wspace=0.2)
axes = {}
axes["mu_mu"] = fig.add_subplot(gs[0, 0])
axes["phi"] = fig.add_subplot(gs[0, 1])
axes["eta_1"] = fig.add_subplot(gs[1, 0])
axes["eta_2"] = fig.add_subplot(gs[1, 1])
axes["rho_1"] = fig.add_subplot(gs[2, 0])
axes["rho_2"] = fig.add_subplot(gs[2, 1])


axes["mu_mu"].set_title(r"A | GP mean $\mu_{\mu}$", loc="left")
axes["phi"].set_title(r"B | Dispersion $\phi$", loc="left")
axes["eta_1"].set_title(r"C | Kernel variance $\eta_{short}^2$", loc="left")
axes["eta_2"].set_title(r"D | Kernel variance $\eta_{long}^2$", loc="left")
axes["rho_1"].set_title(r"E | Kernel length scale $\ell_{short}$", loc="left")
axes["rho_2"].set_title(r"F | Kernel length scale $\ell_{long}$", loc="left")


# Priors
prior_mu_mu = yeojohnson_inv(prior.mu_mu, lmbda)
kwargs = {"clip": [0, None], "color": ".5", "fill": True}
kwargs1 = {"clip": [0, None], "color": ".5", "fill": True}
sns.kdeplot(data=prior_mu_mu, ax=axes["mu_mu"], label="Prior", **kwargs1)
sns.kdeplot(data=prior.phi, ax=axes["phi"], label="Prior", **kwargs)
sns.kdeplot(data=prior.eta_1, ax=axes["eta_1"], **kwargs)
sns.kdeplot(data=prior.eta_2, ax=axes["eta_2"], **kwargs)
sns.kdeplot(data=prior.rho_1, ax=axes["rho_1"], **kwargs)
sns.kdeplot(data=prior.rho_2, ax=axes["rho_2"], **kwargs)


for i, (season, id) in enumerate(idata.items()):
    post = az.extract(
        id,
        group="posterior",
        combined=True,
        var_names=["mu_mu", "phi", "eta_1", "eta_2", "rho_1", "rho_2"],
    )

    post_mu_mu = yeojohnson_inv(post.mu_mu, lmbda)

    kwargs = dict(label=season, color=COLORS[i], fill=False, alpha=0.8)
    sns.kdeplot(post_mu_mu, ax=axes["mu_mu"], **kwargs)
    sns.kdeplot(post.phi, ax=axes["phi"], **kwargs)
    sns.kdeplot(post.eta_1, ax=axes["eta_1"], **kwargs)
    sns.kdeplot(post.eta_2, ax=axes["eta_2"], **kwargs)
    sns.kdeplot(post.rho_1, ax=axes["rho_1"], **kwargs)
    sns.kdeplot(post.rho_2, ax=axes["rho_2"], **kwargs)

axes["mu_mu"].set_xlim(0, 1000)
axes["phi"].set_xlim(0, 8)
axes["eta_1"].set_xlim(0, 2)
axes["eta_2"].set_xlim(0, 2)
axes["rho_1"].set_xlim(0, 100)
axes["rho_2"].set_xlim(0, 4000)

axes["phi"].legend()

sns.despine(fig=fig, trim=True, left=False)

save_to_vector = get_figure_path("chapter7", "vector", "lgcp_model_trace_plots.svg")
save_to_raster = get_figure_path("chapter7", "raster", "lgcp_model_trace_plots.png")
plt.savefig(save_to_vector, bbox_inches="tight", dpi=300)
plt.savefig(save_to_raster, bbox_inches="tight", dpi=300)
plt.show()


# %%
