# %%
import arviz as az
import cmocean.cm as cmo
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import pandas as pd
import pymc as pm
import seaborn as sns
import utils.visualization as viz
import xarray as xr
from matplotlib.colors import LogNorm
from scipy.spatial.distance import cdist
from scipy.stats import invgamma, poisson
from utils.tools import get_figure_path

viz.set_style()

numpyro.set_host_device_count(4)


# Define the Matérn kernel
def matern_kernel(d, variance=0.1, length_scale=1.0, nu=1.5):
    scale_factor = variance
    d = np.maximum(d, 1e-10)  # Avoid division by zero
    if nu == 0.5:  # Exponential kernel
        return scale_factor * np.exp(-d / length_scale)
    elif nu == 1.5:
        sqrt_3 = np.sqrt(3)
        return (
            scale_factor
            * (1 + sqrt_3 * d / length_scale)
            * np.exp(-sqrt_3 * d / length_scale)
        )
    elif nu == 2.5:
        sqrt_5 = np.sqrt(5)
        return (
            scale_factor
            * (1 + sqrt_5 * d / length_scale + (5 * d**2) / (3 * length_scale**2))
            * np.exp(-sqrt_5 * d / length_scale)
        )
    else:
        raise NotImplementedError("Only nu=0.5, 1.5, and 2.5 are supported.")


# %%

RANDOM_SEED = 12345

# Define the domain
xmax, ymax = 12, 12
xstep = 0.5
ystep = 0.5
x = np.arange(0, xmax + xstep, xstep, dtype=float)
y = np.arange(0, ymax + ystep, ystep, dtype=float)
xbounds = np.array((x - xstep / 2).tolist() + [x[-1] + xstep / 2])
ybounds = np.array((y - ystep / 2).tolist() + [y[-1] + ystep / 2])
x_grid, y_grid = np.meshgrid(x, y)
coords = np.column_stack([x_grid.ravel(), y_grid.ravel()])

# Set the kernel hyperparameters
variance_f = 1
length_scale_f = 3
mean_intensity_f = 3 * np.ones(len(coords))

# Compute the covariance matrix
pairwise_distances = cdist(coords, coords)
cov_matrix = matern_kernel(
    pairwise_distances, variance=variance_f, length_scale=length_scale_f
)

# Simulate from the Gaussian Process
rng = np.random.default_rng(RANDOM_SEED)
latent_gp = rng.multivariate_normal(mean=mean_intensity_f, cov=cov_matrix)


# Transform GP to intensity via exp
intensity_f = np.exp(latent_gp).reshape(x_grid.shape)

# Simulate the point pattern (Poisson process)
poisson_rate = intensity_f * xstep * ystep  # Scale by grid cell area
counts = poisson.rvs(poisson_rate)


# create dummy points and add some noise for visualization
individual_points = np.repeat(coords, counts.ravel(), axis=0)
xnoise = rng.uniform(0, xstep, size=individual_points.shape[0])
ynoise = rng.uniform(0, ystep, size=individual_points.shape[0])
individual_points += np.column_stack([xnoise, ynoise])


# Plot the simulated intensity_f and points
xb, yb = np.meshgrid(xbounds, ybounds)
plt.figure(figsize=(7, 6))
plt.contourf(x, y, poisson_rate, vmin=0, levels=50, cmap="viridis")
plt.colorbar(label="Intensity")
plt.scatter(
    individual_points[:, 0][::-1],
    individual_points[:, 1][::-1],
    s=1,
    alpha=0.7,
    color="red",
)
plt.title("Simulated Intensity Function")
plt.xlabel("x")
plt.ylabel("y")
plt.xlim(0, xmax)
plt.ylim(0, ymax)
plt.tight_layout()
plt.show()

# %%
simulated = pd.DataFrame(
    np.concatenate([coords, counts.ravel().reshape(-1, 1)], axis=1),
    columns=["x", "y", "count"],
)
# simulate missing data
simulated["prob_sampling"] = np.linspace(0.05, 0.95, coords.shape[0]) ** (2)
observed = simulated.sample(
    frac=0.2, replace=False, weights="prob_sampling", random_state=RANDOM_SEED
)

ax = plt.axes()
simulated.plot.scatter(x="x", y="y", c=".9", ax=ax)
observed.plot.scatter(x="x", y="y", c="count", ec=".3", ax=ax)


# %%

try:
    trace = az.from_netcdf("../data/lgcp_trace.nc")
except FileNotFoundError:
    # Fit LGCP using PyMC
    with pm.Model() as model:
        # Priors for hyperparameters
        variance = pm.InverseGamma("variance", alpha=1.0, beta=1.0)
        length_scale = pm.Uniform("length_scale", lower=0.5, upper=10.0)

        # Covariance function
        mean = pm.gp.mean.Constant(3)
        cov = pm.gp.cov.Matern32(input_dim=2, ls=length_scale) * variance

        # Gaussian process prior
        gp = pm.gp.Latent(mean_func=mean, cov_func=cov)
        latent_field = gp.prior("latent_field", X=observed[["x", "y"]].values)

        # Likelihood
        intensity = pm.Deterministic(
            "intensity", pm.math.exp(latent_field) * xstep * ystep
        )
        pm.Poisson("obs", mu=intensity, observed=observed["count"].values)

        # Inference
        trace = pm.sample(
            draws=3000,
            tune=1000,
            chains=1,
            cores=2,
            return_inferencedata=True,
            nuts_sampler="numpyro",
            random_seed=RANDOM_SEED,
        )

    # Prior predictive checks
    with model:
        prior_predictive = pm.sample_prior_predictive(
            var_names=["variance", "length_scale"]
        )

    # Posterior predictive
    with model:
        latent_field_pred = gp.conditional(
            "latent_field_pred", Xnew=simulated[["x", "y"]].values
        )
        intensity_pred = pm.Deterministic(
            "intensity_pred", pm.math.exp(latent_field_pred) * xstep * ystep
        )
        posterior_predictive = pm.sample_posterior_predictive(
            trace,
            var_names=["intensity_pred"],
            extend_inferencedata=True,
            predictions=True,
            random_seed=RANDOM_SEED,
        )

    trace.to_netcdf("../data/lgcp_trace.nc")


# %%
# Posterior analysis
pm.plot_trace(trace, var_names=["variance", "length_scale"])
plt.show()

# %%
pm.summary(trace, var_names=["variance", "length_scale"])

# %%


xy_coords_obs = pd.MultiIndex.from_arrays(
    observed[["y", "x"]].values.T, names=["x2", "x1"]
)
xy_coords_obs = xr.Coordinates.from_pandas_multiindex(xy_coords_obs, "xy")

xy_coords_sim = pd.MultiIndex.from_arrays(
    simulated[["y", "x"]].values.T, names=["x2", "x1"]
)
xy_coords_sim = xr.Coordinates.from_pandas_multiindex(xy_coords_sim, "xy")


intensity_model = trace.posterior.intensity.sel(chain=0).rename(
    {"intensity_dim_0": "xy"}
)
intensity_model = intensity_model.assign_coords(xy=xy_coords_obs["xy"]).unstack("xy")

hdi_spread = pm.hdi(
    trace, hdi_prob=0.95, group="predictions", var_names="intensity_pred"
)
hdi_spread = hdi_spread.intensity_pred.diff("hdi").squeeze()
hdi_spread = (
    hdi_spread.rename({"intensity_pred_dim_2": "xy"})
    .assign_coords(xy=xy_coords_sim["xy"])
    .unstack("xy")
)
hdi_spread = hdi_spread.assign_coords(x1=x[::-1]).sortby("x1")


intensity_pred = trace.predictions.intensity_pred.rename({"intensity_pred_dim_2": "xy"})
intensity_pred = intensity_pred.assign_coords(xy=xy_coords_sim["xy"]).unstack("xy")
intensity_pred = intensity_pred.squeeze()
intensity_pred = intensity_pred.assign_coords(x1=x[::-1]).sortby("x1")


intensity_pred.mean("draw").plot.contourf(vmax=15)
# %%
plt.figure(figsize=(6, 6))
intensity_pred.squeeze().mean("draw").plot(cmap="viridis", vmin=0, vmax=16)
plt.title("Posterior Intensity Function")
plt.xlabel("x")
plt.ylabel("y")
plt.xlim(0, xmax)
plt.ylim(0, ymax)
plt.tight_layout()
plt.show()


# %%

clr_highlight = "#c33c54"
clr_secondary = ".3"
cmap = viz.get_sequential_color_palette(as_cmap=True)
clrs = viz.get_sequential_color_palette(as_cmap=False, n_colors=4)
cmap_uncertainty = cmo.matter

lvls = np.arange(0, 32.0, 0.5)

# Plot the simulated intensity and points
fig = plt.figure(figsize=(7.2, 9))
gs = fig.add_gridspec(
    4,
    5,
    hspace=0.3,
    wspace=0.1,
    height_ratios=[0.33, 0.33, 0.01, 0.33],
    width_ratios=[0.425, 0.025, 0.1, 0.425, 0.025],
)
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 3])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 3])
ax4 = fig.add_subplot(gs[3, 0])
ax5 = fig.add_subplot(gs[3, 3])

cax0 = fig.add_subplot(gs[0, 1])
cax1 = fig.add_subplot(gs[0, 4])
cax2 = fig.add_subplot(gs[1, 1])
cax3 = fig.add_subplot(gs[1, 4])


# A | Simulated intensity
# -----------------------------------------------------------------------------
cbar_lvls = np.linspace(0, 50.0)
cbar_ticks = np.arange(0, 51, 10)
im0 = ax0.contourf(x, y, poisson_rate, levels=cbar_lvls, cmap=cmap, vmin=0, vmax=50)
plt.colorbar(
    im0, cax=cax0, label="Poisson rate [counts per (0.5 km)$^2$]", ticks=cbar_ticks
)


# Simulated points
ax0.scatter(
    individual_points[:, 0][::-1],
    individual_points[:, 1][::-1],
    s=0.3,
    alpha=0.2,
    color=clr_highlight,
)

# B | Observed counts
# -----------------------------------------------------------------------------
ax1.scatter(
    observed["x"] + 0.25,
    observed["y"] + 0.25,
    c=observed["count"],
    s=20,
    marker="s",
    cmap=cmap,
    edgecolor=".3",
    linewidth=0.5,
    zorder=2,
)
ax1.contourf(x, y, poisson_rate, cmap=cmo.gray_r, vmin=0, vmax=50, alpha=0.5)
plt.colorbar(im0, cax=cax1, label="Counts per (0.5 km)$^2$]", ticks=cbar_ticks)

# C | Predicted intensity
# -----------------------------------------------------------------------------
ax2.scatter(
    observed["x"] + 0.25,
    observed["y"] + 0.25,
    c=observed["count"],
    s=20,
    marker="s",
    cmap=cmap,
    edgecolor=".3",
    linewidth=0.5,
    zorder=2,
)
im2 = ax2.contourf(
    x,
    y,
    intensity_pred.mean("draw"),
    levels=cbar_lvls,
    cmap=cmap,
    vmin=0,
    vmax=50,
    extend="max",
)
plt.colorbar(
    im2,
    cax=cax2,
    label="Poisson rate [counts per (0.5 km)$^2$]",
    ticks=cbar_ticks,
    extend="max",
)


# D | Predicted uncertainty
# -----------------------------------------------------------------------------
logstart = -1
logend = 1
lnorm = LogNorm(vmin=10**logstart, vmax=10**logend)

uncertainty_ratio = intensity_pred.std("draw") / intensity_pred.mean("draw")
im3 = ax3.contourf(
    x,
    y,
    uncertainty_ratio,
    cmap=cmap_uncertainty,
    norm=lnorm,
    levels=np.logspace(logstart, logend, 50),
    zorder=1,
)
ax3.contour(
    x, y, uncertainty_ratio, levels=[1], linewidths=[0.5], linestyles=["--"], zorder=2
)
ax3.scatter(
    observed["x"] + 0.25,
    observed["y"] + 0.25,
    c="none",
    s=20,
    marker="s",
    edgecolor=".3",
    linewidth=0.5,
    zorder=3,
)
plt.colorbar(
    im3,
    cax=cax3,
    label="Coefficient of variation [normalized]",
    ticks=np.logspace(logstart, logend, 5).round(2),
)

# E,F | Hyperparameters
# -----------------------------------------------------------------------------
clr_prior = clrs[1]
clr_posterior = clrs[2]

# Prior
prior_variance = invgamma.rvs(a=1.0, scale=1.0, size=4000, random_state=5)
sns.kdeplot(
    prior_variance,
    ax=ax4,
    color=clr_prior,
    fill=True,
    cut=0,
    clip=(0, 12),
    bw_adjust=0.02,
    zorder=1,
)
ax5.fill_betweenx(
    (0.0, 0.22),
    0.5,
    10,
    color=clr_prior,
    alpha=0.2,
    zorder=1,
)
ax5.plot([0.5, 10], [0.22, 0.22], color=clr_prior, lw=0.8)
ax5.plot([0, 0.5], [0, 0.0], color=clr_prior, lw=0.8)
ax5.plot([10, 12], [0, 0.0], color=clr_prior, lw=0.8)

# Posterior
sns.kdeplot(
    trace.posterior["variance"].sel(chain=0).values,
    ax=ax4,
    color=clr_posterior,
    fill=True,
    zorder=1,
)
sns.kdeplot(
    trace.posterior["length_scale"].sel(chain=0).values,
    ax=ax5,
    color=clr_posterior,
    fill=True,
    zorder=1,
)

# Posterior HDI and Median
hdi_var = pm.hdi(trace, hdi_prob=0.95)["variance"]
hdi_ls = pm.hdi(trace, hdi_prob=0.95)["length_scale"]

ax4.vlines(hdi_var, 0.60, 0.65, color=clr_posterior, lw=1.5)
ax5.vlines(hdi_ls, 0.60, 0.65, color=clr_posterior, lw=1.5)

ax4.plot(hdi_var, [0.625, 0.625], color=clr_posterior, lw=1)
ax5.plot(hdi_ls, [0.625, 0.625], color=clr_posterior, lw=1)

ax4.text(
    hdi_var[1] + 0.5, 0.625, "95% HDI", color=clr_secondary, ha="left", va="center"
)
ax5.text(hdi_ls[1] + 0.5, 0.625, "95% HDI", color=clr_secondary, ha="left", va="center")

ax4.vlines(
    trace.posterior.variance.median(),
    0.60,
    0.65,
    color=clr_posterior,
    lw=3,
)
ax5.vlines(
    trace.posterior.length_scale.median(),
    0.60,
    0.65,
    color=clr_posterior,
    lw=3,
)
# True values
ax4.vlines(variance_f, 0, 0.65, color=clr_secondary, lw=0.5, ls="--", zorder=50)
ax5.vlines(length_scale_f, 0, 0.65, color=clr_secondary, lw=0.5, ls="--", zorder=50)

# Annotations
kws_annots = dict(textcoords="data", ha="left", va="bottom", weight="bold")
ax4.annotate(
    "Prior",
    xy=(8, 0.05),
    color=clr_prior,
    **kws_annots,
)
ax4.annotate(
    "Posterior",
    xy=(3, 0.3),
    color=clr_posterior,
    **kws_annots,
)
ax5.annotate(
    "Prior",
    xy=(8, 0.25),
    color=clr_prior,
    **kws_annots,
)
ax5.annotate(
    "Posterior",
    xy=(4.4, 0.4),
    color=clr_posterior,
    **kws_annots,
)


ax0.set_title("A | Simulated Intensity Function")
ax1.set_title("B | Observed Counts")
ax2.set_title("C | Predicted Intensity Function")
ax3.set_title("D | Predicted Uncertainty")
ax4.set_title("E | Kernel Variance")
ax5.set_title("F | Kernel Length Scale")

ax0.set_xlabel("")
ax0.set_xlabel("")
ax2.set_xlabel("$x_1$ [km]")
ax3.set_xlabel("$x_1$ [km]")
ax0.set_ylabel("$x_2$ [km]")
ax1.set_ylabel("")
ax2.set_ylabel("$x_2$ [km]")
ax3.set_ylabel("")

ax4.set_xlabel("Variance $\eta^2$ [$(events/km^2)^2$]")
ax5.set_xlabel("Length scale $\ell$ [km]")
ax4.set_ylabel("PDF (normalized)")
ax5.set_ylabel("PDF (normalized)")

for ax in [ax0, ax1, ax2, ax3]:
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)

ax4.set_xlim(0, 12)
ax4.set_ylim(0, 0.8)
ax5.set_xlim(0, 12)
ax5.set_ylim(0, 0.8)

path_to_raster = get_figure_path("chapter4", "raster", "lgcp_model.png")
path_to_vector = get_figure_path("chapter4", "vector", "lgcp_model.svg")

fig.savefig(path_to_raster, dpi=300, bbox_inches="tight")
fig.savefig(path_to_vector, bbox_inches="tight")

plt.show()

# %%
pm.summary(trace, var_names=["variance", "length_scale"], hdi_prob=0.95, round_to=2)

# %%
