# %%
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import pymc as pm
import seaborn as sns
import utils.visualization as viz
from scipy.spatial.distance import cdist
from scipy.stats import invgamma, poisson, uniform
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
# Define the domain
xmax, ymax = 10, 10
xstep = 0.75
ystep = 0.75
x = np.arange(0, xmax + xstep, xstep, dtype=float)
y = np.arange(0, ymax + ystep, ystep, dtype=float)
xbounds = np.array((x - xstep / 2).tolist() + [x[-1] + xstep / 2])
ybounds = np.array((y - ystep / 2).tolist() + [y[-1] + ystep / 2])
x_grid, y_grid = np.meshgrid(x, y)
coords = np.column_stack([x_grid.ravel(), y_grid.ravel()])
print(coords.shape)

# Set the kernel hyperparameters
variance_f = 1.5
length_scale_f = 3
mean_intensity_f = 1 * np.ones(len(coords))

# Compute the covariance matrix
pairwise_distances = cdist(coords, coords)
cov_matrix = matern_kernel(
    pairwise_distances, variance=variance_f, length_scale=length_scale_f
)
print(cov_matrix.shape)

# Simulate from the Gaussian Process
rng = np.random.default_rng(5)
latent_gp = rng.multivariate_normal(mean=mean_intensity_f, cov=cov_matrix)

# %%
# Transform GP to intensity via exp
intensity_f = np.exp(latent_gp).reshape(x_grid.shape)

# Simulate the point pattern (Poisson process)
poisson_rate = intensity_f  # * xstep * ystep  # Scale by grid cell area
point_counts = poisson.rvs(poisson_rate)
points = np.column_stack((np.repeat(x, len(y)), np.tile(y, len(x))))

sampled_points = np.repeat(coords, point_counts.ravel(), axis=0)

# add some noise to the points
xnoise = rng.uniform(0, xstep, size=sampled_points.shape[0])
ynoise = rng.uniform(0, ystep, size=sampled_points.shape[0])
sampled_points += np.column_stack([xnoise, ynoise])

# sampled_points = np.clip(sampled_points, 0.05, xmax - 0.05)

# %%
# Plot the simulated intensity_f and points
xb, yb = np.meshgrid(xbounds, ybounds)
plt.figure(figsize=(6, 6))
plt.contourf(x, y, intensity_f, levels=50, cmap="viridis")
plt.colorbar(label="Intensity")
plt.scatter(
    sampled_points[:, 0][::-1], sampled_points[:, 1][::-1], s=1, alpha=0.7, color="red"
)
plt.title("Simulated Intensity Function")
plt.xlabel("x")
plt.ylabel("y")
plt.xlim(0, xmax)
plt.ylim(0, ymax)
plt.tight_layout()
plt.show()

# %%

try:
    trace = az.from_netcdf("../data/lgcp_trace.nc")
except FileNotFoundError:
    # Fit LGCP using PyMC
    with pm.Model() as model:
        # Priors for hyperparameters
        variance = pm.InverseGamma("variance", alpha=1.0, beta=1.0)
        length_scale = pm.Uniform("length_scale", lower=0.5, upper=5.0)

        # Covariance function
        mean = pm.gp.mean.Constant(1)
        cov = pm.gp.cov.Matern32(input_dim=2, ls=length_scale) * variance

        # Gaussian process prior
        gp = pm.gp.Latent(mean_func=mean, cov_func=cov)
        latent_field = gp.prior("latent_field", X=coords)

        # Likelihood
        intensity = pm.Deterministic("intensity", pm.math.exp(latent_field))
        pm.Poisson("obs", mu=intensity, observed=point_counts.ravel())

        # Inference
        trace = pm.sample(
            1000, chains=1, cores=2, return_inferencedata=True, nuts_sampler="numpyro"
        )

    # Prior predictive checks
    with model:
        prior_predictive = pm.sample_prior_predictive(
            var_names=["variance", "length_scale"]
        )

    trace.to_netcdf("../data/lgcp_trace.nc")


# %%
# Posterior analysis
pm.plot_trace(trace, var_names=["variance", "length_scale"])
plt.show()

# %%
pm.summary(trace, var_names=["variance", "length_scale"])

# %%


prior_variance = invgamma.rvs(a=1.0, scale=1.0, size=4000, random_state=5)
prior_length_scale = uniform.rvs(loc=0.5, scale=4.5, size=4000, random_state=5)
cmap = viz.get_sequential_color_palette(as_cmap=True, n_colors=4)
clrs = viz.get_sequential_color_palette(as_cmap=False, n_colors=4)
clr_highlight = "#c33c54"
clr_secondary = ".3"
lvls = np.arange(0, 32.0, 0.5)

# Plot the simulated intensity and points
fig = plt.figure(figsize=(7.2, 3.6))
gs = fig.add_gridspec(
    2, 4, hspace=0.4, wspace=0.05, width_ratios=[0.5, 0.025, 0.15, 0.325]
)
ax1 = fig.add_subplot(gs[:, 0])
cax = fig.add_subplot(gs[:, 1])
ax2 = fig.add_subplot(gs[0, 3])
ax3 = fig.add_subplot(gs[1, 3])

# Simulated intensity
im = ax1.contourf(
    x, y, intensity_f * xstep * ystep, levels=50, cmap=cmap, vmin=0, vmax=25
)
plt.colorbar(im, cax=cax, label="Poisson rate [events per square km]")
ax1.scatter(
    sampled_points[:, 0][::-1],
    sampled_points[:, 1][::-1],
    s=1,
    alpha=0.5,
    color=clr_highlight,
)


# Prior of hyperparameters
sns.kdeplot(
    prior_variance,
    ax=ax2,
    color=clrs[1],
    fill=True,
    cut=0,
    clip=(0, 10),
    bw_adjust=0.02,
)
ax3.fill_betweenx((0.0, 0.22), 0.5, 5, color=clrs[1], alpha=0.2)
ax3.plot([0.5, 5], [0.22, 0.22], color=clrs[1], lw=0.8)
ax3.plot([0, 0.5], [0, 0.0], color=clrs[1], lw=0.8)
ax3.plot([5, 10], [0, 0.0], color=clrs[1], lw=0.8)

# Posterior of hyperparameters
sns.kdeplot(
    trace.posterior["variance"].sel(chain=0).values, ax=ax2, color=clrs[2], fill=True
)
sns.kdeplot(
    trace.posterior["length_scale"].sel(chain=0).values,
    ax=ax3,
    color=clrs[2],
    fill=True,
)

# Posterior HDI and Median
ax2.vlines(
    pm.hdi(trace, hdi_prob=0.95)["variance"],
    0,
    0.8,
    color=clr_secondary,
    lw=0.5,
    ls="--",
    alpha=0.8,
)
ax3.vlines(
    pm.hdi(trace, hdi_prob=0.95)["length_scale"],
    0,
    0.6,
    color=clr_secondary,
    lw=0.5,
    ls="--",
    alpha=0.8,
)

ax2.vlines(
    trace.posterior.variance.median(), 0, 0.8, color=clr_secondary, lw=1, alpha=0.8
)
ax3.vlines(
    trace.posterior.length_scale.median(), 0, 0.6, color=clr_secondary, lw=1, alpha=0.8
)

# Annotations
kws_annots = dict(textcoords="data", ha="center", va="center", weight="bold")
ax2.annotate(
    "Prior",
    xy=(6, 0.15),
    color=clrs[1],
    **kws_annots,
)
ax2.annotate(
    "Posterior",
    xy=(2, 0.9),
    color=clrs[2],
    **kws_annots,
)
ax3.annotate(
    "Prior",
    xy=(6, 0.2),
    color=clrs[1],
    **kws_annots,
)
ax3.annotate(
    "Posterior",
    xy=(3, 0.7),
    color=clrs[2],
    **kws_annots,
)


ax1.set_title("A | Simulated Intensity Function")
ax2.set_title("B | Model Hyperparamters")
ax1.set_xlabel("x [km]")
ax1.set_ylabel("y [km]")
ax2.set_ylabel("")
ax3.set_ylabel("")
ax1.set_xlim(0, xmax)
ax1.set_ylim(0, ymax)
ax2.set_xlabel("Variance $\eta^2$ [$(events/km^2)^2$]")
ax3.set_xlabel("Length scale $\sigma$ [km]")
ax2.set_xlim(0, 10)
ax3.set_xlim(0, 10)
ax2.set_ylim(0, 1)
ax3.set_ylim(0, 0.8)
plt.show()

pm.summary(trace, var_names=["variance", "length_scale"], hdi_prob=0.95, round_to=2)

path_to_raster = get_figure_path("chapter4", "raster", "lgcp_model.png")
path_to_vector = get_figure_path("chapter4", "vector", "lgcp_model.svg")

fig.savefig(path_to_raster, dpi=300, bbox_inches="tight")
fig.savefig(path_to_vector, bbox_inches="tight")


# %%
