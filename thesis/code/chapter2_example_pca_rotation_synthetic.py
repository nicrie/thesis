# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import xeofs as xe
import xskillscore as xs
from matplotlib.gridspec import GridSpec
from statsmodels.tsa.arima_process import arma_generate_sample
from utils.tools import get_figure_path

plt.style.use("style/tex.mplstyle")
plt.style.use("style/thesis.mplstyle")

mpl.rcParams["font.size"] = 5
mpl.rcParams["axes.linewidth"] = 0.5
mpl.rcParams["xtick.major.width"] = 0.5
mpl.rcParams["ytick.major.width"] = 0.5
# Set the length of the ticks
plt.rcParams["xtick.major.size"] = 0
plt.rcParams["ytick.major.size"] = 0


# %%
# Create synthetic data
# =============================================================================
# Define am ARMA(1,0) model for simulation
# Note: The AR parameter array must have 1 as the first element according to the convention used by the function
def simulate_ar(n_samples, ar_params):
    # Define the MA parameters
    ma_params = np.array([0.8, 0.5])

    # Define the AR and MA parameters
    ar = np.r_[1, -ar_params]  # add zero-lag and negate
    ma = np.r_[1, ma_params]  # add zero-lag

    # Generate the random time series
    simulated_time_series = arma_generate_sample(ar, ma, n_samples)

    return simulated_time_series


def normalize(signal, dim):
    return (signal - signal.mean(dim)) / signal.std(dim)


def convert_signal_to_xarray(signal, coords):
    return xr.DataArray(
        signal,
        dims=["time"],
        coords={"time": coords},
    )


def convert_scaling_to_xarray(scaling, xcoords, ycoords):
    return xr.DataArray(
        scaling,
        dims=["y", "x"],
        coords={"y": ycoords, "x": xcoords},
    )


# Define the number of sampleslatex detexify
n_samples = 1000

# Set the random seed for reproducibility
np.random.seed(40)

# Generate the random time series
signal1 = simulate_ar(n_samples, 0.99)
signal2 = simulate_ar(n_samples, 0.98)
signal3 = simulate_ar(n_samples, 0.97)

signal1 = normalize(signal1, 0)
signal2 = normalize(signal2, 0)
signal3 = normalize(signal3, 0)

# Print or plot the time series
fig = plt.figure(figsize=(6.3, 6.3 * 3 / 5))
plt.plot(signal1)
plt.plot(signal2)
plt.plot(signal3)


# %%
print(np.corrcoef(signal1, signal2))
print(np.corrcoef(signal1, signal3))
print(np.corrcoef(signal2, signal3))
# %%
# Define 3D data matrix with the three signals
time = np.arange(0, n_samples, 1)
lats = np.arange(51)
lons = np.arange(51)

X = np.ones((n_samples, len(lats), len(lons)))

scaling1 = np.exp(-((lats[:, None] - 40) ** 2) / 200) * np.exp(
    -((lons[None, :] - 40) ** 2) / 200
)
scaling2 = np.exp(-((lats[:, None] - 25) ** 2) / 200) * np.exp(
    -((lons[None, :] - 25) ** 2) / 200
)
scaling3 = np.exp(-((lats[:, None] - 10) ** 2) / 200) * np.exp(
    -((lons[None, :] - 10) ** 2) / 200
)

X1 = signal1[:, None, None] * X * scaling1 * 1.5
X2 = signal2[:, None, None] * X * scaling2 * 1
X3 = signal3[:, None, None] * X * scaling3 * 0.9


# Variance of each signal
print("s1: ", X1.var(0).sum())
print("s2: ", X2.var(0).sum())
print("s3: ", X3.var(0).sum())

X_ = X1 + X2 + X3

# Add noise
X_ = X_ + np.random.normal(0, 0.5, X.shape)

expvar_true = np.array([X1.var(0).sum(), X2.var(0).sum(), X3.var(0).sum()])
expvar_true = expvar_true / X_.var(0).sum()
expvar_true = xr.DataArray(
    expvar_true, dims=["signal"], coords={"signal": ["s1", "s2", "s3"]}
)

X = xr.DataArray(
    X_,
    dims=["time", "y", "x"],
    coords={"time": time, "y": lats, "x": lons},
)

# %%
# Perform PCA variants
# =============================================================================
pca = xe.models.EOF(n_modes=10).fit(X, "time")
scores = pca.scores()
comps = pca.components()
expvar = pca.explained_variance_ratio()

rpca = xe.models.EOFRotator(n_modes=10).fit(pca)
rotated_scores = rpca.scores()
rotated_comps = rpca.components()
rotated_expvar = rpca.explained_variance_ratio()

spca = xe.models.SparsePCA(n_modes=3, alpha=5e-4)
spca.fit(X, "time")
spca_scores = spca.scores(normalized=False).isel(mode=slice(0, 3))
spca_comps = spca.components().isel(mode=slice(0, 3))
spca_expvar = spca.explained_variance_ratio().isel(mode=slice(0, 3))

spca_comps.plot(col="mode")

print(expvar.values)
print(spca_expvar.values)

# %%


S1 = convert_signal_to_xarray(signal1, time)
S2 = convert_signal_to_xarray(signal2, time)
S3 = convert_signal_to_xarray(signal3, time)

scaling1 = convert_scaling_to_xarray(scaling1, lons, lats)
scaling2 = convert_scaling_to_xarray(scaling2, lons, lats)
scaling3 = convert_scaling_to_xarray(scaling3, lons, lats)

S = xr.concat([S1, S2, S3], "signal")
scalings = xr.concat([scaling1, scaling2, scaling3], "signal")

S = S.assign_coords(signal=["s1", "s2", "s3"])
scalings = scalings.assign_coords(signal=["s1", "s2", "s3"])

scores_norm = normalize(scores, "time")
rotated_scores_norm = normalize(rotated_scores, "time")
spca_scores_norm = normalize(spca_scores, "time")


pearson_correlation = xs.pearson_r(S, scores_norm.sel(mode=slice(1, 3)), dim="time")
pearson_correlation_rotated = xs.pearson_r(
    S, rotated_scores_norm.sel(mode=slice(1, 3)), dim="time"
)
pearson_correlation_spca = xs.pearson_r(
    S, spca_scores_norm.sel(mode=slice(1, 3)), dim="time"
)
# Sparse PCA has sometimes flipped components; align the sign
for i in range(3):
    if pearson_correlation_spca.isel(signal=i, mode=i) < 0:
        spca_scores_norm.loc[{"mode": i + 1}] *= -1
        spca_comps.loc[{"mode": i + 1}] *= -1
        pearson_correlation_spca.loc[{"signal": f"s{i+1}", "mode": i + 1}] *= -1


# %%
# Create Figure
# =============================================================================

fig = plt.figure(figsize=(6.3, 6.3 * 3 / 8), dpi=500)
gs = GridSpec(3, 10, figure=fig, width_ratios=[1, 0.1, 1, 1, 0.1, 1, 1, 0.1, 1, 1])
ax_comps_true = [fig.add_subplot(gs[i, 0]) for i in range(3)]
ax_comps_pca = [fig.add_subplot(gs[i, 2]) for i in range(3)]
ax_scores = [fig.add_subplot(gs[i, 3]) for i in range(3)]
ax_comps_pca_rotated = [fig.add_subplot(gs[i, 5]) for i in range(3)]
ax_scores_rotated = [fig.add_subplot(gs[i, 6]) for i in range(3)]
ax_comps_spca = [fig.add_subplot(gs[i, 8]) for i in range(3)]
ax_scores_spca = [fig.add_subplot(gs[i, 9]) for i in range(3)]


ticks = np.arange(0, 51, 10)
levels = np.array([-0.07, -0.05, -0.03, -0.01, 0.01, 0.03, 0.05, 0.07])
yoffset = 0.92
# Plot the true components
for i, ax in enumerate(ax_comps_true):
    scalings.sel(signal=f"s{i+1}").plot.contourf(
        ax=ax, add_colorbar=False, vmin=0, vmax=1, cmap="cividis"
    )
    ax.set_title(
        "{:.1f} \%".format(100 * expvar_true.sel(signal=f"s{i+1}").values), y=yoffset
    )

# Plot the PCA components
for i, ax in enumerate(ax_comps_pca):
    comps.sel(mode=i + 1).plot.contourf(
        ax=ax, add_colorbar=False, levels=levels, cmap="RdBu"
    )
    ax.set_title("{:.1f} \%".format(100 * expvar.sel(mode=i + 1).values), y=yoffset)

# Plot the PCA rotated components
for i, ax in enumerate(ax_comps_pca_rotated):
    rotated_comps.sel(mode=i + 1).plot.contourf(
        ax=ax, add_colorbar=False, levels=levels, cmap="RdBu"
    )
    ax.set_title(
        "{:.1f} \%".format(100 * rotated_expvar.sel(mode=i + 1).values), y=yoffset
    )

# Plot the Sparse PCA components
for i, ax in enumerate(ax_comps_spca):
    spca_comps.isel(mode=i).plot.contourf(
        ax=ax, add_colorbar=False, levels=levels * 2, cmap="RdBu"
    )
    ax.set_title("{:.1f} \%".format(100 * spca_expvar.isel(mode=i).values), y=yoffset)

for ax in ax_comps_true + ax_comps_pca + ax_comps_pca_rotated + ax_comps_spca:
    ax.set_aspect("equal")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(["" for t in ticks])
    ax.set_yticklabels(["" for t in ticks])


# Plot scores
for i, ax in enumerate(ax_scores):
    S.sel(signal=f"s{i+1}").plot(ax=ax, color=".2", lw=0.5)
    scores_norm.sel(mode=i + 1).plot(ax=ax, color="C0", lw=0.5)

    ax.set_xticks([])
    ax.set_yticks([])
    if i in [0, 1]:
        ax.set_xlabel("")
    if i in [0, 2]:
        ax.set_ylabel("")
    else:
        ax.set_ylabel("Scores", labelpad=0.1)

    # Add correlation coefficient
    ax.set_title(
        "r={:.2f}".format(pearson_correlation.isel(signal=i, mode=i).values),
        y=yoffset,
    )


# Plot rotated scores
for i, ax in enumerate(ax_scores_rotated):
    S.sel(signal=f"s{i+1}").plot(ax=ax, color=".2", lw=0.5)
    rotated_scores_norm.sel(mode=i + 1).plot(ax=ax, color="C0", lw=0.5)

    ax.set_title("")
    ax.set_xticks([])
    ax.set_yticks([])
    if i in [0, 1]:
        ax.set_xlabel("")
    if i in [0, 2]:
        ax.set_ylabel("")
    else:
        ax.set_ylabel("Scores", labelpad=0.1)

    # Add correlation coefficient
    ax.set_title(
        "r={:.2f}".format(pearson_correlation_rotated.isel(signal=i, mode=i).values),
        y=yoffset,
    )

# Plot Sparse PCA scores
for i, ax in enumerate(ax_scores_spca):
    S.sel(signal=f"s{i+1}").plot(ax=ax, color=".2", lw=0.5)
    spca_scores_norm.isel(mode=i).plot(ax=ax, color="C0", lw=0.5)

    ax.set_title("")
    ax.set_xticks([])
    ax.set_yticks([])
    if i in [0, 1]:
        ax.set_xlabel("")
    if i in [0, 2]:
        ax.set_ylabel("")
    else:
        ax.set_ylabel("Scores", labelpad=0.1)

    # Add correlation coefficient
    ax.set_title(
        "r={:.2f}".format(pearson_correlation_spca.isel(signal=i, mode=i).values),
        y=yoffset,
    )

ax_comps_true[1].set_ylabel("y", labelpad=0.1)
ax_comps_pca[1].set_ylabel("y", labelpad=0.1)
ax_comps_pca_rotated[1].set_ylabel("y", labelpad=0.1)

ax_comps_true[2].set_xlabel("x", labelpad=0.1)
ax_comps_pca[2].set_xlabel("x", labelpad=0.1)
ax_comps_pca_rotated[2].set_xlabel("x", labelpad=0.1)

title_kws = dict(x=0.5, y=1.2, ha="center", va="bottom", weight=1000)
ax_comps_true[0].text(
    s="True patterns", transform=ax_comps_true[0].transAxes, **title_kws
)
title_kws["x"] = 1.1
ax_comps_pca[0].text(s="PCA", transform=ax_comps_pca[0].transAxes, **title_kws)
ax_comps_pca_rotated[0].text(
    s="Sparse PCA (Varimax)", transform=ax_comps_pca_rotated[0].transAxes, **title_kws
)

ax_comps_spca[0].text(
    s="Sparse PCA (Variable Projection)",
    transform=ax_comps_spca[0].transAxes,
    **title_kws,
)
path_fig = get_figure_path("chapter2", "example_synthetic_pca_rotated.pdf")
fig.savefig(path_fig, format="pdf", bbox_inches="tight")

# %%
