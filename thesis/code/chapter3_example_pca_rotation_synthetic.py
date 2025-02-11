# %%
from string import ascii_uppercase as ABC

import cmocean.cm as cmo
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import utils.visualization as viz
import xarray as xr
import xeofs as xe
import xskillscore as xs
from matplotlib.gridspec import GridSpec
from statsmodels.tsa.arima_process import arma_generate_sample
from utils.tools import get_figure_path

viz.set_style()


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
pca = xe.single.EOF(n_modes=10).fit(X, "time")
scores = pca.scores()
comps = pca.components()
expvar = pca.explained_variance_ratio()

rpca = xe.single.EOFRotator(n_modes=10).fit(pca)
rotated_scores = rpca.scores()
rotated_comps = rpca.components()
rotated_expvar = rpca.explained_variance_ratio()

spca = xe.single.SparsePCA(n_modes=3, alpha=5e-4)
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
        pearson_correlation_spca.loc[{"signal": f"s{i + 1}", "mode": i + 1}] *= -1


# %%
# Create Figure
# =============================================================================
cmap = viz.get_sequential_color_palette()
cmap_div = cmo.curl_r
clrs = viz.get_sequential_color_palette(as_cmap=False)
clr_true = clrs[-2]
clr_score = clrs[3]

fig = plt.figure(figsize=(6, 9), dpi=300)
gs = GridSpec(
    10,
    5,
    figure=fig,
    height_ratios=[1, 0.1, 1, 0.6, 0.1, 1, 0.6, 0.1, 1, 0.6],
    width_ratios=[1, 1, 1, 1, 0.05],
    hspace=0.05,
    wspace=0.05,
)
ax_comps_true = [fig.add_subplot(gs[0, i]) for i in range(1, 4)]
ax_comps_pca = [fig.add_subplot(gs[2, i]) for i in range(1, 4)]
ax_scores_pca = [fig.add_subplot(gs[3, i]) for i in range(1, 4)]
ax_comps_rpca = [fig.add_subplot(gs[5, i]) for i in range(1, 4)]
ax_scores_rpca = [fig.add_subplot(gs[6, i]) for i in range(1, 4)]
ax_comps_spca = [fig.add_subplot(gs[8, i]) for i in range(1, 4)]
ax_scores_spca = [fig.add_subplot(gs[9, i]) for i in range(1, 4)]

cax_true = fig.add_subplot(gs[0, 4])
cax_pca = fig.add_subplot(gs[2, 4])
cax_rpca = fig.add_subplot(gs[5, 4])
cax_spca = fig.add_subplot(gs[8, 4])


axes_comps = ax_comps_true + ax_comps_pca + ax_comps_rpca + ax_comps_spca
axes_scores = ax_scores_pca + ax_scores_rpca + ax_scores_spca
axes = axes_comps + axes_scores

ticks = np.arange(0, 51, 10)
levels = np.array([-0.07, -0.05, -0.03, -0.01, 0.01, 0.03, 0.05, 0.07])

txt_info_kws = dict(x=0.17, y=0.98, va="top", ha="left")

# Plot the true components
for i, ax in enumerate(ax_comps_true):
    im0 = scalings.sel(signal=f"s{i + 1}").plot.contourf(
        ax=ax, add_colorbar=False, vmin=0, vmax=1, cmap=cmap
    )
    plt.colorbar(im0, cax=cax_true)
    ax.text(
        s="{:.1f} %".format(100 * expvar_true.sel(signal=f"s{i + 1}").values),
        transform=ax.transAxes,
        **txt_info_kws,
    )

# Plot the PCA components
for i, ax in enumerate(ax_comps_pca):
    im1 = comps.sel(mode=i + 1).plot.contourf(
        ax=ax, add_colorbar=False, levels=levels, cmap=cmap_div
    )
    plt.colorbar(im1, cax=cax_pca)
    ax.text(
        s="{:.1f} %".format(100 * expvar.sel(mode=i + 1).values),
        transform=ax.transAxes,
        **txt_info_kws,
    )

# Plot the PCA rotated components
for i, ax in enumerate(ax_comps_rpca):
    im2 = rotated_comps.sel(mode=i + 1).plot.contourf(
        ax=ax, add_colorbar=False, levels=levels, cmap=cmap_div
    )
    plt.colorbar(im2, cax=cax_rpca)
    ax.text(
        s="{:.1f} %".format(100 * rotated_expvar.sel(mode=i + 1).values),
        transform=ax.transAxes,
        **txt_info_kws,
    )

# Plot the Sparse PCA components
for i, ax in enumerate(ax_comps_spca):
    im3 = spca_comps.isel(mode=i).plot.contourf(
        ax=ax, add_colorbar=False, levels=levels * 2, cmap=cmap_div
    )
    plt.colorbar(im3, cax=cax_spca)
    ax.text(
        s="{:.1f} %".format(100 * spca_expvar.sel(mode=i + 1).values),
        transform=ax.transAxes,
        **txt_info_kws,
    )

for ax in axes_comps:
    ax.set_aspect("equal")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(["" for t in ticks])
    ax.set_yticklabels(["" for t in ticks])

# Plot scores
for i, ax in enumerate(ax_scores_pca):
    S.sel(signal=f"s{i + 1}").plot(ax=ax, color=clr_true, lw=0.5, zorder=2)
    scores_norm.sel(mode=i + 1).plot(ax=ax, color=clr_score, lw=0.5, zorder=3)

    # Add correlation coefficient
    ax.text(
        0.17,
        0.98,
        "{:.2f}".format(pearson_correlation.isel(signal=i, mode=i).values),
        ha="left",
        va="top",
        transform=ax.transAxes,
    )


# Plot rotated scores
for i, ax in enumerate(ax_scores_rpca):
    S.sel(signal=f"s{i + 1}").plot(ax=ax, color=clr_true, lw=0.5, zorder=2)
    rotated_scores_norm.sel(mode=i + 1).plot(ax=ax, color=clr_score, lw=0.5, zorder=3)

    # Add correlation coefficient
    ax.set_title("")
    ax.text(
        0.17,
        0.98,
        "{:.2f}".format(pearson_correlation_rotated.isel(signal=i, mode=i).values),
        ha="left",
        va="top",
        transform=ax.transAxes,
    )

# Plot Sparse PCA scores
for i, ax in enumerate(ax_scores_spca):
    S.sel(signal=f"s{i + 1}").plot(ax=ax, color=clr_true, lw=0.5, zorder=2)
    spca_scores_norm.isel(mode=i).plot(ax=ax, color=clr_score, lw=0.5, zorder=3)

    # Add correlation coefficient
    ax.set_title("")
    ax.text(
        0.17,
        0.98,
        "{:.2f}".format(pearson_correlation_spca.isel(signal=i, mode=i).values),
        ha="left",
        va="top",
        transform=ax.transAxes,
    )

axes_wo_cax = (
    ax_comps_true
    + ax_comps_pca
    + ax_scores_pca
    + ax_comps_rpca
    + ax_scores_rpca
    + ax_comps_spca
    + ax_scores_spca
)

for i, ax in enumerate(axes_wo_cax):
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    ax.text(
        0.02, 0.98, "({:})".format(ABC[i]), transform=ax.transAxes, ha="left", va="top"
    )

for ax in axes_scores:
    ax.set_ylim(-4, 4)


for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    sns.despine(ax=ax, right=False, top=False)

for ax in axes_scores:
    sns.despine(ax=ax, left=True, bottom=True)
    ax.axhline(0, lw=0.5, color=".8", zorder=1)

title_kws = dict(x=-0.5, y=0.5, ha="center", va="center", weight=500)
ax_comps_true[0].text(
    s="True patterns", transform=ax_comps_true[0].transAxes, **title_kws
)
ax_comps_pca[0].text(s="PCA", transform=ax_comps_pca[0].transAxes, **title_kws)
ax_comps_rpca[0].text(
    s="Sparse PCA\n(Rotated)", transform=ax_comps_rpca[0].transAxes, **title_kws
)

ax_comps_spca[0].text(
    s="Sparse PCA\n(VP)",
    transform=ax_comps_spca[0].transAxes,
    **title_kws,
)
path_vector = get_figure_path("chapter3", "vector", "example_synthetic_pca_rotated.svg")
path_raster = get_figure_path("chapter3", "raster", "example_synthetic_pca_rotated.png")
fig.savefig(path_vector, format="svg", bbox_inches="tight")
fig.savefig(path_raster, format="png", bbox_inches="tight")

# %%
