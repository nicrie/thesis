# %%

from string import ascii_uppercase as LETTERS

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns
import utils.visualization as viz
import xarray as xr
import xeofs as xe
from cartopy.feature import LAND, OCEAN
from cycler import cycler
from matplotlib.gridspec import GridSpec
from statsmodels.stats.multitest import multipletests
from utils.tools import get_figure_path

viz.set_style()

clrs = sns.color_palette("tab20", n_colors=8, desat=0.9)
default_cycler = cycler(color=[clrs[0], clrs[1], clrs[6], clrs[7]])
plt.rc("axes", prop_cycle=default_cycler)


def compute_angle(x):
    return xr.apply_ufunc(np.angle, x, dask="allowed", output_dtypes=[float])


def _np_pvalue_correction(p_values, alpha=0.05, method="fdr_bh"):
    shape_in = p_values.shape
    p_values = p_values.reshape(-1)
    pvals_corrected = np.zeros_like(p_values) * np.nan

    # Remove NaNs
    mask = np.isnan(p_values)
    p_values = p_values[~mask]

    # Apply correction
    reject, pvals_corrected[~mask], _, _ = multipletests(
        p_values, alpha=alpha, method=method
    )
    return pvals_corrected.reshape(shape_in)


def pvalue_correction(pvalues):
    return xr.apply_ufunc(
        _np_pvalue_correction,
        pvalues,
        input_core_dims=[["lat", "lon"]],
        output_core_dims=[["lat", "lon"]],
        kwargs={"alpha": 0.05, "method": "fdr_bh"},
        dask="allowed",
        output_dtypes=[float],
        vectorize=True,
    )


# %%
# Hilbert Analysis
# =============================================================================
case_study = "tele"
alpha = 1.00  # 1.00 or 0.00
n_rot = 22  # 22 or 28
power = 2  # 1 or 2
model = "rotated_hilbert_cpcca"
root_dir = f"/home/nrieger/Projects/cpcca/{case_study}/"
rot = xe.cross.HilbertCPCCARotator.load(
    root_dir + f"models/{model}_a{alpha:.2f}_r{n_rot}_p{power}"
)
dt = xr.open_datatree(
    root_dir + f"models/{model}_a{alpha:.2f}_r{n_rot}_p{power}_individual",
    engine="zarr",
)

model = xe.cross.HilbertCPCCA.load(root_dir + f"models/hilbert_cpcca_{alpha:.2f}")
model.compute()
model.data["input_data1"] = model.data["input_data1"].load()
model.data["input_data2"] = model.data["input_data2"].load()

tsc = model.data["total_squared_covariance"].load()
model_SCF = model.squared_covariance_fraction()
singular_values_rotated = np.sqrt(rot.data["squared_covariance"].load())
singular_values = rot.model_data["singular_values"].load()

# %%
# Coefficients of congruence // pattern stability
# =============================================================================
coef_cong = xr.open_dataset(root_dir + f"data/congruence_coefficient_{alpha:.2f}.nc")


# %%
# Sort according to SCF
lbda = np.sqrt(rot.data["squared_covariance"].load())
idx_sorted = np.argsort(dt["scf"].values)[::-1]
dt = dt.isel(mode=idx_sorted).update({"mode": dt.mode})

# %%
# Load climate indices
indexes = xr.open_dataset(root_dir + "data/index/climate_indexes.nc")
indexes = indexes.to_array("index")

# Smooth as preprocessed data
indexes = indexes.rolling(time=7, center=True).mean()
# Restrict to analized period
indexes = indexes.sel(time=slice("1940", "2022"))
# Remove trend from all inideces except OWI (global warming)
fit = indexes.polyfit("time", deg=1)
index_trend = xr.polyval(indexes.time, fit.polyfit_coefficients)
index_trend.loc[{"index": "OWI"}] = 0
indexes = indexes - index_trend
# Normalize
indexes = (indexes - indexes.mean("time")) / indexes.std("time")

# Flip IOD for better comparison
indexes.loc[{"index": "IOD"}] = -indexes.loc[{"index": "IOD"}]


# %%
# Components
Q0, Q1 = dt["sst"]["components"], dt["prcp"]["components"]
Qa0, Qa1 = abs(Q0), abs(Q1)

# Heterogeneous patterns
Qhet0 = dt["sst"]["heterogeneous_patterns"]
Qhet1 = dt["prcp"]["heterogeneous_patterns"]

Qhet0_pvals = pvalue_correction(dt["sst"]["pvalues"])
Qhet1_pvals = pvalue_correction(dt["prcp"]["pvalus"])


Qhet0 = abs(Qhet0)
Qhet1 = abs(Qhet1)

# Scores
P0, P1 = dt["sst"]["scores"], dt["prcp"]["scores"]
# normalize scores
P0 = P0 / P0.std("time")
P1 = P1 / P1.std("time")

# Explained variance etc.
SCF = dt["scf"]
CORR = dt["corr"]
FVE_X = dt["sst"]["fve"]
FVE_Y = dt["prcp"]["fve"]


# Due to a bug in the computation of the correlation coefficient (xeofs v3.0.1)
# we need to correct the correlation coefficient by the number of samples
n = P0.time.size
corr_factor = (n - 1) / n
CORR = CORR * corr_factor

# Apply phase shift so that the dominant patterns has a phase of 0
lonlats_max = Qhet0.fillna(0).argmax(dim=["lat", "lon"])
lon_p = lonlats_max["lon"]
lat_p = lonlats_max["lat"]
phi = np.angle(Q0.isel(lon=lon_p, lat=lat_p))

shift = np.exp(-1j * phi)
shift = xr.DataArray(shift, dims="mode", coords={"mode": Q0.mode})


P0 = P0 * shift
P1 = P1 * shift

Q0 = Q0 * shift
Q1 = Q1 * shift


# Correlation between indexes and (phase-shifted) scores
def compute_corr(indexes, scores, detrend=False):
    if detrend:
        pfit = scores.polyfit("time", deg=1)
        regression_line = xr.polyval(
            coord=scores.time, coeffs=pfit.polyfit_coefficients
        )
        scores = scores - regression_line
    return xr.corr(indexes, scores, dim="time")


all_corrs = compute_corr(indexes, P0.real, detrend=False)
all_corrs_detrended = compute_corr(indexes, P0.real, detrend=True)
all_corrs_detrended_1950 = compute_corr(
    indexes, P0.real.sel(time=slice("1950", None)), detrend=True
)

# Phases
Qp0, Qp1 = compute_angle(Q0), compute_angle(Q1)
Qhet0 = Qhet0.where(Qhet0_pvals <= 0.05)
Qhet1 = Qhet1.where(Qhet1_pvals <= 0.05)

phase_threshold = 0.125
Qp0 = Qp0.where(Qhet0 > phase_threshold)
Qp1 = Qp1.where(Qhet1 > phase_threshold)


# What is the correlation between the expansion coefficients?
corr_coef_scores = abs(np.corrcoef(P0, P1))[:n_rot, n_rot:].round(2)[:10, :10]
corr_coef_scores[np.tril_indices_from(corr_coef_scores)] = 0
print("Correlation coefficient scores: \n", corr_coef_scores)


# %%
# Plotting
# =============================================================================
mode = 10


def get_vmax(mode: int):
    if mode in [1]:
        vmax = 1.0
    elif mode in [2]:
        vmax = 0.6
    elif mode in [3, 4]:
        vmax = 0.4
    else:
        vmax = 0.3

    return vmax


def get_mode2index(alpha, n_rot, power):
    mode2index = dict(zip(np.arange(1, 11), [None] * 10))

    if np.isclose(alpha, 1.00) and power == 1:
        mode2index.update(
            {2: "ONI", 3: "OWI", 4: "EMI", 6: "AMMSST", 9: "PDO", 10: "IOD"}
        )
    elif np.isclose(alpha, 1.00) and power == 2:
        mode2index.update({2: "ONI", 3: "OWI", 6: "PDO", 7: "AMMSST", 8: "EMI"})

    return mode2index


mode2index = get_mode2index(alpha, n_rot, power)


scf = SCF.sel(mode=mode).values
ccoeff = CORR.sel(mode=mode).values
fve_x = FVE_X.sel(mode=mode).values
fve_y = FVE_Y.sel(mode=mode).values

cmap_twilight = plt.get_cmap("twilight")
cmap_twilight_shifted = viz.shift_cmap(cmap_twilight, 0.35)
cmap_amplitude = sns.color_palette("mako_r", as_cmap=True)
cmap_amplitude = sns.color_palette("Blues", as_cmap=True)
cmap = {"amplitude": cmap_amplitude, "phase": cmap_twilight_shifted}


data_proj = ccrs.PlateCarree()
map_proj = {
    "sst": ccrs.EqualEarth(central_longitude=200),
    "prcp": ccrs.EqualEarth(central_longitude=0),
}

fig = plt.figure(figsize=(7, 4.7))
gs = GridSpec(3, 3, figure=fig, width_ratios=[1, 1, 0.02], hspace=0.15, wspace=0.00)
ax1 = fig.add_subplot(gs[0, 0], projection=map_proj["sst"])
ax2 = fig.add_subplot(gs[0, 1], projection=map_proj["prcp"])
ax3 = fig.add_subplot(gs[1, 0], projection=map_proj["sst"])
ax4 = fig.add_subplot(gs[1, 1], projection=map_proj["prcp"])
ax5 = fig.add_subplot(gs[2, :2])

cax_amp = fig.add_subplot(gs[0, 2])
cax_phase = fig.add_subplot(gs[1, 2])

ax_comps = [ax1, ax2, ax3, ax4]
ax_scores = [ax5]

vmax = get_vmax(mode)

levels = np.arange(0, vmax, 0.125)
limits = {
    "vmin": 0,
    "vmax": vmax,
    # "levels": levels,
    "cmap": cmap["amplitude"],
    "transform": data_proj,
    "cbar_ax": cax_amp,
    "cbar_kwargs": {
        "label": "Amplitude [no units]",
        "ticks": [0, 0.2, 0.4, 0.6, 0.8, 1.0],
    },
}
Qhet0.sel(mode=mode).plot(ax=ax1, **limits)
Qhet1.sel(mode=mode).plot(ax=ax2, **limits)


levels = np.arange(-7 / 8, 7 / 8 + 0.001, 2 / 8) * np.pi
limits = {
    "levels": levels,
    "cmap": cmap["phase"],
    "transform": data_proj,
    "cbar_ax": cax_phase,
    "cbar_kwargs": {
        "label": "Phase [rad]",
        "ticks": [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
    },
}
Qp0.real.sel(mode=mode).plot(ax=ax3, **limits)
Qp1.real.sel(mode=mode).plot(ax=ax4, **limits)
cax_phase.set_yticklabels(["$-\pi$", "$-\pi/2$", "0", "$+\pi/2$", "$\pi$"])
cax_phase.tick_params(size=0)


(P0.real / P0.real.std("time")).sel(mode=mode).plot(ax=ax5, alpha=0.5, label="SST")
(P1.real / P1.real.std("time")).sel(mode=mode).plot(ax=ax5, alpha=0.5, label="PRCP")
index = mode2index[mode]
if index is not None:
    max_corr = all_corrs.sel(mode=mode, index=mode2index[mode]).item()
    label = f"{index} ($r_p$: {max_corr:.2f})"
    indexes.sel(index=index).plot(ax=ax5, lw=1, color=clrs[2], label=label, alpha=0.7)


ax1.set_title(f"Sea Surface Temperature \n{fve_x * 100:.1f} %", loc="center")
ax2.set_title(f"Precipitation \n{fve_y * 100:.1f} %", loc="center")
ax5.set_title("")
ax5.text(
    0.02,
    1,
    "(E) | Expansion Coefficients",
    ha="left",
    va="top",
    transform=ax5.transAxes,
)
ax5.text(0.5, 1, f"Mode {mode}", ha="center", va="top", transform=ax5.transAxes)
ax5.text(
    0.98,
    1,
    f"SCF: {scf * 100:.1f} %, Corr: {ccoeff:.2f}",
    ha="right",
    va="top",
    transform=ax5.transAxes,
)


for ax in ax_comps:
    ax.coastlines(lw=0.3, color=".5")
    ax.add_feature(LAND, facecolor=".95", zorder=0)
    ax.add_feature(OCEAN, facecolor=".95", zorder=0)

    ax.set_title("")

for ax in ax_scores:
    ax.set_ylim(-3, 4)
    ax.set_xlabel("")
    ax.set_ylabel("")


# Add gridlines
gl_kws = dict(linestyle=":", color=".3", linewidth=0.2)
gl1 = ax1.gridlines(draw_labels=["left"], **gl_kws)
gl2 = ax2.gridlines(draw_labels=False, **gl_kws)
gl3 = ax3.gridlines(draw_labels={"left": "y", "bottom": "x"}, **gl_kws)
gl4 = ax4.gridlines(draw_labels=["bottom"], **gl_kws)
for gl in [gl1, gl2, gl3, gl4]:
    #     gl.xlocator = mticker.FixedLocator([-120, -60, 0, 60, 120, 180])
    gl.ylocator = mticker.FixedLocator([-30, 0, 30])
    gl.xlabel_style = {"size": "small"}
    gl.ylabel_style = {"size": "small"}


# Add legend to scores plot
ax5.legend(loc="upper left", frameon=False, bbox_to_anchor=(0.95, 0.9))
ax5.axhline(0, color=".8", lw=0.7, ls="--", zorder=0)
sns.despine(ax=ax5, trim=True)

# Add letters for subplots
for ax, letter in zip(ax_comps, LETTERS):
    ax.text(
        0.02,
        1.02,
        f"({letter})",
        transform=ax.transAxes,
        va="bottom",
        ha="left",
    )


# Save figure
figname = f"tele_a{int(alpha * 100):03d}_r{n_rot}_p{power}_mode{mode:02d}"
save_to_vector = get_figure_path("chapter4", f"pdf/{figname}.pdf")
save_to_raster = get_figure_path("chapter4", f"raster/{figname}.png")

plt.savefig(save_to_vector, bbox_inches="tight")
plt.savefig(save_to_raster, bbox_inches="tight")


# %%
# Figure Squared Covariance Fraction and related measures
# =============================================================================
default_cycler = cycler(color=clrs)
plt.rc("axes", prop_cycle=default_cycler)

fig = plt.figure(figsize=(7.2, 6.5))
gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 2], hspace=0.25)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
ax4 = fig.add_subplot(gs[1, 0:1])
ax5 = fig.add_subplot(gs[1, 1:])
axes_top = [ax1, ax2, ax3]

SCF.plot(ax=ax1, marker=".")
model_SCF.plot(ax=ax1, marker=".")
SCF.cumsum().plot(ax=ax1)
model_SCF.cumsum().plot(ax=ax1)

(singular_values_rotated**2).plot(ax=ax2)
(singular_values**2).plot(ax=ax2)
(singular_values_rotated**2).cumsum().plot(ax=ax2)
(singular_values**2).cumsum().plot(ax=ax2)

singular_values_rotated.plot(ax=ax3, label="Rotated", color=clrs[0])
singular_values.plot(ax=ax3, label="Non-rotated", color=clrs[1])
singular_values_rotated.cumsum().plot(
    ax=ax3, label="Rotated (cumulative)", color=clrs[2]
)
singular_values.cumsum().plot(ax=ax3, label="Non-rotated (cumulative)", color=clrs[3])


ax1.set_xlim(0.5, 40)
ax2.set_xlim(0, 40)
ax3.set_xlim(0, 40)

ax1.set_ylim(1e-4, 1.15)
ax2.set_ylim(1e-3, 1e4)
ax3.set_ylim(1e-2, 1e2)

for ax in axes_top:
    ax.set_yscale("log")
    ax.set_xlabel("Mode")
    ax.set_ylabel("")
    ax.grid(True, which="major", axis="y", **{"lw": 0.2})

ax1.set_yticks([1e-4, 1e-3, 1e-2, 0.1, 1])
ax1.set_yticklabels(["0.01 %", "0.1 %", "1%", "10%", "100 %"])

ax1.set_title("A | Squared Covariance Fraction (SCF)")
ax2.set_title("B | Squared Singular Values $\sigma^2$")
ax3.set_title("C | Singular Values $\sigma$")

ax3.legend(ncols=2, bbox_to_anchor=[0, 0], loc="lower left", frameon=False)


# inset axes
x1, x2 = 20.5, 23.5
axins1 = ax1.inset_axes([0.4, 0.55, 0.6, 0.2], xlim=(0.5, 10.5), ylim=(0.98, 1.1))
axins2 = ax2.inset_axes(
    [0.7, 0.55, 0.3, 0.2],
    xlim=(x1, x2),
    ylim=(800, 1500),
    xticklabels=[],
    yticklabels=[],
)
axins3 = ax3.inset_axes(
    [0.7, 0.55, 0.3, 0.2], xlim=(x1, x2), ylim=(30, 50), xticklabels=[], yticklabels=[]
)
axins = [axins1, axins2, axins3]

model_SCF.cumsum().plot(ax=axins1, marker=".", color=clrs[3])
SCF.cumsum().plot(ax=axins1, marker=".", color=clrs[2])

(singular_values).cumsum().plot(ax=axins3, marker=".", color=clrs[3])
(singular_values_rotated).cumsum().plot(ax=axins3, marker=".", color=clrs[2])

(singular_values**2).cumsum().plot(ax=axins2, marker=".", color=clrs[3])
(singular_values_rotated**2).cumsum().plot(ax=axins2, marker=".", color=clrs[2])

for axr, ax in zip([ax1, ax2, ax3], axins):
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    axr.indicate_inset_zoom(ax)
    ax.set_xlabel("")
    ax.set_ylabel("")

axins1.set_xticks(np.arange(1, 11, 1))
axins1.set_yticks([1, 1.05, 1.10])
axins1.set_yticklabels(["100 %", "105 %", "110 %"])

# Stability of patterns
# -----------------------------------------------------------------------------
found_pattern = (coef_cong > 0.9).sum("mode2")
found_pattern.mean(["n_rot2", "mode1"]).to_array("variable").plot.line(
    ax=ax4, marker=".", x="n_rot1"
)
ax4.legend(
    ["SST", "PRCP"],
    title="Variable",
    ncols=1,
    frameon=False,
    loc="upper left",
    title_fontsize="medium",
)
ax4.set_ylim(0.5, 1)
ax4.set_xlabel("Number of Rotated Modes")
ax4.set_ylabel("Mode Stability $\chi$")
ax4.set_title("D | Stability of Rotated Patterns")
# -----------------------------------------------------------------------------

# Add Correlation Matrix
# -----------------------------------------------------------------------------
corr_mat = abs(np.corrcoef(P0))

# Replance diagonal with NaN
np.fill_diagonal(corr_mat, np.nan)

labels = np.arange(1, 11)
sns.heatmap(
    corr_mat[:10, :10],
    ax=ax5,
    vmin=0,
    vmax=0.3,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels,
    cbar_kws={"label": "Pearson Correlation Coefficient"},
)
ax5.set_xlabel("Mode")
ax5.set_ylabel("Mode")
ax5.set_title("E | Correlation Matrix of Expansion Coefficients")
# -----------------------------------------------------------------------------

# Save figure
save_to_pdf = get_figure_path("chapter7", "vectortele_singular_spectrum.svg")
save_to_raster = get_figure_path("chapter7", "raster", "tele_singular_spectrum.png")
plt.savefig(save_to_pdf, bbox_inches="tight")
plt.savefig(save_to_raster, bbox_inches="tight")


# %%

# %%
