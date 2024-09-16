# %%

from string import ascii_uppercase as LETTERS

import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns
import xarray as xr
import xeofs as xe
from cartopy.feature import LAND, OCEAN
from cycler import cycler
from matplotlib.gridspec import GridSpec
from xarray.backends.api import open_datatree

from utils.plotting import shift_cmap
from utils.tools import get_figure_path

# plt.style.use("style/tex.mplstyle")
sns.set_context("paper")
plt.style.use("style/thesis.mplstyle")

clrs = sns.color_palette("tab20", n_colors=8, desat=0.9)

default_cycler = cycler(color=[clrs[0], clrs[1], clrs[6], clrs[7]])
plt.rc("axes", prop_cycle=default_cycler)
mpl.rcParams["font.size"] = 7
mpl.rcParams["axes.linewidth"] = 0.5
mpl.rcParams["xtick.major.width"] = 0.5
mpl.rcParams["ytick.major.width"] = 0.5


def compute_angle(x):
    return xr.apply_ufunc(np.angle, x, dask="allowed", output_dtypes=[float])


# %%
# Hilbert Analysis
# =============================================================================
case_study = "tele"
alpha = 0.2
model = "rotated_hilbert_cpcca"
root_dir = f"/home/nrieger/Projects/cpcca/{case_study}/"
rot = xe.cross.HilbertCPCCARotator.load(root_dir + f"models/{model}_{alpha:.2f}")
dt = open_datatree(root_dir + f"models/{model}_{alpha:.2f}_individual", engine="zarr")

lbda = np.sqrt(rot.data["squared_covariance"].load())

idx_sorted = np.argsort(dt["scf"].values)[::-1]
dt = dt.isel(mode=idx_sorted).assign_coords(mode=dt.mode)


indexes = xr.open_dataset(root_dir + "data/index/all_indexes_normed.nc")

# %%

# Components
Q0, Q1 = dt["sst"]["components"], dt["prcp"]["components"]
Qa0, Qa1 = abs(Q0), abs(Q1)

# Heterogeneous patterns
Qhet0 = dt["sst"]["heterogeneous_patterns"]
Qhet1 = dt["prcp"]["heterogeneous_patterns"]

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
# compute lagged correlation between P0.real and indexes phase shifting P0 from -PI to PI to find the best phase shift
phis = np.linspace(-np.pi, np.pi, 100)
possible_shifts = np.exp(-1j * phis)
possible_shifts = xr.DataArray(possible_shifts, dims=["shift"], coords={"shift": phis})
P0_shifted = P0 * possible_shifts
all_corrs = xr.corr(indexes.to_array("index"), P0_shifted.real, dim="time")
max_idx = all_corrs.argmax("shift").sel(index="OWI", mode=2)
phis[max_idx]


phi = np.zeros(Q0.mode.size, dtype=float)
phi[1] = 0.2856
phi[2] = 1.3645
phi[3] = 2.6793
phi[4] = -1.428
phi[6] = 0.5 * np.pi
phi[7] = 1.872
shift = np.exp(-1j * phi)
shift = xr.DataArray(shift, dims="mode", coords={"mode": Q0.mode})


P0 = P0 * shift
P1 = P1 * shift

Q0 = Q0 * shift
Q1 = Q1 * shift

# Phases

Qp0, Qp1 = compute_angle(Q0), compute_angle(Q1)
Qp0 = Qp0.where(Qhet0 > 0.2)
Qp1 = Qp1.where(Qhet1 > 0.2)

Qhet0 = Qhet0.where(Qhet0 > 0.2)
Qhet1 = Qhet1.where(Qhet1 > 0.2)

mode2index = {
    1: None,
    2: "OWI",
    3: "ONI",
    4: "EMI",
    5: "AMMSST",
    6: None,
    7: None,
    8: "PDO",
}


# %%
# Plotting
# =============================================================================
mode = 1

scf = SCF.sel(mode=mode).values
ccoeff = CORR.sel(mode=mode).values
fve_x = FVE_X.sel(mode=mode).values
fve_y = FVE_Y.sel(mode=mode).values

cmap_twilight = plt.get_cmap("twilight")
cmap_twilight_shifted = shift_cmap(cmap_twilight, 0.35)
cmap = {"amplitude": "Blues", "phase": cmap_twilight_shifted}

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


limits = {
    "levels": [0, 0.2, 0.4, 0.6, 0.8, 1],
    "cmap": cmap["amplitude"],
    "transform": data_proj,
    "cbar_ax": cax_amp,
    "cbar_kwargs": {"label": "Amplitude [no units]"},
}
Qhet0.sel(mode=mode).plot(ax=ax1, **limits)
Qhet1.sel(mode=mode).plot(ax=ax2, **limits)


limits = {
    "vmin": -np.pi,
    "vmax": np.pi,
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
cax_phase.set_yticklabels(["$-\pi$", "$-\pi/2$", "0", "$\pi/2$", "$\pi$"])


P0.real.sel(mode=mode).plot(ax=ax5, alpha=0.5, label="SST")
P1.real.sel(mode=mode).plot(ax=ax5, alpha=0.5, label="PRCP")
index = mode2index[mode]
if index is not None:
    max_corr = all_corrs.sel(mode=mode).max(("shift", "index"))
    label = f"{index} ($r_p$: {max_corr:.2f})"
    indexes[index].plot(ax=ax5, lw=1, color=clrs[2], label=label, alpha=0.7)


ax1.set_title(f"Sea Surface Temperature \n{fve_x*100:.1f} %", loc="center")
ax2.set_title(f"Precipitation \n{fve_y*100:.1f} %", loc="center")
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
    f"SCF: {scf*100:.1f} %, Corr: {ccoeff:.2f}",
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
gl1 = ax1.gridlines(draw_labels=["left"], linewidth=0.3)
gl2 = ax2.gridlines(draw_labels=False, linewidth=0.3)
gl3 = ax3.gridlines(draw_labels={"left": "y", "bottom": "x"}, linewidth=0.3)
gl4 = ax4.gridlines(draw_labels=["bottom"], linewidth=0.3)
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
save_to = get_figure_path("chapter4", f"tele_mode{mode:02d}.pdf")
plt.savefig(save_to, bbox_inches="tight")

# %%
