# %%

from string import ascii_uppercase as LETTERS

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler
from xarray.backends.api import open_datatree

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


# %%
# Hilbert Analysis
# =============================================================================
case_study = "tele"
model = "rotated_hilbert_cpcca"
root_dir = f"/home/nrieger/Projects/cpcca/{case_study}/"
dt = open_datatree(root_dir + "data/power_spectral_density", engine="zarr")

# %%
psd = dt["scores"]
psd_red_noise = dt["red_noise"]


# %%
# Plotting
# =============================================================================
modes = [1, 2, 3, 4, 5, 6, 7, 8]

fig, axes = plt.subplots(
    nrows=4, ncols=2, figsize=(7, 8), sharex=True, sharey=True, squeeze=True
)

for ax, mode, letter in zip(axes.flatten(), modes, LETTERS):
    psd["sst"].sel(mode=mode, confidence="mean").plot(ax=ax, zorder=5, label="SST")
    psd["prcp"].sel(mode=mode, confidence="mean").plot(ax=ax, zorder=4, label="PRCP")

    ax.fill_between(
        psd.frequency,
        psd["sst"].sel(mode=mode, confidence="lower"),
        psd["sst"].sel(mode=mode, confidence="upper"),
        color=".80",
        alpha=0.5,
        zorder=2,
        label=r"95% CI",
    )
    psd_red_noise["sst"].sel(mode=mode, confidence="mean").plot(
        ax=ax,
        color="darkred",
        linestyle="-",
        zorder=6,
        lw=0.5,
        alpha=0.7,
        label="Red-noise fit",
    )
    psd_red_noise["sst"].sel(mode=mode, confidence="95").plot(
        ax=ax,
        color="darkred",
        linestyle="--",
        zorder=6,
        lw=0.5,
        alpha=0.5,
        label="95% CI",
    )
    psd_red_noise["sst"].sel(mode=mode, confidence="99").plot(
        ax=ax,
        color="darkred",
        linestyle=":",
        zorder=6,
        lw=0.5,
        alpha=0.5,
        label="99% CI",
    )
    # Add parameter rho for red noise model
    rho_sst = psd_red_noise["rho_sst"].sel(mode=mode).item()
    if mode == 1:
        red_noise_label = f"Red-noise model \n $\\rho$ = {rho_sst:.2f}"
    else:
        red_noise_label = f"\n $\\rho$ = {rho_sst:.2f}"
    ax.text(
        1,
        0.95,
        red_noise_label,
        ha="right",
        va="top",
        transform=ax.transAxes,
        fontsize=6,
    )

    # Set title
    ax.set_xlabel("")
    ax.set_title(f"({letter}) | Mode {mode}")

ax1 = axes[0, 0].twiny()
ax2 = axes[0, 1].twiny()
for ax in [ax1, ax2]:
    ax.set_xlabel("Period (months)")
    ax.set_xscale("log")
    ax.set_xlim(5e-3, 0.3)
    periods = [6, 12, 18, 24, 36, 48, 6 * 12, 10 * 12]
    freqs = [1 / p for p in periods]
    ax.set_xticks(freqs)
    ax.set_xticklabels(periods)
    ax.minorticks_off()  # Remove minor ticks
    ax.spines["top"].set_visible(True)

axes[0, 0].legend(loc="upper left", ncols=2, frameon=False)
axes[0, 0].set_xscale("log")
axes[0, 0].set_xlim(5e-3, 0.3)
axes[0, 0].set_ylim(0, 0.5)
axes[0, 0].set_ylabel("Power [$1/$ cycles month$^{-1}$]")
axes[1, 0].set_ylabel("Power [$1/$ cycles month$^{-1}$]")
axes[2, 0].set_ylabel("Power [$1/$ cycles month$^{-1}$]")
axes[3, 0].set_ylabel("Power [$1/$ cycles month$^{-1}$]")
axes[0, 1].set_ylabel("")
axes[1, 1].set_ylabel("")
axes[2, 1].set_ylabel("")
axes[3, 1].set_ylabel("")
axes[3, 0].set_xlabel("Frequency [cycles month$^{-1}$]")
axes[3, 1].set_xlabel("Frequency [cycles month$^{-1}$]")

# Save figure
save_to = get_figure_path("chapter4", "tele_psd.pdf")
plt.savefig(save_to, bbox_inches="tight")

# %%
