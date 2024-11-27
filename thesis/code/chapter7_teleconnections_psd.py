# %%

from string import ascii_uppercase as LETTERS

import matplotlib.pyplot as plt
import seaborn as sns
import utils.visualization as viz
from cycler import cycler
from matplotlib.gridspec import GridSpec
from utils.tools import get_figure_path
from xarray.backends.api import open_datatree

viz.set_style()

clrs = sns.color_palette("tab20", n_colors=8, desat=0.9)
default_cycler = cycler(color=[clrs[0], clrs[1], clrs[6], clrs[7], clrs[2]])
plt.rc("axes", prop_cycle=default_cycler)


# %%
# Hilbert Analysis
# =============================================================================
case_study = "tele"
alpha = 1.00  # 1.00 or 0.00
n_rot = 22  # 22 or 28
power = 1  # 1 or 2
model = "rotated_hilbert_cpcca"
root_dir = f"/home/nrieger/Projects/cpcca/{case_study}/"
dt = open_datatree(root_dir + "data/power_spectral_density", engine="zarr")

# %%
psd = dt["scores"]
psd_red_noise = dt["red_noise"]
psd_bootstraps = dt["bootstraps"]


# %%
# Plotting
# =============================================================================
MODES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1]

clr_climate_index = ".5"

# xlim = (5e-3, 0.1)
# ylim = (1e-3, 10)
xlim = (1e-2, 0.1)
ylim = (0, 0.2)


fig = plt.figure(figsize=(7.2, 12))
gs = GridSpec(15, 2, figure=fig, hspace=1.5)
# create axes as a grid of subplots each covering 2 rows and 1 column

ax1 = [fig.add_subplot(gs[0, 0])]
ax1.append(fig.add_subplot(gs[1:3, 0]))

ax2 = [fig.add_subplot(gs[:3, 1])]
axes = [ax1[0], ax2[0]]
axes = (
    axes
    + [
        fig.add_subplot(gs[i : i + 3, j], sharex=ax2[0], sharey=ax2[0])
        for i in range(3, 15, 3)
        for j in range(2)
    ]
    + [ax1[1]]
)
idx_axes2mode = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 1}

# Create split axis
ax1[0].set_ylim(0.57, 0.7)
ax1[0].set_yticks([0.6, 0.65, 0.7])

ax1[1].set_ylim(0.0, 0.19)

# Now, let's turn towards the cut-out slanted lines.
# We create line objects in axes coordinates, in which (0,0), (0,1),
# (1,0), and (1,1) are the four corners of the Axes.
# The slanted lines themselves are markers at those locations, such that the
# lines keep their angle and position, independent of the Axes size or scale
# Finally, we need to disable clipping.
d = 0.5  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(
    marker=[(-1, -d), (1, d)],
    markersize=8,
    linestyle="none",
    color=".7",
    mec=".7",
    mew=0.8,
    clip_on=False,
)
ax1[0].plot([0], [0], transform=ax1[0].transAxes, **kwargs)
ax1[1].plot([0], [1], transform=ax1[1].transAxes, **kwargs)


# Add PSD of known climate indices
ax_idx = [axes[1], axes[3], axes[5], axes[8], axes[9]]
climate_indices = ["ONI", "EMI", "AMM", "PDO", "IOD"]
for ax, index in zip(ax_idx, climate_indices):
    psd["indexes"].sel(index=index).plot(ax=ax, color=clr_climate_index, label=index)
    ax.text(
        0.98,
        0.95,
        index,
        ha="right",
        va="top",
        transform=ax.transAxes,
        color=clr_climate_index,
        weight="semibold",
    )

for i, ax in enumerate(axes):
    letter = LETTERS[i]
    mode = MODES[i]

    psd["sst"].sel(mode=mode).plot(ax=ax, zorder=5, lw=1.5, label="SST")
    psd["prcp"].sel(mode=mode).plot(ax=ax, zorder=4, lw=1.5, label="Precipitation")

    # Add dummy label for climate indices
    if i == 0:
        axes[0].plot([], [], color=clr_climate_index, label="Climate Index")

    # Add bootstrapped PSDs
    bstplt = psd_bootstraps["bootstraps"].sel(
        mode=mode, bootstrap=slice(None, None, 25)
    )
    bstplt.plot.line(
        x="frequency", ax=ax, color="C0", alpha=0.025, zorder=0, add_legend=False
    )
    psd_bootstraps["quantiles"].sel(mode=mode, quantile=[0.025, 0.975]).plot.line(
        x="frequency",
        add_legend=False,
        ax=ax,
        color="C0",
        linestyle="--",
        linewidth=0.7,
        zorder=1,
        alpha=0.5,
    )
    psd_red_noise["sst"].sel(mode=mode, quantile=0.5).plot(
        ax=ax,
        color="darkred",
        linestyle="-",
        zorder=6,
        lw=0.8,
        alpha=0.7,
        label="AR(1)",
    )
    psd_red_noise["sst"].sel(mode=mode, quantile=0.95).plot(
        ax=ax,
        color="darkred",
        linestyle="--",
        zorder=6,
        lw=0.7,
        alpha=0.5,
        label=r"$\alpha=0.05$",
    )
    psd_red_noise["sst"].sel(mode=mode, quantile=0.99).plot(
        ax=ax,
        color="darkred",
        linestyle=":",
        zorder=6,
        lw=0.7,
        alpha=0.5,
        label=r"$\alpha=0.01$",
    )

    # Set title
    ax.set_xlabel("")
    ax.set_title(f"({letter}) | Mode {mode}")

ax_top1 = axes[0].twiny()
ax_top2 = axes[1].twiny()
# Fine-tune figure and axes
for ax in [ax_top1, ax_top2]:
    ax.set_xlabel("Period (months/cycle)")
    # ax.set_xscale("log")
    ax.set_xlim(xlim)
    periods = [10, 12, 18, 24, 36, 48, 100]
    freqs = [1 / p for p in periods]
    ax.set_xticks(freqs)
    ax.set_xticklabels(periods)
    ax.minorticks_off()  # Remove minor ticks
    ax.spines["top"].set_visible(True)
    ax.spines["bottom"].set_visible(False)

axes[0].legend(loc="upper left", ncols=2, frameon=False)
ax1[0].set_xlim(xlim)
ax1[1].set_xlim(xlim)
ax2[0].set_xlim(xlim)
ax2[0].set_ylim(ylim)
ax1[1].set_title("")
axes[1].set_yticks([0, 0.05, 0.1, 0.15, 0.2])
# axes[0, 0].set_xscale("log")
# axes[0, 0].set_yscale("log")
ax1[1].set_ylabel("Power [$1/$ cycles/month]", loc="bottom")
ax1[0].set_ylabel("")
axes[2].set_ylabel("Power [$1/$ cycles/month]")
axes[4].set_ylabel("Power [$1/$ cycles/month]")
axes[6].set_ylabel("Power [$1/$ cycles/month]")
axes[8].set_ylabel("Power [$1/$ cycles/month]")
axes[1].set_ylabel("")
axes[3].set_ylabel("")
axes[5].set_ylabel("")
axes[7].set_ylabel("")
axes[9].set_ylabel("")
axes[8].set_xlabel("Frequency [cycles/month]")
axes[9].set_xlabel("Frequency [cycles/month]")

# Fine-tuning of split axis
ax1[0].set_xticks([])
ax1[0].spines["bottom"].set_visible(False)
ax1[1].spines["top"].set_visible(False)
ax1[0].spines["right"].set_visible(False)
ax1[1].spines["right"].set_visible(False)
ax1[0].set_xlabel("")
ax1[1].set_xlabel("")


# Save figure
save_to_pdf = get_figure_path("chapter7", "pdf/tele_psd.svg")
save_to_raster = get_figure_path("chapter7", "raster/tele_psd.png")
plt.savefig(save_to_pdf, bbox_inches="tight")
plt.savefig(save_to_raster, bbox_inches="tight")

plt.show()

# %%
