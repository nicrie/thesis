# %%
import matplotlib.pyplot as plt
import seaborn as sns
import utils.visualization as viz
import xarray as xr
from utils.tools import get_figure_path

viz.set_style()

project_path = "/home/nrieger/Projects/MINKE/seasonality_ospar/data/"
fao = xr.open_dataset(project_path + "economic/production/fao_fish_production.nc")
live_weight = fao["live_weight"]
live_weight = live_weight.sel(year=slice(1950, 2020))
total_aqua_fish_production = live_weight.sum(("species", "country"))
total = total_aqua_fish_production.sum("production_type")
aqua = total_aqua_fish_production.sel(production_type="aqua")
wild = total_aqua_fish_production.sel(production_type="wild")

palette = viz.get_sequential_color_palette(as_cmap=False, n_colors=4)

clrs = {"wild": palette[2], "aqua": palette[1], "text": "0.9"}

# %%
# Figure
# =============================================================================

fig = plt.figure(figsize=(7.2, 3.5), dpi=300)
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.fill_between(
    total.year,
    total / 1e6,
    aqua / 1e6,
    color=clrs["wild"],
    label="Wild",
    alpha=0.7,
    ec="0.9",
)
ax1.fill_between(
    total.year, aqua / 1e6, color=clrs["aqua"], label="Aquaculture", alpha=0.7, ec="0.9"
)

# relative
ax2.fill_between(
    total.year,
    100 * total / total,
    100 * aqua / total,
    color=clrs["wild"],
    alpha=0.7,
    ec=".9",
)
ax2.fill_between(total.year, 100 * aqua / total, color=clrs["aqua"], alpha=0.7, ec=".9")

# add horizontal lines at 2, 4, 6, 8, 10 million tonnes
for y in range(2, 13, 2):
    ax1.axhline(y, color=".3", lw=0.3)
    ax2.axhline(y * 10, color=".3", lw=0.3)

# shorten the length of y label ticks to zero
ax1.tick_params(axis="y", length=0)
ax2.tick_params(axis="y", length=0)

# add the value of aquaculture in the last year as a marker with number, use annotate
ax1.annotate(
    f"{aqua[-1].values / 1e6:.1f} Mt in 2020",
    (2020, aqua[-1] / 1e6),
    (1980, 3),
    # make a curved arrow
    arrowprops=dict(
        arrowstyle="->", connectionstyle="arc3,rad=-0.2", color=clrs["text"]
    ),
    fontweight="bold",
    color=clrs["text"],
    ha="center",
)
ax2.annotate(
    f"{100 * aqua[-1] / total[-1].values:.1f}% in 2020",
    (2020, 100 * aqua[-1] / total[-1]),
    (1980, 30),
    # make a curved arrow
    arrowprops=dict(
        arrowstyle="->", connectionstyle="arc3,rad=-0.2", color=clrs["text"]
    ),
    fontweight="bold",
    color=clrs["text"],
    ha="center",
)


# for both ax1 and ax2 place the label "Wild capture" and "Aquaculture" at the right end of the plot in white
ax1.text(
    0.05,
    0.4,
    "Wild capture",
    color=clrs["text"],
    ha="left",
    va="center",
    fontweight="bold",
    transform=ax1.transAxes,
)
ax1.text(
    0.95,
    0.05,
    "Aquaculture",
    color=clrs["text"],
    ha="right",
    va="bottom",
    fontweight="bold",
    transform=ax1.transAxes,
)
ax2.text(
    0.05,
    0.95,
    "Wild capture",
    color=clrs["text"],
    ha="left",
    va="top",
    fontweight="bold",
    transform=ax2.transAxes,
)
ax2.text(
    0.95,
    0.05,
    "Aquaculture",
    color=clrs["text"],
    ha="right",
    va="bottom",
    fontweight="800",
    transform=ax2.transAxes,
)
ax1.set_title("A | Total production (million tonnes)", loc="left")
ax2.set_title("B | Relative production (%) ", loc="left")
ax1.set_ylabel("")
ax1.set_xlabel("")
ax1.set_ylim(0, 12)
ax2.set_ylim(0, 100)
ax1.set_xlim(1950, 2020)
ax2.set_xlim(1950, 2020)


sns.despine(fig, trim=True, left=True)

save_to_vector = get_figure_path(
    "chapter7", "vector", "production_nea_wild_capture_aquaculture.svg"
)
save_to_raster = get_figure_path(
    "chapter7", "raster", "production_nea_wild_capture_aquaculture.png"
)
plt.savefig(save_to_raster, bbox_inches="tight", dpi=300)
plt.savefig(save_to_vector, bbox_inches="tight", dpi=300)
plt.show()

# %%
