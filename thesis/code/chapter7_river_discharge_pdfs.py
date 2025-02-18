# %%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import utils.visualization as viz
from utils.tools import get_figure_path

viz.set_style()

# %%
# Load data
# =============================================================================
path_project = "/home/nrieger/Projects/MINKE/seasonality_ospar/data/"
path_data = path_project + "physical/river/river_discharge_nea_pdf.csv"

dist = pd.read_csv(path_data)

threshold = dist.threshold.dropna().unique()

n_samples = []
for th in threshold:
    sub = dist.discharge[dist.threshold == th]
    n_samples.append(len(sub))
    print("Threshold:", th, "Samples:", len(sub))

# D8ummy data for labels
# columns x; y; threshold, n_samples
# where x,y are just -1 for all threshold
dummy_data = pd.DataFrame(
    {
        "x": [-1] * len(threshold),
        "y": [-1] * len(threshold),
        "threshold": threshold,
        "n_samples": n_samples,
    }
)

dist_all_rivers = dist[dist.threshold.isna()]
dist_threshold = dist[dist.threshold != "None"]

# %%
# Figure
# =============================================================================
cmap = "crest_r"
clrs = sns.color_palette("crest_r", as_cmap=False, n_colors=len(dummy_data) - 1)

plt.figure(figsize=(7.2, 5))

# Add sample sizes
plot_lines = []
dd = dummy_data.iloc[1:].reset_index()
for i, row in dd.iterrows():
    (lp,) = plt.plot(row.x, row.y, color=clrs[i], label=row.n_samples)
    plot_lines.append(lp)
legend1 = plt.legend(
    [pl for pl in plot_lines],
    dd["n_samples"],
    loc="upper right",
    title="Number of rivers $n$",
    title_fontsize=8,
    ncol=3,
    bbox_to_anchor=(1.0, 0.8),
    columnspacing=2.5,
)

fg1 = sns.kdeplot(
    data=dist_all_rivers,
    x="discharge",
    log_scale=[True, False],
    common_norm=False,
    clip=[0, None],
    color="k",
    ls="--",
)
fg2 = sns.kdeplot(
    data=dist_threshold,
    x="discharge",
    hue="threshold",
    log_scale=[True, False],
    common_norm=False,
    clip=[0, None],
    palette=cmap,
)


plt.xlim(1, 10000)
plt.ylim(0, 1.201)
plt.xlabel("Discharge [m³/s]")
plt.ylabel("Probability density function")
plt.title("")
sns.move_legend(
    fg2,
    "upper right",
    **{"title": "Probability threshold $p$", "ncols": 3, "title_fontsize": 8},
)

sns.despine(trim=True)

plt.gca().add_artist(legend1)


plt.annotate(
    text=f"All rivers in \nWestern Europe \n$n={n_samples[0]}$",
    xy=(2, 1.0),
    xytext=(10, 0.9),
    xycoords="data",
    ha="left",
    va="bottom",
    fontsize=8,
    arrowprops=dict(arrowstyle="-", lw=0.5, color=".3", connectionstyle="arc3,rad=.4"),
)
plt.annotate(
    text="Only rivers near OSPAR \nbeaches with a cluster \nmembership probability $p\geq 0.4$.",
    xy=(10, 0.5),
    xytext=(30, 0.4),
    xycoords="data",
    ha="left",
    va="bottom",
    fontsize=8,
    arrowprops=dict(arrowstyle="-", lw=0.5, color=".3", connectionstyle="arc3,rad=.4"),
)

save_to_raster = get_figure_path("chapter7", "raster", "river_discharge_pdfs.png")
save_to_vector = get_figure_path("chapter7", "vector", "river_discharge_pdfs.svg")
plt.savefig(save_to_raster, bbox_inches="tight", dpi=300)
plt.savefig(save_to_vector, bbox_inches="tight", dpi=300)
plt.show()

# %%
