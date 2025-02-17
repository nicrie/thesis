# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import utils.visualization as vis
from utils.tools import get_figure_path

vis.set_style()

# Get Data
# =============================================================================

projects = pd.read_csv(
    "~/Data/opensustain.tech/projects_20240906.csv",
    usecols=[
        "project_names",
        "category",
        "sub_category",
        "language",
        "score",
        "project_created_at",
    ],
)

CATEGORIES = [
    "Atmosphere",
    "Climate Change",
    "Cryosphere",
    "Hydrosphere",
    "Biosphere",
]

# %%
pr = projects[projects["category"].isin(CATEGORIES)]
pr["project_created_year"] = pd.to_datetime(pr["project_created_at"]).dt.year

# %%
nb_projects = pr.groupby(["project_created_year", "language"]).size().reset_index()
nb_projects = nb_projects.sort_values("project_created_year")
nb_projects_total = nb_projects.groupby("project_created_year").sum()[0].reset_index()
nb_projects_py_only = nb_projects[nb_projects["language"] == "Python"]

nb_projects_total_acc = nb_projects.groupby("language").sum()[0]
nb_projects_total_acc = nb_projects_total_acc.sort_values(ascending=False)
nb_projects_total_prct = nb_projects_total_acc / nb_projects_total_acc.sum() * 100
nb_projects_total_prct = nb_projects_total_prct.reset_index()
nb_projects_total_prct = nb_projects_total_prct.where(nb_projects_total_prct[0] > 0.5)


# %%
# Create the figure
# =============================================================================
clrs = vis.get_sequential_color_palette(as_cmap=False)
color_primary = clrs[5]
color_secondary = clrs[3]

# Initialize the matplotlib figure
f, ax = plt.subplots(
    ncols=2, figsize=(7, 3), gridspec_kw={"width_ratios": [4, 1], "wspace": 0.5}
)


# Plot the total number of projects
sns.set_color_codes("pastel")
sns.barplot(
    x="project_created_year",
    y=0,
    data=nb_projects_total,
    label="Total",
    color=color_primary,
    ec="w",
    ax=ax[0],
)

# Plot the number of projects based on Python
sns.set_color_codes("muted")
g = sns.barplot(
    x="project_created_year",
    y=0,
    data=nb_projects_py_only,
    label="Python",
    color=color_secondary,
    ec="w",
    ax=ax[0],
)

# Plot the percentage of projects based on language
sns.barplot(
    x=0,
    y="language",
    data=nb_projects_total_prct,
    color=color_secondary,
    ec="w",
    ax=ax[1],
)

# Add a legend and informative axis label
ax[0].legend(ncol=1, loc="upper left", frameon=False)
xticklabels = np.arange(2010, 2025)
xticklabels = [str(tick) for tick in xticklabels]
xticklabels[-1] = "2024*"

ax[0].set(xticklabels=xticklabels, xlim=(-1, 14.5), ylabel="", xlabel="")
g.set_xticklabels(g.get_xticklabels(), rotation=30)
ax[0].set_title("A | Number of OSS Created Per Year", y=1.05)


ax[1].set(xticks=[0, 20, 40], xlim=(0, 50), ylabel="", xlabel="")
ax[1].set_title("B | Share of OSS [%]", y=1.05)
ax[1].text(
    0,
    1,
    f"Total Number of Projects: {nb_projects_total_acc.sum()}",
    transform=ax[1].transAxes,
    ha="left",
    va="bottom",
    size="small",
    style="italic",
)

sns.despine(left=True, bottom=True)

path_vector = get_figure_path("chapter5", "vector", "overview_oss.svg")
path_raster = get_figure_path("chapter5", "raster", "overview_oss.png")
f.savefig(path_vector, bbox_inches="tight", format="svg")
f.savefig(path_raster, bbox_inches="tight", format="png")


# %%
ok = pr[pr["category"].isin(CATEGORIES)]
ok = ok[ok["language"].isin(["Python"])]
ok = ok[ok["sub_category"] == "Climate Data Processing and Analysis"]
ok.sub_category.unique()
sns.boxplot(ok["score"], whis=[0, 100])
plt.axhline(13.68)

# %%
