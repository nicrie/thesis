# %%
import matplotlib.pyplot as plt
import numpy as np
import utils.visualization as viz

cmap = viz.get_sequential_color_palette()

x = np.arange(-10.0, 5.0, 0.2)
y = np.arange(-5.0, 5.0, 0.2)
X, Y = np.meshgrid(x, y)
Z1 = np.exp(-0.15 * ((X - 2) ** 2) - 0.5 * (Y + 2) ** 2)
Z3 = np.exp(-0.2 * ((X + 2) ** 2) - 0.1 * (Y - 4) ** 2)
Z = (Z1 - 0.4 * Z3) * 2

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

ax.view_init(elev=10, azim=80, roll=0)

# draw surface plot
surf = ax.plot_surface(X, Y, Z, lw=0.05, cmap=cmap, edgecolor="w")

# add color bar
# fig.colorbar(surf, shrink=0.5, aspect=10)

# projecting the contour with an offset
ax.contour(X, Y, Z, 20, zdir="z", offset=-1.5, cmap=cmap, linewidths=0.5)

# match the lower bound of zlim to the offset
ax.set(zlim=(-1, 2))
ax.set_axis_off()

plt.tight_layout()
plt.savefig("../frontmatter/surface.svg")
plt.show()

# %%
