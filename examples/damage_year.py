"""Amplitude, fictive stress, strain and yearly damage against wind speed.

`dohl.lifespan.damage_year` for an Aster 570 conductor at several tensions,
with the 'olla_approx' damping, for each fictive stress method (colours) and
bending stiffness (markers).
"""

import matplotlib.pyplot as plt
import numpy as np

from dohl import damage
from dohl.lifespan import damage_year

# Aster 570 conductor
mu, D, rts = 1.571, 3.105e-02, 1.853e05
EImin, EImax, alpha = 28.28, 2155.0, 0.140652249164
young, d = 6.8e10, 3.45e-03  # Young modulus (Pa), last-strand diameter (m)

u = np.linspace(0, 5, 21)[1:]
ratios = [0.05, 0.10, 0.15, 0.20, 0.30, 0.35]
methods = damage._fictive_stress_methods()
fsbas = ["min", "avg", "max"]
markers = ["s", "o", "d"]

fig, ax = plt.subplots(nrows=4, ncols=len(ratios), figsize=(19.2, 12.8), sharex=True)
for i, ratio in enumerate(ratios):
    tension = ratio * rts
    for j, method in enumerate(methods):
        for k, fsba in enumerate(fsbas):
            ym, _, sg, dmg = damage_year(
                u,
                tension,
                D,
                d,
                mu,
                young,
                EImin,
                EImax,
                wind_method="cigre",
                damping_method="olla_approx",
                fictive_stress_method=method,
                fsba=fsba,
                multilayer=False,
                rts=rts,
                x0=alpha * tension / rts,
            )
            ax[0, i].plot(u, ym / D, c=f"C{j}", lw=1)
            ax[1, i].plot(u, sg * 1.0e-06, marker=markers[k], c=f"C{j}", lw=1)
            ax[2, i].plot(u, sg / young * 1.0e06, marker=markers[k], c=f"C{j}", lw=1)
            ax[3, i].semilogy(u, dmg, marker=markers[k], c=f"C{j}", lw=1)
    ax[0, i].set_title(f"T = {ratio:.0%} RTS")
    ax[-1, i].set_xlabel("Wind speed (m/s)")
for a in ax.flat:
    a.grid(True)
ax[0, 0].set_ylabel("Vibration amplitude (ymax/D)")
ax[1, 0].set_ylabel(r"Fictive stress ($\sigma$, MPa)")
ax[2, 0].set_ylabel("Micro strain")
ax[3, 0].set_ylabel("Estimated damage per year")

# legend: colours for the fictive stress method, markers for the stiffness
for j, method in enumerate(methods):
    ax[1, 0].plot([], [], c=f"C{j}", label=method)
for fsba, marker in zip(fsbas, markers, strict=True):
    ax[1, 0].plot([], [], c="gray", marker=marker, label=f"EI {fsba}")
ax[1, 0].legend(fontsize="small")

fig.tight_layout()
plt.show()
