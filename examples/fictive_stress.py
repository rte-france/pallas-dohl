"""Fictive stress against the antinode amplitude, for the three methods.

`dohl.damage.fictive_stress` with 'fymax', 'ba_cigre' and 'ba_pfs', for the
minimal, mean and maximal bending stiffness of an Aster 570 conductor, at
several frequencies and tensions.
"""

import matplotlib.pyplot as plt
import numpy as np

from dohl import damage

# Aster 570 conductor
mu, D, rts = 1.571, 3.105e-02, 1.853e05
EImin, EImax = 28.28, 2155.0
young, d = 6.8e10, 3.45e-03  # Young modulus (Pa), last-strand diameter (m)

stiffness = {"EImin": EImin, "EIavg": 0.5 * (EImin + EImax), "EImax": EImax}
methods = damage._fictive_stress_methods()
markers = ["s", "o", "d"]

ymax = np.linspace(0, D, 11)
freqs = [1, 2, 5, 10, 20, 50]
tensions = np.array([0.05, 0.15, 0.25]) * rts

fig, ax = plt.subplots(
    nrows=len(tensions), ncols=len(freqs), figsize=(19.2, 9.6), sharex=True, sharey=True
)
for i, tension in enumerate(tensions):
    for j, f in enumerate(freqs):
        for k, method in enumerate(methods):
            for m, ei in enumerate(stiffness.values()):
                s = damage.fictive_stress(ymax, f, tension, d, mu, young, ei, method=method)
                ax[i, j].semilogy(ymax / D, s * 1.0e-06, marker=markers[k], c=f"C{m}")
        ax[i, j].grid(True)
        ax[i, j].set_title(
            f"f={f:.1f} Hz, H={tension / 1000.0:.1f} kN ({tension / rts:.0%} RTS)",
            fontsize="small",
        )
for a in ax[:, 0]:
    a.set_ylabel(r"$\sigma$ (MPa)")
for a in ax[-1, :]:
    a.set_xlabel("ymax / D")

# legend: markers for the method, colours for the stiffness
for method, marker in zip(methods, markers, strict=True):
    ax[0, 0].plot([], [], marker=marker, c="gray", label=method)
for m, name in enumerate(stiffness):
    ax[0, 0].plot([], [], c=f"C{m}", label=name)
ax[0, 0].legend(fontsize="small")

fig.tight_layout()
plt.show()
