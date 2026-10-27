"""Bending amplitude near the clamp, CIGRE and Poffenberger-Swart formulas.

Amplitude of the conductor at a distance x from the clamp, for an antinode
amplitude of one diameter, with the minimal, mean and maximal bending
stiffness. These are the amplitudes behind the 'ba_cigre' and 'ba_pfs'
fictive stress methods of `dohl.damage.fictive_stress`.
"""

import matplotlib.pyplot as plt
import numpy as np

from dohl import damage

# Aster 570 conductor
mu, D, rts = 1.571, 3.105e-02, 1.853e05
EImin, EImax = 28.28, 2155.0

x = np.linspace(0, 0.2, 21)
freqs = [1, 2, 5, 10, 20, 50]
tensions = np.array([0.05, 0.15, 0.25]) * rts
stiffness = {"EImin": EImin, "EIavg": 0.5 * (EImin + EImax), "EImax": EImax}
markers = ["s", None, "o"]

fig, ax = plt.subplots(
    nrows=len(tensions), ncols=len(freqs), figsize=(19.2, 9.6), sharex=True, sharey=True
)
for i, tension in enumerate(tensions):
    for j, f in enumerate(freqs):
        for k, ei in enumerate(stiffness.values()):
            a = damage._ampl_cigre(D, f, tension, mu, ei, x)
            b = damage._ampl_pbs(D, f, tension, mu, ei, x)
            ax[i, j].plot(x, a, c="C0", marker=markers[k])
            ax[i, j].plot(x, b, c="C1", marker=markers[k])
        ax[i, j].grid(True)
        ax[i, j].set_title(
            f"f={f:.1f} Hz, H={tension / 1000.0:.1f} kN ({tension / rts:.0%} RTS)",
            fontsize="small",
        )
for a in ax[:, 0]:
    a.set_ylabel("amplitude (m)")
for a in ax[-1, :]:
    a.set_xlabel("distance to the clamp (m)")

# legend: colours for the formula, markers for the stiffness
ax[0, 0].plot([], [], c="C0", label="CIGRE")
ax[0, 0].plot([], [], c="C1", label="Poffenberger-Swart")
for name, marker in zip(stiffness, markers, strict=True):
    ax[0, 0].plot([], [], c="gray", marker=marker, label=name)
ax[0, 0].legend(fontsize="small")

fig.tight_layout()
plt.show()
