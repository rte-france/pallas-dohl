"""Vibration amplitude with the damping methods, on random frequencies and tensions.

With the CIGRE wind power (in the balance, only 'cigre' and 'foti' give
realistic results with every damping method, and they are identical without
turbulence), the amplitude is computed for random frequencies and tensions
with each damping method. The Kendall tau between methods shows they rank
the cases in the same order, although their levels differ.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kendalltau

from dohl import ebp

# Aster 570 conductor
mu, D, rts = 1.571, 3.105e-02, 1.853e05
EImin, EImax, alpha = 28.28, 2155.0, 0.140652249164

conductor = ebp.Conductor(d=D, mu=mu, rts=rts)
damping_methods = ebp._balance_damping_methods()

rng = np.random.default_rng(0)
n = 300
freqs = rng.uniform(1, 50, n)
tensions = rng.uniform(0.05, 0.25, n) * rts

ymax = {dm: np.zeros(n) for dm in damping_methods}
for i in range(n):
    bending = ebp.Bending(EImin=EImin, EImax=EImax, x0=alpha * tensions[i] / rts)
    for dm in damping_methods:
        ymax[dm][i] = ebp.ymax(
            freqs[i],
            tensions[i],
            conductor,
            bending=bending,
            wind_method="cigre",
            damping_method=dm,
        )

# all methods
fig, ax = plt.subplots(1, 2, figsize=(12.8, 4.8))
for k, dm in enumerate(damping_methods):
    ax[0].scatter(freqs, ymax[dm] / D, marker="o", alpha=0.8, c=f"C{k}", label=dm)
ax[0].set_title("EBP with various damping methods")

# a selection from high to low
for k, dm in enumerate(["claren", "foti", "olla", "tompkins"]):
    ax[1].scatter(freqs, ymax[dm] / D, marker=".", alpha=0.8, c=f"C{k}", label=dm)
ax[1].set_title("EBP with various damping methods [selection]")
for a in ax:
    a.set(xlabel="Frequency (Hz)", ylabel="ymax / D")
    a.grid(True)
    a.legend()
fig.tight_layout()

# all methods have a similar ranking
m = len(damping_methods)
kt = np.zeros((m, m))
for i, di in enumerate(damping_methods):
    for j, dj in enumerate(damping_methods):
        kt[i, j] = kendalltau(ymax[di], ymax[dj]).correlation

fig, ax = plt.subplots()
q = ax.pcolormesh(kt, cmap="PuBu")
fig.colorbar(q)
ax.set_xticks(np.arange(m) + 0.5)
ax.set_yticks(np.arange(m) + 0.5)
ax.set_xticklabels(damping_methods, rotation=45)
ax.set_yticklabels(damping_methods)
ax.set_title("Kendall tau between damping methods")
fig.tight_layout()

plt.show()
