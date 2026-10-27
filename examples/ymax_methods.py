"""Vibration amplitude for every pair of wind and damping methods.

One panel per pair accepted by `dohl.ebp.ymax` (rows: wind methods, columns:
damping methods), for an Aster 570 conductor at 5, 15 and 25 % of its rated
tensile strength.
"""

import matplotlib.pyplot as plt
import numpy as np

from dohl import ebp

# Aster 570 conductor
mu, D, rts = 1.571, 3.105e-02, 1.853e05
EImin, EImax, alpha = 28.28, 2155.0, 0.140652249164

conductor = ebp.Conductor(d=D, mu=mu, rts=rts)
freq = np.linspace(0.0, 50, 51)[1:]
ratios = [0.05, 0.15, 0.25]
wind_methods = ebp._balance_wind_methods()
damping_methods = ebp._balance_damping_methods()

fig, ax = plt.subplots(
    nrows=len(wind_methods),
    ncols=len(damping_methods),
    figsize=(24, 11),
    sharex=True,
    sharey=True,
)
for i, wm in enumerate(wind_methods):
    for j, dm in enumerate(damping_methods):
        for k, ratio in enumerate(ratios):
            bending = ebp.Bending(EImin=EImin, EImax=EImax, x0=alpha * ratio)
            y = ebp.ymax(
                freq,
                ratio * rts,
                conductor,
                bending=bending,
                wind_method=wm,
                damping_method=dm,
            )
            ax[i, j].plot(freq, y / D, c=f"C{k}", label=f"T = {ratio:.0%} RTS")
        ax[i, j].set_title(f"{wm} / {dm}", fontsize="small")
        ax[i, j].grid(True)
ax[0, 0].set_ylim([0, 1])
ax[0, 0].legend(fontsize="small")
for a in ax[:, 0]:
    a.set_ylabel("ymax / D")
for a in ax[-1, :]:
    a.set_xlabel("freq (Hz)")

fig.tight_layout()
plt.show()
