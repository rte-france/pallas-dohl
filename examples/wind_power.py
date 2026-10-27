"""Wind input power: EPRI experimental fits, OLLA and Foti.

Top: the experimental points of the EPRI tables A2.1-1 and A2.1-2 drawn over
the figure of Brika and Laneville (1995), and the polynomial fits of
`dohl.ebp.wind_power_adim` over their validity range. Bottom: the OLLA and
Foti models against the same data.

Run from the repository root: the background figure is read from
examples/data.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from dohl import ebp
from dohl.config import _EpriData

img = plt.imread(os.path.join("examples", "data", "wind_power_input_brika_1995.png"))

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12.8, 9.6))
fig.suptitle("Maximum wind power input coefficients")
for a in ax.flat:
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlim([0.01, 1])
    a.set_ylim([0.01, 100])
    a.set_xlabel("Relative amplitude, ($A/D$)")
    a.set_ylabel("Power ($W.m^{-1}.Hz^{-3}.m^{-4})$")
    a.grid(True)
for a in ax[:, 0]:
    a.imshow(img, aspect="auto", origin="upper", transform=a.transAxes, extent=[0, 1, 0, 1])
ax[0, 0].set_title("[Brika & Laneville 1995]")
ax[0, 1].set_title("EPRI 2009 data (tables A2.1-1 and A2.1-2)")

# experimental points and their polynomial fits
dat = _EpriData.wind_power()
for k, key in enumerate(dat):
    elm = dat[key]
    for a in ax[0]:
        a.plot(elm["ampl"], elm["power"], "o", c=f"C{k}", label=elm["label"])
    x = np.linspace(elm["range"][0], elm["range"][1], 101)
    y = ebp.wind_power_adim(x, method=key)
    ax[0, 1].plot(x, y, "--", c=f"C{k}")
    ax[1, 1].plot(x, y, ":", c=f"C{k}", label=elm["label"])
ax[0, 1].legend()

# analytical models
x = np.linspace(0.01, 100, 1001)
for a in ax[1]:
    a.plot(x, ebp.wind_power_adim_olla(x), c="red", lw=2, label="OLLA")
    a.plot(x, ebp.wind_power_adim_foti(x), c="blue", lw=2, label="Foti")
    a.legend()

fig.tight_layout()
plt.show()
