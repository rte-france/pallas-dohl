"""Impedance of a Stockbridge damper against frequency.

`dohl.stockbridge.impedance` (Politecnico di Milano model) for a Z5 damper:
modulus and phase.
"""

import matplotlib.pyplot as plt
import numpy as np

from dohl.stockbridge import impedance

# Z5 damper: offsets (m), moments of inertia (kg.m**2), messenger lengths (m),
# end masses (kg), centre mass (kg), messenger density (kg/m**3), section
# (m**2), bending stiffness (N.m**2) and damping coefficient
z5 = dict(
    e1=0.02,
    j1=1.815e-03,
    l1=0.157,
    m1=0.8633,
    e2=0.01,
    j2=9.614e-04,
    l2=0.113,
    m2=0.7652,
    mm=0.3532,
    rho=7800.0,
    a=9.85e-05,
    ej=4.903,
    c=1.962,
)

f = np.linspace(0.0, 50, 501)[1:]
z = impedance(**z5, W=2.0 * np.pi * f)

fig, ax = plt.subplots(nrows=2, sharex=True)
ax[0].plot(f, np.abs(z))
ax[0].set_ylabel("Modulus (N.s/m)")
ax[1].plot(f, np.angle(z, deg=True))
ax[1].set_ylabel("Phase (deg)")
ax[1].set_xlabel("Freq (Hz)")
for a in ax:
    a.grid(True)
plt.show()
