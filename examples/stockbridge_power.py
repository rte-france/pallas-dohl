"""Power dissipated by a Stockbridge damper against its position (work in progress).

`dohl.stockbridge.power_foti` (solid) and `dohl.stockbridge.power_wolf`
(dashed) for an SDM damper on a 350 m Aster 570 span at 15 % of its rated
tensile strength, vibrating at one diameter, for three wind speeds. The sign
of the Wolf power depends on the impedance phase; its absolute value is
plotted.
"""

import matplotlib.pyplot as plt
import numpy as np

from dohl.stockbridge import impedance, power_foti, power_wolf

# Aster 570 conductor on a 350 m span
mu, D, rts = 1.571, 3.105e-02, 1.853e05
L = 350.0
tension = 0.15 * rts

# SDM damper (see examples/stockbridge_impedance.py for the parameters)
sdm = dict(
    e1=0.0163,
    j1=0.00841,
    l1=0.15,
    m1=2.65,
    e2=0.0222,
    j2=0.0165,
    l2=0.197,
    m2=3.115,
    mm=0.835,
    rho=7800.0,
    a=0.0001327,
    ej=7.7945,
    c=1.7983,
)

u = np.array([1.0, 3.0, 5.0])
f = u * 0.185 / D
z = impedance(**sdm, W=2.0 * np.pi * f)

ymax = D
xs = np.linspace(0, 12, 601)
p_foti = np.array([power_foti(ymax, f, tension, L, mu, x, z) for x in xs])
p_wolf = np.array([np.abs(power_wolf(ymax, f, tension, D, mu, x, z)) for x in xs])

for j in range(len(f)):
    plt.plot(xs, p_foti[:, j], c=f"C{j}", label=f"u={u[j]:.1f} m/s")
    plt.plot(xs, p_wolf[:, j], c=f"C{j}", ls="--")
plt.grid(True)
plt.xlabel("Stockbridge position (m)")
plt.ylabel("Power dissipated (W)")
plt.legend()
plt.show()
