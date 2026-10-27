"""Yearly damage of many conductors under a Weibull wind, scalar and array versions.

Conductor properties are drawn around the Aster 570 values (+/- 10 %, clipped
to +/- 33 %) and tensions between 5 and 45 % of the rated tensile strength.
`dohl.lifespan.damage_year_weibull` is called once per conductor, and
`dohl.lifespan.damage_year_weibull_array` once for all of them: both give the
same damage, the array version faster.
"""

import time

import matplotlib.pyplot as plt
import numpy as np

from dohl.lifespan import damage_year_weibull, damage_year_weibull_array

# Aster 570 conductor, perturbed
n = 999
rng = np.random.default_rng(0)
nominal = dict(
    mu=1.571,
    D=3.105e-02,
    rts=1.853e05,
    EImin=28.28,
    EImax=2155.0,
    alpha=0.140652249164,
    young=6.8e10,
    d=3.45e-03,
)
c = {k: v * np.clip(1.0 + 0.1 * rng.standard_normal(n), 0.67, 1.33) for k, v in nominal.items()}

shape, scale = 2.12, 4.41
tension = rng.uniform(0.05, 0.45, n) * c["rts"]
x0 = c["alpha"] * tension / c["rts"]
args = (tension, c["D"], c["d"], c["mu"], c["young"], c["EImin"], c["EImax"])
kwargs = dict(
    multilayer=False,
    wind_method="cigre",
    damping_method="foti",
    fictive_stress_method="ba_pfs",
    fsba="avg",
)

t1 = time.time()
d1 = np.array(
    [
        damage_year_weibull(
            shape, scale, *[a[i] for a in args], rts=c["rts"][i], x0=x0[i], **kwargs
        )
        for i in range(n)
    ]
)
t1 = time.time() - t1

t2 = time.time()
d2 = damage_year_weibull_array(shape, scale, *args, rts=c["rts"], x0=x0, **kwargs)
t2 = time.time() - t2

print(f"max relative difference {np.max(np.abs(1 - d2 / d1)):.2E}")
print(f"scalar loop {t1:.1f} s, array version {t1 / t2:.1f} times faster")

plt.scatter(100 * tension / c["rts"], 1.0 / d2, marker=".")
plt.yscale("log")
plt.xlabel("tension (% RTS)")
plt.ylabel("lifespan (years)")
plt.title(f"Weibull wind, shape {shape} and scale {scale} m/s")
plt.grid(True)
plt.show()
