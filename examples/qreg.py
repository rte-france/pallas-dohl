"""Number of cycles to failure by quantile regression, against the fatigue tests.

`dohl.damage.num_cycles_qreg_said` evaluated on the fatigue tests it is
fitted on, for the 0.5, 0.9 and 0.99 quantiles and their complements.
"""

import matplotlib.pyplot as plt

from dohl import damage
from dohl.config import _FatigueData

df = _FatigueData.qreg_input()
x = df[["ten_rat", "yad_log", "ybga", "ka"]].values
y = df["logN"].values

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12.8, 4.8), sharey=True)
for a, q in zip(ax, [0.5, 0.9, 0.99], strict=True):
    yp1, yp2 = damage.num_cycles_qreg_said(x[:, 0], x[:, 1], x[:, 2], x[:, 3], q=q, log=True)
    a.scatter(y, yp1, label=f"QReg ({q:.3f})", marker=".")
    a.scatter(y, yp2, label=f"QReg ({1.0 - q:.3f})", marker=".")
    a.plot([y.min(), y.max()], [y.min(), y.max()], color="grey", linestyle="--", label="y=x")
    a.set(xlabel="experimental log(N)", ylabel="predicted log(N)", ylim=[5.0, 9.0])
    a.legend()
    a.grid(True)
fig.tight_layout()
plt.show()
