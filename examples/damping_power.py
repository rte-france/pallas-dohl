import matplotlib.pyplot as plt
import numpy as np

from dohl import ebp


def _aster570_data():
    """ "Conductor data for aster570."""
    m = 1.571
    D = 3.105e-02
    rts = 1.853e05
    EImin = 2.828e01
    EImax = 2.155e03
    alpha = 0.140652249164
    young = 6.8e10
    d = 3.45e-03
    ml = False
    return m, D, rts, EImin, EImax, alpha, young, d, ml


if __name__ == "__main__":
    # get conductor data
    m, d, rts, EImin, EImax, alpha, _, _, _ = _aster570_data()

    # variation of ymax, frequency and mechanical tension
    y = np.linspace(0.0, 1.0, 1001)[1:] * d
    f = np.linspace(0.0, 100, 1001)[1:]
    t = np.linspace(0.05, 0.30, 1001) * rts
    x0 = alpha * t / rts
    conductor = ebp.Conductor(d=d, mu=m, rts=rts)

    # get list of damping power methods, remove olla (slow), use olla-approx only instead
    mth = ebp._damping_power_methods()
    _ = mth.pop(6)

    # loop on methods, plots
    fig, ax = plt.subplots(nrows=1, ncols=3)
    ix = 499
    for mt in ["foti", "olla_approx", "pdm"]:
        X = y
        bending = ebp.Bending(EImin=EImin, EImax=EImax, x0=x0[ix])
        Y = ebp.damping_power(X, f[ix], t[ix], conductor, bending=bending, method=mt)
        ax[0].loglog(X, Y, label=mt)

        X = f
        bending = ebp.Bending(EImin=EImin, EImax=EImax, x0=x0[ix])
        Y = ebp.damping_power(y[ix], X, t[ix], conductor, bending=bending, method=mt)
        ax[1].loglog(X, Y, label=mt)

        X = t
        bending = ebp.Bending(EImin=EImin, EImax=EImax, x0=x0)
        Y = ebp.damping_power(y[ix], f[ix], X, conductor, bending=bending, method=mt)
        ax[2].loglog(X, Y, label=mt)

    ax[0].set_xlabel("ymax (m)")
    ax[1].set_xlabel("freq (Hz)")
    ax[2].set_xlabel("tension (N)")
    ax[0].set_ylabel("dissipated power (W.m**-1)")
    for i in range(3):
        ax[i].grid(True)
        ax[i].legend()

    plt.show()
