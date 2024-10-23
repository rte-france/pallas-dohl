import matplotlib.pyplot as plt
import numpy as np
from dohl import ebp


def _aster570_data():
    """"Conductor data for aster570."""
    m = 1.571
    D = 3.105E-02
    rts = 1.853E+05
    EImin = 2.828E+01
    EImax = 2.155E+03
    alpha = 0.140652249164
    young = 6.8E+10
    d = 3.45E-03
    ml = False
    return m, D, rts, EImin, EImax, alpha, young, d, ml


if __name__ == '__main__':

    import matplotlib

    matplotlib.use('TkAgg')
    plt.close('all')

    # get conductor data
    m, d, rts, EImin, EImax, alpha, _, _, _ = _aster570_data()

    # variation of ymax, frequency and mechanical tension
    y = np.linspace(0., 1., 1001)[1:] * d
    f = np.linspace(0., 100, 1001)[1:]
    t = np.linspace(0.05, 0.30, 1001) * rts
    x0 = alpha * t / rts
    dc = dict(d=d, mu=m, rts=rts, EImin=EImin, EImax=EImax)

    # get list of damping power methods, remove olla (slow), use olla-approx only instead
    mth = ebp._damping_power_methods()
    _ = mth.pop(6)

    # loop on methods, plots
    fig, ax = plt.subplots(nrows=1, ncols=3)
    ix = 499
    for mt in ['foti', 'olla_approx', 'pdm']:
        X = y
        Y = ebp.damping_power(X, f[ix], t[ix], method=mt, x0=x0[ix], **dc)
        ax[0].loglog(X, Y, label=mt)

        X = f
        Y = ebp.damping_power(y[ix], X, t[ix], method=mt, x0=x0[ix], **dc)
        ax[1].loglog(X, Y, label=mt)

        X = t
        Y = ebp.damping_power(y[ix], f[ix], X, method=mt, x0=x0, **dc)
        ax[2].loglog(X, Y, label=mt)

    ax[0].set_xlabel('ymax (m)')
    ax[1].set_xlabel('freq (Hz)')
    ax[2].set_xlabel('tension (N)')
    ax[0].set_ylabel('dissipated power (W.m**-1)')
    for i in range(3):
        ax[i].grid(True)
        ax[i].legend()

    plt.show()
