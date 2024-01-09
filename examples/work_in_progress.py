# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import os.path
import time

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dohl import damage as dmg
from dohl import ebp
from dohl.lifespan import damage_year, damage_year_weibull, damage_year_weibull_array
from dohl.stockbridge import impedance, power_foti, power_wolf
from scipy.stats import kendalltau

_datadir = 'examples/data'


def _aster570_data():
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


def _stockbridge_data(id='Z5'):
    if id == 'sdm':
        e1 = 0.0163
        j1 = 0.00841
        l1 = 0.15
        m1 = 2.65
        e2 = 0.0222
        j2 = 0.0165
        l2 = 0.197
        m2 = 3.115
        mm = 0.835
        rho = 7800.
        a = 0.0001327
        ej = 7.7945
        c = 1.7983
    elif id == 'Z5':
        e1 = 0.02
        j1 = 1.815E-03
        l1 = 0.157
        m1 = 0.8633
        e2 = 0.01
        j2 = 9.614E-04
        l2 = 0.113
        m2 = 0.7652
        mm = 0.3532
        rho = 7800
        a = 9.85E-05
        ej = 4.903
        c = 1.962
    else:
        return _stockbridge_data()

    return e1, j1, l1, m1, e2, j2, l2, m2, mm, rho, a, ej, c


def test_wind_power():
    img = plt.imread(os.path.join(_datadir, 'wind_power_input_brika_1995.png'))

    fig, ax = plt.subplots(nrows=2, ncols=2)
    fig.suptitle('Maximum wind power input coefficients')

    for i in range(2):
        for j in range(2):
            ax[i, j].set_xscale('log')
            ax[i, j].set_yscale('log')
            ax[i, j].set_xlim([0.01, 1])
            ax[i, j].set_ylim([0.01, 100])
            ax[i, j].set_xlabel('Relative amplitude, ($A/D$)')
            ax[i, j].set_ylabel('Power ($W.m^{-1}.Hz^{-3}.m^{-4})$')
            ax[i, j].grid(True)

    ax[0, 0].set_title('[Brika & Laneville 1995]')
    ax[0, 0].imshow(img, aspect='auto', origin="upper", transform=ax[0, 0].transAxes, extent=[0, 1, 0, 1])
    ax[1, 0].imshow(img, aspect='auto', origin="upper", transform=ax[1, 0].transAxes, extent=[0, 1, 0, 1])

    ax[0, 1].set_title('EPRI 2009 data (tables A2.1-1 and A2.1-2)')
    dat = ebp._EpriData.wind_power()
    for k, key in enumerate(dat):
        elm = dat[key]
        for j in range(2):
            ax[0, j].plot(elm['ampl'], elm['power'], 'o', c='C%d' % (k,), label=elm['label'])
        x = np.linspace(elm['range'][0], elm['range'][1], 101)
        y = ebp.wind_power_adim(x, method=key)
        ax[0, 1].plot(x, y, '--', c='C%d' % (k,))
        ax[0, 1].legend()
        ax[1, 1].plot(x, y, ':', c='C%d' % (k,), label=elm['label'])

    x = np.linspace(0.01, 100, 1001)
    ax[1, 0].plot(x, ebp.wind_power_adim_olla(x), c='red', lw=2, label='OLLA')
    ax[1, 0].plot(x, ebp.wind_power_adim_foti(x), c='blue', lw=2, label='Foti')
    ax[1, 0].legend()

    ax[1, 1].plot(x, ebp.wind_power_adim_olla(x), c='red', lw=2, label='OLLA')
    ax[1, 1].plot(x, ebp.wind_power_adim_foti(x), c='blue', lw=2, label='Foti')
    ax[1, 1].legend()

    return


def test_damping_power():
    m, d, rts, EImin, EImax, alpha, _, _, _ = _aster570_data()
    y = np.linspace(0., 1., 1001)[1:] * d
    f = np.linspace(0., 100, 1001)[1:]
    t = np.linspace(0.05, 0.30, 1001) * rts
    x0 = alpha * t / rts
    dc = dict(d=d, mu=m, rts=rts, EImin=EImin, EImax=EImax)

    mth = ebp._damping_power_methods()
    _ = mth.pop(6)

    fig, ax = plt.subplots(nrows=1, ncols=3)
    ix = 499

    for mt in mth:
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
    return


def test_ymax_methods():
    m, d, rts, EImin, EImax, alpha, _, _, _ = _aster570_data()
    f = np.linspace(0., 50, 51)[1:]

    fig, ax = plt.subplots(nrows=5, ncols=10)
    for k, r in enumerate([0.05, 0.15, 0.25]):
        t = r * rts
        x0 = r * alpha
        for i, wm in enumerate(ebp._balance_wind_methods()):
            for j, dm in enumerate(ebp._balance_damping_methods()):
                y = ebp.ymax(f, t, wind_method=wm, damping_method=dm, d=d, mu=m, rts=rts, EImin=EImin, EImax=EImax,
                             x0=x0)
                if k == 0:
                    ax[i, j].plot(f, y / d, '-', label=wm + '/' + dm)
                else:
                    ax[i, j].plot(f, y / d, '-')
                ax[i, j].grid(True)
                ax[i, j].legend()
                ax[i, j].set_ylim([0, 1])
        for k in range(5):
            ax[k, 0].set_ylabel('y/d')
        for i in range(10):
            ax[-1, k].set_ylabel('freq')

    return


def test_ymax_methods_2():
    m, D, rts, EImin, EImax, alpha, _, _, _ = _aster570_data()

    # in balance, only 'cigre' and 'foti' methods gives realistic results with all damping methods; ymax from 'cigre'
    # and 'foti' are identical, hence we chose 'cigre' as the unique wind method

    N = 999
    freqs = np.random.uniform(1, 50, N)
    tensions = np.random.uniform(0.05, 0.25, N) * rts
    damp = ebp._balance_damping_methods()

    x = np.zeros((N, 2 + len(damp)))
    df = pd.DataFrame(columns=['frequency', 'tension'] + damp, data=x)
    del (x)

    for i in range(N):
        f = freqs[i]
        h = tensions[i]
        x0 = h * alpha / rts
        df.loc[i, 'frequency'] = f
        df.loc[i, 'tension'] = h
        for j, dm in enumerate(damp):
            ymax = ebp.ymax(f, h, wind_method='cigre', damping_method=dm, d=D, mu=m, rts=rts, EImin=EImin,
                            EImax=EImax, x0=x0)
            df.loc[i, dm] = ymax

    # plot all methods
    plt.figure()
    for i, mth in enumerate(damp):
        plt.scatter(df['frequency'], df[mth] / D, marker='o', alpha=0.8, c='C%1d' % (i,), label=mth)
    plt.grid(True)
    plt.legend()
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Ymax / d')
    plt.title('EPB with various damping methods [selection]')

    # show that all methods have similar ranking
    kt = np.zeros((len(damp), len(damp)))
    for i in range(len(damp)):
        for j in range(len(damp)):
            kt[i, j] = kendalltau(df[damp[i]].values, df[damp[j]].values).correlation

    fig, ax = plt.subplots()
    q = ax.pcolormesh(kt, cmap=cm.PuBu)
    fig.colorbar(q)
    ax.set_xticks(np.linspace(0, 9, 10) + 0.5)
    ax.set_yticks(np.linspace(0, 9, 10) + 0.5)
    ax.set_xticklabels(damp, rotation=45)
    ax.set_yticklabels(damp)
    ax.set_title('Kendall Tau for damping methods')

    # plot a selection of methods from high to low
    plt.figure()
    for i, mth in enumerate(['claren', 'foti', 'olla', 'tompkins']):
        plt.scatter(df['frequency'], df[mth] / D, marker='.', alpha=0.8, c='C%1d' % (i,), label=mth)
    plt.grid(True)
    plt.legend()
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Ymax / d')
    plt.title('EPB with various damping methods')

    return


def test_bending_amplitude():
    m, d, rts, EImin, EImax, alpha, _, _, _ = _aster570_data()
    x = np.linspace(0, 0.2, 21)
    freqs = np.array([1, 2, 5, 10, 20, 50])
    tensions = np.array([0.05, 0.15, 0.25]) * rts

    fig, ax = plt.subplots(nrows=len(tensions), ncols=len(freqs))
    for i, h in enumerate(tensions):
        for j, f in enumerate(freqs):
            a1 = dmg._ampl_cigre(d, f, h, m, EImin, x)
            a2 = dmg._ampl_cigre(d, f, h, m, 0.5 * (EImin + EImax), x)
            a3 = dmg._ampl_cigre(d, f, h, m, EImax, x)
            b1 = dmg._ampl_pbs(d, f, h, m, EImin, x)
            b2 = dmg._ampl_pbs(d, f, h, m, 0.5 * (EImin + EImax), x)
            b3 = dmg._ampl_pbs(d, f, h, m, EImax, x)
            ax[i, j].plot(x, a1, c='C0', marker='s', label='cigre (EImin)')
            ax[i, j].plot(x, a2, c='C0', marker=None, label='cigre')
            ax[i, j].plot(x, a3, c='C0', marker='o', label='cigre (EImax)')
            ax[i, j].plot(x, b1, c='C1', marker='s', label='PBS (EImin)')
            ax[i, j].plot(x, b2, c='C1', marker=None, label='PBS')
            ax[i, j].plot(x, b3, c='C1', marker='o', label='PBS (EImax)')
            ax[i, j].grid(True)
            ax[i, j].legend()
            ax[i, j].set_title('f=%.1f Hz & H=%.1f kN (%.0f %% RTS)'
                               % (f, h / 1000., 100. * h / rts))

    return


def test_fictive_stress():
    m, D, rts, EImin, EImax, alpha, young, d, _ = _aster570_data()
    EI = np.linspace(EImin, EImax, 3)

    ymax = np.linspace(0, D, 11)
    frqs = np.array([1, 2, 5, 10, 20, 50])
    tens = np.array([0.05, 0.15, 0.25]) * rts

    fig, ax = plt.subplots(nrows=len(tens), ncols=len(frqs))
    for i, h in enumerate(tens):
        for j, f in enumerate(frqs):
            for k, mt in enumerate(dmg._fictive_stress_methods()):
                for l, ei in enumerate(EI):
                    s = dmg.fictive_stress(ymax, f, h, d, m, young, ei, method=mt)
                    ax[i, j].semilogy(ymax / D, s * 1.0E-06, marker=['s', 'o', 'd'][k], c='C%d' % (l,))
            ax[i, j].grid(True)
            ax[i, j].set_title('f=%.1f Hz & H=%.1f kN (%.0f %% RTS)' % (f, h / 1000., 100. * h / rts))

    for i, h in enumerate(tens):
        ax[i, 0].set_ylabel('$\sigma$ (MPa)')
    for j, f in enumerate(frqs):
        ax[-1, j].set_xlabel('ymax / D')

    for k, mt in enumerate(dmg._fictive_stress_methods()):
        ax[0, 0].plot(np.nan, np.nan, marker=['s', 'o', 'd'][k], c='gray', label=mt)
    for l, ei in enumerate(EI):
        ax[0, 0].plot(np.nan, np.nan, c='C%d' % (l,), label=['EImin', 'EIavg', 'EImax'][l])
    ax[0, 0].legend()

    return


def test_sbd():
    im1 = plt.imread(os.path.join(_datadir, 'fatigue_test_1_layer.png'))
    im2 = plt.imread(os.path.join(_datadir, 'fatigue_test_2_layers.png'))
    im3 = plt.imread(os.path.join(_datadir, 'fatigue_test_3_layers.png'))
    img = [im1, im2, im3]

    N = np.logspace(5, 9, 41)

    fig, ax = plt.subplots(nrows=1, ncols=3)
    plt.suptitle('EPRI 2009 data (figures 3.2-23 to 3.2-25)')
    for i in range(3):
        ax[i].set_xscale('log')
        ax[i].set_xlim([1E+05, 1E+09])
        ax[i].set_xlabel('Num cycles at first wire break')
        ax[i].set_ylabel('Fictive stress (MPa)')
        ax[i].set_title('Fatigue test of %s-layer ACSR' % (['one', 'two', 'three'][i],))
        ax[i].grid(True)
    ax[0].set_ylim([0, 80])
    ax[1].set_ylim([0, 45])
    ax[2].set_ylim([0, 40])
    for i in range(3):
        ax[i].imshow(img[i], aspect='auto', origin="upper", transform=ax[i].transAxes, extent=[0, 1, 0, 1])
        ax[i].semilogx(N, dmg.safe_border_line(N, False), label='single layer acsr')
        ax[i].semilogx(N, dmg.safe_border_line(N, True), label='multi layer acsr')
        ax[i].legend()

    # --

    N = np.logspace(0, 16, 1601)
    s = np.logspace(-2, +3, 1501)

    plt.figure()
    plt.loglog(N, dmg.safe_border_line(N, False), label='single layer acsr')
    plt.loglog(N, dmg.safe_border_line(N, True), label='multi layer acsr')
    plt.grid(True)
    plt.legend()
    plt.title('Safe border line')
    plt.xlabel('Number of cycles (N)')
    plt.ylabel('Fictive stress ($\sigma$, MPa)')

    plt.figure()
    plt.loglog(s, dmg.inv_safe_border_line(s, False), label='single layer acsr')
    plt.loglog(s, dmg.inv_safe_border_line(s, True), label='multi layer acsr')
    plt.grid(True)
    plt.legend()
    plt.title('Inverse Safe border line')
    plt.xlabel('Fictive stress ($\sigma$, MPa)')
    plt.ylabel('Number of cycles (N)')

    return


def test_dmg():
    mu, D, rts, EImin, EImax, alpha, young, d, ml = _aster570_data()

    u = np.linspace(0, 5, 21)[1:]
    r = np.array([0.05, 0.10, 0.15, 0.20, 0.30, 0.35])
    m = ['s', 'o', 'd']

    fig, ax = plt.subplots(nrows=4, ncols=len(r))
    for i, ratio in enumerate(r):
        for dm in ['olla_approx']:
            for j, fsm in enumerate(dmg._fictive_stress_methods()):
                for k, fsba in enumerate(['min', 'avg', 'max']):
                    h = ratio * rts
                    x0 = alpha * h / rts
                    ym, _, sg, ls = damage_year(u, h, D, d, mu, young, EImin, EImax,
                                                wind_method='cigre', damping_method=dm,
                                                fictive_stress_method=fsm, fsba=fsba,
                                                multilayer=ml, rts=rts, x0=x0)

                    ax[0, i].plot(u, ym / D, c='C%d' % (j,), lw=1)
                    ax[0, i].grid(True)
                    ax[0, i].set_title('With H=%.0f %% RTD' % (100. * ratio,))

                    ax[1, i].plot(u, sg * 1.0E-06, marker=m[k], c='C%d' % (j,), lw=1)
                    ax[1, i].grid(True)

                    ax[2, i].plot(u, sg / young * 1.0E+06, marker=m[k], c='C%d' % (j,), lw=1)
                    ax[2, i].grid(True)

                    ax[3, i].semilogy(u, ls, marker=m[k], c='C%d' % (j,), lw=1)
                    ax[3, i].grid(True)

        ax[-1, i].set_xlabel('Wind speed (m/s)')
        if i == 0:
            ax[0, i].set_ylabel('Vibration amplitude (ymax/D)')
            ax[1, i].set_ylabel('Fictive stress ($\sigma$, MPa)')
            ax[2, i].set_ylabel('Micro strain')
            ax[3, i].set_ylabel('Estimated damage per year')

    return


def test_dmg_weibull(n=9999):
    mu, D, rts, EImin, EImax, alpha, young, d, ml = _aster570_data()

    l = [mu, D, rts, EImin, EImax, alpha, young, d]
    for i in range(len(l)):
        l[i] *= np.minimum(np.maximum(1. + 0.1 * np.random.randn(n), 0.67), 1.33)
    mu, D, rts, EImin, EImax, alpha, young, d = tuple(l)

    shape, scale = 2.12, 4.41
    tension = np.random.uniform(0.05, 0.45, n) * rts
    x0 = alpha * tension

    d1 = np.zeros(n)
    t1 = time.time()
    for i in range(n):
        d1[i] = damage_year_weibull(shape, scale, tension[i], D[i], d[i], mu[i], young[i], EImin[i], EImax[i],
                                    multilayer=ml, strouhal=0.2, wind_method='cigre', damping_method='foti', iv=0.,
                                    fictive_stress_method='ba_pfs',
                                    fsba='avg', rts=rts[i], x0=x0[i])
    t1 = time.time() - t1
    t2 = time.time()
    d2 = damage_year_weibull_array(shape, scale, tension, D, d, mu, young, EImin, EImax, multilayer=ml, strouhal=0.2,
                                   wind_method='cigre', damping_method='foti', iv=0., fictive_stress_method='ba_pfs',
                                   fsba='avg', rts=rts, x0=x0)
    t2 = time.time() - t2

    print('test_dmg_weibull | Max err is %.2E, normal time %.1f s, array version is %.1f faster'
          % (np.max(np.abs(1 - d2 / d1)), t1, t1 / t2))

    return


def test_stockbridge_impedance(id='Z5'):
    e1, j1, l1, m1, e2, j2, l2, m2, mm, rho, a, ej, c = _stockbridge_data(id)

    f = np.linspace(0., 50, 501)[1:]
    w = 2. * np.pi * f
    z = impedance(e1, j1, l1, m1, e2, j2, l2, m2, mm, rho, a, ej, c, w)

    fig, ax = plt.subplots(nrows=2)
    ax[0].plot(f, np.abs(z))
    ax[0].set_xlabel('Freq (Hz)')
    ax[0].set_ylabel('Module')
    ax[0].grid(True)

    ax[1].plot(f, np.angle(z) * 180 / np.pi)
    ax[1].set_xlabel('Freq (Hz)')
    ax[1].set_ylabel('Phase (deg)')
    ax[1].grid(True)

    return


def test_stockbridge_power():
    # [WORK IN PROGRESS] test stockbridge power
    # sign problem depending on phase values :(

    mu, D, rts, _, _, _, _, _, _ = _aster570_data()
    L = 350.

    e1, j1, l1, m1, e2, j2, l2, m2, mm, rho, a, ej, c = _stockbridge_data(id='sdm')

    u = np.array([1., 3., 5.])
    f = u * 0.185 / D
    w = 2. * np.pi * f
    z = impedance(e1, j1, l1, m1, e2, j2, l2, m2, mm, rho, a, ej, c, w)

    ymax = D
    h = 0.15 * rts
    xs = np.linspace(0, 12, 601)
    dp = np.zeros((len(xs), len(f)))
    dq = np.zeros((len(xs), len(f)))
    for i in range(len(xs)):
        dp[i, :] = power_foti(ymax, f, h, L, mu, xs[i], z)
        dq[i, :] = np.abs(power_wolf(ymax, f, h, D, mu, xs[i], z))
    plt.figure()
    for j in range(len(f)):
        cj = 'C%1d' % (j,)
        plt.plot(xs, dp[:, j], c=cj, label='u=%.1f m/s' % (u[j],))
        plt.plot(xs, dq[:, j], c=cj, ls='--')
    plt.grid(True)
    plt.xlabel('Stockbridge position')
    plt.ylabel('Power dissipated')
    plt.legend()

    return


def test_qreg():
    from dohl.config import _FatigueData

    df = _FatigueData.qreg_input()
    inputs = ['ten_rat', 'yad_log', 'ybga', 'ka']
    output = ['logN']
    x = df[inputs].values
    y = df[output].values.ravel()

    fig, ax = plt.subplots(nrows=1, ncols=4)
    for i, q in enumerate([0.5, 0.9, 0.99]):
        yp1, yp2 = dmg.num_cycles_qreg_said(x[:, 0], x[:, 1], x[:, 2], x[:, 3], q=q, log=True)
        ax[i].scatter(y, yp1, label="QReg (%.3f)" % (q,), marker='.')
        ax[i].scatter(y, yp2, label="QReg (%.3f)" % (1. - q,), marker='.')
        ax[i].plot([y.min(), y.max()], [y.min(), y.max()], color='grey', linestyle='--', label="y=x")
        ax[i].set(xlabel="experimental log(N)", ylabel="predicted log(N)")
        ax[i].legend()
        ax[i].set_ylim([5., 9.])
        ax[i].grid(True)
    return


if __name__ == "__main__":

    import matplotlib

    matplotlib.use('TkAgg')
    plt.close('all')

    _b = False
    if _b:
        test_wind_power()
        test_damping_power()
        test_ymax_methods()
        test_ymax_methods_2()
        test_bending_amplitude()
        test_fictive_stress()
        test_sbd()
        test_dmg()
        test_dmg_weibull()
        test_stockbridge_impedance()
        test_stockbridge_power()
        test_qreg()
