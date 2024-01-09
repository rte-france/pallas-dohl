# !/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from dohl.config import _EpriData
from dohl.stockbridge import power
from scipy.optimize import bisect


def _wind_power_methods():
    return ['bate', 'brika', 'carroll', 'cigre', 'diana', 'farquharson', 'foti', 'olla', 'pon',
            'rawlins58', 'rawlins83']


def _damping_power_methods():
    return ['claren', 'foti', 'kraus', 'mocks', 'noiseux1991', 'noiseux1992', 'olla', 'olla_approx',
            'pdm', 'seppa', 'tompkins']


def wind_power_adim_foti(yad: np.ndarray, iv: float = 0.) -> np.ndarray:
    """Estimate the wind input power.

    Foti & Martinelli (2018) *An enhanced unified model for the self-damping
    of stranded cables under aeolian vibrations* -- eqs (4) & (5).

    NB: with iv=0, it is the same formulation as CIGRE's one

    Parameters
    ----------
    yad : float or numpy.ndarray
        Antinode vibration amplitude divided with cable diameter (no unit).
    iv : float
        Turbulence intensity of the wind (no unit).

    Returns
    -------
    float or numpy.array
        Array of wind input power (in W.m**-5.f**-3).

    """
    il = 0.09
    bw = 1. / np.sqrt(1. + (iv / il)**2)
    return bw * np.polyval([-99.73, 101.62, 0.1627, 0.2256], yad)


def wind_power_adim_olla(yad: np.ndarray) -> np.ndarray:
    """Estimate the wind input power.

    Fit from OLLA research program.

    Parameters
    ----------
    yad : float or numpy.ndarray
        Antinode vibration amplitude divided with cable diameter (no unit).

    Returns
    -------
    float or numpy.array
        Array of wind input power (in W.m**-5.f**-3).

    """
    x = yad
    return ((0.669 * x + 12.843 * x**2 - 13.223 * x**3) / (
            0.168 - 0.097 * x + 0.119 * x**2 - 0.057 * x**3))


def wind_power_adim(yad: np.ndarray, iv: float = 0, method: str = 'cigre',
                    extend: bool = True) -> np.ndarray:
    """Estimate the wind input power.

    Parameters
    ----------
    yad : float or numpy.ndarray
        Antinode vibration amplitude divided with cable diameter (no unit).
    iv : float
        Turbulence intensity of the wind (no unit). Only used if method='foti'.
        Default is 0.
    method: str
        Choice of method to get wind power. Default is 'cigre'.
    extend : bool
        If false, put nans if yad is out of method bounds
    Returns
    -------
    float or numpy.ndarray
        Array of wind input power (in W.m**-5.f**-3).

    """
    if method not in _wind_power_methods():
        raise ValueError('Unknown wind power method')

    if method == 'cigre':
        return wind_power_adim_foti(yad, iv=0.)
    elif method == 'foti':
        return wind_power_adim_foti(yad, iv=iv)
    elif method == 'olla':
        return wind_power_adim_olla(yad)
    else:
        dat = _EpriData.wind_power()[method]
        wp = yad * np.polyval(dat['poly'][::-1], yad)
        if not extend:
            np.where(yad < dat['range'][0], np.nan, yad)
            np.where(yad > dat['range'][1], np.nan, yad)
        return wp


def wind_power(y: np.ndarray, f: np.ndarray, d: np.ndarray, iv: float = 0,
               method: str = 'cigre') -> np.ndarray:
    """Estimate the wind input power.

    Parameters
    ----------
    y : float or numpy.ndarray
        Antinode vibration (in m).
    f : float or numpy.ndarray
        Vibration frequency (in Hz).
    d : float or numpy.ndarray
        Cable diameter (in m).
    iv : float
        Turbulence intensity of the wind (no unit). Only used if method='foti'.
        Default is 0.
    method: str
        Choice of method to get wind power. Default is 'cigre'.

    Returns
    -------
    float or numpy.ndarray
        Array of wind input power (in W.m**-1).

    """
    return f**3 * d**4 * wind_power_adim(y / d, iv=iv, method=method)


def _k_lilien(d, mu, rts):
    return (d * 1000.) / np.sqrt(mu * rts / 1000.)


def _damping_power_coefficients(method):
    dat = _EpriData.damping_power()[method]
    res = []
    for k in ['l', 'm', 'n']:
        if type(dat[k]) is list:
            res.append(np.mean(dat[k]))
        else:
            res.append(dat[k])
    return res[0], res[1], res[2]


def damping_power_foti(ymax: np.ndarray, freq: np.ndarray, tension: np.ndarray, mu: np.ndarray,
                       EImax: np.ndarray, x0: np.ndarray, mode: str = 'full') -> np.ndarray:
    """Evaluate the power dissipated by self-damping.

    Foti & Martinelli (2018) *An enhanced unified model for the self-damping
    of stranded cables under aeolian vibrations*.

    Parameters
    ----------
    ymax : float or numpy.ndarray
        Antinode vibration amplitude (m).
    freq : float or numpy.ndarray
        Frequencies relative to vibration (Hz).
    tension : float or numpy.ndarray
        Mechanical tension (N).
    mu : float or numpy.ndarray
        Lineic mass of the cable (kg.m**-1).
    EImax: float or numpy.ndarray
        Maximal tangential bending stiffness of the cable (N.m**2).
    x0 : float or numpy.ndarray
        Limit curvature (m**-1).
    mode : str
        Formulation used for the dissipation. The value must be in
        ['full', 'gross', 'micro']. Default is 'full'.

    Returns
    -------
    float or numpy.ndarray
        Array of power dissipated by damping (in W.m**-1).

    """

    def G1(kn):
        # eq (29)
        return 1. / 3. - 0.375 * np.cos(2. * np.pi * kn) + 1. / 24. * np.cos(6. * np.pi * kn)

    def G2(kn):
        # eq (30)
        return np.pi - 4. * np.pi * kn + np.sin(4. * np.pi * kn)

    # eq (27)
    Pms = 128. * np.pi**5 * mu**3 * EImax / x0 * ymax**3 * freq**7 / tension**3
    # eq (28)
    Pgs = 4. * np.pi**3 * mu**2 * EImax * ymax**2 * freq**5 / tension**2
    if mode == 'full':
        # eq (23)
        lbd = 1. / freq * np.sqrt(tension / mu)
        # eq (25)
        rsin = lbd**2 * x0 / (4 * np.pi**2 * ymax)
        rsin = np.where(rsin > 1, 1, rsin)
        rsin = np.where(rsin < 0, 0, rsin)
        kn = np.arcsin(rsin) / (2. * np.pi)
        return G1(kn) * Pms + G2(kn) * Pgs
    elif mode == 'gross':
        return G2(0.) * Pgs
    elif mode == 'micro':
        return G1(0.25) * Pms
    else:
        raise ValueError("'%s' unknown" % mode)


def damping_power_cieren(ymax: np.ndarray, freq: np.ndarray, tension: np.ndarray, mu: float,
                         EImin: np.ndarray, EImax: np.ndarray, x0: np.ndarray, approx: bool = False,
                         N: int = 999) -> np.ndarray:
    """Evaluate the power dissipated by self-damping.

    Cieren (2020) *AAA*.

    Using numpy.ndarrays is only possible with approx=True

    Parameters
    ----------
    ymax : float or numpy.ndarray
        Antinode vibration amplitude (m).
    freq : float or numpy.ndarray
        Frequencies relative to vibration (Hz).
    tension : float or numpy.ndarray
        Mechanical tension (N).
    mu : float
        Lineic mass of the cable (kg.m**-1).
    EImin: float or numpy.ndarray
        Minimal tangential bending stiffness of the cable (N.m**2).
    EImax: float or numpy.ndarray
        Maximal tangential bending stiffness of the cable (N.m**2).
    x0 : float or numpy.ndarray
        Limit curvature (m**-1).
    approx: bool
        Use (or not) an approximation for small curvatures. Default is False.
    N : int
        Number of discretization points in integral calculation. Only used when
        approx is False. Default is 999.

    Returns
    -------
    float or numpy.ndarray
        Array of power dissipated by damping.

    """
    lm = np.sqrt(tension / mu) / freq
    if approx:
        # valid if ymax < (lm/(2*np.pi))**2 * x0/2
        return freq * 8 * EImax / (9 * np.pi * x0) * (4 * np.pi**2 * ymax / lm**2)**3

    def eds(xz, cm, cM, x):
        ep = cm / cM
        xb = (1. - ep) * xz
        y = x / xb
        return cM * xb**2 * (
                (8 * (1 + ep) + 4 * (1 + 2 * ep) * y + 4 * ep * y**2) * np.exp(-y) - 8 * (
                1 + ep) + 4 * y)

    x = np.linspace(0, 0.5, N) * lm
    c = 4. * np.pi**2 * ymax / lm**2 * np.sin(2. * np.pi * x / lm)
    e = eds(x0, EImin, EImax, c)
    return freq * 2. * np.sum(0.5 * (e[1:] + e[:-1]) * np.diff(x)) / lm


def damping_power_noiseux1992(ymax: np.ndarray, freq: np.ndarray, tension: np.ndarray,
                              d: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Evaluate the power dissipated by self-damping.

    Noiseux (1992) *AAA*

    Parameters
    ----------
    ymax : float or numpy.ndarray
        Antinode vibration amplitude (m).
    freq : float or numpy.ndarray
        Frequencies relative to vibration (Hz).
    tension : float or numpy.ndarray
        Mechanical tension (N).
    d : float or numpy.ndarray
        Cable external diameter (m)
    mu : float or numpy.ndarray
        Lineic mass of the cable (kg.m**-1).

    Returns
    -------
    float or numpy.ndarray
        Array of power dissipated by damping (in W.m**-1).

    """
    lm = np.sqrt(tension / mu) / freq
    return d**4 / lm**5.52 * freq**0.11 * ymax**2.44


def damping_power_klmn(ymax: np.ndarray, freq: np.ndarray, tension: np.ndarray, k: np.ndarray,
                       l: np.ndarray, m: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Evaluate the power dissipated by self-damping.

    Empiric formula found in various experimental studies

    Parameters
    ----------
    ymax : float or numpy.ndarray
        Antinode vibration amplitude (m).
    freq : float or numpy.ndarray
        Frequencies relative to vibration (Hz).
    tension : float or numpy.ndarray
        Mechanical tension (N).
    k : float or numpy.ndarray
        Coefficient
    l : float or numpy.ndarray
        Exponent for ymax
    m : float or numpy.ndarray
        Exponent for freq
    n : float or numpy.ndarray
        Exponent for 1/tension

    Returns
    -------
    float or numpy.ndarray
        Array of power dissipated by damping (in W.m**-1).

    """
    return k * ymax**l * freq**m / (tension / 1000.)**n


def damping_power(ymax: np.ndarray, freq: np.ndarray, tension: np.ndarray, method: str = '',
                  k: np.ndarray = None, l: np.ndarray = None, m: np.ndarray = None,
                  n: np.ndarray = None, d: np.ndarray = None, mu: np.ndarray = None,
                  rts: np.ndarray = None, EImin: np.ndarray = None, EImax: np.ndarray = None,
                  x0: np.ndarray = None) -> np.ndarray:
    """Evaluate the power dissipated by self-damping.

    If method='olla', all inputs must be floats (no numpy.ndarray). Not all args
    must be specified (see below).

    'claren', 'foti', 'kraus', 'mocks', 'noiseux1991', 'noiseux1992',
            'olla', 'olla_approx', 'pdm', 'seppa', 'tompkins'

    If method is in ['claren', 'kraus', 'mocks', 'noiseux1991', 'pdm', 'seppa',
    'tompkins'], either arg k or the triplet d, mu and rts must be specified.

    If method is 'foti', args mu, EImax and x0 must be specified.

    If method is 'noiseux1992', args d and mu must be specified.

    If method is 'olla', args mu, EImin, EImax and x0 must be specified.

    If method is 'olla_approx', args mu, EImax and x0 must be specified.

    If method is not specified, either arg k or the triplet d, mu and rts must
    be specified, along with l, m, n exponents.

    Parameters
    ----------
    ymax : float or numpy.ndarray
        Antinode vibration amplitude (m).
    freq : float or numpy.ndarray
        Frequencies relative to vibration (Hz).
    tension : float or numpy.ndarray
        Mechanical tension (N).
    method : str
        Choice of method to get damping power.
    k : float or numpy.ndarray
        Coefficient.
    l : float or numpy.ndarray
        Exponent for ymax
    m : float or numpy.ndarray
        Exponent for freq
    n : float or numpy.ndarray
        Exponent for 1/tension
    d : float or numpy.ndarray
        Cable external diameter (m)
    mu : float or numpy.ndarray
        Lineic mass of the cable (kg.m**-1).
    rts : float or numpy.ndarray
        Rated tensile strength of the cable (N)
    EImin: float or numpy.ndarray
        Minimal tangential bending stiffness of the cable (N.m**2).
    EImax: float or numpy.ndarray
        Maximal tangential bending stiffness of the cable (N.m**2).
    x0 : float or numpy.ndarray
        Limit curvature (m**-1).

    Returns
    -------
    float or numpy.ndarray
        Array of power dissipated by damping.

    """
    if method == 'foti':
        if mu is None or EImax is None or x0 is None:
            raise ValueError('Args mu, EImax and x0 must be specified when using foti method')
        return damping_power_foti(ymax, freq, tension, mu, EImax, x0)
    elif method == 'noiseux1992':
        if d is None or mu is None or rts is None:
            raise ValueError('Args d and mu must be specified when using noiseux1992 method')
        return damping_power_noiseux1992(ymax, freq, tension, d, mu)
    elif method == 'olla':
        if mu is None or EImin is None or EImax is None or x0 is None:
            raise ValueError(
                'Args mu, EImin, EImax and x0 must be specified when using olla method')
        return damping_power_cieren(ymax, freq, tension, mu, EImin, EImax, x0, approx=False)
    elif method == 'olla_approx':
        if mu is None or EImax is None or x0 is None:
            raise ValueError('Args mu, EImax and x0 must be specified when using olla method')
        return damping_power_cieren(ymax, freq, tension, mu, EImin, EImax, x0, approx=True)

    if l is None or m is None or n is None:
        if method not in _damping_power_methods():
            raise ValueError('Unknown damping power method')
        l, m, n = _damping_power_coefficients(method)

    if k is None:
        if d is None or mu is None or rts is None:
            raise ValueError('Either k or (d, mu and rts) must be specified')
        else:
            k = _k_lilien(d, mu, rts)

    return damping_power_klmn(ymax, freq, tension, k, l, m, n)


def _vect_bisection(fun, a, b, tol, maxiter):
    """Hand-made bisection method (vector mode)."""
    e = np.abs(b - a)
    c = 1
    while np.nanmax(e) > tol and c <= maxiter:
        x = 0.5 * (a + b)
        y = fun(x)
        i = y > 0
        a[i] = x[i]
        b[~i] = x[~i]
        e = np.abs(b - a)
        c = c + 1
    x = 0.5 * (a + b)
    return x, e


def _balance_wind_methods():
    return ['cigre', 'foti', 'olla', 'pon', 'rawlins83']


def _balance_damping_methods():
    return ['claren', 'foti', 'kraus', 'mocks', 'noiseux1991', 'olla', 'olla_approx', 'pdm', 'seppa', 'tompkins']


def ymax(freq: np.ndarray, tension: np.ndarray, d: np.ndarray, wind_method: str = 'cigre',
         damping_method: str = 'foti', iv: float = 0., k: np.ndarray = None, l: np.ndarray = None,
         m: np.ndarray = None, n: np.ndarray = None, mu: np.ndarray = None, rts: np.ndarray = None,
         EImin: np.ndarray = None, EImax: np.ndarray = None, x0: np.ndarray = None,
         stockbridge_cfg: dict = {}, tol: float = 1.0E-06, maxiter: int = 64) -> np.ndarray:
    """Computes vibration amplitude by balancing wind power and damping terms.

    Parameters
    ----------
    freq : float or numpy.ndarray
        Frequencies relative to vibration (Hz).
    tension : float or numpy.ndarray
        Mechanical tension (N).
    wind_method: str
        Choice of method to get wind power. Default is 'cigre'.
    damping_method : str
        Choice of method to get damping power. Default is 'foti'
    iv : float
        Turbulence intensity of the wind (no unit). Only used if method='foti'.
        Default is 0.
    k : float or numpy.ndarray
        Coefficient.
    l : float or numpy.ndarray
        Exponent for ymax
    m : float or numpy.ndarray
        Exponent for freq
    n : float or numpy.ndarray
        Exponent for 1/tension
    d : float or numpy.ndarray
        Cable external diameter (m)
    mu : float or numpy.ndarray
        Lineic mass of the cable (kg.m**-1).
    rts : float or numpy.ndarray
        Rated tensile strength of the cable (N)
    EImin : float or numpy.ndarray
        Minimal tangential bending stiffness of the cable (N.m**2).
    EImax : float or numpy.ndarray
        Maximal tangential bending stiffness of the cable (N.m**2).
    x0 : float or numpy.ndarray
        Limit curvature (m**-1).
    stockbridge_cfg : dict
        Stockbridge configuration. The default is {} (empty dict ie no stockbridges).
    tol : float
        Convergence parameter for bisection (tolerance on error)
    maxiter: int
        Convergence parameter for bisection (Max. number of iterations)

    Returns
    -------
    float or numpy.ndarray
        Vibration amplitude (m)

    """
    if wind_method not in _balance_wind_methods():
        raise ValueError('')
    if damping_method not in _balance_damping_methods():
        raise ValueError('')

    if len(stockbridge_cfg) == 0:
        stockbridge_cfg = dict(method='foti', tension=0.0, L=1.0, d=1.0, mu=1.0, xs=0.0, fz=[], z=[])

    if (damping_method == 'olla' and type(freq) is np.ndarray) or len(stockbridge_cfg['z']) > 0:

        def fun(y):
            wp = wind_power(y, freq, d, iv=iv, method=wind_method)
            dp = np.zeros_like(y)
            sp = np.zeros_like(y)
            for i in range(len(y)):
                dp[i] = damping_power(y[i], freq[i], tension, method=damping_method, k=k, l=l, m=m,
                                      n=n, d=d, mu=mu, rts=rts, EImin=EImin, EImax=EImax, x0=x0)
                sp[i] = power(ymax=y[i], freq=freq[i], **stockbridge_cfg)

            return wp - dp - sp

    else:

        def fun(y):
            wp = wind_power(y, freq, d, iv=iv, method=wind_method)
            dp = damping_power(y, freq, tension, method=damping_method, k=k, l=l, m=m, n=n, d=d,
                               mu=mu, rts=rts, EImin=EImin, EImax=EImax, x0=x0)
            return wp - dp

    ymin = 0. * np.ones_like(freq) * d
    ymax = 5. * np.ones_like(freq) * d

    if type(fun(ymin)) is np.ndarray:
        y, _ = _vect_bisection(fun, ymin, ymax, tol, maxiter)
    else:
        y = bisect(fun, ymin, ymax)

    y = np.where(np.sign(fun(ymin)) != np.sign(fun(ymax)), y, np.nan)

    return y
