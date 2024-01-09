# !/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Union

import numpy as np


def fictive_stress_fymax(ymax: np.ndarray, freq: np.ndarray, d: np.ndarray, mu: np.ndarray,
                         young: np.ndarray, EI: np.ndarray) -> float:
    """Compute the fictive stress.

    Orange Book -- chap 'fatigue of overhead conductors'
    "eq 3.2-14" 3-12 -- page 196/614.

    Parameters
    ----------
    ymax : float or numpy.ndarray
        Antinode vibration amplitude (m).
    freq : float or numpy.ndarray
        Frequencies relative to vibration (Hz).
    d : float or numpy.ndarray
        Diameter of the last strand of the cable (m)
    mu : float or numpy.ndarray
        Lineic mass of the cable (kg.m**-1).
    young : float or numpy.ndarray
        Young modulus of the cable (Pa)
    EI : float or numpy.ndarray
        Bending stiffness (N.m**2).

    Returns
    -------
    float
        Fictive stress (Pa.s).
    """
    return np.pi * d * young * np.sqrt(mu / EI) * freq * ymax


def _wavelength(freq: np.ndarray, tension: np.ndarray, mu: np.ndarray,
                EI: np.ndarray) -> np.ndarray:
    """Get cable wavelength.

    Based on a beam vibration model, from CIGRE, *Report on aeolian vibration*,
    Electra, 1986.

    Parameters
    ----------
    freq : float or numpy.ndarray
        Frequencies relative to vibration (Hz).
    tension : float or numpy.ndarray
        Mechanical tension (N).
    mu : float or numpy.ndarray
        Lineic mass of the cable (kg.m**-1).
    EI: float or numpy.ndarray
        Tangential bending stiffness of the cable (N.m**2).

    Returns
    -------
    float or numpy.ndarray
        Wavelength (m)

    """
    s = 1 / (2 * mu * freq**2)
    return np.sqrt(tension * s + np.sqrt(s * (tension + 8 * np.pi**2 * EI)))


def _ampl_cigre(ymax: np.ndarray, freq: np.ndarray, tension: np.ndarray, mu: np.ndarray,
                EI: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Corrected amplitude.

    Based on a beam vibration model, from CIGRE, *Report on aeolian vibration*,
    Electra, 1986.

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
    EI : float or numpy.ndarray
        Tangential bending stiffness of the cable (N.m**2).
    x : float or numpy.ndarray
        Abcissa where to evaluate the amplitude (m)

    Returns
    -------
    float or numpy.ndarray
        Vibration amplitude at abcissa x (m)

    """
    w = _wavelength(freq, tension, mu, EI)
    q = 0.5 * tension / EI
    r = (2. * np.pi * freq)**2 * mu / EI
    a = np.sqrt(+q + np.sqrt(r + q**2))
    b = np.sqrt(-q + np.sqrt(r + q**2))
    y = np.sin(b * x) - b * (
            np.sin(a * x) + np.tanh(0.5 * a * w) * (np.cos(b * x) - np.cosh(a * x))) / a
    return ymax * y


def _ampl_pbs(ymax: np.ndarray, freq: np.ndarray, tension: np.ndarray, mu: np.ndarray,
              EI: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Corrected amplitude.

    Based on J.C. Poffenberger, R.L. Swart, *Differential displacement and dynamic
    conductor strain*, IEEE Transactions on Power Apparatus and Systems, 1965.

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
    EI : float or numpy.ndarray
        Tangential bending stiffness of the cable (N.m**2).
    x : float or numpy.ndarray
        Abcissa where to evaluate the amplitude (m)

    Returns
    -------
    float or numpy.ndarray
        Vibration amplitude at abcissa x (m)

    """
    w = _wavelength(freq, tension, mu, EI)
    p = np.sqrt(EI / tension)
    q = (w - 2. * p) / (2. * np.pi * p)
    num = q * np.sin(2. * np.pi * x / w) + np.exp(-x / p) - 1 + 2. * x / w
    den = q + np.exp(-0.25 * w / p) - 0.5
    return ymax * num / den


def fictive_stress_xyb(yb: np.ndarray, tension: np.ndarray, d: np.ndarray, young: np.ndarray,
                       EI: float, xb: float) -> np.ndarray:
    """Compute the fictive stress ratio

    (Orange Book -- chap 'fatigue of overhead conductors'
    "eq 3.2-15" 3-12 -- page 196/614).

    Parameters
    ----------
    yb : float or numpy.ndarray
        Vibration amplitude at xb (m).
    tension : float or numpy.ndarray
        Mechanical tension (N).
    d : float or numpy.ndarray
        Diameter of the last strand of the cable (m)
    young : float or numpy.ndarray
        Young modulus of the cable (Pa)
    EI : float
        Bending stiffness (Pa.m**4).
    xb : float
        Abscissa where yb is given (m).

    Returns
    -------
    float
        Fictive stress ratio (Pa/m)
    """
    p = np.sqrt(tension / EI)
    num = 0.25 * d * young * p**2
    den = np.exp(-p * xb) - 1. + p * xb
    return yb * num / den


def _fictive_stress_methods() -> list:
    """List of available fictive stress computation methods.

     Parameters
     ----------
     None

     Returns
     -------
     list
         List of available fictive stress computation methods.
     """
    return ['fymax', 'ba_cigre', 'ba_pfs']


def fictive_stress(ymax: np.ndarray, freq: np.ndarray, tension: np.ndarray, d: np.ndarray,
                   mu: np.ndarray, young: np.ndarray, EI: np.ndarray, method: str = 'fymax',
                   xb: np.ndarray = 0.089, return_yb=False) -> Union[float, tuple]:
    """Compute the fictive stress.

    Parameters
    ----------
    ymax : float or numpy.ndarray
        Antinode vibration amplitude (m).
    freq : float or numpy.ndarray
        Frequencies relative to vibration (Hz).
    tension : float or numpy.ndarray
        Mechanical tension (N).
    d : float or numpy.ndarray
        Diameter of the last strand of the cable (m)
    mu : float or numpy.ndarray
        Lineic mass of the cable (kg.m**-1).
    young : float or numpy.ndarray
        Young modulus of the cable (Pa)
    EI : float or numpy.ndarray
        Bending stiffness (N.m**2).
    method : str
        Choice of method to compute fictive stress. Must be in ['fymax',
        'ba_cigre', 'ba_pfs'].
    xb : float or numpy.ndarray
        Abcissa where to evaluate the amplitude (m)
    return_yb : bool
        If set to True the function returns both fictive stress and yb value,
        which is the vibration amplitude in meters at xb.

    Returns
    -------
    float
        Fictive stress ratio (Pa/m)
    -------

    """
    if method == 'fymax':
        sigma = fictive_stress_fymax(ymax, freq, d, mu, young, EI)
        yb = None
    else:
        if method == 'ba_cigre':
            yb = _ampl_cigre(ymax, freq, tension, mu, EI, xb)
        elif method == 'ba_pfs':
            yb = _ampl_pbs(ymax, freq, tension, mu, EI, xb)
        else:
            raise ValueError('Unrecognized method %s' % (method,))
        sigma = fictive_stress_xyb(yb, tension, d, young, EI, xb=xb)

    if return_yb:
        return sigma, yb

    return sigma


def _mlc() -> tuple:
    """Safe border line coefficients for multi-layer acsr conductor.

    (Orange Book -- chap 'Safe Border Line Method'
    3-30 -- page 214).

     Parameters
     ----------
     None

     Returns
     -------
     tuple
         N Threshold, a1, b1, a2, b2
     """
    return 1.56E+07, 450., -0.20, 263., -0.17


def _slc() -> tuple:
    """Safe border line coefficients for single-layer acsr conductor.

    (Orange Book -- chap 'Safe Border Line Method'
    3-30 -- page 214).

     Parameters
     ----------
     None

     Returns
     -------
     tuple
         N Threshold, a1, b1, a2, b2
     """
    return 2.0E+07, 730., -0.20, 430., -0.17


def _sbd(a, b, N) -> float:
    """Computes safe border line formula.

    Parameters
    ----------
    a : float or numpy.ndarray
        Coefficient (MPa).
    b : float or numpy.ndarray
        Coefficient adim.
    N : float or numpy.ndarray
        Number of cycles.

    Returns
    -------
    float
        Stress (MPa)
    -------

    """
    return a * np.power(N, b)


def _isbd(a, b, sigma) -> np.ndarray:
    """Computes inverse of safe border line formula.

    Parameters
    ----------
    a : float or numpy.ndarray
        Coefficient (MPa).
    b : float or numpy.ndarray
        Coefficient adim.
    sigma : float or numpy.ndarray
        Stress (MPa).

    Returns
    -------
    float or numpy.ndarrau
        Number of cycles
    -------

    """
    return np.exp(np.log(sigma / a) / b)


def _coeffs(multilayer) -> np.ndarray:
    """Coefficients of safe border line formula.

    Generates coefficients for safe border line calculation depending on type fo conductor (single layer or multi layer)

    Parameters
    ----------
    multilayer : bool or numpy.ndarray
        Whether or not the conductor is a multi-layer ACSR.

    Returns
    -------
    tuple or numpy.ndarray
        Array of poffenberger-swart stresses -- sigma_a(Yb) (MPa).
    """
    try:
        if multilayer:
            n0, a1, b1, a2, b2 = _mlc()
        else:
            n0, a1, b1, a2, b2 = _slc()
    except ValueError:
        n0, a1, b1, a2, b2 = _slc()
        n0 *= np.ones_like(multilayer)
        a1 *= np.ones_like(multilayer)
        b1 *= np.ones_like(multilayer)
        a2 *= np.ones_like(multilayer)
        b2 *= np.ones_like(multilayer)
        ix = np.where(multilayer)[0]
        n0_, a1_, b1_, a2_, b2_ = _mlc()
        n0[ix] = n0_
        a1[ix] = a1_
        b1[ix] = b1_
        a2[ix] = a2_
        b2[ix] = b2_

    return n0, a1, b1, a2, b2


def safe_border_line(N, multilayer) -> np.ndarray:
    """Safe border line method.

    Evaluate the safe border line for a given number of cyclic vibration. From
    Orange Book -- chap *'fatigue of overhead conductors'* "Safe Border Line
    Method" 3-30 -- page 214/614.

    Parameters
    ----------
    N : int or float or numpy.ndarray
        Number of cyclic vibrations (no unit).
    multilayer : bool or numpy.ndarray
        Whether or not the conductor is a multi-layer ACSR.

    Returns
    -------
    float or numpy.ndarray
        Array of poffenberger-swart stresses -- sigma_a(Yb) (MPa).
    """
    n0, a1, b1, a2, b2 = _coeffs(multilayer)
    return np.where(N <= n0, _sbd(a1, b1, N), _sbd(a2, b2, N))


def inv_safe_border_line(sigma, multilayer) -> np.ndarray:
    """Inverse safe border line method.

    Parameters
    ----------
    sigma : float or numpy.ndarray
        poffenberger-swart stress (MPa).
    multilayer : bool or numpy.ndarray
        Whether or not the conductor is a multi-layer ACSR.

    Returns
    -------
    float or numpy.ndarray
        Number of cyclic vibrations until break (no unit).

    """
    n0, a1, b1, a2, b2 = _coeffs(multilayer)

    smax = _sbd(a1, b1, n0)
    smin = _sbd(a2, b2, n0)

    n = n0 * np.ones_like(sigma)
    n = np.where(sigma < smin, _isbd(a2, b2, sigma), n)
    n = np.where(sigma > smax, _isbd(a1, b1, sigma), n)

    return n


def _qreg_apply(x, y, z, t, c) -> np.ndarray:
    """Quantile regression function

    Parameters
    ----------
    x : float or numpy.ndarray

    y : float or numpy.ndarray

    z : float or numpy.ndarray

    t : float or numpy.ndarray

    c : numpy.ndarray
        list of cofficient of the fited line

    Returns
    -------
    float or numpy.ndarray


    """
    return c[0] + c[1] * x + c[2] * y + c[3] * z + c[4] * t


def _qreg_fit_coeffs(q):
    from sklearn.linear_model import QuantileRegressor
    from dohl.config import _FatigueData

    if q < 0. or q > 1:
        raise ValueError()
    qs = min(q, 1. - q)

    df = _FatigueData.qreg_input()
    inputs = ['ten_rat', 'yad_log', 'ybga', 'ka']
    output = ['logN']
    x = df[inputs].values
    y = df[output].values.ravel()

    m1 = QuantileRegressor(quantile=qs, alpha=0.0, solver='highs', fit_intercept=True)
    m1.fit(x, y)
    c1 = (m1.intercept_,) + tuple(m1.coef_)

    qc = 1. - qs
    m2 = QuantileRegressor(quantile=qc, alpha=0.0, solver='highs', fit_intercept=True)
    m2.fit(x, y)
    c2 = (m2.intercept_,) + tuple(m2.coef_)

    return c1, c2


def num_cycles_qreg_said(ratio: np.ndarray, yad: np.ndarray, ybga: np.ndarray, ka: np.ndarray,
                         q: float = 0.025, log: bool = False) -> tuple:
    # TODO : doc
    """...

    See [Said2022], METAMODEL APPLIED TO FATIGUE DAMAGE IN OVERHEAD LINES
    CONDUCTORS, Cigre, 2022.

    Parameters
    ----------
    ratio : float or numpy.ndarray
        Mechanical tension divided by rated tensile strength (no unit).
    yad : float or numpy.ndarray
        Adim. vibration amplitude with cable external diameter.
    ybga : float or numpy.ndarray
        Adim. catenary offset at xb=89 mm with cable external diameter.
    ka : float or numpy.ndarray
        Adim. fictive stress (no unit)
    q : float
        Quantile for regression (must be between 0 and 1)

    Returns
    -------
    float or numpy.ndarray
        Number of cyclic vibrations until break (no unit).
    """
    cq, dq = _qreg_fit_coeffs(q=q)
    n1 = _qreg_apply(ratio, yad, ybga, ka, cq)
    n2 = _qreg_apply(ratio, yad, ybga, ka, dq)
    if log:
        return n1, n2
    else:
        return 10**n1, 10**n2
