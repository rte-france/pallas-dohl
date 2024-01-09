# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import scipy.stats

from dohl import damage
from dohl import ebp

_wind_speed_min = 0.
_wind_speed_max = 15.
_wind_speed_num = 150


def damage_year(wind_speed: np.ndarray, tension: np.ndarray, external_diameter: np.ndarray,
                last_strand_diameter: np.ndarray, lineic_mass: np.ndarray, young: np.ndarray,
                EImin: np.ndarray, EImax: np.ndarray, multilayer: bool = True,
                strouhal: float = 0.2, wind_method: str = 'cigre', damping_method: str = 'foti',
                iv: float = 0., fictive_stress_method: str = 'ba_pfs', fsba: str = 'avg',
                rts: np.ndarray = None, x0: np.ndarray = None, stockbridge_cfg: dict = {}) -> tuple:
    """Estimate sustained damage in a year.

    Combining vibrations amplitudes using Energy Balance Principle from
    dohl.ebp.ymax, fictive stress from dohl.damage.fictive_stress, and endurance
    estimations from dohl.damage.inv_safe_border_line, compute the damage
    sustained in a year by a conductor at a given wind speed.

    Parameters
    ----------
    wind_speed : float or numpy.ndarray
        Wind speed (m.s**-1).
    tension : float or numpy.ndarray
        Mechanical tension (N).
    external_diameter : float or numpy.ndarray
        External diameter of the cable (m).
    last_strand_diameter : float or numpy.ndarray
        Diameter of the last strand of the cable (m).
    lineic_mass : float or numpy.ndarray
        Lineic mass of the cable (kg.m**-1).
    young : float or numpy.ndarray
        Young modulus of the cable (Pa).
    EImin : float or numpy.ndarray
        Minimal tangential bending stiffness of the cable (N.m**2).
    EImax : float or numpy.ndarray
        Maximal tangential bending stiffness of the cable (N.m**2).
    multilayer : bool
        Whether the conductor is a multi-layer ACSR.
    strouhal : float
        Strouhal number to compute frequency from wind speed. Default is 0.2.
    wind_method: str
        Choice of method to get wind power. Default is 'cigre'.
    damping_method : str
        Choice of method to get damping power. Default is 'foti'.
    iv : float
        Turbulence intensity of the wind (no unit). Only used if method='foti'.
    fictive_stress_method : str
        Choice of method to compute fictive stress. Must be in ['fymax',
        'ba_cigre', 'ba_pfs'].
    fsba : str
        Bending stiffness to use in fictive stress estimations. Must be in
        ['min', 'avg', 'max'] and correspond to EImin, (EImin+EImax)/2, EImax.
    rts : float or numpy.ndarray
        Rated tensile strength of the cable (N).
    x0 : float or numpy.ndarray
        Limit curvature (m**-1).
    stockbridge_cfg : dict
        Stockbridge configuration. The default is {} (empty dict ie no stockbridges).

    Returns
    -------
    ymax : float or numpy.ndarray
        Vibration amplitude at antinode (m)
    yb : float or numpy.ndarray
        Vibration amplitude at xb (m)
    sigma: float or numpy.ndarray
        Strain.
    dmg: float or numpy.ndarray
        Damage in a year.

    """
    u = wind_speed
    D = external_diameter
    d = last_strand_diameter
    mu = lineic_mass

    if fsba == 'min':
        ei = EImin
    elif fsba == 'max':
        ei = EImax
    elif fsba == 'avg':
        ei = 0.5 * (EImin + EImax)
    else:
        raise ValueError('')

    freq = u * strouhal / D
    ymax = ebp.ymax(freq, tension, D, wind_method=wind_method, damping_method=damping_method, iv=iv,
                    mu=mu, rts=rts, EImin=EImin, EImax=EImax, x0=x0, stockbridge_cfg=stockbridge_cfg)
    sigma, yb = damage.fictive_stress(ymax, freq, tension, d, mu, young, ei, method=fictive_stress_method,
                                      return_yb=True)
    nbcyc = damage.inv_safe_border_line(1.0E-06 * sigma, multilayer)
    dmg = 365.25 * 24 * 3600 * freq / nbcyc

    return ymax, yb, sigma, dmg


def damage_history(time: np.ndarray, wind_speed: np.ndarray, tension: np.ndarray,
                   external_diameter: np.ndarray, last_strand_diameter: np.ndarray,
                   lineic_mass: np.ndarray, young: np.ndarray, EImin: np.ndarray, EImax: np.ndarray,
                   multilayer: bool = True, strouhal: float = 0.2, wind_method: str = 'cigre',
                   damping_method: str = 'foti', iv: float = 0.,
                   fictive_stress_method: str = 'ba_pfs', fsba: str = 'avg', rts: np.ndarray = None,
                   beta: np.ndarray = None, stockbridge_cfg={}) -> tuple:
    """Estimate sustained damage over a time-period.

    Combining vibrations amplitudes using Energy Balance Principle from
    dohl.ebp.ymax, fictive stress from dohl.damage.fictive_stress, and endurance
    estimations from dohl.damage.inv_safe_border_line, compute the damage
    sustained in a year by a conductor at a given wind speed.

    Parameters
    ----------
    time : numpy.ndarray
        Time vector (s).
    wind_speed : float or numpy.ndarray
        Wind speed (m.s**-1).
    tension : float or numpy.ndarray
        Mechanical tension (N).
    external_diameter : float or numpy.ndarray
        External diameter of the cable (m).
    last_strand_diameter : float or numpy.ndarray
        Diameter of the last strand of the cable (m).
    lineic_mass : float or numpy.ndarray
        Lineic mass of the cable (kg.m**-1).
    young : float or numpy.ndarray
        Young modulus of the cable (Pa).
    EImin : float or numpy.ndarray
        Minimal tangential bending stiffness of the cable (N.m**2).
    EImax : float or numpy.ndarray
        Maximal tangential bending stiffness of the cable (N.m**2).
    multilayer : bool
        Whether the conductor is a multi-layer ACSR.
    strouhal : float
        Strouhal number to compute frequency from wind speed. Default is 0.2.
    wind_method: str
        Choice of method to get wind power. Default is 'cigre'.
    damping_method : str
        Choice of method to get damping power. Default is 'foti'.
    iv : float
        Turbulence intensity of the wind (no unit). Only used if method='foti'.
    fictive_stress_method : str
        Choice of method to compute fictive stress. Must be in ['fymax',
        'ba_cigre', 'ba_pfs'].
    fsba : str
        Bending stiffness to use in fictive stress estimations. Must be in
        ['min', 'avg', 'max'] and correspond to EImin, (EImin+EImax)/2, EImax.
    rts : float or numpy.ndarray
        Rated tensile strength of the cable (N)
    beta : float or numpy.ndarray
        Coefficient to compute limit curvature from tension (kg**-1.m**-2.s**2).
    stockbridge_cfg : dict
        Stockbridge configuration. The default is {} (empty dict ie no stockbridges).

    Returns
    -------
    ymax : float or numpy.ndarray
        Vibration amplitude at antinode (m)
    sigma: float or numpy.ndarray
        Strain.
    dmg: float or numpy.ndarray
        Damage in a year.

    """
    u = wind_speed
    D = external_diameter
    d = last_strand_diameter
    mu = lineic_mass

    dmg = np.zeros_like(time)
    for i in range(len(time)):
        # compute damage per year
        _, _, _, tmp, = damage_year(u[i], tension[i], D, d, mu, young, EImin, EImax, multilayer=multilayer,
                                    strouhal=strouhal, wind_method=wind_method, damping_method=damping_method, iv=iv,
                                    fictive_stress_method=fictive_stress_method, fsba=fsba, rts=rts,
                                    x0=beta * tension[i], stockbridge_cfg=stockbridge_cfg)
        # back in damage per second
        dmg[i] = tmp / (365.25 * 24 * 3600)

    # integrate over time
    return 0.5 * np.sum((dmg[1:] + dmg[:-1]) * np.diff(time))


def damage_year_weibull(shape: float, scale: float, tension: np.ndarray,
                        external_diameter: np.ndarray, last_strand_diameter: np.ndarray,
                        lineic_mass: np.ndarray, young: float, EImin: float, EImax: float,
                        multilayer: str = True, strouhal: float = 0.2, wind_method: str = 'cigre',
                        damping_method: str = 'foti', iv: float = 0.,
                        fictive_stress_method: str = 'ba_pfs', fsba: str = 'avg', rts: float = None,
                        x0: float = None, stockbridge_cfg: dict = {}) -> float:
    """Sustained damage of a cable with wind history as a Weibull distribution.

    Using the damage computation function, an integration is performed over
    a wind distribution to get total damage before being converted to a
    lifespan estimation.

    Parameters
    ----------
    shape : float
        Weibull shape parameter.
    scale : float
        Weibull scale parameter.
    tension : float or numpy.ndarray
        Mechanical tension (N).
    external_diameter : float or numpy.ndarray
        External diameter of the cable (m).
    last_strand_diameter : float or numpy.ndarray
        Diameter of the last strand of the cable (m).
    lineic_mass : float or numpy.ndarray
        Lineic mass of the cable (kg.m**-1).
    young : float
        Young modulus of the cable (Pa)
    EImin : float
        Minimal tangential bending stiffness of the cable (N.m**2).
    EImax : float
        Maximal tangential bending stiffness of the cable (N.m**2).
    multilayer : bool
        Whether the conductor is a multi-layer ACSR.
    strouhal : float
        Strouhal number to compute frequency from wind speed. Default is 0.2.
    wind_method: str
        Choice of method to get wind power. Default is 'cigre'.
    damping_method : str
        Choice of method to get damping power. Default is 'foti'.
    iv : float
        Turbulence intensity of the wind (no unit). Only used if method='foti'.
    fictive_stress_method : str
        Choice of method to compute fictive stress. Must be in ['fymax',
        'ba_cigre', 'ba_pfs'].
    fsba : str
        Bending stiffness to use in fictive stress estimations. Must be in
        ['min', 'avg', 'max'] and correspond to EImin, (EImin+EImax)/2, EImax.
    rts : float
        Rated tensile strength of the cable (N)
    x0 : float
        Limit curvature (m**-1).
    stockbridge_cfg : dict
        Stockbridge configuration. The default is {} (empty dict ie no stockbridges).

    Returns
    -------
    float:
        Estimated lifespan in years.

    """
    D = external_diameter
    d = last_strand_diameter
    mu = lineic_mass

    # wind speed discretization
    u = np.linspace(_wind_speed_min, _wind_speed_max, _wind_speed_num + 1)[1:]
    # compute damage for wind speeds
    _, _, _, dmg = damage_year(u, tension, D, d, mu, young, EImin, EImax, multilayer=multilayer, strouhal=strouhal,
                               wind_method=wind_method, damping_method=damping_method, iv=iv,
                               fictive_stress_method=fictive_stress_method, fsba=fsba, rts=rts, x0=x0,
                               stockbridge_cfg=stockbridge_cfg)
    # wind distributions
    p = scipy.stats.exponweib.pdf(u, 1., shape, 0., scale)
    # integrate damage over distribution (cumulative damage using miner's law)
    dmg *= p
    return np.sum(0.5 * (dmg[1:] + dmg[:-1]) * np.diff(u))


def damage_year_weibull_array(shape: np.ndarray, scale: np.ndarray, tension: np.ndarray,
                              external_diameter: np.ndarray, last_strand_diameter: np.ndarray,
                              lineic_mass: np.ndarray, young: np.ndarray, EImin: np.ndarray,
                              EImax: np.ndarray, multilayer: bool = True, strouhal: float = 0.2,
                              wind_method: str = 'cigre', damping_method: str = 'foti',
                              iv: float = 0., fictive_stress_method: str = 'ba_pfs',
                              fsba: str = 'avg', rts: np.ndarray = None,
                              x0: np.ndarray = None, stockbridge_cfg: dict = {}) -> np.ndarray:
    """Sustained damage of a cable with wind history as a Weibull distribution.

    Using the damage computation function, an integration is performed over
    a wind distribution to get total damage before being converted to a
    lifespan estimation.

    Parameters
    ----------
    shape : float or numpy.ndarray
        Weibull shape parameter.
    scale : float or numpy.ndarray
        Weibull scale parameter.
    tension : float or numpy.ndarray
        Mechanical tension (N).
    external_diameter : float or numpy.ndarray
        External diameter of the cable (m).
    last_strand_diameter : float or numpy.ndarray
        Diameter of the last strand of the cable (m).
    lineic_mass : float or numpy.ndarray
        Lineic mass of the cable (kg.m**-1).
    young : float or numpy.ndarray
        Young modulus of the cable (Pa)
    EImin : float or numpy.ndarray
        Minimal tangential bending stiffness of the cable (N.m**2).
    EImax : float or numpy.ndarray
        Maximal tangential bending stiffness of the cable (N.m**2).
    multilayer : bool
        Whether the conductor is a multi-layer ACSR.
    strouhal : float or numpy.ndarray
        Strouhal number to compute frequency from wind speed. Default is 0.2.
    wind_method: str
        Choice of method to get wind power. Default is 'cigre'.
    damping_method : str
        Choice of method to get damping power. Default is 'foti'.
    iv : float or numpy.ndarray
        Turbulence intensity of the wind (no unit). Only used if method='foti'.
    fictive_stress_method : str
        Choice of method to compute fictive stress. Must be in ['fymax',
        'ba_cigre', 'ba_pfs'].
    fsba : str
        Bending stiffness to use in fictive stress estimations. Must be in
        ['min', 'avg', 'max'] and correspond to EImin, (EImin+EImax)/2, EImax.
    rts : float or numpy.ndarray
        Rated tensile strength of the cable (N)
    x0 : float or numpy.ndarray
        Limit curvature (m**-1).
    stockbridge_cfg : dict
        Stockbridge configuration. The default is {} (empty dict ie no stockbridges).

    Returns
    -------
    float or numpy.ndarray:
        Estimated lifespan in years.

    """
    D = external_diameter
    d = last_strand_diameter
    mu = lineic_mass

    l = [shape, scale, tension, D, d, mu, young, EImin, EImax, multilayer, rts, x0]

    # size of input array
    n = 1
    for x in l:
        try:
            n = max(n, len(x))
        except:
            pass

    # if no array, use std function
    if n == 1:
        return damage_year_weibull(shape, scale, tension, D, d, mu, young, EImin, EImax,
                                   multilayer=multilayer, strouhal=strouhal,
                                   wind_method=wind_method, damping_method=damping_method, iv=iv,
                                   fictive_stress_method=fictive_stress_method, fsba=fsba, rts=rts,
                                   x0=x0, stockbridge_cfg=stockbridge_cfg)

    # ...
    for i in range(len(l)):
        l[i] *= np.ones(n)
    shape, scale, tension, D, d, mu, young, EImin, EImax, multilayer, rts, x0 = tuple(l)
    multilayer = multilayer.astype(bool)
    u = np.linspace(_wind_speed_min, _wind_speed_max, _wind_speed_num + 1)[1:]

    # if small array, loop over it
    if n < len(u):
        dmg = np.zeros(n)
        for i in range(n):
            dmg[i] = damage_year_weibull(shape[i], scale[i], tension[i], D[i], d[i], mu[i],
                                         young[i], EImin[i], EImax[i], multilayer=multilayer[i],
                                         strouhal=strouhal, wind_method=wind_method,
                                         damping_method=damping_method, iv=iv,
                                         fictive_stress_method=fictive_stress_method, fsba=fsba,
                                         rts=rts[i], x0=x0[i], stockbridge_cfg=stockbridge_cfg)
        return dmg

    # large array: loop on wind speed items
    p = np.zeros((n, len(u)))
    for i in range(n):
        p[i, :] = scipy.stats.exponweib.pdf(u, 1., shape[i], 0., scale[i])

    dmg = np.zeros_like(p)
    for j in range(len(u)):
        _, _, _, dmg[:, j] = damage_year(u[j], tension, D, d, mu, young, EImin, EImax, multilayer=multilayer,
                                         strouhal=strouhal, wind_method=wind_method, damping_method=damping_method,
                                         iv=iv, fictive_stress_method=fictive_stress_method, fsba=fsba, rts=rts, x0=x0,
                                         stockbridge_cfg=stockbridge_cfg)
    dmg *= p
    return np.sum(0.5 * (dmg[:, 1:] + dmg[:, :-1]) * np.diff(u), axis=1)
