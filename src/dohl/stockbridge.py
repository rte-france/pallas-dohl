# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np


def impedance(e1, j1, l1, m1, e2, j2, l2, m2, mm, rho, a, ej, c, W):
    """Model form polimi scilab file in 2010 RTE note [H-R27-2010-02975-FR].

    Parameters
    ----------
    e1, e2 : float
        Offsets (m).
    j1, j2 : float
        Moments of inertia (kgm2)
    l1, l2 : float
        Messenger cable length (m)
    m1, m2 : float
        End masses (kg)
    mm : float
        Centre mass (kg)
    rho : float
        Messenger cable Material density (kg/m3)
    a : float
        Area of messenger cables (m2)
    ej : float
        Bending stiffness of messanger cables (Nm2)
    c : float
        Damping constant of messanger cables (?)
    W : numpy.ndarray
        Vibration frequencies (rad/s)

    Returns
    -------
    numpy.ndarray
        Impedance as a function of frequency (Ns/m)

    """
    b1 = rho * a * l1
    b2 = rho * a * l2
    k1 = 3 * ej / l1**3
    k2 = 3 * ej / l2**3
    h1 = c / l1**3
    h2 = c / l2**3

    Ma = np.array([[13 * b1 / 35 + m1, -11 * b1 * l1 / 210 - m1 * e1, 0., 0.],
                   [-11 * b1 * l1 / 210 - m1 * e1, b1 * l1**2 / 105 + m1 * e1**2 + j1, 0, 0],
                   [0, 0, 13 * b2 / 35 + m2, -11 * b2 * l2 / 210 - m2 * e2],
                   [0, 0, -11 * b2 * l2 / 210 - m2 * e2, b2 * l2**2 / 105 + m2 * e2**2 + j2]])
    Mb = np.array([9 * b1 / 70, -13 * b1 * l1 / 420, 9 * b2 / 70, +13 * b2 * l2 / 420])
    Md = 13 * (b1 + b2) / 35 + mm

    Ka = np.array([[4 * k1, -2 * k1 * l1, 0, 0], [-2 * k1 * l1, 4 * k1 * l1**2 / 3, 0, 0],
                   [0, 0, 4 * k2, 2 * k2 * l2], [0, 0, 2 * k2 * l2, 4 * k2 * l2**2 / 3]])
    Kb = np.array([-4 * k1, 2 * k1 * l1, -4 * k2, -2 * k2 * l2])
    Kd = 4 * (k1 + k2)

    Z = np.zeros_like(W, dtype=complex)
    for i, w in enumerate(W):
        Ca = np.array([[4 * h1 / w, -2 * h1 / w * l1, 0, 0],
                       [-2 * h1 / w * l1, 4 * h1 / w * l1**2 / 3, 0, 0],
                       [0, 0, 4 * h2 / w, 2 * h2 / w * l2],
                       [0, 0, 2 * h2 / w * l2, 4 * h2 / w * l2**2 / 3]])
        Cb = np.array([-4 * h1 / w, 2 * h1 / w * l1, -4 * h2 / w, -2 * h2 / w * l2])
        Cd = 4 * (h1 / w + h2 / w)

        A = -w**2 * Ma + 1j * w * Ca + Ka
        B = +w**2 * Mb - 1j * w * Cb - Kb
        C = -B
        D = -w**2 * Md + 1j * w * Cd + Kd

        z = C @ np.linalg.solve(A, B) + D
        Z[i] = z

    return Z


def power_foti(ymax, freq, tension, L, mu, xs, z):
    # equation 2.34 in Foti report
    omega = 2. * np.pi * freq
    omega_c = np.sqrt(tension / mu) / L
    mode_n = np.round(omega / (np.pi * omega_c))
    omega_n = mode_n * np.pi * omega_c
    bar_omega_n = mode_n * np.pi

    alphad = xs / L
    p = 0.5 * np.real(z) * np.sin(bar_omega_n * (1. - alphad))**2 * omega_n**2 * ymax**2

    return p


def power_wolf(ymax, freq, tension, d, mu, xs, z):
    # [WORK IN PROGRESS] Model from Wolf, *Using the Energy Balance Method in Estimation of
    # Overhead Transmission Line Aeolian Vibrations*, in Strojarstvo 50 (5) 269-276 (2008) - ISSN 0562-1887

    # Problem : when tested, the sign of dissipated power is not always consistent, hence the np.abs(p) in the
    # return statement which is not in the paper

    w = 2. * np.pi * freq
    z_ = np.abs(z) / freq
    a = np.angle(z)

    k = w * np.sqrt(mu / tension)
    C = np.sqrt(tension / mu)
    gm = tension / (z_ * C)

    b = k * xs
    h = -1 * np.sin(b)**2 * (np.sin(2 * b) + 2 * gm * np.sin(a)) / (
            np.sin(b)**2 + gm**2 + 2 * gm * np.sin(b) * np.sin(b + a))
    g = (np.sin(b)**2 * np.cos(2 * b) + gm**2 + gm * np.sin(2 * b) * np.sin(a)) / (
            np.sin(b)**2 + gm**2 + 2 * gm * np.sin(b) * np.sin(b + a))
    p = 0.25 * tension * C * k**2 * d**2 * (ymax / d)**2 * (1 - h**2 - g**2) / (
            1 + h**2 + g**2)

    return np.abs(p)


def power(ymax=0., freq=0., method='foti', tension=0.0, L=1.0, d=1.0, mu=1.0, xs=0.0, fz=[], z=[]):
    if len(z) > 0:
        # find z at target frequency
        z_ = np.interp(freq, fz, z)
    else:
        return 0.

    # compute sum of powers
    if method == 'foti':
        p = power_foti(ymax, freq, tension, L, mu, xs, z_)
    elif method == 'wolf':
        p = power_wolf(ymax, freq, tension, d, mu, xs, z_)
    else:
        raise ValueError

    return p / L
