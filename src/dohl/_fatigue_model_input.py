# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd


def _p(h, m, g):
    """Mechanical parameter."""
    return h / (m * g)


def _cat(x, p, l):
    """Catenary equation.

    Parameters
    ----------
    x
    p
    l
    """
    return p * (np.cosh((x - 0.5 * l) / p) - 1)


def _span_length(angle, h, m, g):
    """Get span length from input angle.

    Args:
        angle: output angle from the clamp (degre)
        h: mechanical tension (N)
        m: lineic mass (kg/m)

    Returns:
        equivalent length (m)
    """
    p = _p(h, m, g)
    leq = 2. * np.arcsinh(np.deg2rad(angle)) * p
    return leq


def _catenary(x, h, m, l, g):
    """Catenary equation.

    Give the vertical displacement at `x` according to the catenary equation (no
    inclination)

    Args:
        x: abcissa (m)
        h: mechanical tension (N)
        m: lineic mass (kg/m)
        l: span length (m)

    Returns:
        vertical displacement (m)
    """
    p = _p(h, m, g)
    return _cat(x, p, l) - _cat(0., p, l)


def _fatigue_model_input(file_fatigue, file_conductor, xb=0.089, g=9.81):
    """Create input data for ML fatigue model.

        This function should not be used as data is directly provided. Later
        it should not depend from strdcable but directly uses ollabdd results.
    """
    from strdcable.cable import Conductor

    # read raw data
    df = pd.read_csv(file_fatigue, sep=',')
    # remove unreliable rows
    df = df.loc[df['olla_reliability'], :]
    df = df.drop(columns='olla_reliability')
    # keep only experiments with conductor break
    df = df.dropna(subset=['num_cycles'])
    # reset index
    df = df.reset_index(drop=True)
    # get tension in newton
    df['h'] = df['mech_tension_kn'] * 1.0E+03
    # get yb in mm
    df['yb'] = df['yb_mm'] * 1.0E-03
    df.drop(columns=['mech_tension_kn', 'yb_mm'], inplace=True)

    # read strand database
    sdb = pd.read_csv(file_conductor, sep=',')

    # loop on conductors, create data with properties
    for name in np.unique(df['conductor']):

        c_data = sdb.loc[sdb['conductor'] == name]
        if len(c_data) == 0:
            # print("No data found for conductor %s, these data points are skipped." % (name,))
            continue

        # get conductor properties
        cnd = Conductor(dwires=c_data['d_strand'].values,
                        nbwires=c_data['n_strand'].values,
                        material=c_data['nuance'].values,
                        compute_physics=False)
        cnd.set_normative_lay_length(cmin=0.5, cmax=0.5)
        cnd.compute_all(compute_physics=True, set_usual_values=True,
                        formulEI='PAPAILIOU', formulEP='rte')

        rows = np.where(df['conductor'] == name)[0]
        df.loc[rows, 'young_l'] = cnd.young[-1]
        df.loc[rows, 'D'] = cnd.D
        df.loc[rows, 'RTS'] = cnd.RTS
        df.loc[rows, 'm'] = cnd.m
        # compute ka (fictive stress adim coeff)
        p = np.sqrt(df.loc[rows, 'h'] / cnd.EImin)
        ka = 0.25 * (cnd.D * cnd.dwires[-1] * p**2) / (np.exp(-p * xb) - 1. + p * xb)
        df.loc[rows, 'ka'] = ka

        df.loc[rows, 'nb_layers'] = len(c_data.n_strand)

    # drop incomplete rows (because of unknown conductor)
    df = df.dropna(axis='index')

    # drop unnecessary columns
    df = df.drop(columns=['bench_type', 'num_cycles_exp'])

    # physic quantities
    df['span_length'] = _span_length(df['output_angle'], df['h'], df['m'], g)
    df['upp89'] = np.abs(_catenary(xb, df['h'], df['m'], df['span_length'], g))
    df['sigma'] = df['ka'] * df['young_l'] * df['yb'] / df['D']

    # input dataframe for fit
    dg = pd.DataFrame()

    # ..
    dg['ka'] = df['ka']
    # adim vibration amplitude (log)
    dg['yad_log'] = np.log10(df.yb / df.D)
    # catenary gravity at xb
    dg['ybga'] = df['upp89'] / df['D']
    # tension ratio
    dg['ten_rat'] = df['h'] / df['RTS']
    # ..
    dg['logN'] = np.log10(df['num_cycles'])

    return dg


if __name__ == '__main__':
    import os.path
    from dohl.config import cfg

    if1 = os.path.join(os.path.dirname(os.path.realpath(__file__)), cfg['fatigue_model']['file_raw_exp'])
    if2 = os.path.join(os.path.dirname(os.path.realpath(__file__)), cfg['fatigue_model']['file_raw_cly'])
    if3 = os.path.join(os.path.dirname(os.path.realpath(__file__)), cfg['fatigue_model']['file_qreg'])

    df = _fatigue_model_input(if1, if2)
    df.to_csv(if3, index=False)
