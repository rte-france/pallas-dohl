# -*- coding: utf-8 -*-
"""Tests for dohl.stockbridge."""

import numpy as np
import pytest

from dohl import stockbridge


@pytest.fixture
def stockbridge_geom():
    """Typical stockbridge damper parameters (small/symmetric)."""
    return dict(
        e1=0.02,
        j1=1.0e-5,
        l1=0.10,
        m1=0.4,
        e2=0.02,
        j2=1.0e-5,
        l2=0.12,
        m2=0.5,
        mm=0.3,
        rho=7800.0,
        a=1.0e-5,
        ej=5.0,
        c=1.0e-2,
        W=2.0 * np.pi * np.array([5.0, 10.0, 20.0, 40.0]),
    )


def test_impedance_returns_complex_array(stockbridge_geom):
    Z = stockbridge.impedance(**stockbridge_geom)
    assert Z.shape == stockbridge_geom["W"].shape
    assert np.iscomplexobj(Z)
    assert np.all(np.isfinite(Z))


def test_power_foti_zero_at_zero_amplitude():
    p = stockbridge.power_foti(
        ymax=0.0, freq=20.0, tension=1.0e4, L=30.0, mu=1.5, xs=1.0, z=1.0 + 0.0j
    )
    assert p == pytest.approx(0.0)


def test_power_foti_scales_with_amplitude_squared():
    base = stockbridge.power_foti(
        ymax=1.0e-3, freq=20.0, tension=1.0e4, L=30.0, mu=1.5, xs=1.0, z=1.0 + 0.5j
    )
    doubled = stockbridge.power_foti(
        ymax=2.0e-3, freq=20.0, tension=1.0e4, L=30.0, mu=1.5, xs=1.0, z=1.0 + 0.5j
    )
    np.testing.assert_allclose(doubled, 4.0 * base, rtol=1e-10)


def test_power_returns_zero_when_z_empty():
    """Without an impedance table, power() returns 0.0."""
    out = stockbridge.power(
        ymax=1.0e-3,
        freq=20.0,
        method="foti",
        tension=1.0e4,
        L=30.0,
        d=0.03,
        mu=1.5,
        xs=1.0,
        fz=[],
        z=[],
    )
    assert out == 0.0


def test_power_unknown_method_raises():
    fz = np.array([5.0, 10.0, 20.0])
    z = np.array([1.0 + 0.5j, 1.2 + 0.4j, 1.5 + 0.3j])
    with pytest.raises(ValueError):
        stockbridge.power(
            ymax=1.0e-3,
            freq=15.0,
            method="bogus",
            tension=1.0e4,
            L=30.0,
            d=0.03,
            mu=1.5,
            xs=1.0,
            fz=fz,
            z=z,
        )


def test_power_foti_method_dispatch_finite():
    fz = np.array([5.0, 10.0, 20.0, 40.0])
    z = np.array([1.0 + 0.5j, 1.2 + 0.4j, 1.5 + 0.3j, 1.8 + 0.2j])
    out = stockbridge.power(
        ymax=1.0e-3,
        freq=15.0,
        method="foti",
        tension=1.0e4,
        L=30.0,
        d=0.03,
        mu=1.5,
        xs=1.0,
        fz=fz,
        z=z,
    )
    assert np.isfinite(out)


def test_power_wolf_method_finite_and_nonnegative():
    fz = np.array([5.0, 10.0, 20.0, 40.0])
    z = np.array([1.0 + 0.5j, 1.2 + 0.4j, 1.5 + 0.3j, 1.8 + 0.2j])
    out = stockbridge.power(
        ymax=1.0e-3,
        freq=15.0,
        method="wolf",
        tension=1.0e4,
        L=30.0,
        d=0.03,
        mu=1.5,
        xs=1.0,
        fz=fz,
        z=z,
    )
    assert np.isfinite(out)
    assert out >= 0.0
