"""Tests for dohl.damage."""

import numpy as np
import pytest

from dohl import damage

# ---------- fictive_stress -----------------------------------------------------


def test_fictive_stress_fymax_known_formula(aster570):
    c = aster570
    ymax = 1.0e-03
    freq = 20.0
    expected = np.pi * c["d"] * c["young"] * np.sqrt(c["mu"] / c["EImin"]) * freq * ymax
    got = damage.fictive_stress_fymax(
        np.asarray(ymax), np.asarray(freq), c["d"], c["mu"], c["young"], c["EImin"]
    )
    np.testing.assert_allclose(got, expected)


def test_fictive_stress_fymax_array_shape(aster570):
    c = aster570
    ymax = np.linspace(1e-4, 1e-3, 6)
    freq = np.full_like(ymax, 20.0)
    out = damage.fictive_stress_fymax(ymax, freq, c["d"], c["mu"], c["young"], c["EImin"])
    assert out.shape == ymax.shape
    assert np.all(out > 0)


def test_fictive_stress_dispatch_methods_finite(aster570):
    c = aster570
    ymax = 5.0e-04
    freq = 20.0
    tension = 0.15 * c["rts"]
    for method in ["fymax", "ba_cigre", "ba_pfs"]:
        sigma = damage.fictive_stress(
            np.asarray(ymax),
            np.asarray(freq),
            np.asarray(tension),
            c["d"],
            c["mu"],
            c["young"],
            c["EImin"],
            method=method,
        )
        assert np.isfinite(sigma)
        assert sigma > 0


def test_fictive_stress_unknown_method_raises(aster570):
    c = aster570
    with pytest.raises(ValueError):
        damage.fictive_stress(
            np.asarray(1e-3),
            np.asarray(20.0),
            np.asarray(0.15 * c["rts"]),
            c["d"],
            c["mu"],
            c["young"],
            c["EImin"],
            method="not_a_method",
        )


def test_fictive_stress_return_yb_tuple(aster570):
    c = aster570
    sigma, yb = damage.fictive_stress(
        np.asarray(1e-3),
        np.asarray(20.0),
        np.asarray(0.15 * c["rts"]),
        c["d"],
        c["mu"],
        c["young"],
        c["EImin"],
        method="ba_pfs",
        return_yb=True,
    )
    assert np.isfinite(sigma) and sigma > 0
    assert np.isfinite(yb)


def test_fictive_stress_return_yb_fymax_yields_none(aster570):
    c = aster570
    sigma, yb = damage.fictive_stress(
        np.asarray(1e-3),
        np.asarray(20.0),
        np.asarray(0.15 * c["rts"]),
        c["d"],
        c["mu"],
        c["young"],
        c["EImin"],
        method="fymax",
        return_yb=True,
    )
    assert np.isfinite(sigma)
    assert yb is None


# ---------- safe border line --------------------------------------------------


def test_sbl_single_layer_continuity_at_n0():
    """At the threshold N0, both formula branches should match."""
    n0, a1, b1, a2, b2 = damage._slc()
    left = a1 * n0**b1
    right = a2 * n0**b2
    np.testing.assert_allclose(left, right, rtol=5e-2)


def test_sbl_multi_layer_continuity_at_n0():
    n0, a1, b1, a2, b2 = damage._mlc()
    left = a1 * n0**b1
    right = a2 * n0**b2
    np.testing.assert_allclose(left, right, rtol=5e-2)


def test_safe_border_line_monotone_decreasing():
    N = np.logspace(5, 9, 50)
    sigma_sl = damage.safe_border_line(N, multilayer=False)
    sigma_ml = damage.safe_border_line(N, multilayer=True)
    assert np.all(np.diff(sigma_sl) <= 0)
    assert np.all(np.diff(sigma_ml) <= 0)


def test_safe_border_line_single_vs_multi_layer_differ():
    N = np.logspace(5, 9, 20)
    sigma_sl = damage.safe_border_line(N, multilayer=False)
    sigma_ml = damage.safe_border_line(N, multilayer=True)
    assert np.any(sigma_sl != sigma_ml)


def test_safe_border_line_array_multilayer():
    """multilayer argument can be an array (vectorized)."""
    N = np.logspace(5, 9, 10)
    ml = np.zeros_like(N, dtype=bool)
    ml[5:] = True
    sigma = damage.safe_border_line(N, multilayer=ml)
    assert sigma.shape == N.shape
    assert np.all(np.isfinite(sigma))


def test_inv_safe_border_line_roundtrip_single_layer():
    N = np.logspace(5, 9, 25)
    sigma = damage.safe_border_line(N, multilayer=False)
    N_back = damage.inv_safe_border_line(sigma, multilayer=False)
    np.testing.assert_allclose(N_back, N, rtol=1e-6)


def test_inv_safe_border_line_roundtrip_multi_layer():
    N = np.logspace(5, 9, 25)
    sigma = damage.safe_border_line(N, multilayer=True)
    N_back = damage.inv_safe_border_line(sigma, multilayer=True)
    np.testing.assert_allclose(N_back, N, rtol=1e-6)


def test_inv_safe_border_line_monotone_decreasing():
    sigma = np.linspace(1.0, 80.0, 60)
    N = damage.inv_safe_border_line(sigma, multilayer=False)
    assert np.all(np.diff(N) <= 0)


# ---------- internal amplitude functions --------------------------------------


def test_wavelength_positive(aster570):
    c = aster570
    freq = np.array([10.0, 20.0, 40.0])
    tension = 0.15 * c["rts"]
    w = damage._wavelength(freq, tension, c["mu"], c["EImin"])
    assert np.all(w > 0)
    assert np.all(np.isfinite(w))


def test_ampl_cigre_zero_at_clamp(aster570):
    """Beam-vibration amplitude should vanish at x=0."""
    c = aster570
    ymax = np.array([1.0e-3])
    freq = np.array([20.0])
    tension = np.array([0.15 * c["rts"]])
    x = np.array([0.0])
    y = damage._ampl_cigre(ymax, freq, tension, c["mu"], c["EImin"], x)
    np.testing.assert_allclose(y, 0.0, atol=1e-12)


def test_ampl_pbs_zero_at_clamp(aster570):
    c = aster570
    ymax = np.array([1.0e-3])
    freq = np.array([20.0])
    tension = np.array([0.15 * c["rts"]])
    x = np.array([0.0])
    y = damage._ampl_pbs(ymax, freq, tension, c["mu"], c["EImin"], x)
    np.testing.assert_allclose(y, 0.0, atol=1e-12)


# ---------- coefficients ------------------------------------------------------


def test_coeffs_scalar_bool():
    n0, a1, b1, a2, b2 = damage._coeffs(False)
    assert all(isinstance(v, float) for v in (n0, a1, b1, a2, b2))


def test_coeffs_array_bool_dispatches():
    ml = np.array([False, True, False])
    n0, a1, b1, a2, b2 = damage._coeffs(ml)
    assert n0.shape == ml.shape
    n0_sl, *_ = damage._slc()
    n0_ml, *_ = damage._mlc()
    np.testing.assert_allclose(n0, [n0_sl, n0_ml, n0_sl])
