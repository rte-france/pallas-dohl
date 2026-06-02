"""Tests for dohl.ebp (Energy Balance Principle)."""

import numpy as np
import pytest

from dohl import ebp

# ---------- wind power -------------------------------------------------------


def test_wind_power_methods_list_non_empty():
    methods = ebp._wind_power_methods()
    assert len(methods) > 0
    assert "cigre" in methods
    assert "foti" in methods
    assert "olla" in methods


def test_wind_power_adim_cigre_equals_foti_iv0():
    """CIGRE is foti at zero turbulence intensity by construction."""
    yad = np.linspace(1e-4, 0.5, 30)
    cigre = ebp.wind_power_adim(yad, method="cigre")
    foti = ebp.wind_power_adim(yad, iv=0.0, method="foti")
    np.testing.assert_allclose(cigre, foti)


def test_wind_power_adim_foti_iv_reduces_power():
    """Higher turbulence intensity should reduce wind input."""
    yad = np.linspace(0.05, 0.5, 20)
    low = ebp.wind_power_adim(yad, iv=0.0, method="foti")
    high = ebp.wind_power_adim(yad, iv=0.2, method="foti")
    assert np.all(high <= low)


def test_wind_power_adim_olla_finite():
    yad = np.linspace(1e-3, 0.5, 25)
    out = ebp.wind_power_adim(yad, method="olla")
    assert np.all(np.isfinite(out))


def test_wind_power_adim_epri_methods_finite():
    yad = np.linspace(0.05, 0.5, 25)
    for method in ["bate", "brika", "carroll", "diana", "farquharson", "pon"]:
        out = ebp.wind_power_adim(yad, method=method)
        assert out.shape == yad.shape
        assert np.all(np.isfinite(out))


def test_wind_power_adim_unknown_method_raises():
    with pytest.raises(ValueError):
        ebp.wind_power_adim(np.array([0.1]), method="not_a_method")


def test_wind_power_adim_extend_false_masks_out_of_range():
    """With extend=False, yad values outside method range become NaN in the output."""
    from dohl.config import _EpriData

    dat = _EpriData.wind_power()["carroll"]
    lo, hi = dat["range"]
    yad = np.array([lo - 0.5 * lo, 0.5 * (lo + hi), hi + 0.5 * (hi - lo)])
    wp = ebp.wind_power_adim(yad, method="carroll", extend=False)
    assert np.isnan(wp[0])
    assert np.isfinite(wp[1])
    assert np.isnan(wp[2])


def test_wind_power_scaling(aster570):
    """wind_power = f**3 * d**4 * wind_power_adim(y/d)."""
    c = aster570
    y = np.array([1.0e-3])
    f = np.array([20.0])
    d = c["D"]
    expected = f**3 * d**4 * ebp.wind_power_adim(y / d, method="cigre")
    got = ebp.wind_power(y, f, d, method="cigre")
    np.testing.assert_allclose(got, expected)


# ---------- damping power ----------------------------------------------------


def test_damping_power_methods_list_non_empty():
    methods = ebp._damping_power_methods()
    assert "foti" in methods
    assert "olla" in methods
    assert "olla_approx" in methods


def test_damping_power_unknown_method_raises(aster570):
    c = aster570
    with pytest.raises(ValueError):
        ebp.damping_power(
            np.array([1e-3]),
            np.array([20.0]),
            np.array([0.15 * c["rts"]]),
            method="not_a_method",
            d=c["d"],
            mu=c["mu"],
            rts=c["rts"],
        )


def test_damping_power_foti_missing_args_raises():
    with pytest.raises(ValueError, match="mu, EImax and x0"):
        ebp.damping_power(
            np.array([1e-3]),
            np.array([20.0]),
            np.array([1e4]),
            method="foti",
        )


def test_damping_power_olla_missing_args_raises():
    with pytest.raises(ValueError, match="mu, EImin, EImax and x0"):
        ebp.damping_power(
            np.array([1e-3]),
            np.array([20.0]),
            np.array([1e4]),
            method="olla",
        )


def test_damping_power_noiseux1992_missing_args_raises():
    with pytest.raises(ValueError, match="d and mu"):
        ebp.damping_power(
            np.array([1e-3]),
            np.array([20.0]),
            np.array([1e4]),
            method="noiseux1992",
        )


def test_damping_power_foti_modes_finite(aster570):
    c = aster570
    ymax = np.array([1.0e-3])
    freq = np.array([20.0])
    tension = np.array([0.15 * c["rts"]])
    x0 = c["alpha"] * tension / c["rts"]
    for mode in ["full", "gross", "micro"]:
        p = ebp.damping_power_foti(ymax, freq, tension, c["mu"], c["EImax"], x0, mode=mode)
        assert np.all(np.isfinite(p))
        assert np.all(p >= 0)


def test_damping_power_foti_invalid_mode_raises(aster570):
    c = aster570
    ymax = np.array([1.0e-3])
    freq = np.array([20.0])
    tension = np.array([0.15 * c["rts"]])
    x0 = c["alpha"] * tension / c["rts"]
    with pytest.raises(ValueError):
        ebp.damping_power_foti(ymax, freq, tension, c["mu"], c["EImax"], x0, mode="bogus")


def test_damping_power_klmn_formula():
    ymax = np.array([1e-3])
    freq = np.array([20.0])
    tension = np.array([1.0e4])
    k = 2.0
    l = 2.0
    m = 5.0
    n = 3.0
    expected = k * ymax**l * freq**m / (tension / 1000.0) ** n
    got = ebp.damping_power_klmn(ymax, freq, tension, k, l, m, n)
    np.testing.assert_allclose(got, expected)


def test_damping_power_increases_with_amplitude(aster570):
    """Damping power must be increasing in ymax for any model."""
    c = aster570
    ymax = np.linspace(1e-4, 5e-3, 12)
    freq = np.full_like(ymax, 20.0)
    tension = np.full_like(ymax, 0.15 * c["rts"])
    x0 = c["alpha"] * tension / c["rts"]
    for method in ["foti", "olla_approx", "claren", "pdm"]:
        p = ebp.damping_power(
            ymax,
            freq,
            tension,
            method=method,
            d=c["d"],
            mu=c["mu"],
            rts=c["rts"],
            EImin=c["EImin"],
            EImax=c["EImax"],
            x0=x0,
        )
        assert np.all(np.diff(p) > 0), f"method {method} not monotonic"


# ---------- ymax balance -----------------------------------------------------


def test_ymax_unknown_wind_method_raises(aster570):
    c = aster570
    with pytest.raises(ValueError):
        ebp.ymax(
            np.array([20.0]),
            np.array([0.15 * c["rts"]]),
            c["D"],
            wind_method="not_a_method",
            damping_method="foti",
            mu=c["mu"],
            EImax=c["EImax"],
            x0=np.array([c["alpha"] * 0.15]),
        )


def test_ymax_unknown_damping_method_raises(aster570):
    c = aster570
    with pytest.raises(ValueError):
        ebp.ymax(
            np.array([20.0]),
            np.array([0.15 * c["rts"]]),
            c["D"],
            wind_method="cigre",
            damping_method="not_a_method",
        )


def test_damping_power_cieren_exact_matches_approx_in_limit(aster570):
    """For small ymax (small curvatures), exact integral and approx should be close."""
    c = aster570
    ymax = 1.0e-5
    freq = 20.0
    tension = 0.15 * c["rts"]
    x0 = c["alpha"] * tension / c["rts"]
    approx = ebp.damping_power_cieren(
        ymax, freq, tension, c["mu"], c["EImin"], c["EImax"], x0, approx=True
    )
    exact = ebp.damping_power_cieren(
        ymax, freq, tension, c["mu"], c["EImin"], c["EImax"], x0, approx=False, N=2001
    )
    np.testing.assert_allclose(exact, approx, rtol=0.1)


def test_ymax_finite_for_reasonable_inputs(aster570):
    c = aster570
    freq = np.linspace(5.0, 40.0, 8)
    tension = np.full_like(freq, 0.15 * c["rts"])
    x0 = c["alpha"] * tension / c["rts"]
    y = ebp.ymax(
        freq,
        tension,
        c["D"],
        wind_method="cigre",
        damping_method="foti",
        mu=c["mu"],
        EImax=c["EImax"],
        x0=x0,
    )
    assert y.shape == freq.shape
    # all positive and bounded by 5*D (the bisection upper bound)
    assert np.all((y > 0) & (y < 5.0 * c["D"]))
