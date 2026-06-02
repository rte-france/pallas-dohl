"""Tests for dohl.lifespan."""

import numpy as np
import pytest

from dohl.lifespan import (
    damage_history,
    damage_year,
    damage_year_weibull,
    damage_year_weibull_array,
)


def _common_kwargs(c):
    tension = np.full(5, 0.15 * c["rts"])
    return dict(
        wind_speed=np.linspace(1.0, 5.0, 5),
        tension=tension,
        external_diameter=c["D"],
        last_strand_diameter=c["d"],
        lineic_mass=c["mu"],
        young=c["young"],
        EImin=c["EImin"],
        EImax=c["EImax"],
        multilayer=c["multilayer"],
        wind_method="cigre",
        damping_method="olla_approx",
        fictive_stress_method="fymax",
        fsba="min",
        rts=c["rts"],
        x0=c["alpha"] * tension / c["rts"],
    )


def test_damage_year_returns_finite_positive(aster570):
    """Smoke test: full pipeline produces finite damage."""
    kw = _common_kwargs(aster570)
    ymax, yb, sigma, dmg = damage_year(**kw)
    assert ymax.shape == kw["wind_speed"].shape
    assert sigma.shape == kw["wind_speed"].shape
    assert dmg.shape == kw["wind_speed"].shape
    assert np.all(np.isfinite(dmg))
    assert np.all(dmg >= 0)


def test_damage_year_fymax_yb_is_none(aster570):
    kw = _common_kwargs(aster570)
    _, yb, _, _ = damage_year(**kw)
    assert yb is None


def test_damage_year_bapfs_yb_is_array(aster570):
    kw = _common_kwargs(aster570)
    kw["fictive_stress_method"] = "ba_pfs"
    _, yb, _, _ = damage_year(**kw)
    assert isinstance(yb, np.ndarray)
    assert yb.shape == kw["wind_speed"].shape


def test_damage_year_fsba_choices_differ(aster570):
    """The three bending-stiffness choices should produce different damage."""
    base = _common_kwargs(aster570)
    out = {}
    for fsba in ["min", "avg", "max"]:
        kw = dict(base)
        kw["fsba"] = fsba
        _, _, _, dmg = damage_year(**kw)
        out[fsba] = dmg
    # min and max should differ wherever damage is non-zero
    assert not np.allclose(out["min"], out["max"])


def test_damage_year_invalid_fsba_raises(aster570):
    kw = _common_kwargs(aster570)
    kw["fsba"] = "not_a_choice"
    with pytest.raises(ValueError):
        damage_year(**kw)


def test_damage_year_weibull_positive_scalar(aster570):
    """Integrated damage over a Weibull wind distribution is a non-negative scalar."""
    c = aster570
    dmg = damage_year_weibull(
        shape=1.5,
        scale=4.0,
        tension=0.15 * c["rts"],
        external_diameter=c["D"],
        last_strand_diameter=c["d"],
        lineic_mass=c["mu"],
        young=c["young"],
        EImin=c["EImin"],
        EImax=c["EImax"],
        multilayer=c["multilayer"],
        wind_method="cigre",
        damping_method="olla_approx",
        fictive_stress_method="fymax",
        fsba="min",
        rts=c["rts"],
        x0=c["alpha"] * 0.15,
    )
    assert np.isscalar(dmg) or (hasattr(dmg, "shape") and dmg.shape == ())
    assert np.isfinite(dmg)
    assert dmg >= 0


def test_damage_year_weibull_array_single_element_matches_scalar(aster570):
    """damage_year_weibull_array with n=1 should delegate to damage_year_weibull."""
    c = aster570
    kwargs = dict(
        tension=0.15 * c["rts"],
        external_diameter=c["D"],
        last_strand_diameter=c["d"],
        lineic_mass=c["mu"],
        young=c["young"],
        EImin=c["EImin"],
        EImax=c["EImax"],
        multilayer=c["multilayer"],
        wind_method="cigre",
        damping_method="olla_approx",
        fictive_stress_method="fymax",
        fsba="min",
        rts=c["rts"],
        x0=c["alpha"] * 0.15,
    )
    scalar = damage_year_weibull(shape=1.5, scale=4.0, **kwargs)
    arr = damage_year_weibull_array(shape=1.5, scale=4.0, **kwargs)
    np.testing.assert_allclose(arr, scalar)


def test_damage_history_integrates_to_finite(aster570):
    """damage_history integrates damage rate over a time vector."""
    c = aster570
    n = 4
    time = np.linspace(0.0, 3.0 * 3600.0, n)
    u = np.full(n, 3.0)
    tension = np.full(n, 0.15 * c["rts"])
    out = damage_history(
        time=time,
        wind_speed=u,
        tension=tension,
        external_diameter=c["D"],
        last_strand_diameter=c["d"],
        lineic_mass=c["mu"],
        young=c["young"],
        EImin=c["EImin"],
        EImax=c["EImax"],
        multilayer=c["multilayer"],
        wind_method="cigre",
        damping_method="olla_approx",
        fictive_stress_method="fymax",
        fsba="min",
        rts=c["rts"],
        beta=c["alpha"] / c["rts"],
    )
    assert np.isfinite(out)
    assert out >= 0


def test_damage_year_weibull_array_small(aster570):
    """Array variant with several conductors returns an array of same length."""
    c = aster570
    n = 3
    tension = np.full(n, 0.15 * c["rts"])
    dmg = damage_year_weibull_array(
        shape=np.full(n, 1.5),
        scale=np.full(n, 4.0),
        tension=tension,
        external_diameter=np.full(n, c["D"]),
        last_strand_diameter=np.full(n, c["d"]),
        lineic_mass=np.full(n, c["mu"]),
        young=np.full(n, c["young"]),
        EImin=np.full(n, c["EImin"]),
        EImax=np.full(n, c["EImax"]),
        multilayer=np.full(n, c["multilayer"]),
        wind_method="cigre",
        damping_method="olla_approx",
        fictive_stress_method="fymax",
        fsba="min",
        rts=np.full(n, c["rts"]),
        x0=c["alpha"] * tension / c["rts"],
    )
    assert dmg.shape == (n,)
    assert np.all(np.isfinite(dmg))
    assert np.all(dmg >= 0)
