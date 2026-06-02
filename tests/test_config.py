"""Tests for dohl.config data loaders."""

import pandas as pd

from dohl.config import _EpriData, _FatigueData


def test_wind_power_data_loads():
    dat = _EpriData.wind_power()
    assert isinstance(dat, dict)
    assert len(dat) > 0
    sample = next(iter(dat.values()))
    assert "poly" in sample
    assert "range" in sample


def test_damping_power_data_loads():
    dat = _EpriData.damping_power()
    assert isinstance(dat, dict)
    assert len(dat) > 0
    sample = next(iter(dat.values()))
    for key in ("l", "m", "n"):
        assert key in sample


def test_wind_power_data_cached():
    """Second call should return the same dict object (lazy cache)."""
    first = _EpriData.wind_power()
    second = _EpriData.wind_power()
    assert first is second


def test_damping_power_data_cached():
    first = _EpriData.damping_power()
    second = _EpriData.damping_power()
    assert first is second


def test_fatigue_qreg_input_loads():
    df = _FatigueData.qreg_input()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    for col in ("ka", "yad_log", "ybga", "ten_rat", "logN"):
        assert col in df.columns
