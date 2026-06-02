# -*- coding: utf-8 -*-
"""Shared fixtures for dohl tests.

The Aster570 conductor parameters used throughout the test suite mirror
the example values from the project README.
"""

import numpy as np
import pytest


@pytest.fixture(scope="session")
def aster570():
    """Aster570 conductor data (lineic mass in kg/m, lengths in m, stresses in Pa)."""
    return dict(
        mu=1.571,
        D=3.105e-02,
        rts=1.853e05,
        EImin=2.828e01,
        EImax=2.155e03,
        alpha=0.140652249164,
        young=6.8e10,
        d=3.45e-03,
        multilayer=False,
    )


@pytest.fixture(scope="session")
def freq_grid():
    return np.linspace(5.0, 50.0, 10)


@pytest.fixture(scope="session")
def wind_grid():
    return np.linspace(0.5, 5.0, 10)
