# Damage in Overhead Lines

[![ci](https://github.com/rte-france/pallas-dohl/actions/workflows/ci.yml/badge.svg)](https://github.com/rte-france/pallas-dohl/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MPL--2.0-green)

_**dohl**_ is a python package to evaluate fatigue damage in overhead transmission lines from aeolian vibration. It implements established methods from the EPRI "Orange Book" and CIGRE (safe-border-line, fictive stress, energy balance principle, Stockbridge dampers, Weibull-integrated lifespan).

## Table of contents

* [Installation](#installation)
* [Basic usage](#basic-usage)
* [Testing](#testing)
* [Building the documentation](#building-the-documentation)

## Installation

Install from GitHub:

```shell
python -m pip install git+https://github.com/rte-france/pallas-dohl
```

Or from a local clone:

```shell
python -m pip install .
```

Two helper scripts (`create_uvenv` for linux, `create_uvenv.ps1` for windows) at project root build a development environment with the `uv` package manager.

## Basic usage

Estimate annual fatigue damage of an Aster 570 conductor across a range of wind speeds:

```python
import numpy as np
from dohl.lifespan import damage_year

# Aster 570 conductor data
mu, D, d = 1.571, 3.105e-2, 3.45e-3        # lineic mass (kg/m), external diameter (m), last-layer strand diameter (m)
rts = 1.853e5                              # rated tensile strength (N)
EImin, EImax = 28.28, 2155.0               # bending stiffness range (N.m**2)
young = 6.8e10                             # Young's modulus (Pa)
alpha = 0.140652                           # tension-to-curvature coefficient

# Wind speeds (m/s) and span tension (arbitrary 15% of RTS)
u = np.linspace(0.5, 5.0, 10)
H = 0.15 * rts * np.ones_like(u)

ymax, yb, sigma, dmg = damage_year(
    u, H, D, d, mu, young, EImin, EImax,
    multilayer=False,
    rts=rts,
    x0=alpha * H / rts,
)

print("annual damage per wind speed:", dmg)
```

## Testing

The test suite is in `tests/` and runs with pytest:

```shell
uv sync --group test
uv run pytest --cov=dohl
```

Current state (61 tests, 1.0s runtime):

| Module                                          | Coverage |
| ----------------------------------------------- | -------- |
| `dohl/stockbridge.py`                           | 100%     |
| `dohl/config.py`                                | 97%      |
| `dohl/ebp.py`                                   | 89%      |
| `dohl/lifespan.py`                              | 86%      |
| `dohl/damage.py`                                | 75%      |
| **Total** (excluding `_fatigue_model_input.py`) | **~88%** |

The internal `_fatigue_model_input.py` depends on the `strdcable` package which is not a runtime dependency, so it is excluded from coverage targets.

Continuous integration runs the suite on Ubuntu, Windows and macOS for Python 3.10, 3.11 and 3.12, plus a lint job (`ruff check`, `ruff format --check`, `mypy` advisory). See `.github/workflows/ci.yml`.

## Building the documentation

Install the doc dependency group, then build:

```shell
uv sync --group doc
cd doc
make html
```

The documentation can then be accessed from `doc/build/html/index.html`.

## Acknowledgements

_**dohl**_ is developed by [Mews Labs](https://www.mews-labs.com) (formerly _Eurobios_) and supported by [Rte-R&D](https://www.rte-france.com/)
_via_ the OLLA & PALLAS projects (see [ResearchGate](https://www.researchgate.net/project/OLLA-overhead-lines-lifespan-assessment)).
