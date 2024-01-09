# Damage in Overhead Lines

_**dohl**_ is a python package proposed to evaluate damage in overhead lines.

## Table of contents

* [Installation](#installation)
* [Basic usage](#basic-usage)
* [Building the documentation](#building-the-documentation)

## Installation

### Using pip

To install the package using pip, execute the following command :

```shell script
python -m pip install git+https://gitlab.eurobios.com/rte/dohl.git
```

Or by downloading the package and executing in the repo :

```shell script
python -m pip install .
```

A version of python 3.8 or above is required you can create such environment using conda :

```shell script
conda create -n dohl_env python=3.8
```

To activate your environement :

```shell script
conda activate dohl_env
```

## Basic usage

### Calculate amplitude of vibrations based on CIGRE model

```python script
import numpy as np
from dohl import damage as dmg


def _aster570_data():
    m = 1.571
    D = 3.105E-02
    rts = 1.853E+05
    EImin = 2.828E+01
    EImax = 2.155E+03
    alpha = 0.140652249164
    young = 6.8E+10
    d = 3.45E-03
    ml = False
    return m, D, rts, EImin, EImax, alpha, young, d, ml


# Get you conductor data
m, d, rts, EImin, EImax, alpha, _, _, _ = _aster570_data()

# Define your abscisse where to evaluate amplitude
x = np.linspace(0, 0.2, 21)

# Define a frequencie of vibration
f = np.array([10])

# Calculate your tension using rts and a ratio
ratio = np.array([0.15])
t = ratio * rts

# Calculations based on a beam vibration model, from CIGRE with minimal tangential bending stiffness of the cable (EImin)
a = dmg._ampl_cigre(d, f, t, m, EImin, x)
```

### Calculate damage

```python script
import numpy as np
from dohl import damage as dmg
from dohl.lifespan import damage_year


def _aster570_data():
    m = 1.571
    D = 3.105E-02
    rts = 1.853E+05
    EImin = 2.828E+01
    EImax = 2.155E+03
    alpha = 0.140652249164
    young = 6.8E+10
    d = 3.45E-03
    ml = False
    return m, D, rts, EImin, EImax, alpha, young, d, ml


# Get your conductor data
mu, D, rts, EImin, EImax, alpha, young, d, ml = _aster570_data()

# Generate your wind
u = np.linspace(0, 5, 21)[1:]

# Calculate your tension using rts and a ratio
ratio = np.array([0.15])
h = ratio * rts

# Set your damping method
dm = 'olla_approx'

# Set your fictive stress method
fsm = 'fymax'

# Set bending stiffness to use
fsba = 'min'

# Calculate your curvature
x0 = alpha * h / rts

# Calculate your damage
ym, sg, ls = damage_year(u, h, D, d, mu, young, EImin, EImax,
                         wind_method='cigre', damping_method=dm,
                         fictive_stress_method=fsm, fsba=fsba,
                         multilayer=ml, rts=rts, x0=x0)

```

## Building the documentation

Make sure you have sphinx and the Readthedocs theme installed:

```shell script
pip install sphinx
pip install sphinx_rtd_theme
```

Go to the doc directory and build:

```shell script
cd doc
make html
```

The documentation can then be accessed from `doc/build/html/index.html`.

## Acknowledgements

_**dohl**_ is developed by [Eurobios](http://www.eurobios.com/) and supported by [Rte-R&D](https://www.rte-france.com/)
_via_ the OLLA project (
see [ResearchGate](https://www.researchgate.net/project/OLLA-overhead-lines-lifespan-assessment)).
