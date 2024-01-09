Install
=======

If you have access to eurobios gitlab, just type ``pip install -e
git+https://gitlab.eurobios.com/rte/dohl.git``
in a terminal; you will be asked your login credentials. 

If you got this packages from the sources, type ``pip install -e .`` at
the root of the repository.

A version of python 3.8 or above is required you can create such environement using conda :

``conda create -n dohl_env python=3.8``

To activate your environement :

``conda activate dohl_env``


Simple usage
============

Calculate amplitude of vibrations based on CIGRE model 

.. code-block:: python

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
