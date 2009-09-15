Timestepping
============

Fourth-order LSERK
------------------

.. module:: hedge.timestep.rk4

.. autoclass:: RK4TimeStepper
    :members: __init__, __call__
    :undoc-members:

Adams-Bashforth
---------------

.. module:: hedge.timestep.ab

.. autofunction:: make_generic_ab_coefficients
.. autofunction:: make_ab_coefficients

.. autoclass:: AdamsBashforthTimeStepper
    :members: __init__, __call__
    :undoc-members:

Multirate Adams-Bashforth
-------------------------

.. module:: hedge.timestep.multirate_ab

.. autoclass:: TwoRateAdamsBashforthTimeStepper
    :members: __init__, __call__
    :undoc-members:
