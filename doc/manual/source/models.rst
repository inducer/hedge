Canned PDE Models
=================

.. automodule:: hedge.models
    :members:

Advection
---------

.. module:: hedge.models.advection

.. class:: StrongAdvectionOperator

    .. attribute:: flux_types
    .. automethod:: __init__
    .. automethod:: max_eigenvalue
    .. automethod:: bind

.. class:: WeakAdvectionOperator

    .. attribute:: flux_types
    .. automethod:: __init__
    .. automethod:: max_eigenvalue
    .. automethod:: bind

.. autoclass:: VariableCoefficientAdvectionOperator
    :show-inheritance:

    .. attribute:: flux_types
    .. automethod:: __init__
    .. automethod:: max_eigenvalue
    .. automethod:: bind

:math:`n`-dimensional Calculus
------------------------------

.. module:: hedge.models.nd_calculus

.. autoclass:: GradientOperator
    :show-inheritance:

    .. automethod:: __init__
    .. automethod:: bind

.. autoclass:: DivergenceOperator
    :show-inheritance:

    .. automethod:: __init__
    .. automethod:: bind

Waves
-----

.. module:: hedge.pde

.. autoclass:: StrongWaveOperator
    :show-inheritance:

    .. automethod:: __init__
    .. automethod:: bind

.. autoclass:: VariableVelocityStrongWaveOperator
    :show-inheritance:

    .. automethod:: __init__
    .. automethod:: bind


Electromagnetism
----------------

.. module:: hedge.models.em

.. autoclass:: MaxwellOperator
    :show-inheritance:

    .. automethod:: __init__
    .. automethod:: bind
    .. automethod:: partial_to_eh_subsets
    .. automethod:: split_eh
    .. automethod:: assemble_eh
    .. automethod:: get_eh_subset
    .. automethod:: max_eigenvalue

.. autoclass:: TMMaxwellOperator
    :show-inheritance:

.. autoclass:: TEMaxwellOperator
    :show-inheritance:

.. autoclass:: TE1DMaxwellOperator
    :show-inheritance:

.. autoclass:: SourceFree1DMaxwellOperator
    :show-inheritance:

Electromagnetism with PMLs
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. module:: hedge.models.pml

.. autoclass:: AbarbanelGottliebPMLMaxwellOperator
    :show-inheritance:

    .. automethod:: assemble_ehpq
    .. automethod:: split_ehpq
    .. automethod:: bind
    .. automethod:: coefficients_from_boxes
    .. automethod:: coefficients_from_width

.. autoclass:: AbarbanelGottliebPMLTEMaxwellOperator
    :show-inheritance:

.. autoclass:: AbarbanelGottliebPMLTMMaxwellOperator
    :show-inheritance:

Diffusion
---------

.. module:: hedge.models.diffusion

.. autoclass:: StrongHeatOperator
    :show-inheritance:

    .. automethod:: __init__
    .. automethod:: bind

Poisson
-------

.. module:: hedge.models.poisson

.. autoclass:: WeakPoissonOperator
    :show-inheritance:

    .. automethod:: __init__
    .. automethod:: bind

.. autoclass:: BoundPoissonOperator

    .. automethod:: grad
    .. automethod:: div
    .. automethod:: op
    .. automethod:: prepare_rhs


Gas Dynamics
------------
