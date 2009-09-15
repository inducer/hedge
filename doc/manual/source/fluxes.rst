.. _fluxspec:

Flux Specification Language
===========================

.. module:: hedge.flux

Summary
-------

An "flux expression" is an expression tree that represents
a numerical DG flux. Flux expressions may be passed to
:func:`hedge.optemplate.get_flux_operator` to integrate
with the :ref:`optemplate`.  The following components can be 
used to build flux expressions in :mod:`hedge`:

* Scalar constants, may be of type
  :class:`int`, :class:`float`, :class:`complex`,
  :class:`numpy.number`. These occur directly as part of the
  expression tree. :func:`pymbolic.primitives.is_constant`
  is a predicate to test for constant-ness.

* :class:`numpy.ndarray` with *dtype=:class:`object`* (called
  "object array" in :mod:`numpy`-speak). When processed by
  :func:`hedge.optemplate.get_flux_operator`, vector fluxes
  result in an object array of scalar fluxes.

  Use :func:`hedge.tools.join_fields` to easily create
  object arrays from scalars and other arrays.

* :class:`Normal` represents one axial (i.e. *x*, *y* or *z*)
  component of the face normal for which the flux is to be
  computed.

  Use :func:`make_normal` to generate an object array of
  normal components for more natural, vectorial use of the
  normal.

* :class:`FieldComponent` allows access to the DG-discretized
  fields for which the flux is to be computed. If the
  flux operator is bound to a scalar field, it refers to it
  by the index 0. If it is bound to a vector field, indices
  are subscripts into this vector field.

  Further, :class:`FieldComponent` lets you specify which
  side of an element interface the placeholder stands by
  its `is_interior` parameter.

  .. seealso:: :ref:`flux-simp-placeholder`.

* :class:`PenaltyTerm`: Evaluates to the conventional penalty
  term :math:`(N^2/h)^k`, where the power :math:`k` is adjustable
  by the user, :math:`N` represents the order of the discretization,
  and :math:`h` represents the surface Jacobian.

* :class:`pymbolic.primitives.Sum`,
  :class:`pymbolic.primitives.Product`,
  :class:`pymbolic.primitives.Quotient`,
  :class:`pymbolic.primitives.Power`: These are created implicitly
  when :class:`~pymbolic.primitives.Expression` objects are combined
  using the `+`, `-`, `*`, `/` and `**` operators.
  These are all interpreted in a node-by-node fashion.

* :class:`pymbolic.primitives.IfPositive` offers a simple way
  to build conditionals and is interpreted in a node-by-node
  fashion.

* :class:`pymbolic.primitives.CommonSubexpression` (CSE for short):
  Prevents double evaluation of identical subexpressions
  when the flux expression tree is walked to evaluate
  the operator.

  Use :func:`make_common_subexpression` to wrap each component
  of an object array in a CSE.

* :class:`hedge.flux.FluxScalarParameter`: A placeholder for a
  user-supplied scalar value, drawn from the same namespace
  as :class:`hedge.optemplate.ScalarParameter`.

* :class:`pymbolic.primitives.Call`: The *function* attribute
  must evaluate to one of a number of predefined 
  :class:`pymbolic.primitive.FunctionSymbol` instances.

  .. seealso:: :ref:`flux-function-symbols`.

.. _flux-simp-placeholder:

Simplifying Flux Notation with Placeholders
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Suppose the state vector :math:`[u,v_x,v_y]` is contained
in the variable *q* and the flux to be calculated is

.. math::

    \{v\} \cdot \hat n - \frac 12 (u^- - u^+),

using the convetional flux notation (see Hesthaven/Warburton,
"Nodal Discontinuous Galerkin Methods").

Then this can be expressed as::

    flux = 0.5* (
      (FieldComponent(1, True) + FieldComponent(1, False)) * Normal(0)
      + (FieldComponent(2, True)+ FieldComponent(2, False)) * Normal(1)
      - (FieldComponent(0, True) - FieldComponent(0, False))
    )

and the flux is then bound to the components of *q* using::

    get_flux_operator(flux)*q

This is however rather cumbersome.
We may use :class:`FluxVectorPlaceholder` to simplify the notation.

We begin by introducing flux placeholders, specifying
that our state vector contains three components, as above, and
extract slices to get symbolic, vectorial representations of u, v,
and the normal::

    q_ph = FluxVectorPlaceholder(3)
    u = q_ph[0]
    v = q_ph[1:]
    n = make_normal(2) # normal has two dimensions

Then the flux simplifies to::

    flux = numpy.dot(v.avg, n) - 0.5*(u.int - u.ext)

The resulting flux expression will be the same as above, but
notational effort is greatly reduced.

Detailed Documentation
----------------------

.. autoclass:: FieldComponent

.. autoclass:: FluxScalarParameter

.. autoclass:: Normal

.. autofunction:: make_normal

.. autoclass:: PenaltyTerm

.. autoclass:: FluxZeroPlaceholder
    :members: int, ext, avg
    :undoc-members:

.. autoclass:: FluxScalarPlaceholder
    :members: int, ext, avg
    :undoc-members:

.. autoclass:: FluxVectorPlaceholder
    :members: __len__, __getitem__, int, ext, avg
    :undoc-members:

.. _flux-function-symbols:

Predefined Flux Function Symbols
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. data:: flux_abs

    One argument, returns its absolute value.

.. data:: flux_min

    Two arguments, returns the smaller of the two.

.. data:: flux_max

    Two arguments, returns the larger of the two.
