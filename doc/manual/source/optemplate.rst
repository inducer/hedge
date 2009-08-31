.. _optemplate:

Operator Specification Language
===============================

Summary
-------

An "operator template" is an expression tree that represents
a DG operator. The following components can be used to build 
one in :mod:`hedge`:

* Scalar constants, may be of type 
  :class:`int`, :class:`float`, :class:`complex`,
  :class:`numpy.number`. These occur directly as part of the
  expression tree. :func:`pymbolic.primitives.is_constant`
  is a predicate to test for constant-ness.

* :class:`numpy.ndarray` with `dtype=object` (called
  "object array" in :mod:`numpy`-speak). These evaluate to the
  same-shape array with each component evaluated.

  Use :func:`hedge.tools.join_fields` to easily create
  object arrays from scalars and other arrays.

* :class:`hedge.optemplate.Field`: A placeholder for a 
  user-supplied field.

  :class:`~hedge.optemplate.Field` instances or 
  :class:`pymbolic.primitives.Subscript` instances involving 
  them are interpreted as scalar DG-discretized fields.

* :class:`hedge.optemplate.ScalarParameter`: A placeholder for a 
  user-supplied scalar value.

* :class:`pymbolic.primitives.Sum`,
  :class:`pymbolic.primitives.Product`,
  :class:`pymbolic.primitives.Quotient`,
  :class:`pymbolic.primitives.Power`: These are created implicitly
  when :class:`~pymbolic.primitives.Expression` objects are combined
  using the `+`, `-`, `*`, `/` and `**` operators.

* :class:`pymbolic.primitives.CommonSubexpression` (CSE for short):
  Prevents double evaluation of identical subexpressions
  when the operator expression tree is walked to evaluate
  the operator.

  Use :func:`make_common_subexpression` to wrap each component
  of an object array in a CSE.

* :class:`pymbolic.primitives.Call`: The function attribute must
  evaluate to a name that was registered by calling
  :meth:`hedge.discretization.Discretization.add_function`.

* Operators may be left-multiplied with other field expressions.
  See :ref:`optemplate-operators` for an overview.

Detailed Documentation
----------------------

Leaf Nodes
^^^^^^^^^^

.. module:: hedge.optemplate

.. autoclass:: Field

.. autofunction:: make_vector_field

.. autoclass:: ScalarParameter

"Wrapper" Nodes
^^^^^^^^^^^^^^^

.. autofunction:: make_common_subexpression

.. autoclass:: PrioritizedSubexpression

.. _optemplate-operators:

Operators
^^^^^^^^^
.. autoclass:: OperatorBinding
.. autoclass:: ElementwiseMaxOperator

Differentiation Operators
"""""""""""""""""""""""""

.. autoclass:: StiffnessOperator
.. autoclass:: StiffnessTOperator
.. autoclass:: DifferentiationOperator
.. autoclass:: MInvSTOperator

.. autofunction:: make_stiffness
.. autofunction:: make_stiffness_t
.. autofunction:: make_nabla
.. autofunction:: make_minv_stiffness_t

Mass Operators
""""""""""""""

.. autoclass:: MassOperator
.. autoclass:: InverseMassOperator

Flux Operators and Related Functionality
""""""""""""""""""""""""""""""""""""""""

.. autoclass:: BoundaryPair

.. autofunction:: get_flux_operator

Boundary-valued operators
"""""""""""""""""""""""""

These operators are only allowed within the *bfield*
argument of :class:`BoundaryPair`, because they 
evaluate to boundary vectors of the given *tag*.

.. autoclass:: BoundarizeOperator
.. autofunction:: make_normal

