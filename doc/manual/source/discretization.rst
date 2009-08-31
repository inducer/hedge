DG Discretizations and related services
=======================================
.. module:: hedge.discretization

.. class:: Discretization

.. method:: Discretization.close()
.. method:: Discretization.add_instrumentation(mgr)
.. method:: Discretization.is_boundary_tag_nonempty(tag)

.. method:: Discretization.all_debug_flags
.. automethod:: Discretization.noninteractive_debug_flags

.. _vector-kinds:

Vector Kinds
------------

When computing on GPU hardware, the :class`Discretization` operates
on fields that are not represented by :mod:`numpy` arrays.
Vector kinds make this explicit. Every discretization is
guaranteed to support converting its preferred vector kind,
given by :attr:`Discretization.compute_kind`, to the kind `"numpy"`.

.. attribute:: Discretization.compute_kind

    The array kind that this discretization can evaluate
    operators on.

.. method:: Discretization.convert_volume(field, kind)
.. method:: Discretization.convert_boundary(field, tag, kind)

Vector Creation
---------------

.. method:: Discretization.volume_empty(shape=(), dtype=None, kind=None)
.. method:: Discretization.volume_zeros(shape=(), dtype=None, kind=None)
.. method:: Discretization.interpolate_volume_function(f, dtype=None, kind=None)

.. method:: Discretization.boundary_empty(tag, shape=(), dtype=None, kind=None)
.. method:: Discretization.boundary_zeros(tag, shape=(), dtype=None, kind=None)
.. method:: Discretization.interpolate_boundary_function(f, tag, dtype=None, kind=None)

.. method:: Discretization.boundary_normals(tag, dtype=None, kind=None)

Vector Conversion
-----------------

.. method:: Discretization.volumize_boundary_field(bfield, tag, kind=None)
.. method:: Discretization.boundarize_boundary_field(bfield, tag, kind=None)

Vector Reductions
-----------------

.. method:: Discretization.nodewise_dot_product(a, b)
.. method:: Discretization.mesh_volume()
.. method:: Discretization.integral(volume_vector)
.. method:: Discretization.norm(volume_vector, p=2)
.. method:: Discretization.inner_product(a, b)

Miscellanea
-----------

.. method:: Discretization.dt_factor(max_system_ev, stepper_class, *stepper_args)
.. method:: Discretization.get_point_evaluator(point)

Compilation of :ref:`operator templates <optemplate>`
-----------------------------------------------------

.. method:: Discretization.compile(optemplate, post_bind_mapper=lambda x: x)
.. method:: Discretization.add_function(name, func)

Projection between :class:`Discretization` instances
----------------------------------------------------

.. class:: Projector(from_discr, to_discr)

    .. method:: __call__(self, from_vec)

Filtering
---------

.. autoclass:: ExponentialFilterResponseFunction
    :members: __init__

.. class:: Filter

    .. automethod:: __init__
