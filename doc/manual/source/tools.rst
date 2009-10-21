Useful Helper Routines
======================

.. module:: hedge.tools

Object Arrays
-------------

.. autofunction:: is_obj_array
.. autofunction:: to_obj_array
.. autofunction:: join_fields
.. autofunction:: log_shape

Operator Expression Helpers
---------------------------

.. autofunction:: ptwise_mul
.. autofunction:: ptwise_dot
.. autofunction:: make_common_subexpression

Operator Subsetting
-------------------

.. autofunction:: count_subset
.. autofunction:: full_to_subset_indices
.. autofunction:: full_to_all_subset_indices
.. autofunction:: partial_to_all_subset_indices

Mathematical Helpers
--------------------

.. autoclass:: SubsettableCrossProduct
.. autofunction:: normalize
.. autofunction:: sign
.. autofunction:: orthonormalize
.. autofunction:: permutation_matrix
.. autofunction:: leftsolve
.. autofunction:: unit_vector
.. autofunction:: relative_error
.. autoclass:: AffineMap

Convergence Tests
-----------------

.. autofunction:: estimate_order_of_convergence
.. autoclass:: EOCRecorder

Mesh Helpers
------------

.. autofunction:: cuthill_mckee

Flux Helpers
------------

.. autofunction:: make_lax_friedrichs_flux

Miscellaneous
-------------

.. autofunction:: get_spherical_coord
