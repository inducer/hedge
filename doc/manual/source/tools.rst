Useful Helper Routines
======================

.. module:: hedge.tools

Operator Subsetting
-------------------

.. module:: hedge.tools.indexing

.. autofunction:: count_subset
.. autofunction:: full_to_subset_indices
.. autofunction:: full_to_all_subset_indices
.. autofunction:: partial_to_all_subset_indices

Mathematical Helpers
--------------------

.. module:: hedge.tools.math

.. autoclass:: SubsettableCrossProduct
.. autofunction:: normalize
.. autofunction:: sign
.. autofunction:: orthonormalize
.. autofunction:: permutation_matrix
.. autofunction:: leftsolve
.. autofunction:: unit_vector
.. autofunction:: relative_error
.. autofunction:: get_spherical_coord

Convergence Tests
-----------------

.. module:: hedge.tools.convergence

.. autofunction:: estimate_order_of_convergence
.. autoclass:: EOCRecorder
