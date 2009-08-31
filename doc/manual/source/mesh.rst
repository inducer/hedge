Mesh Data Structure
===================

.. module:: hedge.mesh

Tags
----

.. autoclass:: TAG_NONE
.. autoclass:: TAG_ALL
.. autoclass:: TAG_REALLY_ALL
.. autoclass:: TAG_NO_BOUNDARY
.. autoclass:: TAG_RANK_BOUNDARY(rank)

    .. attribute:: rank

Element Types
-------------

.. class:: Element

    .. attribute:: id
    .. attribute:: vertex_indices
    .. method:: bounding_box(vertices)
    .. method:: centroid(vertices)
    .. attribute:: map
    .. attribute:: inverse_map
    .. attribute:: face_normals
    .. attribute:: face_jacobians

Meshes
------

.. autoclass:: Mesh
    :members:
    :undoc-members:

.. autoclass:: ConformalMesh()
    :show-inheritance:

    .. method:: __init__()
    .. method:: reordered_by
    .. method:: reordered

.. autofunction:: make_conformal_mesh
.. autofunction:: check_bc_coverage

Mesh Generation
===============

1D Meshes
---------

.. autofunction:: make_1d_mesh
.. autofunction:: make_uniform_1d_mesh

2D Meshes
---------

.. autofunction:: make_regular_rect_mesh
.. autofunction:: make_centered_regular_rect_mesh
.. autofunction:: make_regular_square_mesh
.. autofunction:: make_rect_mesh
.. autofunction:: make_regular_rect_mesh
.. autofunction:: make_square_mesh
.. autofunction:: make_disk_mesh

3D Meshes
---------

.. autofunction:: make_ball_mesh
.. autofunction:: make_cylinder_mesh
.. autofunction:: make_box_mesh
