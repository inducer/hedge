"""Mesh generation for simple geometries."""

from __future__ import division

__copyright__ = "Copyright (C) 2009 Andreas Kloeckner"

__license__ = """
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see U{http://www.gnu.org/licenses/}.
"""




import numpy




class MeshPyFaceMarkerLookup:
    def __init__(self, meshpy_output):
        self.fvi2fm = dict((frozenset(fvi), marker) for fvi, marker in
                zip(meshpy_output.facets, meshpy_output.facet_markers))

    def __call__(self, fvi):
        return self.fvi2fm[frozenset(fvi)]




def make_1d_mesh(points, left_tag=None, right_tag=None, periodic=False,
        boundary_tagger=None, element_tagger=None):
    def force_array(pt):
        if not isinstance(pt, numpy.ndarray):
            return numpy.array([pt])
        else:
            return pt

    def my_boundary_tagger(fvi, el, fn, all_v):
        if el.face_normals[fn][0] < 0:
            return [left_tag]
        else:
            return [right_tag]

    kwargs = {}
    if element_tagger is not None:
        kwargs["element_tagger"] = element_tagger

    if periodic:
        left_tag = "x_minus"
        right_tag = "x_plus"
        from hedge.mesh import make_conformal_mesh
        return make_conformal_mesh(
                [force_array(pt) for pt in points],
                [(i,i+1) for i in range(len(points)-1)],
                periodicity=[("x_minus", "x_plus")],
                boundary_tagger=my_boundary_tagger,
                **kwargs)
    else:
        from hedge.mesh import make_conformal_mesh
        return make_conformal_mesh(
                [force_array(pt) for pt in points],
                [(i,i+1) for i in range(len(points)-1)],
                boundary_tagger=boundary_tagger or my_boundary_tagger,
                **kwargs)





def make_uniform_1d_mesh(a, b, el_count, left_tag=None, right_tag=None, periodic=False,
        boundary_tagger=None):
    dx = (b-a)/el_count
    return make_1d_mesh(
            [a+dx*i for i in range(el_count+1)],
            left_tag=left_tag,
            right_tag=right_tag,
            periodic=periodic,
            boundary_tagger=boundary_tagger)




def make_single_element_mesh(a=-0.5, b=0.5,
        boundary_tagger=(lambda vertices, face_indices: [])):
    n = 2
    node_dict = {}
    points = []
    points_1d = numpy.linspace(a, b, n)
    for j in range(n):
        for i in range(n):
            node_dict[i,j] = len(points)
            points.append(numpy.array([points_1d[i], points_1d[j]]))

    elements = [(
                node_dict[1,1],
                node_dict[0,1],
                node_dict[1,0],
                )]

    boundary_faces = [(3,1), (1,2), (2,3)]

    boundary_tags = dict(
            (frozenset(seg),
                boundary_tagger(points, seg))
                for seg in  boundary_faces)

    from hedge.mesh import make_conformal_mesh
    return make_conformal_mesh(
            points,
            elements,
            boundary_tags)




def make_regular_rect_mesh(a=(0,0), b=(1,1), n=(5,5), periodicity=None,
        boundary_tagger=(lambda fvi, el, fn, all_v: [])):
    """Create a semi-structured rectangular mesh.

    :param a: the lower left hand point of the rectangle
    :param b: the upper right hand point of the rectangle
    :param n: a tuple of integers indicating the total number of points
      on [a,b].
    :param periodicity: either None, or a tuple of bools specifying whether
      the mesh is to be periodic in x and y.
    """
    if min(n) < 2:
        raise ValueError("need at least two points in each direction")

    node_dict = {}
    points = []
    points_1d = [numpy.linspace(a_i, b_i, n_i)
            for a_i, b_i, n_i in zip(a, b, n)]

    for j in range(n[1]):
        for i in range(n[0]):
            node_dict[i,j] = len(points)
            points.append(numpy.array([points_1d[0][i], points_1d[1][j]]))

    elements = []

    if periodicity is None:
        periodicity = (False, False)

    axes = ["x", "y"]
    mesh_periodicity = []
    periodic_tags = set()
    for i, axis in enumerate(axes):
        if periodicity[i]:
            minus_tag = "minus_"+axis
            plus_tag = "plus_"+axis
            mesh_periodicity.append((minus_tag, plus_tag))
            periodic_tags.add(minus_tag)
            periodic_tags.add(plus_tag)
        else:
            mesh_periodicity.append(None)

    fvi2fm = {}

    for i in range(n[0]-1):
        for j in range(n[1]-1):

            # c--d
            # |  |
            # a--b

            a = node_dict[i,j]
            b = node_dict[i+1,j]
            c = node_dict[i,j+1]
            d = node_dict[i+1,j+1]

            elements.append((a,b,c))
            elements.append((d,c,b))

            if i == 0: fvi2fm[frozenset((a,c))] = "minus_x"
            if i == n[0]-2: fvi2fm[frozenset((b,d))] = "plus_x"
            if j == 0: fvi2fm[frozenset((a,b))] = "minus_y"
            if j == n[1]-2: fvi2fm[frozenset((c,d))] = "plus_y"

    def wrapped_boundary_tagger(fvi, el, fn, all_v):
        btag = fvi2fm[frozenset(fvi)]
        if btag in periodic_tags:
            return [btag]
        else:
            return [btag] + boundary_tagger(fvi, el, fn, all_v)

    from hedge.mesh import make_conformal_mesh
    return make_conformal_mesh(points, elements, wrapped_boundary_tagger,
            periodicity=mesh_periodicity)




def make_centered_regular_rect_mesh(a=(0,0), b=(1,1), n=(5,5), periodicity=None,
        post_refine_factor=1, boundary_tagger=(lambda fvi, el, fn, all_v: [])):
    """Create a semi-structured rectangular mesh.

    :param a: the lower left hand point of the rectangle
    :param b: the upper right hand point of the rectangle
    :param n: a tuple of integers indicating the total number of points
      on [a,b].
    :param periodicity: either None, or a tuple of bools specifying whether
      the mesh is to be periodic in x and y.
    """
    if min(n) < 2:
        raise ValueError("need at least two points in each direction")

    node_dict = {}
    centered_node_dict = {}
    points = []
    points_1d = [numpy.linspace(a_i, b_i, n_i)
            for a_i, b_i, n_i in zip(a, b, n)]
    dx = (numpy.array(b, dtype=numpy.float64)
            - numpy.array(a, dtype=numpy.float64)) / (numpy.array(n)-1)
    half_dx = dx/2

    for j in range(n[1]):
        for i in range(n[0]):
            node_dict[i,j] = len(points)
            points.append(numpy.array([points_1d[0][i], points_1d[1][j]]))

    centered_points = []
    for j in range(n[1]-1):
        for i in range(n[0]-1):
            centered_node_dict[i,j] = len(points)
            points.append(numpy.array([points_1d[0][i], points_1d[1][j]]) + half_dx)

    elements = []

    if periodicity is None:
        periodicity = (False, False)

    axes = ["x", "y"]
    mesh_periodicity = []
    periodic_tags = set()
    for i, axis in enumerate(axes):
        if periodicity[i]:
            minus_tag = "minus_"+axis
            plus_tag = "plus_"+axis
            mesh_periodicity.append((minus_tag, plus_tag))
            periodic_tags.add(minus_tag)
            periodic_tags.add(plus_tag)
        else:
            mesh_periodicity.append(None)

    fvi2fm = {}

    for i in range(n[0]-1):
        for j in range(n[1]-1):

            # c---d
            # |\ /|
            # | m |
            # |/ \|
            # a---b

            a = node_dict[i,j]
            b = node_dict[i+1,j]
            c = node_dict[i,j+1]
            d = node_dict[i+1,j+1]

            m = centered_node_dict[i,j]

            elements.append((a,b,m))
            elements.append((b,d,m))
            elements.append((d,c,m))
            elements.append((c,a,m))

            if i == 0: fvi2fm[frozenset((a,c))] = "minus_x"
            if i == n[0]-2: fvi2fm[frozenset((b,d))] = "plus_x"
            if j == 0: fvi2fm[frozenset((a,b))] = "minus_y"
            if j == n[1]-2: fvi2fm[frozenset((c,d))] = "plus_y"

    def wrapped_boundary_tagger(fvi, el, fn, all_v):
        btag = fvi2fm[frozenset(fvi)]
        if btag in periodic_tags:
            return [btag]
        else:
            return [btag] + boundary_tagger(fvi, el, fn, all_v)

    if post_refine_factor > 1:
        from meshpy.tools import uniform_refine_triangles
        points, elements, of2nf = uniform_refine_triangles(
                points, elements, post_refine_factor)
        old_fvi2fm = fvi2fm
        fvi2fm = {}

        for fvi, fm in old_fvi2fm.iteritems():
            for new_fvi in of2nf[fvi]:
                fvi2fm[frozenset(new_fvi)] = fm

    from hedge.mesh import make_conformal_mesh
    return make_conformal_mesh(points, elements, wrapped_boundary_tagger,
            periodicity=mesh_periodicity)





def make_regular_square_mesh(a=-0.5, b=0.5, n=5, periodicity=None,
        boundary_tagger=(lambda fvi, el, fn, all_v: [])):
    """Create a semi-structured square mesh.

    :param a: the lower x and y coordinate of the square
    :param b: the upper x and y coordinate of the square
    :param n: integer indicating the total number of points on [a,b].
    :param periodicity: either None, or a tuple of bools specifying whether
      the mesh is to be periodic in x and y.
    """
    return make_regular_rect_mesh(
            (a,a), (b,b), (n,n), periodicity, boundary_tagger)




def finish_2d_rect_mesh(points, facets, facet_markers, marker2tag, refine_func,
        periodicity, boundary_tagger):
    """Semi-internal bottom-half routine for generation of rectangular 2D meshes."""
    import meshpy.triangle as triangle

    mesh_info = triangle.MeshInfo()
    mesh_info.set_points(points)
    mesh_info.set_facets(facets, facet_markers)

    #triangle.write_gnuplot_mesh("mesh.dat", mesh_info, True)

    if periodicity is None:
        periodicity = (False, False)

    axes = ["x", "y"]
    mesh_periodicity = []
    periodic_tags = set()
    for i, axis in enumerate(axes):
        if periodicity[i]:
            minus_tag = "minus_"+axis
            plus_tag = "plus_"+axis
            mesh_periodicity.append((minus_tag, plus_tag))
            periodic_tags.add(minus_tag)
            periodic_tags.add(plus_tag)
        else:
            mesh_periodicity.append(None)

    generated_mesh = triangle.build(mesh_info,
            refinement_func=refine_func,
            allow_boundary_steiner=not (periodicity[0] or periodicity[1]))

    fmlookup = MeshPyFaceMarkerLookup(generated_mesh)

    def wrapped_boundary_tagger(fvi, el, fn, all_v):
        btag = marker2tag[fmlookup(fvi)]
        if btag in periodic_tags:
            return [btag]
        else:
            return [btag] + boundary_tagger(fvi, el, fn, all_v)

    from hedge.mesh import make_conformal_mesh
    return make_conformal_mesh(
            generated_mesh.points,
            generated_mesh.elements,
            wrapped_boundary_tagger,
            periodicity=mesh_periodicity)




def _round_trip_connect(start, end):
    for i in range(start, end):
        yield i, i+1
    yield end, start




def make_rect_mesh(a=(0,0), b=(1,1), max_area=None,
        boundary_tagger=(lambda fvi, el, fn, all_v: []),
        periodicity=None, subdivisions=None,
        refine_func=None):
    """Create an unstructured rectangular mesh.

    :param a: the lower left hand point of the rectangle
    :param b: the upper right hand point of the rectangle
    :param max_area: maximum area of each triangle.
    :param periodicity: either None, or a tuple of bools specifying whether
      the mesh is to be periodic in x and y.
    :param subdivisions: If not *None*, this is a 2-tuple specifying
      the number of facet subdivisions in X and Y.
    :param refine_func: A refinement function as taken by
      :func:`meshpy.triangle.build`.
    """
    import meshpy.triangle as triangle

    if max_area is not None:
        if refine_func is not None:
            raise ValueError, "cannot specify both refine_func and max_area"
        def refine_func(vertices, area):
            return area > max_area

    marker2tag = {
            1: "minus_x",
            2: "minus_y",
            3: "plus_x",
            4: "plus_y",
            }

    points = [a, (b[0],a[1]), b, (a[0],b[1])]
    facets = list(_round_trip_connect(0, 3))
    facet_markers = [2,3,4,1]

    if subdivisions is not None:
        points, facets, facet_markers = triangle.subdivide_facets(
                [subdivisions[0], subdivisions[1],
                    subdivisions[0], subdivisions[1]],
                points, facets, facet_markers)

    return finish_2d_rect_mesh(points, facets, facet_markers, marker2tag,
            refine_func, periodicity, boundary_tagger)





def make_rect_mesh_with_corner(a=(0,0), b=(1,1), max_area=None,
        boundary_tagger=(lambda fvi, el, fn, all_v: []),
        corner_fraction=(0.3, 0.3),
        refine_func=None):
    """Create an unstructured rectangular mesh with a reentrant
    corner at (-x, -y).

    :param a: the lower left hand point of the rectangle
    :param b: the upper right hand point of the rectangle
    :param max_area: maximum area of each triangle.
    :param refine_func: A refinement function as taken by :func:`meshpy.triangle.build`.
    :param corner_fraction: Tuple of fraction of the width taken up by
      the rentrant corner.
    """
    if max_area is not None:
        if refine_func is not None:
            raise ValueError, "cannot specify both refine_func and max_area"
        def refine_func(vertices, area):
            return area > max_area

    marker2tag = {
            1: "minus_x",
            2: "minus_y",
            3: "plus_x",
            4: "plus_y",
            4: "plus_y",
            5: "corner_plus_y",
            6: "corner_plus_x",
            }

    a = numpy.asarray(a)
    b = numpy.asarray(b)
    diag =  b-a
    w = diag.copy(); w[1] = 0
    h = diag.copy(); h[0] = 0

    points = [
            a+h*corner_fraction[1],
            a+h*corner_fraction[1]+w*corner_fraction[0],
            a+w*corner_fraction[0],
            a+w,
            a+w+h,
            a+h,
            ]
    facets = list(_round_trip_connect(0, 5))
    facet_markers = [5,6,2,3,4,1]

    import meshpy.triangle as triangle
    mesh_info = triangle.MeshInfo()
    mesh_info.set_points(points)
    mesh_info.set_facets(facets, facet_markers)

    generated_mesh = triangle.build(mesh_info,
            refinement_func=refine_func)

    from hedge.mesh import make_conformal_mesh
    return make_conformal_mesh(
            generated_mesh.points,
            generated_mesh.elements,
            boundary_tagger)





def make_square_mesh(a=-0.5, b=0.5, max_area=4e-3,
        boundary_tagger=(lambda fvi, el, fn, all_v: [])):
    """Create an unstructured square mesh.

    :param a: the lower x and y coordinate of the square
    :param b: the upper x and y coordinate of the square
    :param max_area: maximum area of each triangle
    """
    return make_rect_mesh((a,a), (b,b), max_area, boundary_tagger)




def make_disk_mesh(r=0.5, faces=50, max_area=4e-3,
        boundary_tagger=(lambda fvi, el, fn, all_v: [])):
    from math import cos, sin, pi

    def needs_refinement(vertices, area):
        return area > max_area

    points = [(r*cos(angle), r*sin(angle))
            for angle in numpy.linspace(0, 2*pi, faces, endpoint=False)]

    import meshpy.triangle as triangle

    mesh_info = triangle.MeshInfo()
    mesh_info.set_points(points)
    mesh_info.set_facets(
            list(_round_trip_connect(0, faces-1)),
            faces*[1]
            )

    generated_mesh = triangle.build(mesh_info, refinement_func=needs_refinement)

    from hedge.mesh import make_conformal_mesh_ext
    from hedge.mesh.element import Triangle
    vertices = numpy.asarray(generated_mesh.points, dtype=float, order="C")
    return make_conformal_mesh_ext(
            vertices,
            [Triangle(i, el_idx, vertices)
                for i, el_idx in enumerate(generated_mesh.elements)],
            boundary_tagger)




def make_ball_mesh(r=0.5, subdivisions=10, max_volume=None,
        boundary_tagger=(lambda fvi, el, fn, all_v: [])):
    from meshpy.tet import MeshInfo, build
    from meshpy.geometry import make_ball

    points, facets, facet_holestarts, facet_markers = \
            make_ball(r, subdivisions)

    mesh_info = MeshInfo()

    mesh_info.set_points(points)
    mesh_info.set_facets_ex(facets, facet_holestarts, facet_markers)
    generated_mesh = build(mesh_info, max_volume=max_volume)

    from hedge.mesh import make_conformal_mesh
    return make_conformal_mesh(
            generated_mesh.points,
            generated_mesh.elements,
            boundary_tagger)








def _make_z_periodic_mesh(points, facets, facet_holestarts, facet_markers, height,
        max_volume, boundary_tagger):
    from meshpy.tet import MeshInfo, build
    from meshpy.geometry import Marker

    mesh_info = MeshInfo()
    mesh_info.set_points(points)
    mesh_info.set_facets_ex(facets, facet_holestarts, facet_markers)

    mesh_info.pbc_groups.resize(1)
    pbcg = mesh_info.pbc_groups[0]

    pbcg.facet_marker_1 = Marker.MINUS_Z
    pbcg.facet_marker_2 = Marker.PLUS_Z

    pbcg.set_transform(translation=[0,0,height])

    def zper_boundary_tagger(fvi, el, fn, all_v):
        # we only ask about *boundaries*
        # we should not try to have the user tag
        # the (periodicity-induced) interior faces

        face_marker = fvi2fm[frozenset(fvi)]

        if face_marker == Marker.MINUS_Z:
            return ["minus_z"]
        if face_marker == Marker.PLUS_Z:
            return ["plus_z"]

        result = boundary_tagger(fvi, el, fn, all_v)
        if face_marker == Marker.SHELL:
            result.append("shell")
        return result

    generated_mesh = build(mesh_info, max_volume=max_volume)
    fvi2fm = generated_mesh.face_vertex_indices_to_face_marker

    from hedge.mesh import make_conformal_mesh
    return make_conformal_mesh(
            generated_mesh.points,
            generated_mesh.elements,
            zper_boundary_tagger,
            periodicity=[None, None, ("minus_z", "plus_z")])




def make_cylinder_mesh(radius=0.5, height=1, radial_subdivisions=10,
        height_subdivisions=1, max_volume=None, periodic=False,
        boundary_tagger=(lambda fvi, el, fn, all_v: [])):
    from meshpy.tet import MeshInfo, build
    from meshpy.geometry import make_cylinder

    points, facets, facet_holestarts, facet_markers = \
            make_cylinder(radius, height, radial_subdivisions,
                    height_subdivisions)

    assert len(facets) == len(facet_markers)

    if periodic:
        return _make_z_periodic_mesh(
                points, facets, facet_holestarts, facet_markers,
                height=height,
                max_volume=max_volume,
                boundary_tagger=boundary_tagger)
    else:
        mesh_info = MeshInfo()
        mesh_info.set_points(points)
        mesh_info.set_facets_ex(facets, facet_holestarts, facet_markers)

        generated_mesh = build(mesh_info, max_volume=max_volume)

        from hedge.mesh import make_conformal_mesh
        return make_conformal_mesh(
                generated_mesh.points,
                generated_mesh.elements,
                boundary_tagger)




def make_box_mesh(a=(0,0,0),b=(1,1,1),
        max_volume=None, periodicity=None,
        boundary_tagger=(lambda fvi, el, fn, all_v: []),
        return_meshpy_mesh=False):
    """Return a mesh for a brick from the origin to `dimensions`.

    *max_volume* specifies the maximum volume for each tetrahedron.
    *periodicity* is either None, or a triple of bools, indicating
    whether periodic BCs are to be applied along that axis.
    See :func:`make_conformal_mesh` for the meaning of *boundary_tagger*.

    A few stock boundary tags are provided for easy application
    of boundary conditions, namely plus_[xyz] and minus_[xyz] tag
    the appropriate faces of the brick.
    """

    def count(iterable):
        result = 0
        for i in iterable:
            result += 1
        return result

    from meshpy.tet import MeshInfo, build
    from meshpy.geometry import make_box

    points, facets, facet_markers = make_box(a, b)

    mesh_info = MeshInfo()
    mesh_info.set_points(points)
    mesh_info.set_facets(facets, facet_markers)

    if periodicity is None:
        periodicity = (False, False, False)

    axes = ["x", "y", "z"]

    per_count = count(p for p in periodicity if p)
    mesh_info.pbc_groups.resize(per_count)
    pbc_group_number = 0

    marker_to_tag = {}
    mesh_periodicity = []
    periodic_tags = set()

    for axis, axis_per in enumerate(periodicity):
        minus_marker = 1+2*axis
        plus_marker = 2+2*axis

        minus_tag = "minus_"+axes[axis]
        plus_tag = "plus_"+axes[axis]

        marker_to_tag[minus_marker] = minus_tag
        marker_to_tag[plus_marker] = plus_tag

        if axis_per:
            pbcg = mesh_info.pbc_groups[pbc_group_number]
            pbc_group_number +=1

            pbcg.facet_marker_1 = minus_marker
            pbcg.facet_marker_2 = plus_marker

            translation = [0,0,0]
            translation[axis] = b[axis]-a[axis]
            pbcg.set_transform(translation=translation)

            mesh_periodicity.append((minus_tag, plus_tag))
            periodic_tags.add(minus_tag)
            periodic_tags.add(plus_tag)
        else:
            mesh_periodicity.append(None)

    generated_mesh = build(mesh_info, max_volume=max_volume)

    fvi2fm = generated_mesh.face_vertex_indices_to_face_marker

    def wrapped_boundary_tagger(fvi, el, fn, all_v):
        face_tag = marker_to_tag[fvi2fm[frozenset(fvi)]]

        if face_tag in periodic_tags:
            return [face_tag]
        else:
            return [face_tag] + boundary_tagger(fvi, el, fn, all_v)

    from hedge.mesh import make_conformal_mesh_ext
    from hedge.mesh.element import Tetrahedron
    vertices = numpy.asarray(generated_mesh.points, dtype=float, order="C")
    result = make_conformal_mesh_ext(
            vertices,
            [Tetrahedron(i, el_idx, vertices)
                for i, el_idx in enumerate(generated_mesh.elements)],
            wrapped_boundary_tagger,
            periodicity=mesh_periodicity)

    if return_meshpy_mesh:
        return result, generated_mesh
    else:
        return result




# poke generator bits into hedge.mesh for backwards compatibility -------------
def _add_depr_generator_functions():
    from pytools import MovedFunctionDeprecationWrapper

    import hedge.mesh
    for name in globals():
        if name.startswith("make_") or name.startswith("finish"):
            setattr(hedge.mesh, name, 
                    MovedFunctionDeprecationWrapper(globals()[name]))

_add_depr_generator_functions()
