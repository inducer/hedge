"""Mesh topology representation."""

from __future__ import division

__copyright__ = "Copyright (C) 2007 Andreas Kloeckner"

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




import pytools
import numpy

# make sure AffineMap monkeypatch happens
import hedge.tools




class TAG_NONE(object):
    """A boundary or volume tag representing an empty boundary or volume."""
    pass
class TAG_ALL(object):
    """A boundary or volume tag representing the entire boundary or volume.

    In the case of the boundary, TAG_ALL does not include rank boundaries,
    or, more generally, anything tagged with TAG_NO_BOUNDARY."""
    pass
class TAG_REALLY_ALL(object):
    """A boundary tag representing the entire boundary.

    Unlike :class:`TAG_ALL`, this includes rank boundaries,
    or, more generally, everything tagged with :class:`TAG_NO_BOUNDARY`."""
    pass
class TAG_NO_BOUNDARY(object):
    """A boundary tag indicating that this edge should not fall under
    :class:`TAG_ALL`."""
    pass
class TAG_RANK_BOUNDARY(object):
    """A boundary tag indicating the boundary with a neighboring rank."""
    def __init__(self, rank):
        self.rank = rank

    def __repr__(self):
        return "TAG_RANK_BOUNDARY(%d)" % self.rank

    def __eq__(self, other):
        return isinstance(other, TAG_RANK_BOUNDARY) and self.rank == other.rank

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return 0xaffe ^ hash(self.rank)






def make_element(class_, id, vertex_indices, all_vertices):
    vertices = [all_vertices[v] for v in vertex_indices]
    map = class_.get_map_unit_to_global(vertices)
    new_vertex_indices = \
            class_._reorder_vertices(vertex_indices, vertices, map)
    if new_vertex_indices:
        vertex_indices = new_vertex_indices
        vertices = [all_vertices[v] for v in vertex_indices]
        map = class_.get_map_unit_to_global(vertices)

    vertex_indices = numpy.array(vertex_indices, dtype=numpy.intp)

    face_normals, face_jacobians = \
            class_.face_normals_and_jacobians(vertices, map)

    return class_(id, vertex_indices, map, map.inverted(),
            face_normals, face_jacobians)




class Element(pytools.Record):
    __slots__ = ["id", "vertex_indices", "map", "inverse_map", "face_normals",
            "face_jacobians"]

    def __init__(self, id, vertex_indices, map, inverse_map, face_normals,
            face_jacobians):
        pytools.Record.__init__(self, locals())

    @staticmethod
    def _reorder_vertices(vertex_indices, vertices):
        return vertex_indices

    def bounding_box(self, vertices):
        my_verts = numpy.array([vertices[vi] for vi in self.vertex_indices])
        return numpy.min(my_verts, axis=0), numpy.max(my_verts, axis=0)

    def centroid(self, vertices):
        my_verts = numpy.array([vertices[vi] for vi in self.vertex_indices])
        return numpy.average(my_verts, axis=0)






class SimplicialElement(Element):
    __slots__ = []

    @property
    def faces(self):
        return self.face_vertices(self.vertex_indices)

    @classmethod
    def get_map_unit_to_global(cls, vertices):
        """Return an affine map that maps the unit coordinates of the reference
        element to a global element at a location given by its `vertices'.
        """
        from hedge._internal import get_simplex_map_unit_to_global
        return get_simplex_map_unit_to_global(cls.dimensions, vertices)

    def contains_point(self, x):
        unit_coords = self.inverse_map(x)
        for xi in unit_coords:
            if xi < -1:
                return False
        return sum(unit_coords) <= -(self.dimensions-2)




class Interval(SimplicialElement):
    dimensions = 1
    @staticmethod
    def face_vertices(vertices):
        return [(vertices[0],), (vertices[1],) ]

    @classmethod
    def _reorder_vertices(cls, vertex_indices, vertices, map):
        vi = vertex_indices
        if vertices[0][0] > vertices[1][0]: # make sure we're ordered left-right
            return (vi[1], vi[0])
        else:
            return None

    @staticmethod
    def face_normals_and_jacobians(vertices, affine_map):
        """Compute the normals and face jacobians of the unit element
        transformed according to `affine_map'.

        Returns a pair of lists [normals], [jacobians].
        """
        if affine_map.jacobian() < 0:
            return [
                    numpy.array([1], dtype=float),
                    numpy.array([-1], dtype=float)
                    ], [1, 1]
        else:
            return [
                    numpy.array([-1], dtype=float),
                    numpy.array([1], dtype=float)
                    ], [1, 1]




class Triangle(SimplicialElement):
    dimensions = 2

    __slots__ = []

    @staticmethod
    def face_vertices(vertices):
        return [(vertices[0], vertices[1]),
                (vertices[1], vertices[2]),
                (vertices[0], vertices[2])
                ]

    @classmethod
    def _reorder_vertices(cls, vertex_indices, vertices, map):
        vi = vertex_indices
        if map.jacobian() > 0:
            return (vi[0], vi[2], vi[1])
        else:
            return None

    @staticmethod
    def face_normals_and_jacobians(vertices, affine_map):
        """Compute the normals and face jacobians of the unit element
        transformed according to `affine_map'.

        Returns a pair of lists [normals], [jacobians].
        """
        from hedge.tools import sign

        m = affine_map.matrix
        orient = sign(affine_map.jacobian())
        face1 = m[:, 1] - m[:, 0]
        raw_normals = [
                orient*numpy.array([m[1, 0], -m[0, 0]]),
                orient*numpy.array([face1[1], -face1[0]]),
                orient*numpy.array([-m[1, 1], m[0, 1]]),
                ]

        face_lengths = [numpy.linalg.norm(fn) for fn in raw_normals]
        return [n/fl for n, fl in zip(raw_normals, face_lengths)], \
                face_lengths




class Tetrahedron(SimplicialElement):
    dimensions = 3

    __slots__ = []

    def _face_vertices(vertices):
        return [(vertices[0], vertices[1], vertices[2]),
                (vertices[0], vertices[1], vertices[3]),
                (vertices[0], vertices[2], vertices[3]),
                (vertices[1], vertices[2], vertices[3]),
                ]
    face_vertices = staticmethod(_face_vertices)

    face_vertex_numbers = _face_vertices([0, 1, 2, 3])

    @classmethod
    def _reorder_vertices(cls, vertex_indices, vertices, map):
        vi = vertex_indices
        if map.jacobian() > 0:
            return (vi[0], vi[1], vi[3], vi[2])
        else:
            return None

    @classmethod
    def face_normals_and_jacobians(cls, vertices, affine_map):
        """Compute the normals and face jacobians of the unit element
        transformed according to `affine_map'.

        Returns a pair of lists [normals], [jacobians].
        """
        from hedge._internal import tetrahedron_fj_and_normal
        from hedge.tools import sign

        return tetrahedron_fj_and_normal(
                sign(affine_map.jacobian()),
                cls.face_vertex_numbers,
                vertices)




class Mesh(pytools.Record):
    """Information about the geometry and connectivity of a finite
    element mesh. (Note: no information about the discretization
    is stored here.)

    :ivar points: list of Pylinear vectors of node coordinates
    :ivar elements: list of Element instances
    :ivar interfaces: a list of pairs:

          ((element instance 1, face index 1), (element instance 2, face index 2))

      enumerating elements bordering one another.  The relation "element 1 touches
      element 2" is always reflexive, but this list will only contain one entry
      per element pair.
    :ivar tag_to_boundary: a mapping of the form:
          boundary_tag -> [(element instance, face index)])

      The boundary tag :class:`TAG_NONE` always refers to an empty boundary.
      The boundary tag :class:`TAG_ALL` always refers to the entire boundary.
    :ivar tag_to_elements: a mapping of the form
      element_tag -> [element instances]

      The element tag :class:`TAG_NONE` always refers to an empty domain.
      The element tag :class:`TAG_ALL` always refers to the entire domain.
    :ivar periodicity: A list of tuples (minus_tag, plus_tag) or None
      indicating the tags of the boundaries to be matched together
      as periodic. There is one tuple per axis, so that for example
      a 3D mesh has three tuples.
    :ivar periodic_opposite_faces: a mapping of the form:
          (face_vertex_indices) ->
            (opposite_face_vertex_indices), axis

      This maps a face to its periodicity-induced opposite.

    :ivar periodic_opposite_vertices: a mapping of the form:
          vertex_index -> [(opposite_vertex_index, axis), ...]

      This maps one vertex to a list of its periodicity-induced
      opposites.
    """

    def both_interfaces(self):
        for face1, face2 in self.interfaces:
            yield face1, face2
            yield face2, face1

    @property
    def dimensions(self):
        return self.points.shape[1]

    def bounding_box(self):
        try:
            return self._bounding_box
        except AttributeError:
            self._bounding_box = (
                    numpy.min(self.points, axis=0),
                    numpy.max(self.points, axis=0),
                    )
            return self._bounding_box

    def element_adjacency_graph(self):
        """Return a dictionary mapping each element id to a
        list of adjacent element ids.
        """
        adjacency = {}
        for (e1, f1), (e2, f2) in self.interfaces:
            adjacency.setdefault(e1.id, set()).add(e2.id)
            adjacency.setdefault(e2.id, set()).add(e1.id)
        return adjacency





def _build_mesh_data_dict(points, elements, boundary_tagger, periodicity, is_rankbdry_face):
    # create face_map, which is a mapping of
    # (vertices on a face) ->
    #  [(element, face_idx) for elements bordering that face]
    face_map = {}
    for el in elements:
        for fid, face_vertices in enumerate(el.faces):
            face_map.setdefault(frozenset(face_vertices), []).append((el, fid))

    # build non-periodic connectivity structures
    interfaces = []
    tag_to_boundary = {
            TAG_NONE: [],
            TAG_ALL: [],
            TAG_REALLY_ALL: [],
            }

    all_tags = set([TAG_ALL, TAG_REALLY_ALL])

    for face_vertices, els_faces in face_map.iteritems():
        if len(els_faces) == 2:
            interfaces.append(els_faces)
        elif len(els_faces) == 1:
            el_face = el, face = els_faces[0]
            tags = boundary_tagger(face_vertices, el, face, points)

            if isinstance(tags, str):
                raise RuntimeError("Received string as tag list")

            tags = set(tags) - all_tags

            for btag in tags:
                tag_to_boundary.setdefault(btag, []) \
                        .append(el_face)

            if TAG_NO_BOUNDARY not in tags:
                # TAG_NO_BOUNDARY is used to mark rank interfaces
                # as not being part of the boundary
                tag_to_boundary[TAG_ALL].append(el_face)

            tag_to_boundary[TAG_REALLY_ALL].append(el_face)
        else:
            raise RuntimeError("face can at most border two elements")

    # add periodicity-induced connectivity
    from pytools import flatten, reverse_dictionary

    periodic_opposite_faces = {}
    periodic_opposite_vertices = {}

    for tag_bdries in tag_to_boundary.itervalues():
        assert len(set(tag_bdries)) == len(tag_bdries)

    for axis, axis_periodicity in enumerate(periodicity):
        if axis_periodicity is not None:
            # find faces on +-axis boundaries
            minus_tag, plus_tag = axis_periodicity
            minus_faces = tag_to_boundary.get(minus_tag, [])
            plus_faces = tag_to_boundary.get(plus_tag, [])

            # find vertex indices and points on these faces
            minus_vertex_indices = list(set(flatten(el.faces[face]
                for el, face in minus_faces)))
            plus_vertex_indices = list(set(flatten(el.faces[face]
                for el, face in plus_faces)))

            minus_z_points = [points[pi] for pi in minus_vertex_indices]
            plus_z_points = [points[pi] for pi in plus_vertex_indices]

            # find a mapping from -axis to +axis vertices
            from hedge.tools import find_matching_vertices_along_axis

            minus_to_plus, not_found = find_matching_vertices_along_axis(
                    axis, minus_z_points, plus_z_points,
                    minus_vertex_indices, plus_vertex_indices)
            plus_to_minus = reverse_dictionary(minus_to_plus)

            for a, b in minus_to_plus.iteritems():
                periodic_opposite_vertices.setdefault(a, []).append((b, axis))
                periodic_opposite_vertices.setdefault(b, []).append((a, axis))

            # establish face connectivity
            for minus_face in minus_faces:
                minus_el, minus_fi = minus_face
                minus_fvi = minus_el.faces[minus_fi]

                try:
                    mapped_plus_fvi = tuple(minus_to_plus[i] for i in minus_fvi)
                    plus_faces = face_map[frozenset(mapped_plus_fvi)]
                    assert len(plus_faces) == 1
                except KeyError:
                    # is our periodic counterpart is in a different mesh clump?
                    if is_rankbdry_face(minus_face):
                        # if so, cool. parallel handler will take care of it.
                        continue
                    else:
                        # if not, bad.
                        raise

                plus_face = plus_faces[0]
                interfaces.append([minus_face, plus_face])

                plus_el, plus_fi = plus_face
                plus_fvi = plus_el.faces[plus_fi]

                mapped_minus_fvi = tuple(plus_to_minus[i] for i in plus_fvi)

                # periodic_opposite_faces maps face vertex tuples from
                # one end of the periodic domain to the other, while
                # correspondence between each entry

                periodic_opposite_faces[minus_fvi] = mapped_plus_fvi, axis
                periodic_opposite_faces[plus_fvi] = mapped_minus_fvi, axis

                tag_to_boundary[TAG_ALL].remove(plus_face)
                tag_to_boundary[TAG_ALL].remove(minus_face)

                tag_to_boundary[TAG_REALLY_ALL].remove(plus_face)
                tag_to_boundary[TAG_REALLY_ALL].remove(minus_face)

    return {
            "interfaces": interfaces,
            "tag_to_boundary": tag_to_boundary,
            "periodicity": periodicity,
            "periodic_opposite_faces": periodic_opposite_faces,
            "periodic_opposite_vertices": periodic_opposite_vertices,
            }




def make_conformal_mesh(points, elements,
        boundary_tagger=None,
        element_tagger=None,
        periodicity=None,
        _is_rankbdry_face=None,
        ):
    """Construct a simplical mesh.

    Face indices follow the convention for the respective element,
    such as Triangle or Tetrahedron, in this module.

    :param points: an iterable of vertex coordinates, given as vectors.
    :param elements: an iterable of tuples of indices into points,
      giving element endpoints.
    :param boundary_tagger: a function that takes the arguments
      *(set_of_face_vertex_indices, element, face_number, all_vertices)*
      It returns a list of tags that apply to this surface.
    :param element_tagger: a function that takes the arguments
      (element, all_vertices) and returns the a list of tags that apply
      to that element.
    :param periodicity: either None or is a list of tuples
      just like the one documented for the `periodicity`
      member of class :class:`Mesh`.
    :param _is_rankbdry_face: an implementation detail,
      should not be used from user code. It is a function
      returning whether a given face identified by
      *(element instance, face_nr)* is cut by a parallel
      mesh partition.
    """
    if boundary_tagger is None:
        def boundary_tagger(fvi, el, fn, all_v):
            return []

    if element_tagger is None:
        def element_tagger(el, all_v):
            return []

    if _is_rankbdry_face is None:
        def _is_rankbdry_face(el_face):
            return False

    if len(points) == 0:
        raise ValueError("mesh contains no points")

    dim = len(points[0])
    if dim == 1:
        el_class = Interval
    elif dim == 2:
        el_class = Triangle
    elif dim == 3:
        el_class = Tetrahedron
    else:
        raise ValueError("%d-dimensional meshes are unsupported" % dim)

    # build points and elements
    new_points = numpy.array(points, dtype=float, order="C")

    element_objs = [make_element(el_class, id, vert_indices, new_points)
        for id, vert_indices in enumerate(elements)]

    # tag elements
    tag_to_elements = {TAG_NONE: [], TAG_ALL: []}
    for el in element_objs:
        for el_tag in element_tagger(el, new_points):
            tag_to_elements.setdefault(el_tag, []).append(el)
        tag_to_elements[TAG_ALL].append(el)

    # build connectivity
    if periodicity is None:
        periodicity = dim*[None]
    assert len(periodicity) == dim

    mdd = _build_mesh_data_dict(
            new_points, element_objs, boundary_tagger, periodicity, _is_rankbdry_face)
    mdd["tag_to_elements"] = tag_to_elements
    return ConformalMesh(new_points, element_objs, **mdd)




class ConformalMesh(Mesh):
    """A mesh whose elements' faces exactly match up with one another.

    See :class:`Mesh` for attributes provided by this class.
    """

    def __init__(self, points, elements, interfaces, tag_to_boundary, tag_to_elements,
            periodicity, periodic_opposite_faces, periodic_opposite_vertices):
        """This constructor is for internal use only. Use :func:`make_conformal_mesh` instead.
        """
        Mesh.__init__(self, locals())

    def get_reorder_oldnumbers(self, method):
        if method == "cuthill":
            from hedge.tools import cuthill_mckee
            return cuthill_mckee(self.element_adjacency_graph())
        else:
            raise ValueError("invalid mesh reorder method")

    def reordered_by(self, method):
        """Return a reordered copy of *self*.

        :param method: "cuthill"
        """

        old_numbers = self.get_reorder_oldnumbers(method)
        return self.reordered(old_numbers)

    def reordered(self, old_numbers):
        """Return a copy of *self* whose elements are
        reordered using such that for each element *i*,
        *old_numbers[i]* gives the previous number of that
        element.
        """

        elements = [self.elements[old_numbers[i]].copy(id=i)
                for i in range(len(self.elements))]

        old2new_el = dict(
                (self.elements[old_numbers[i]], new_el)
                for i, new_el in enumerate(elements)
                )

        # sort interfaces by element id -- this is actually the most important part
        def face_cmp(face1, face2):
            (face1_el1, _), (face1_el2, _) = face1
            (face2_el1, _), (face2_el2, _) = face2

            return cmp(
                    min(face1_el1.id, face1_el2.id),
                    min(face2_el1.id, face2_el2.id))

        interfaces = [
                ((old2new_el[e1], f1), (old2new_el[e2], f2))
                for (e1, f1), (e2, f2) in self.interfaces]
        interfaces.sort(face_cmp)

        tag_to_boundary = dict(
                (tag, [(old2new_el[old_el], fnr) for old_el, fnr in elfaces])
                for tag, elfaces in self.tag_to_boundary.iteritems())

        tag_to_elements = dict(
                (tag, [old2new_el[old_el] for old_el in tag_els])
                for tag, tag_els in self.tag_to_elements.iteritems())

        return ConformalMesh(
                self.points, elements, interfaces,
                tag_to_boundary, tag_to_elements, self.periodicity,
                self.periodic_opposite_faces, self.periodic_opposite_vertices
                )




def check_bc_coverage(mesh, bc_tags, incomplete_ok=False):
    """Verify boundary condition coverage.

    Given a list of boundary tags as *bc_tags*, this function verifies
    that

     1. the union of all these boundaries gives the complete boundary,
     1. all these boundaries are disjoint.

    :param incomplete_ok: Do not report an error if some faces are not covered
      by the boundary conditions.
    """

    bdry_to_tag = {}
    all_bdry_faces = mesh.tag_to_boundary[TAG_ALL]
    bdry_face_countdown = len(all_bdry_faces)

    for tag in bc_tags:
        for el_face in mesh.tag_to_boundary.get(tag, []):
            if el_face in bdry_to_tag:
                raise RuntimeError("Duplicate BCs %s found on (el=%d,face=%d)"
                        % ([bdry_to_tag[el_face], tag],
                            el_face[0].id, el_face[1]))
            else:
                bdry_to_tag[el_face] = tag
                bdry_face_countdown -= 1

    if bdry_face_countdown > 0 and not incomplete_ok:
        no_bc_faces = set(all_bdry_faces)-set(bdry_to_tag.keys())

        from sys import stderr
        print>>stderr, "Faces without BC:"
        for el, face_nr in no_bc_faces:
            x = mesh.points[el.vertex_indices]
            face_vertices = el.face_vertices(
                    x)[face_nr]
            normal = el.face_normals[face_nr]
            print>>stderr, "  normal: %s vertices: %s" % (
                    normal, ", ".join(str(v) for v in face_vertices))

        raise RuntimeError("Found faces without boundary conditions--see above for list.")
    elif bdry_face_countdown < 0:
        raise RuntimeError("More BCs were assigned than boundary faces are present "
                "(did something screw up your periodicity?)")




class MeshPyFaceMarkerLookup:
    def __init__(self, meshpy_output):
        self.fvi2fm = dict((frozenset(fvi), marker) for fvi, marker in
                zip(meshpy_output.facets, meshpy_output.facet_markers))

    def __call__(self, fvi):
        return self.fvi2fm[frozenset(fvi)]




# mesh producers for simple geometries ----------------------------------------
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
        return make_conformal_mesh(
                [force_array(pt) for pt in points],
                [(i, i+1) for i in range(len(points) - 1)],
                periodicity=[("x_minus", "x_plus")],
                boundary_tagger=my_boundary_tagger,
                **kwargs)
    else:
        return make_conformal_mesh(
                [force_array(pt) for pt in points],
                [(i, i + 1) for i in range(len(points)-1)],
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

            if i == 0:
                fvi2fm[frozenset((a,c))] = "minus_x"
            if i == n[0]-2:
                fvi2fm[frozenset((b,d))] = "plus_x"
            if j == 0:
                fvi2fm[frozenset((a,b))] = "minus_y"
            if j == n[1]-2:
                fvi2fm[frozenset((c,d))] = "plus_y"

    def wrapped_boundary_tagger(fvi, el, fn, all_v):
        btag = fvi2fm[frozenset(fvi)]
        if btag in periodic_tags:
            return [btag]
        else:
            return [btag] + boundary_tagger(fvi, el, fn, all_v)

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

            if i == 0:
                fvi2fm[frozenset((a,c))] = "minus_x"
            if i == n[0]-2:
                fvi2fm[frozenset((b,d))] = "plus_x"
            if j == 0:
                fvi2fm[frozenset((a,b))] = "minus_y"
            if j == n[1]-2:
                fvi2fm[frozenset((c,d))] = "plus_y"

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
            raise ValueError("cannot specify both refine_func and max_area")
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
    import meshpy.triangle as triangle

    if max_area is not None:
        if refine_func is not None:
            raise ValueError("cannot specify both refine_func and max_area")
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
    diag = b-a
    w = diag.copy()
    w[1] = 0
    h = diag.copy()
    h[0] = 0

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

    return make_conformal_mesh(
            generated_mesh.points,
            generated_mesh.elements,
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

    result = make_conformal_mesh(
            generated_mesh.points,
            generated_mesh.elements,
            wrapped_boundary_tagger,
            periodicity=mesh_periodicity)

    if return_meshpy_mesh:
        return result, generated_mesh
    else:
        return result
