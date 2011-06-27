# -*- coding: utf-8 -*-
"""Mesh topology/geometry representation."""

from __future__ import division

__copyright__ = "Copyright (C) 2007 Andreas Kloeckner"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""



import pytools
import numpy
import numpy.linalg as la

# make sure AffineMap monkeypatch happens
import hedge.tools




class MeshOrientationError(ValueError):
    pass



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

SYSTEM_TAGS = set([TAG_NONE, TAG_ALL, TAG_REALLY_ALL, TAG_NO_BOUNDARY])
# tags that are automatically assigned upon mesh creation
MESH_CREATION_TAGS = set([ TAG_ALL, TAG_REALLY_ALL])

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





def find_matching_vertices_along_axis(axis, points_a, points_b, numbers_a, numbers_b):
    a_to_b = {}
    not_found = []

    for i, pi in enumerate(points_a):
        found = False
        for j, pj in enumerate(points_b):
            dist = pi-pj
            dist[axis] = 0
            if la.norm(dist) < 1e-12:
                a_to_b[numbers_a[i]] = numbers_b[j]
                found = True
                break
        if not found:
            not_found.append(numbers_a[i])

    return a_to_b, not_found




def make_conformal_mesh_ext(points, elements,
        boundary_tagger=None,
        volume_tagger=None,
        periodicity=None,
        allow_internal_boundaries=False,
        _is_rankbdry_face=None,
        ):
    """Construct a simplical mesh.

    Face indices follow the convention for the respective element,
    such as Triangle or Tetrahedron, in this module.

    :param points: an array of vertex coordinates, given as vectors.
    :param elements: an iterable of :class:`hedge.mesh.element.Element`
      instances.
    :param boundary_tagger: A function of *(fvi, el, fn, all_v)* 
      that returns a list of boundary tags for a face identified
      by the parameters.

      *fvi* is the set of vertex indices of the face
      in question, *el* is an :class:`Element` instance,
      *fn* is the face number within *el*, and *all_v* is 
      a list of all vertices.
    :param volume_tagger: A function of *(el, all_v)* 
      returning a list of volume tags for the element identified
      by the parameters.

      *el* is an :class:`Element` instance and *all_v* is a list of
      all vertex coordinates.
    :param periodicity: either None or is a list of tuples
      just like the one documented for the `periodicity`
      member of class :class:`Mesh`.
    :param allow_internal_boundaries: Calls the boundary tagger
      for element-element interfaces as well. If the tagger returns
      an empty list of tags for an internal interface, it remains
      internal.
    :param _is_rankbdry_face: an implementation detail,
      should not be used from user code. It is a function
      returning whether a given face identified by
      *(element instance, face_nr)* is cut by a parallel
      mesh partition.
    """

    # input validation 
    if (not isinstance(points, numpy.ndarray) 
            or not points.dtype == numpy.float64):
        raise TypeError("points must be a float64 array")

    if boundary_tagger is None:
        def boundary_tagger(fvi, el, fn, all_v):
            return []

    if volume_tagger is None:
        def volume_tagger(el, all_v):
            return []

    if _is_rankbdry_face is None:
        def _is_rankbdry_face(el_face):
            return False

    dim = max(el.dimensions for el in elements)
    if periodicity is None:
        periodicity = dim*[None]
    assert len(periodicity) == dim

    # tag elements
    tag_to_elements = {TAG_NONE: [], TAG_ALL: []}
    for el in elements:
        for el_tag in volume_tagger(el, points):
            tag_to_elements.setdefault(el_tag, []).append(el)
        tag_to_elements[TAG_ALL].append(el)

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

    for face_vertices, els_faces in face_map.iteritems():
        boundary_el_faces_tags = []
        if len(els_faces) == 2:
            if allow_internal_boundaries:
                el_face_a, el_face_b = els_faces
                el_a, face_a = el_face_a
                el_b, face_b = el_face_b

                tags_a = boundary_tagger(face_vertices, el_a, face_a, points)
                tags_b = boundary_tagger(face_vertices, el_b, face_b, points)

                if not tags_a and not tags_b:
                    interfaces.append(els_faces)
                elif tags_a and tags_b:
                    boundary_el_faces_tags.append((el_face_a, tags_a))
                    boundary_el_faces_tags.append((el_face_b, tags_b))
                else:
                    raise RuntimeError("boundary tagger is inconsistent "
                            "about boundary-ness of interior interface")
            else:
                interfaces.append(els_faces)
        elif len(els_faces) == 1:
            el_face = el, face = els_faces[0]
            tags = boundary_tagger(face_vertices, el, face, points)
            boundary_el_faces_tags.append((el_face, tags))
        else:
            raise RuntimeError("face can at most border two elements")

        for el_face, tags in boundary_el_faces_tags:
            el, face = el_face
            tags = set(tags) - MESH_CREATION_TAGS
            assert not isinstance(tags, str), \
                RuntimeError("Received string as tag list")
            assert TAG_ALL not in tags
            assert TAG_REALLY_ALL not in tags

            for btag in tags:
                tag_to_boundary.setdefault(btag, []) \
                        .append(el_face)

            if TAG_NO_BOUNDARY not in tags:
                # TAG_NO_BOUNDARY is used to mark rank interfaces
                # as not being part of the boundary
                tag_to_boundary[TAG_ALL].append(el_face)

            tag_to_boundary[TAG_REALLY_ALL].append(el_face)

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
                    if _is_rankbdry_face(minus_face):
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

    return ConformalMesh(
            points=points,
            elements=elements,
            interfaces=interfaces,
            tag_to_boundary=tag_to_boundary,
            tag_to_elements=tag_to_elements,
            periodicity=periodicity,
            periodic_opposite_faces=periodic_opposite_faces,
            periodic_opposite_vertices=periodic_opposite_vertices,
            has_internal_boundaries=allow_internal_boundaries,
            )




def make_conformal_mesh(points, elements,
        boundary_tagger=None,
        volume_tagger=None,
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
    :param volume_tagger: a function that takes the arguments
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
    from warnings import warn
    warn("make_conformal_mesh is deprecated. "
            "Use make_conformal_mesh_ext instead.",
            stacklevel=2)

    if len(points) == 0:
        raise ValueError("mesh contains no points")

    from hedge.mesh.element import Interval, Triangle, Tetrahedron
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
    new_points = numpy.asarray(points, dtype=float, order="C")

    element_objs = [el_class(id, vert_indices, new_points)
        for id, vert_indices in enumerate(elements)]

    # call into new interface
    return make_conformal_mesh_ext(
            new_points, element_objs,
            volume_tagger = volume_tagger,
            boundary_tagger=boundary_tagger, 
            periodicity=periodicity, 
            _is_rankbdry_face=_is_rankbdry_face)




class ConformalMesh(Mesh):
    """A mesh whose elements' faces exactly match up with one another.

    See :class:`Mesh` for attributes provided by this class.
    """

    def __init__(self, points, elements, interfaces, tag_to_boundary, tag_to_elements,
            periodicity, periodic_opposite_faces, periodic_opposite_vertices,
            has_internal_boundaries):
        """This constructor is for internal use only. Use :func:`make_conformal_mesh` instead.
        """
        Mesh.__init__(self, locals())

    def get_reorder_oldnumbers(self, method):
        if method == "cuthill":
            from hedge.mesh.tools import cuthill_mckee
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

        elements = [self.elements[old_numbers[i]].copy(
            id=i, all_vertices=self.points)
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
                self.periodic_opposite_faces, self.periodic_opposite_vertices,
                self.has_internal_boundaries)




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
