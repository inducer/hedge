"""Mesh topology/geometry representation."""

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

class MeshOrientationError(Exception):
        pass

class Element(object):
    __slots__ = ["id", "vertex_indices", "map"]

    def __init__(self, id, vertex_indices, map):
        self.id = id
        self.vertex_indices = vertex_indices
        self.map = map

    @property
    def faces(self):
        return self.face_vertices(self.vertex_indices)





class CurvedElement(Element):
    pass




class SimplicialElement(Element):
    __slots__ = ["inverse_map", "face_normals", "face_jacobians"]

    def __init__(self, id, vertex_indices, all_vertices):
        vertex_indices = numpy.asarray(vertex_indices, dtype=numpy.intp)
        vertices = [all_vertices[v] for v in vertex_indices]

        # calculate maps, initialize
        map = self.get_map_unit_to_global(vertices)
        Element.__init__(self, id, vertex_indices, map)
        self.inverse_map = map.inverted()

        # calculate face normals and jacobians
        face_normals, face_jacobians = \
                self.face_normals_and_jacobians(vertices, map)

        self.face_normals = face_normals
        self.face_jacobians = face_jacobians

        #self.check_orientation()

    def copy(self, id, all_vertices):
        """Return a copy of self with id *id*."""

        return self.__class__(id, self.vertex_indices, all_vertices)

    def bounding_box(self, vertices):
        my_verts = numpy.array([vertices[vi] for vi in self.vertex_indices])
        return numpy.min(my_verts, axis=0), numpy.max(my_verts, axis=0)

    def centroid(self, vertices):
        my_verts = numpy.array([vertices[vi] for vi in self.vertex_indices])
        return numpy.average(my_verts, axis=0)

    def check_orientation(self):
        if self.map.jacobian > 0:
            raise MeshOrientationError("element %d is positively oriented"
                    % self.id)

    @classmethod
    def get_map_unit_to_global(cls, vertices):
        """Return an affine map that maps the unit coordinates of the reference
        element to a global element at a location given by its `vertices'.
        """
        from hedge._internal import get_simplex_map_unit_to_global
        return get_simplex_map_unit_to_global(cls.dimensions, vertices)

    def contains_point(self, x, thresh=0):
        unit_coords = self.inverse_map(x)
        for xi in unit_coords:
            if xi < -1-thresh:
                return False
        return sum(unit_coords) <= -(self.dimensions-2)+thresh




class Interval(SimplicialElement):
    dimensions = 1

    def check_orientation(self, vertices):
        if self.map.jacobian < 0:
            raise MeshOrientationError("interval %d is negatively oriented"
                    % self.id)

    @staticmethod
    def face_vertices(vertices):
        return [(vertices[0],), (vertices[1],) ]

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




# triangles -------------------------------------------------------------------
class TriangleBase(object):
    dimensions = 2

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




class Triangle(TriangleBase, SimplicialElement):
    __slots__ = []

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




class CurvedTriangle(TriangleBase, CurvedElement):
    pass




# tetrahedra ------------------------------------------------------------------
class TetrahedronBase(object):
    dimensions = 3

    #@staticmethod
    def _face_vertices(vertices):
        return [(vertices[0], vertices[1], vertices[2]),
                (vertices[0], vertices[1], vertices[3]),
                (vertices[0], vertices[2], vertices[3]),
                (vertices[1], vertices[2], vertices[3]),
                ]
    face_vertices = staticmethod(_face_vertices)

    face_vertex_numbers = _face_vertices([0, 1, 2, 3])




class Tetrahedron(TetrahedronBase, SimplicialElement):
    __slots__ = []

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




class CurvedTetrahedron(TetrahedronBase, CurvedElement):
    pass




TO_CURVED_CLASS = {
        Triangle: CurvedTriangle,
        Tetrahedron: CurvedTetrahedron,
        }
