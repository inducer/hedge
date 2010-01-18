"""Affine maps."""

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





import numpy
import hedge._internal
import numpy.linalg as la




AffineMap = hedge._internal.AffineMap
def _affine_map___getinitargs__(self):
    return self.matrix, self.vector

AffineMap.__getinitargs__ = _affine_map___getinitargs__




class Rotation(AffineMap):
    def __init__(self, angle):
        # FIXME: Add axis, make multidimensional
        from math import sin, cos
        AffineMap.__init__(self,
                numpy.array([
                    [cos(angle), sin(angle)],
                    [-sin(angle), cos(angle)]]),
                numpy.zeros((2,)))




class Reflection(AffineMap):
    def __init__(self, axis, dim):
        mat = numpy.identity(dim)
        mat[axis,axis] = -1
        AffineMap.__init__(self, mat, numpy.zeros((dim,)))




def identify_affine_map(from_points, to_points):
    """Return an affine map that maps *from_points[i]* to *to_points[i]*.
    For an n-dimensional affine map, n+1 points are needed.
    """

    from pytools import single_valued
    dim = single_valued([
        single_valued(len(fp) for fp in from_points),
        single_valued(len(tp) for tp in to_points)])

    if dim == 0:
        return AffineMap(
                numpy.zeros((0,0), dtype=numpy.float64),
                numpy.zeros((0,), dtype=numpy.float64))

    if len(from_points) != dim+1 or len(to_points) != dim+1:
        raise ValueError("need dim+1 points to identify an affine map")

    # columns contain points
    x_mat = numpy.array(from_points).T
    y_mat = numpy.array(to_points).T

    # We are trying to solve 
    # a*x_i + b = y_i 
    # for a and b.  To eliminate b, subtract equation (i+1) from equation i,
    # then chop the last column.
    xdiff_mat = (x_mat - numpy.roll(x_mat, -1, axis=1))[:,:dim]
    ydiff_mat = (y_mat - numpy.roll(y_mat, -1, axis=1))[:,:dim]

    from hedge.tools.linalg import leftsolve
    a = numpy.asarray(leftsolve(xdiff_mat, ydiff_mat), order="C")
    b = to_points[0] - numpy.dot(a, from_points[0])

    return AffineMap(a, b)
