"""Affine maps."""

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
