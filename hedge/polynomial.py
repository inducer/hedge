"""Jacobi polynomials and Vandermonde matrices."""

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




JacobiFunction = hedge._internal.JacobiPolynomial
DiffJacobiFunction = hedge._internal.DiffJacobiPolynomial




class LegendreFunction(JacobiFunction):
    def __init__(self, N):
        JacobiFunction.__init__(self, 0, 0, N)

class VectorLegendreFunction:
    """Same as :class:`LegendreFunction`, but accepts a single-entry
    vector as arugment.
    """
    def __init__(self, n):
        self.lf = LegendreFunction(n)

    def __call__(self, x):
        return self.lf(x[0])

class DiffLegendreFunction(DiffJacobiFunction):
    def __init__(self, N):
        DiffJacobiFunction.__init__(self, 0, 0, N)




def generic_vandermonde(points, functions):
    """Return a Vandermonde matrix.

    The Vandermonde Matrix is given by :math:`V_{i,j} := f_j(x_i)`
    where *functions* is the list of :math:`f_j` and points is
    the list of :math:`x_i`.
    """
    v = numpy.zeros((len(points), len(functions)))
    for i, x in enumerate(points):
        for j, f in enumerate(functions):
            v[i,j] = f(x)
    return v




def generic_multi_vandermonde(points, functions):
    """Return multiple Vandermonde matrices.

    The Vandermonde Matrix is given by :math:`V_{i,j} := f_j(x_i)`
    where *functions* is the list of :math:`f_j` and points is
    the list of :math:`x_i`.

    The functions :math:`f_j` are multi-valued (i.e. return iterables), and one
    matrix is returned for each return value.
    """
    count = len(functions[0](points[0]))
    result = [numpy.zeros((len(points), len(functions))) for n in range(count)]

    for i, x in enumerate(points):
        for j, f in enumerate(functions):
            for n, f_n in enumerate(f(x)):
                result[n][i,j] = f_n
    return result




def legendre_vandermonde(points, N):
    return generic_vandermonde(points,
            [LegendreFunction(i) for i in range(N+1)])




def monomial_vdm(points, max_degree=None):
    class Monomial:
        def __init__(self, expt):
            self.expt = expt
        def __call__(self, x):
            return x**self.expt

    if max_degree is None:
        max_degree = len(points)-1

    return generic_vandermonde(points,
            [Monomial(i) for i in range(max_degree+1)])




def make_interpolation_coefficients(levels, tap):
    point_eval_vec = numpy.array([tap**n for n in range(len(levels))])
    return la.solve(monomial_vdm(levels).T, point_eval_vec)
