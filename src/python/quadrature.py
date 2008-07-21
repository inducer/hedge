"""1D quadrature for Jacobi polynomials. Grundmann-Moeller cubature on the simplex."""

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




def legendre_gauss_quadrature(N):
    return jacobi_quadrature(0, 0, N)




def jacobi_gauss_lobatto_points(alpha, beta, N):
    """Compute the M{N}th order Gauss-Lobatto quadrature
    points, x, associated with the Jacobi polynomial,
    of type (alpha,beta) > -1 ( <> -0.5).
    """

    x = numpy.zeros((N+1,))
    x[0] = -1
    x[-1] = 1

    if N == 1:
        return x

    x[1:-1] = numpy.array(
            JacobiGaussQuadrature(alpha+1, beta+1, N-2).points
            ).real
    return x




def legendre_gauss_lobatto_points(N):
    """Compute the M{N}th order Gauss-Lobatto quadrature
    points, x, associated with the Legendre polynomials.
    """
    return jacobi_gauss_lobatto_points(0, 0, N)




class Quadrature(object):
    """An abstract quadrature rule."""
    def __init__(self, points, weights):
        self.weights = weights
        self.points = points
        self.data = zip(points, weights)

    def __call__(self, f):
        """Integrate the callable f with respect to the given quadrature rule.
        """
        return sum(w*f(x) for x, w in self.data)



class JacobiGaussQuadrature(Quadrature):
    """An M{N}th order Gauss quadrature associated with the Jacobi
    polynomials of type M{(alpha,beta) > -1} 
    
    C{alpha} and C{beta} may not be -0.5.

    Integrates on the interval (-1,1).
    """
    def __init__(self, alpha, beta, N):
        x, w = self.compute_weights_and_nodes(N, alpha, beta)
        Quadrature.__init__(self, x, w)

    @staticmethod
    def compute_weights_and_nodes(N, alpha, beta):
        """Return (nodes, weights) for an n-th order Gauss quadrature
        with the Jacobi polynomials of type (alpha, beta).
        """
        # follows 
        # Gene H. Golub, John H. Welsch, Calculation of Gauss Quadrature Rules, 
        # Mathematics of Computation, Vol. 23, No. 106 (Apr., 1969), pp. 221-230
        # doi:10.2307/2004418

        # see also doc/hedge-notes.tm for correspondence with the Jacobi
        # recursion from Hesthaven/Warburton's book

        from math import sqrt

        apb = alpha+beta

        # see Appendix A of Hesthaven/Warburton for these formulas
        def a(n):
            return (
                    2/(2*n+apb)
                    *
                    sqrt(
                        (n*(n+apb)*(n+alpha)*(n+beta))
                        /
                        ((2*n+apb-1)*(2*n+apb+1))
                        )
                    )

        def b(n):
            if n == 0:
                return (
                        -(alpha-beta)
                        /
                        (apb+2)
                        )
            else:
                return (
                        -(alpha**2-beta**2)
                        /
                        ((2*n+apb)*(2*n+apb+2))
                        )

        T = numpy.zeros((N+1, N+1))

        for n in range(N+1):
            T[n,n] = b(n)
            if n > 0:
                T[n,n-1] = current_a
            if n < N:
                next_a = a(n+1)
                T[n,n+1] = next_a
                current_a = next_a

        assert numpy.linalg.norm(T-T.T) < 1e-12
        eigval, eigvec = numpy.linalg.eigh(T)

        from numpy import dot, diag
        assert numpy.linalg.norm(dot(T, eigvec) -  dot(eigvec, diag(eigval))) < 1e-12

        from hedge.polynomial import JacobiFunction
        p0 = JacobiFunction(alpha, beta, 0)
        nodes = eigval
        weights = [eigvec[0,i]**2 / p0(nodes[i])**2 for i in range(N+1)]

        return nodes, weights


            

class LegendreGaussQuadrature(JacobiGaussQuadrature):
    """An M{N}th order Gauss quadrature associated with the Legendre polynomials.
    """
    def __init__(self, N):
        JacobiGaussQuadrature.__init__(self, 0, 0, N)




class TransformedQuadrature(Quadrature):
    """A quadrature rule on an arbitrary interval M{(a,b)}. """

    def __init__(self, quad, left, right):
        """Transform a given quadrature rule `quad' onto an arbitrary
        interval (left, right).
        """
        self.left = left
        self.right = right

        length = right-left
        assert length > 0
        half_length = length / 2
        Quadrature.__init__(self,
                [left + (p+1)/2*length for p in quad.points],
                [w*half_length for w in quad.weights])




def _extended_euclidean(q, r):
    """Return a tuple (p, a, b) such that p = aq + br, 
    where p is the greatest common divisor.
    """

    # see [Davenport], Appendix, p. 214

    if abs(q) < abs(r):
        p, a, b = _extended_euclidean(r, q)
        return p, b, a
  
    Q = 1, 0
    R = 0, 1
  
    while r:
        quot, t = divmod(q, r)
        T = Q[0] - quot*R[0], Q[1] - quot*R[1]
        q, r = r, t
        Q, R = R, T
  
    return q, Q[0], Q[1]




def _gcd(q, r):
    return _extended_euclidean(q, r)[0]




def _simplify_fraction((a, b)):
    gcd = _gcd(a,b)
    return (a//gcd, b//gcd)




class SimplexCubature(object):
    """Cubature on an M{n}-simplex.

    cf.
    A. Grundmann and H.M. Moeller,  
    Invariant integration formulas for the n-simplex by combinatorial methods, 
    SIAM J. Numer. Anal.  15 (1978), 282--290.

    This cubature rule has both negative and positive weights.
    """
    def __init__(self, order, dimension):
        s = order
        n = dimension
        d = 2*s+1

        from pytools import \
                generate_decreasing_nonnegative_tuples_summing_to, \
                generate_unique_permutations, \
                factorial, \
                wandering_element

        points_to_weights = {}

        for i in xrange(s+1):
            weight = (-1)**i * 2**(-2*s) \
                    * (d + n-2*i)**d \
                    / factorial(i) \
                    / factorial(d+n-i)

            for t in generate_decreasing_nonnegative_tuples_summing_to(s-i, n+1):
                for beta in generate_unique_permutations(t):
                    denominator = d+n-2*i
                    point = tuple(
                            _simplify_fraction((2*beta_i+1, denominator))
                            for beta_i in beta)

                    points_to_weights[point] = points_to_weights.get(point, 0) + weight

        from operator import add

        vertices = [-1 * numpy.ones((n,))] \
                + [numpy.array(x) for x in wandering_element(n, landscape=-1, wanderer=1)]

        self.pos_points = []
        self.pos_weights = []
        self.neg_points = []
        self.neg_weights = []

        dim_factor = 2**n
        for p, w in points_to_weights.iteritems():
            real_p = reduce(add, (a/b*v for (a,b),v in zip(p, vertices)))
            if w > 0:
                self.pos_points.append(real_p)
                self.pos_weights.append(dim_factor*w)
            else:
                self.neg_points.append(real_p)
                self.neg_weights.append(dim_factor*w)

        self.points = self.pos_points + self.neg_points
        self.weights = self.pos_weights + self.neg_weights

        self.pos_info = zip(self.pos_points, self.pos_weights)
        self.neg_info = zip(self.neg_points, self.neg_weights)


    def __call__(self, f):
        return sum(f(x)*w for x, w in self.pos_info) \
                + sum(f(x)*w for x, w in self.neg_info)




if __name__ == "__main__":
    cub = SimplexCubature(2, 3)
    print cub.weights
    for p in cub.points:
        print p
