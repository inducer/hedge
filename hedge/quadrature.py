"""1D quadrature for Jacobi polynomials. Grundmann-Moeller cubature on the simplex."""

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

    *alpha* and *beta* may not be -0.5.

    Integrates on the interval (-1,1).
    The quadrature rule is exact up to degree :math:`2*N+1`.
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
        assert numpy.linalg.norm(dot(T, eigvec) - dot(eigvec, diag(eigval))) < 1e-12

        from hedge.polynomial import JacobiFunction
        p0 = JacobiFunction(alpha, beta, 0)
        nodes = eigval
        weights = numpy.array([eigvec[0,i]**2 / p0(nodes[i])**2 for i in range(N+1)])

        return nodes, weights




class LegendreGaussQuadrature(JacobiGaussQuadrature):
    """An M{N}th order Gauss quadrature associated with the Legendre polynomials.
    """
    def __init__(self, N):
        JacobiGaussQuadrature.__init__(self, 0, 0, N)




class OneDToNDQuadratureAdapter(Quadrature):
    """Augments the :attr:`points` array of a 1D quadrature by an extra dimension.
    """
    def __init__(self, one_d_quad):
        Quadrature.__init__(self, 
                one_d_quad.points[:, numpy.newaxis], 
                one_d_quad.weights)





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
    It is exact for polynomials up to order :math:`2s+1`, where
    :math:`s` is given as *order*.
    The integration domain is the unit simplex

    .. math::

        T_n:=\{(x_1,\dots,x_n) : x_i\ge -1, \sum_i x_i <=0\}
    """
    def __init__(self, order, dimension):
        s = order
        n = dimension
        d = 2*s+1

        self.exact_to = d

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

        self.points = numpy.array(self.pos_points + self.neg_points)
        self.weights = numpy.array(self.pos_weights + self.neg_weights)

        self.pos_points = numpy.array(self.pos_points)
        self.pos_weights = numpy.array(self.pos_weights)
        self.neg_points = numpy.array(self.neg_points)
        self.neg_weights = numpy.array(self.neg_weights)

        self.points = numpy.array(self.points)
        self.weights = numpy.array(self.weights)

        self.pos_info = zip(self.pos_points, self.pos_weights)
        self.neg_info = zip(self.neg_points, self.neg_weights)


    def __call__(self, f):
        return sum(f(x)*w for x, w in self.pos_info) \
                + sum(f(x)*w for x, w in self.neg_info)




class XiaoGimbutasSimplexCubature(Quadrature):
    """
    See

    [1] H. Xiao and Z. Gimbutas, "A numerical algorithm for the construction of
    efficient quadrature rules in two and higher dimensions," Computers &
    Mathematics with Applications, vol. 59, no. 2, pp. 663-676, 2010.
    """

    def __init__(self, order, dimension):
        if dimension == 2:
            from hedge.xg_quad_data import triangle_table as table
            from hedge.discretization.local import TriangleDiscretization
            e2u = TriangleDiscretization.equilateral_to_unit
        elif dimension == 3:
            from hedge.xg_quad_data import tetrahedron_table as table
            from hedge.discretization.local import TetrahedronDiscretization
            e2u = TetrahedronDiscretization.equilateral_to_unit
        else:
            raise ValueError("invalid dimensionality for XG quadrature")

        pts = numpy.array([
                e2u(pt) for pt in table[order]["points"]
                ])
        wts = table[order]["weights"]*e2u.jacobian()

        Quadrature.__init__(self, pts, wts)

        self.exact_to = order





class CoolsSimplexCubature(Quadrature):
    def __init__(self, order, dimension):
        if dimension == 2:
            from hedge.cools_quad_data import triangle_table as table
        else:
            raise ValueError("invalid dimensionality for XG quadrature")

        Quadrature.__init__(self, table[order]["points"], table[order]["weights"])

        self.exact_to = order





def get_simplex_cubature(exact_to_degree, dim):
    """Return a *dim*-dimensional quadrature satisfying
    at least the *exact_to_degree* requirement (but may be more 
    exact by one degree).

    Returns a Gauss quadrature in 1D and a Grundmann-Moeller cubature otherwise.
    All returned quadratures are nD in the sense that their node coordinates
    are arrays.
    """

    from math import ceil
    s = int(ceil((exact_to_degree-1)/2))

    if dim == 0:
        return Quadrature(
                points=numpy.array([[]], dtype=numpy.float64),
                weights=numpy.array([1], dtype=numpy.float64))
    elif dim == 1:
        return OneDToNDQuadratureAdapter(
                LegendreGaussQuadrature(s))
    else:
        return SimplexCubature(s, dim)






if __name__ == "__main__":
    cub = SimplexCubature(2, 3)
    print cub.weights
    for p in cub.points:
        print p
