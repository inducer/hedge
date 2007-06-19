from __future__ import division
import pylinear.array as num




def legendre_gauss_quadrature(N):
    return jacobi_quadrature(0, 0, N)




def jacobi_gauss_lobatto_points(alpha, beta, N):
    """Compute the N'th order Gauss-Lobatto quadrature
    points, x, associated with the Jacobi polynomial,
    of type (alpha,beta) > -1 ( <> -0.5).
    """

    x = num.zeros((N+1,))
    x[0] = -1
    x[-1] = 1

    if N == 1:
        return x

    x[1:-1] = num.array(
            JacobiGaussQuadrature(alpha+1, beta+1, N-2).points
            ).real
    return x




def legendre_gauss_lobatto_points(N):
    """Compute the N'th order Gauss-Lobatto quadrature
    points, x, associated with the Legendre polynomials.
    """
    return jacobi_gauss_lobatto_points(0, 0, N)




class Quadrature:
    """A quadrature rule."""
    def __init__(self, points, weights):
        self.weights = weights
        self.points = points
        self.data = zip(points, weights)

    def __call__(self, f):
        """Integrate the callable f with respect to the given quadrature rule.
        """
        return sum(w*f(x) for x, w in self.data)



class JacobiGaussQuadrature(Quadrature):
    """An N'th order Gauss quadrature associated with the Jacobi
    polynomials of type (alpha,beta) > -1 ( <> -0.5).
    """
    def __init__(self, alpha, beta, N):
        from scipy.special.orthogonal import j_roots
        import numpy
        x, w = j_roots(N+1, alpha, beta)
        Quadrature.__init__(self, numpy.real(x), w)




class LegendreGaussQuadrature(JacobiGaussQuadrature):
    """An N'th order Gauss quadrature associated with the Legendre polynomials.
    """
    def __init__(self, N):
        JacobiGaussQuadrature.__init__(self, 0, 0, N)




class TransformedQuadrature(Quadrature):
    """A quadrature rule on an arbitrary interval. """

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

