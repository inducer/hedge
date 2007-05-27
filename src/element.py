from __future__ import division
import pylinear.array as num
from hedge.tools import AffineMap
from math import sqrt, sin, cos, exp, pi




class WarpFactorCalculator:
    """Calculator for Warburton's warp factor.

    See T. Warburton,
    "An explicit construction of interpolation nodes on the simplex"
    Journal of Engineering Mathematics Vol 56, No 3, p. 247-262, 2006
    """

    def __init__(self, N):
        from hedge.quadrature import legendre_gauss_lobatto_points
        from hedge.interpolation import newton_interpolation_function

        # Find lgl and equidistant interpolation points
        r_lgl = legendre_gauss_lobatto_points(N)
        r_eq  = num.linspace(-1,1,N+1)

        self.int_f = newton_interpolation_function(r_eq, r_lgl - r_eq)

    def __call__(self, x):
        if abs(x) > 1-1e-10:
            return 0
        else:
            return self.int_f(x)/(1-x**2)




class TriangleBasisFunction:
    def __init__(self, (i, j)):
        from hedge.polynomial import jacobi_function
        self.i = i
        self.f = jacobi_function(0, 0, i)
        self.g = jacobi_function(2*i+1, 0, j)

    def __call__(self, (r, s)):
        try:
            a = 2*(1+r)/(1-s)-1
        except ZeroDivisionError:
            a = 1

        return sqrt(2)*self.f(a)*self.g(s)*(1-s)**self.i




class GradTriangleBasisFunction:
    def __init__(self, (i, j)):
        from hedge.polynomial import jacobi_function, diff_jacobi_function
        self.i = i
        self.f  =      jacobi_function(0, 0, i)
        self.df = diff_jacobi_function(0, 0, i)
        self.g  =      jacobi_function(2*i+1, 0, j)
        self.dg = diff_jacobi_function(2*i+1, 0, j)

    def __call__(self, (r, s)):
        try:
            a = 2*(1+r)/(1-s)-1
        except ZeroDivisionError:
            a = 1

        f_a = self.f(a)
        g_s = self.g(s)
        df_a = self.df(a)
        dg_s = self.dg(s)
        one_s = 1-s
        i = self.i

        # see doc/hedge-notes.tm
        return num.array([
            # df/dr
            2*sqrt(2) * g_s * one_s**(i-1) * df_a,
            # df/ds
            sqrt(2)*(
                f_a * one_s**i * dg_s
                +(2*r+2) * g_s * one_s**(i-2) * df_a
                -i * f_a * g_s * one_s**(i-1)
                )])




class Triangle:
    """An arbitrary-order triangular finite element.

    Coordinate systems used:
    ------------------------

    unit coordinates (r,s):

    A
    |\
    | \
    |  O
    |   \
    |    \
    C-----B

    O = (0,0)
    A=(-1,1)
    B=(1,-1)
    C=(-1,-1)

    equilateral coordinates (x,y):

            A
           / \
          /   \
         /     \
        /   O   \
       /         \
      C-----------B

    O = (0,0)
    A = (0,2/sqrt(3))
    B = (1,-1/sqrt(3))
    C = (-1,-1/sqrt(3))
    """

    def __init__(self, order):
        self.order = order

    def indices(self):
        for n in range(0, self.order+1):
            for m in range(0, self.order+1-n):
                yield m,n

    def face_indices(self):
        faces = [[], [], []]

        i = 0
        for m, n in self.indices():
            # face finding
            if n == 0:
                faces[0].append(i)
            if n+m == self.order:
                faces[1].append(i)
            if m == 0:
                faces[2].append(i)

            i += 1

        # make sure faces are numbered counterclockwise
        faces[2] = faces[2][::-1]

        return faces

    def equidistant_barycentric_nodes(self):
        """Compute equidistant nodes in barycentric coordinates
        of order N.
        """
        for m, n in self.indices():
            lambda1 = n/self.order
            lambda3 = m/self.order
            lambda2 = 1-lambda1-lambda3

            yield lambda1, lambda2, lambda3

    @staticmethod
    def barycentric_to_equilateral((lambda1, lambda2, lambda3)):
        return num.array([
            -lambda2+lambda3,
            (-lambda2-lambda3+2*lambda1)/sqrt(3.0)])

    # see doc/hedge-notes.tm
    equilateral_to_unit = AffineMap(
            num.array([[1,-1/sqrt(3)], [0,2/sqrt(3)]]),
                num.array([-1/3,-1/3]))

    def equidistant_equilateral_nodes(self):
        """Compute equidistant nodes in equilateral coordinates."""

        for bary in self.equidistant_barycentric_nodes():
            yield self.barycentric_to_equilateral(bary)

    def equidistant_unit_nodes(self):
        """Compute equidistant nodes in unit coordinates."""

        for bary in self.equidistant_barycentric_nodes():
            yield self.equilateral_to_unit(self.barycentric_to_equilateral(bary))

    def equilateral_nodes(self):
        """Compute warped nodes in equilateral coordinates (x,y)."""

        # port of J. Hesthaven's Nodes2D routine

        # Set optimized parameter alpha, depending on order N
        alpha_opt = [0.0000, 0.0000, 1.4152, 0.1001, 0.2751, 0.9800, 1.0999,
                  1.2832, 1.3648, 1.4773, 1.4959, 1.5743, 1.5770, 1.6223, 1.6258]
                  
        try:
            alpha = alpha_opt[self.order+1]
        except IndexError:
            alpha = 5/3

        warp = WarpFactorCalculator(self.order)

        edge1dir = num.array([1,0])
        edge2dir = num.array([cos(2*pi/3), sin(2*pi/3)])
        edge3dir = num.array([cos(4*pi/3), sin(4*pi/3)])

        for bary in self.equidistant_barycentric_nodes():
            lambda1, lambda2, lambda3 = bary

            # find equidistant (x,y) coordinates in equilateral triangle
            point = self.barycentric_to_equilateral(bary)

            # compute blend factors
            blend1 = 4*lambda2*lambda3
            blend2 = 4*lambda1*lambda3
            blend3 = 4*lambda1*lambda2

            # calculate amount of warp for each node, for each edge
            warp1 = blend1*warp(lambda3 - lambda2)*(1 + (alpha*lambda1)**2)
            warp2 = blend2*warp(lambda1 - lambda3)*(1 + (alpha*lambda2)**2)
            warp3 = blend3*warp(lambda2 - lambda1)*(1 + (alpha*lambda3)**2)

            # return warped point
            yield point + warp1*edge1dir + warp2*edge2dir + warp3*edge3dir

    def unit_nodes(self):
        """Compute the warped nodes in unit coordinates (r,s)."""
        for node in self.equilateral_nodes():
            yield self.equilateral_to_unit(node)

    def basis_functions(self):
        """Get a sequence of functions that form a basis
        of the function space spanned by

          r^i * s ^j for i+j <= N
        """
        for idx in self.indices():
            yield TriangleBasisFunction(idx)

    def vandermonde(self):
        from hedge.polynomial import generic_vandermonde

        return generic_vandermonde(
                list(self.unit_nodes()),
                list(self.basis_functions()))

    def grad_basis_functions(self):
        """Get the gradient functions of the basis_functions(),
        in the same order.
        """
        for idx in self.indices():
            yield GradTriangleBasisFunction(idx)

    def grad_vandermonde(self):
        from hedge.polynomial import generic_vandermonde

        return generic_vandermonde(
                list(self.unit_nodes()),
                list(self.grad_basis_functions()))

