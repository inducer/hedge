from __future__ import division
import pylinear.array as num
import pylinear.computation as comp
from hedge.tools import AffineMap
from math import sqrt, sin, cos, exp, pi
from pytools import memoize




class WarpFactorCalculator:
    """Calculator for Warburton's warp factor.

    See T. Warburton,
    "An explicit construction of interpolation nodes on the simplex"
    Journal of Engineering Mathematics Vol 56, No 3, p. 247-262, 2006
    """

    def __init__(self, N):
        from hedge.quadrature import legendre_gauss_lobatto_points
        from hedge.interpolation import newton_interpolation_function

        import numpy
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
        return [
            # df/dr
            2*sqrt(2) * g_s * one_s**(i-1) * df_a,
            # df/ds
            sqrt(2)*(
                f_a * one_s**i * dg_s
                +(2*r+2) * g_s * one_s**(i-2) * df_a
                -i * f_a * g_s * one_s**(i-1)
                )]




class TriangularElement:
    """An arbitrary-order triangular finite element.

    Coordinate systems used:
    ------------------------

    unit coordinates (r,s):

    C
    |\\
    | \\
    |  O
    |   \\
    |    \\
    A-----B

    O = (0,0)
    A=(-1,-1)
    B=(1,-1)
    C=(-1,1)

    equilateral coordinates (x,y):

            C
           / \\
          /   \\
         /     \\
        /   O   \\
       /         \\
      A-----------B 

    O = (0,0)
    A = (-1,-1/sqrt(3))
    B = (1,-1/sqrt(3))
    C = (0,2/sqrt(3))

    A, B, C is also the ordering of vertices, and
    AB, BC, CA is the ordering of the faces.
    """

    # In case you were wondering: the double backslashes in the docstring
    # are required because single backslashes only escape their subsequent
    # newlines. It looks cool, too.

    def __init__(self, order):
        self.order = order

    def get_map_unit_to_global(self, vertices):
        """Return an affine map that maps the unit coordinates of the reference
        element to a global element at a location given by its `vertices'.
        """
        mat = num.zeros((2,2))
        mat[:,0] = vertices[1] - vertices[0]
        mat[:,1] = vertices[2] - vertices[0]
        return AffineMap(mat, vertices[0])

    # numbering ---------------------------------------------------------------
    def node_indices(self):
        for n in range(0, self.order+1):
            for m in range(0, self.order+1-n):
                yield m,n

    @memoize
    def face_indices(self):
        faces = [[], [], []]

        i = 0
        for m, n in self.node_indices():
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

    # node generators ---------------------------------------------------------
    def equidistant_barycentric_nodes(self):
        """Generate equidistant nodes in barycentric coordinates
        of order N.
        """
        for m, n in self.node_indices():
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
        """Generate equidistant nodes in equilateral coordinates."""

        for bary in self.equidistant_barycentric_nodes():
            yield self.barycentric_to_equilateral(bary)

    def equidistant_unit_nodes(self):
        """Generate equidistant nodes in unit coordinates."""

        for bary in self.equidistant_barycentric_nodes():
            yield self.equilateral_to_unit(self.barycentric_to_equilateral(bary))

    def equilateral_nodes(self):
        """Generate warped nodes in equilateral coordinates (x,y)."""

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

    @memoize
    def unit_nodes(self):
        """Generate the warped nodes in unit coordinates (r,s)."""
        return [self.equilateral_to_unit(node)
                for node in self.equilateral_nodes()]

    # basis functions ---------------------------------------------------------
    def basis_functions(self):
        """Get a sequence of functions that form a basis
        of the function space spanned by

          r^i * s ^j for i+j <= N
        """
        return [TriangleBasisFunction(idx) for idx in self.node_indices()]

    def grad_basis_functions(self):
        """Get the gradient functions of the basis_functions(),
        in the same order.
        """
        return [GradTriangleBasisFunction(idx) for idx in self.node_indices()]

    # matrices ----------------------------------------------------------------
    @memoize
    def vandermonde(self):
        from hedge.polynomial import generic_vandermonde

        return generic_vandermonde(
                list(self.unit_nodes()),
                list(self.basis_functions()))

    @memoize
    def mass_matrix(self):
        """Return the mass matrix of the unit element with respect 
        to the nodal coefficients. Multiply by the Jacobian to obtain
        the global mass matrix.
        """

        # see doc/hedge-notes.tm
        v = self.vandermonde()
        return 1/(v*v.T)

    @memoize
    def grad_vandermonde(self):
        """Compute the Vandermonde matrices of the grad_basis_functions().
        Return a list of these matrices."""

        from hedge.polynomial import generic_multi_vandermonde

        return generic_multi_vandermonde(
                list(self.unit_nodes()),
                list(self.grad_basis_functions()))

    @memoize
    def differentiation_matrices(self):
        """Return matrices that map the nodal values of a function
        to the nodal values of its derivative in each of the unit
        coordinate directions.
        """

        # see doc/hedge-notes.tm
        v = self.vandermonde()
        return [v <<num.leftsolve>> vdiff for vdiff in self.grad_vandermonde()]

    def global_differentiation_matrices(self, inverse_affine_map):
        """Return matrices that map the nodal values of a function
        to the nodal values of its derivative in each of the global
        coordinate directions.

        WARNING: Every possible use of this routine is inefficient.
        You shouldn't store its results, because that's a helluva lot 
        of useless storage. On the other hand, recomputing this
        matrix every time you need it is also obviously silly.

        What you should really be doing is apply the differentiation
        matrices and then linearly combine the results according
        to the matrix.

        It is thus mostly meant as a help to get you off the ground,
        and as a guideline and as verification for your (hopefully more 
        efficient) coding.
        """

        from operator import add

        # see doc/hedge-notes.tm
        dmats = self.differentiation_matrices()
        return [reduce(add, (dmat*coeff for dmat, coeff in zip(dmats, col)))
                for col in inverse_affine_map.matrix.T]

    # face operations ---------------------------------------------------------
    @memoize
    def face_mass_matrix(self):
        from hedge.polynomial import legendre_vandermonde
        unodes = self.unit_nodes()
        face_vandermonde = legendre_vandermonde(
                [unodes[i][0] for i in self.face_indices()[0]],
                self.order)

        return 1/(face_vandermonde*face_vandermonde.T)

    def face_normals_and_jacobians(self, affine_map):
        """Compute the normals and face jacobians of the unit triangle
        transformed according to `affine_map'.

        Returns a pair of lists [normals], [jacobians].
        """
        def sign(x):
            if x > 0: 
                return 1
            else: 
                return -1

        m = affine_map.matrix
        orient = sign(affine_map.jacobian)
        face1 = m[:,1] - m[:,0]
        raw_normals = [
                orient*num.array([m[1,0], -m[0,0]]),
                orient*num.array([face1[1], -face1[0]]),
                orient*num.array([-m[1,1], m[0,1]]),
                ]
        face_jacobians = [comp.norm_2(fn) for fn in raw_normals]
        return [n/fj for n, fj in zip(raw_normals, face_jacobians)], \
                face_jacobians
