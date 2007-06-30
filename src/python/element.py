from __future__ import division
import pylinear.array as num
import pylinear.computation as comp
from hedge.tools import AffineMap
from math import sqrt, sin, cos, exp, pi
from pytools import memoize




__all__ = ["TriangularElement"]




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

    When global vertices are passed in, they are mapped to the 
    reference vertices A, B, C in order.

    Faces are always ordered AB, BC, CA.
    """

    # In case you were wondering: the double backslashes in the docstring
    # are required because single backslashes only escape their subsequent
    # newlines, and thus end up not yielding a correct docstring.

    def __init__(self, order):
        self.order = order
        self.dimensions = 2
        self.has_local_jacobians = False

    def get_map_unit_to_global(self, vertices):
        """Return an affine map that maps the unit coordinates of the reference
        element to a global element at a location given by its `vertices'.
        """
        mat = num.zeros((2,2))
        mat[:,0] = (vertices[1] - vertices[0])/2
        mat[:,1] = (vertices[2] - vertices[0])/2
        return AffineMap(mat, (vertices[1]+vertices[2])/2)

    # numbering ---------------------------------------------------------------
    def node_count(self):
        """Return the number of interpolation nodes in this element."""
        return int((self.order+1)*(self.order+2)/2)

    def node_indices(self):
        """Generate tuples (col,row) enumerating the nodes present
        in this triangle. The order in which these nodes are generated
        also dictates the local node numbering.

        The following invariants hold:
        - col, row >= 0
        - col+row <= self.order
        """

        for n in range(0, self.order+1):
            for m in range(0, self.order+1-n):
                yield m,n

    @memoize
    def face_indices(self):
        """Return a list of face index lists. Each face index list contains
        the local node numbers of the nodes on that face, numbered 
        counterclockwise.
        """

        faces = [[], [], []]

        for i, (m, n) in enumerate(self.node_indices()):
            # face finding
            if n == 0:
                faces[0].append(i)
            if n+m == self.order:
                faces[1].append(i)
            if m == 0:
                faces[2].append(i)

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
        """Return the equilateral (x,y) coordinate corresponding
        to the barycentric coordinates (lambda1..lambdaN)."""
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
            alpha = alpha_opt[self.order-1]
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

    @memoize
    def generate_submesh_indices(self):
        """Return a list of triples of indices into the node list that
        generate a triangulation of the reference triangle, using the
        interpolation nodes."""

        node_dict = dict(
                (ituple, idx) 
                for idx, ituple in enumerate(self.node_indices()))

        result = []
        for i, j in self.node_indices():
            if i+j < self.order:
                result.append(
                        (node_dict[i,j], node_dict[i+1,j], node_dict[i,j+1]))
            if i+j < self.order-1:
                result.append(
                    (node_dict[i+1,j+1], node_dict[i,j+1], node_dict[i+1,j]))
        return result

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
    def inverse_mass_matrix(self):
        """Return the inverse of the mass matrix of the unit element 
        with respect to the nodal coefficients. Divide by the Jacobian 
        to obtain the global mass matrix.
        """

        # see doc/hedge-notes.tm
        v = self.vandermonde()
        return v*v.T

    @memoize
    def mass_matrix(self):
        """Return the mass matrix of the unit element with respect 
        to the nodal coefficients. Multiply by the Jacobian to obtain
        the global mass matrix.
        """

        # see doc/hedge-notes.tm
        return 1/self.inverse_mass_matrix()

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

        face_lengths = [comp.norm_2(fn) for fn in raw_normals]
        return [n/fl for n, fl in zip(raw_normals, face_lengths)], \
                face_lengths

    def shuffle_face_indices_to_match(self, face_1_vertices, face_2_vertices, face_2_indices):
        assert set(face_1_vertices) == set(face_2_vertices)
        if face_1_vertices != face_2_vertices:
            assert face_1_vertices[::-1] == face_2_vertices
            return face_2_indices[::-1]
        else:
            return face_2_indices

    # time step scaling -------------------------------------------------------
    def dt_non_geometric_factor(self):
        unodes = self.unit_nodes()
        return 2/3*min(
                min(comp.norm_2(unodes[fvi+1]-unodes[fvi])
                    for fvi in range(len(face_indices)-1))
                for face_indices in self.face_indices())

    def dt_geometric_factor(self, vertices, map):
        area = abs(2*map.jacobian)
        semiperimeter = sum(comp.norm_2(vertices[vi1]-vertices[vi2]) 
                for vi1, vi2 in [(0,1), (1,2), (2,0)])/2
        return area/semiperimeter

