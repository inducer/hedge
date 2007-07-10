# Hedge - the Hybrid'n'Easy DG Environment
# Copyright (C) 2007 Andreas Kloeckner
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.




from __future__ import division
import pylinear.array as num
import pylinear.computation as comp
from hedge.tools import AffineMap
from math import sqrt, sin, cos, exp, pi
from pytools import memoize
from hedge.mesh import Triangle, Tetrahedron




__all__ = ["TriangularElement", "TetrahedralElement"]




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




class TriangleWarper:
    def __init__(self, alpha, order):
        self.alpha = alpha
        self.warp = WarpFactorCalculator(order)

        cls = TriangularElement

        from pytools import wandering_element
        from tools import normalize

        vertices = [cls.barycentric_to_equilateral(bary)
                for bary in wandering_element(cls.dimensions+1)]
        all_vertex_indices = range(cls.dimensions+1)
        face_vertex_indices = cls.geometry_class \
                .face_vertices(all_vertex_indices)
        faces_vertices = cls.geometry_class.face_vertices(vertices)

        edgedirs = [normalize(v2-v1) for v1, v2 in faces_vertices]
        opp_vertex_indices = [
            (set(all_vertex_indices) - set(fvi)).__iter__().next()
            for fvi in face_vertex_indices]

        self.loop_info = zip(
                face_vertex_indices, 
                edgedirs, 
                opp_vertex_indices)

    def __call__(self, bp):
        shifts = []

        from operator import add, mul

        for fvi, edgedir, opp_vertex_index in self.loop_info:
            blend = 4*reduce(mul, (bp[i] for i in fvi))
            warp_amount = blend*self.warp(bp[fvi[1]]-bp[fvi[0]]) \
                    * (1 + (self.alpha*bp[opp_vertex_index])**2)
            shifts.append(warp_amount*edgedir)

        return reduce(add, shifts)




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




class TetrahedronBasisFunction:
    def __init__(self, (i, j, k)):
        from hedge.polynomial import jacobi_function
        self.i = i
        self.j = j
        self.f = jacobi_function(0, 0, i)
        self.g = jacobi_function(2*i+1, 0, j)
        self.h = jacobi_function(2*i+2*j+2, 0, k)

    def __call__(self, (r, s, t)):
        try:
            a = -2*(1+r)/(s+t) - 1
        except ZeroDivisionError:
            a = -1

        try:
            b = 2*(1+s)/(1-t) - 1
        except ZeroDivisionError:
            b = -1

        c = t

        return sqrt(8) \
                *self.f(a) \
                *self.g(b) \
                *(1-b)**(self.i) \
                *self.h(c) \
                *(1-c)**(self.i+self.j)




class GradTetrahedronBasisFunction:
    def __init__(self, (i, j, k)):
        from hedge.polynomial import jacobi_function, diff_jacobi_function
        self.i = i
        self.j = j
        self.k = k

        self.f  =      jacobi_function(0, 0, i)
        self.df = diff_jacobi_function(0, 0, i)
        self.g  =      jacobi_function(2*i+1, 0, j)
        self.dg = diff_jacobi_function(2*i+1, 0, j)
        self.h  =      jacobi_function(2*i+2*j+2, 0, k)
        self.dh = diff_jacobi_function(2*i+2*j+2, 0, k)

    def __call__(self, (r, s, t)):
        try:
            a = -2*(1+r)/(s+t) - 1
        except ZeroDivisionError:
            a = -1

        try:
            b = 2*(1+s)/(1-t) - 1
        except ZeroDivisionError:
            b = -1

        c = t

        fa = self.f(a)
        gb = self.g(b)
        hc = self.h(c)

        dfa = self.df(a)
        dgb = self.dg(b)
        dhc = self.dh(c)

        id = self.i
        jd = self.j
        kd = self.k

        # shamelessly stolen from Hesthaven/Warburton's GradSimplex3DP

        # r-derivative
        V3Dr = dfa*(gb*hc);
        if id>0:    V3Dr = V3Dr*((0.5*(1-b))**(id-1))
        if id+jd>0: V3Dr = V3Dr*((0.5*(1-c))**(id+jd-1))

        # s-derivative 
        V3Ds = 0.5*(1+a)*V3Dr;
        tmp = dgb*((0.5*(1-b))**id);
        if id>0:
            tmp = tmp+(-0.5*id)*(gb*(0.5*(1-b))**(id-1))
        if id+jd>0: 
            tmp = tmp*((0.5*(1-c))**(id+jd-1))
        tmp = fa*(tmp*hc)
        V3Ds = V3Ds+tmp

        # t-derivative 
        V3Dt = 0.5*(1+a)*V3Dr+0.5*(1+b)*tmp
        tmp = dhc*((0.5*(1-c))**(id+jd))
        if id+jd>0:
            tmp = tmp-0.5*(id+jd)*(hc*((0.5*(1-c))**(id+jd-1)))
        tmp = fa*(gb*tmp)
        tmp = tmp*((0.5*(1-b))**id)
        V3Dt = V3Dt+tmp

        # normalize
        V3Dr = V3Dr*(2**(2*id+jd+1.5));
        V3Ds = V3Ds*(2**(2*id+jd+1.5));
        V3Dt = V3Dt*(2**(2*id+jd+1.5));

        return [V3Dr, V3Ds, V3Dt]




class Element(object):
    pass




class SimplicialElement(Element):
    # numbering ---------------------------------------------------------------
    def node_count(self):
        """Return the number of interpolation nodes in this element."""
        d = self.dimensions
        o = self.order
        from operator import mul
        from pytools import factorial
        return int(reduce(mul, (o+1+i for i in range(d)))/factorial(d))

    def node_tuples(self):
        """Generate tuples enumerating the node indices present
        in this element. Each tuple has a length equal to the dimension
        of the element. The tuples constituents are non-negative integers
        whose sum is less than or equal to the order of the element.
        
        The order in which these nodes are generated dictates the local 
        node numbering.
        """
        from pytools import generate_nonnegative_integer_tuples_summing_to_at_most
        return generate_nonnegative_integer_tuples_summing_to_at_most(
                self.order, self.dimensions)

    @memoize
    def vertex_indices(self):
        """Return the list of the vertices' indices into the elements' volume vector."""
        from pytools import wandering_element

        result = []

        vertex_tuples = [self.dimensions * (0,)] \
                + list(wandering_element(self.dimensions, wanderer=self.order))
        vti = 0

        for i, nt in enumerate(self.node_tuples()):
            if nt == vertex_tuples[vti]:
                result.append(i)
                vti += 1

        assert vti == len(vertex_tuples)

        return result

    # node wrangling ----------------------------------------------------------
    def equidistant_barycentric_nodes(self):
        """Generate equidistant nodes in barycentric coordinates."""
        for indices in self.node_tuples():
            divided = tuple(i/self.order for i in indices)
            yield (1-sum(divided),) + divided

    def equidistant_equilateral_nodes(self):
        """Generate equidistant nodes in equilateral coordinates."""

        for bary in self.equidistant_barycentric_nodes():
            yield self.barycentric_to_equilateral(bary)

    def equidistant_unit_nodes(self):
        """Generate equidistant nodes in unit coordinates."""

        for bary in self.equidistant_barycentric_nodes():
            yield self.equilateral_to_unit(self.barycentric_to_equilateral(bary))

    @memoize
    def unit_nodes(self):
        """Generate the warped nodes in unit coordinates (r,s,...)."""
        return [self.equilateral_to_unit(node)
                for node in self.equilateral_nodes()]

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





class TriangularElement(SimplicialElement):
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
    A = (-1,-1)
    B = (1,-1)
    C = (-1,1)

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

    Faces are always ordered AB, BC, AC.
    """

    # In case you were wondering: the double backslashes in the docstring
    # are required because single backslashes only escape their subsequent
    # newlines, and thus end up not yielding a correct docstring.

    dimensions = 2
    has_local_jacobians = False
    geometry_class = Triangle

    def __init__(self, order):
        self.order = order

    # numbering ---------------------------------------------------------------
    @memoize
    def face_indices(self):
        """Return a list of face index lists. Each face index list contains
        the local node numbers of the nodes on that face.
        """

        faces = [[], [], []]

        for i, (m, n) in enumerate(self.node_tuples()):
            # face finding
            if n == 0:
                faces[0].append(i)
            if n+m == self.order:
                faces[1].append(i)
            if m == 0:
                faces[2].append(i)

        return faces

    # node wrangling ----------------------------------------------------------
    @staticmethod
    def barycentric_to_equilateral((lambda1, lambda2, lambda3)):
        """Return the equilateral (x,y) coordinate corresponding
        to the barycentric coordinates (lambda1..lambdaN)."""

        # reflects vertices in equilateral coordinates
        return num.array([
            (-lambda1  +lambda2            ),
            (-lambda1  -lambda2  +2*lambda3)/sqrt(3.0)])

    # see doc/hedge-notes.tm
    equilateral_to_unit = AffineMap(
            num.array([[1,-1/sqrt(3)], [0,2/sqrt(3)]]),
                num.array([-1/3,-1/3]))

    def equilateral_nodes(self):
        """Generate warped nodes in equilateral coordinates (x,y)."""

        # port of Hesthaven/Warburton's Nodes2D routine
        # note that the order of the barycentric coordinates is changed
        # match the order of the equilateral vertices 
        
        # Not much is left of the original routine--it was very redundant.
        # The test suite still contains the original code and verifies this
        # one against it.

        # Set optimized parameter alpha, depending on order N
        alpha_opt = [0.0000, 0.0000, 1.4152, 0.1001, 0.2751, 0.9800, 1.0999,
                1.2832, 1.3648, 1.4773, 1.4959, 1.5743, 1.5770, 1.6223, 1.6258]
                  
        try:
            alpha = alpha_opt[self.order-1]
        except IndexError:
            alpha = 5/3

        warp = TriangleWarper(alpha, self.order)

        for bp in self.equidistant_barycentric_nodes():
            yield self.barycentric_to_equilateral(bp) + warp(bp)

    @memoize
    def generate_submesh_indices(self):
        """Return a list of tuples of indices into the node list that
        generate a tesselation of the reference element."""

        node_dict = dict(
                (ituple, idx) 
                for idx, ituple in enumerate(self.node_tuples()))

        result = []
        for i, j in self.node_tuples():
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

          r**i * s**j for i+j <= N
        """
        return [TriangleBasisFunction(idx) for idx in self.node_tuples()]

    def grad_basis_functions(self):
        """Get the gradient functions of the basis_functions(),
        in the same order.
        """
        return [GradTriangleBasisFunction(idx) for idx in self.node_tuples()]

    # face operations ---------------------------------------------------------
    @memoize
    def face_mass_matrix(self):
        from hedge.polynomial import legendre_vandermonde
        unodes = self.unit_nodes()
        face_vandermonde = legendre_vandermonde(
                [unodes[i][0] for i in self.face_indices()[0]],
                self.order)

        return 1/(face_vandermonde*face_vandermonde.T)

    @staticmethod
    def shuffle_face_indices_to_match(face_1_vertices, face_2_vertices, face_2_indices):
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

    def dt_geometric_factor(self, vertices, el):
        area = abs(2*el.map.jacobian)
        semiperimeter = sum(comp.norm_2(vertices[vi1]-vertices[vi2]) 
                for vi1, vi2 in [(0,1), (1,2), (2,0)])/2
        return area/semiperimeter




class TetrahedralElement(SimplicialElement):
    """An arbitrary-order tetrahedral finite element.

    Coordinate systems used:
    ------------------------

    unit coordinates (r,s,t):

               ^ s
               |
               C
              /|\\
             / | \\
            /  |  \\
           /   |   \\
          /   O|    \\
         /   __A-----B---> r
        /_--^ ___--^^
       ,D--^^^
      L 
      
      t

    (squint, and it might start making sense...)

    O=( 0, 0, 0)
    A=(-1,-1,-1)
    B=(+1,-1,-1)
    C=(-1,+1,-1)
    D=(-1,-1,+1)

    equilateral coordinates (x,y,z):

    O = (0,0)
    A = (-1,-1/sqrt(3),-1/sqrt(6))
    B = ( 1,-1/sqrt(3),-1/sqrt(6))
    C = ( 0, 2/sqrt(3),-1/sqrt(6))
    D = ( 0,         0, 3/sqrt(6))

    When global vertices are passed in, they are mapped to the 
    reference vertices A, B, C, D in order.

    Faces are always ordered ABC, ABD, ACD, BCD.
    """

    # In case you were wondering: the double backslashes in the docstring
    # above are required because single backslashes only escape their subsequent
    # newlines, and thus end up not yielding a correct docstring.

    dimensions = 3
    has_local_jacobians = False
    geometry_class = Tetrahedron

    def __init__(self, order):
        self.order = order

    # numbering ---------------------------------------------------------------
    @memoize
    def face_indices(self):
        """Return a list of face index lists. Each face index list contains
        the local node numbers of the nodes on that face.
        """

        faces = [[], [], [], []]

        for i, (m,n,o) in enumerate(self.node_tuples()):
            if o == 0:
                faces[0].append(i)
            if n == 0:
                faces[1].append(i)
            if m == 0:
                faces[2].append(i)
            if n+m+o == self.order:
                faces[3].append(i)

        return faces

    # node wrangling ----------------------------------------------------------
    @staticmethod
    def barycentric_to_equilateral((lambda1, lambda2, lambda3, lambda4)):
        """Return the equilateral (x,y) coordinate corresponding
        to the barycentric coordinates (lambda1..lambdaN)."""

        # reflects vertices in equilateral coordinates
        return num.array([
            (-lambda1  +lambda2                        ),
            (-lambda1  -lambda2  +2*lambda3            )/sqrt(3.0),
            (-lambda1  -lambda2  -  lambda3  +3*lambda4)/sqrt(6.0),
            ])

    # see doc/hedge-notes.tm
    equilateral_to_unit = AffineMap(
            num.array([
                [1,-1/sqrt(3),-1/sqrt(6)], 
                [0, 2/sqrt(3),-1/sqrt(6)],
                [0,         0,   sqrt(6)/2]
                ]),
                num.array([-1/2,-1/2,-1/2]))

    def equilateral_nodes(self):
        """Generate warped nodes in equilateral coordinates (x,y)."""

        # port of Hesthaven/Warburton's Nodes3D routine

        # Set optimized parameter alpha, depending on order N
        alpha_opt = [0,0,0,0.1002, 1.1332,1.5608,1.3413,1.2577,1.1603,
                1.10153,0.6080,0.4523,0.8856,0.8717,0.9655]

        try:
            alpha = alpha_opt[self.order-1]
        except IndexError:
            alpha = 1

        from pytools import wandering_element

        vertices = [self.barycentric_to_equilateral(bary)
                for bary in wandering_element(self.dimensions+1)]
        all_vertex_indices = range(self.dimensions+1)
        face_vertex_indices = self.geometry_class \
                .face_vertices(all_vertex_indices)
        faces_vertices = self.geometry_class \
                .face_vertices(vertices)

        bary_points = list(self.equidistant_barycentric_nodes())
        equi_points = [self.barycentric_to_equilateral(bp) 
                for bp in bary_points]

        from tools import normalize
        from operator import add, mul

        tri_warp = TriangleWarper(alpha, self.order)

        for fvi, (v1, v2, v3) in zip(face_vertex_indices, faces_vertices):
            # find directions spanning the face: "base" and "altitude"
            directions = [normalize(v2-v1), normalize((v3)-(v1+v2)/2)]

            # the two should be orthogonal
            assert abs(directions[0]*directions[1]) < 1e-16

            # find the vertex opposite to the current face
            opp_vertex_index = (set(all_vertex_indices) - set(fvi)).__iter__().next()

            shifted = []
            for bp, ep in zip(bary_points, equi_points):
                face_bp = [bp[i] for i in fvi]

                blend = reduce(mul, face_bp) * (1+alpha*bp[opp_vertex_index])**2

                for i in fvi:
                    denom = bp[i] + 0.5*bp[opp_vertex_index]
                    if abs(denom) > 1e-12:
                        blend /= denom
                    else:
                        blend = 0.5 # each edge gets shifted twice
                        break
                    
                shifted.append(ep + blend*reduce(add,
                    (tw*dir for tw, dir in zip(tri_warp(face_bp), directions))))

            equi_points = shifted

        return equi_points

    @memoize
    def generate_submesh_indices(self):
        """Return a list of tuples of indices into the node list that
        generate a tesselation of the reference element."""

        node_dict = dict(
                (ituple, idx) 
                for idx, ituple in enumerate(self.node_tuples()))

        def add_tuples(a, b):
            return tuple(ac+bc for ac, bc in zip(a,b))

        def try_add_tet(d1, d2, d3, d4):
            try:
                result.append((
                    node_dict[add_tuples(current, d1)],
                    node_dict[add_tuples(current, d2)],
                    node_dict[add_tuples(current, d3)],
                    node_dict[add_tuples(current, d4)],
                    ))
            except KeyError, e:
                pass

        result = []
        for current in self.node_tuples():
            # this is a tesselation of a cube into six tets.
            # subtets that fall outside of the master tet are simply not added.

            # positively oriented
            #try_add_tet((0,0,0), (1,0,0), (0,1,0), (0,0,1))
            #try_add_tet((1,0,1), (1,0,0), (0,0,1), (0,1,0))
            #try_add_tet((1,0,1), (0,1,1), (0,1,0), (0,0,1))

            #try_add_tet((1,0,0), (0,1,0), (1,0,1), (1,1,0))
            #try_add_tet((0,1,1), (0,1,0), (1,1,0), (1,0,1))
            #try_add_tet((0,1,1), (1,1,1), (1,0,1), (1,1,0))

            # negatively oriented
            try_add_tet((0,0,0), (1,0,0), (0,0,1), (0,1,0))
            try_add_tet((1,0,1), (1,0,0), (0,1,0), (0,0,1))
            try_add_tet((1,0,1), (0,1,1), (0,0,1), (0,1,0))

            try_add_tet((1,0,0), (0,1,0), (1,1,0), (1,0,1))
            try_add_tet((0,1,1), (0,1,0), (1,0,1), (1,1,0))
            try_add_tet((0,1,1), (1,1,1), (1,1,0), (1,0,1))
        return result

    # basis functions ---------------------------------------------------------
    def basis_functions(self):
        """Get a sequence of functions that form a basis
        of the function space spanned by

          r**i * s**j * t**k  for  i+j+k <= order
        """
        return [TetrahedronBasisFunction(idx) for idx in self.node_tuples()]

    def grad_basis_functions(self):
        """Get the (r,s,...) gradient functions of the basis_functions(),
        in the same order.
        """
        return [GradTetrahedronBasisFunction(idx) for idx in self.node_tuples()]

    # face operations ---------------------------------------------------------
    @memoize
    def face_mass_matrix(self):
        from hedge.polynomial import generic_vandermonde
        unodes = self.unit_nodes()

        node_tuples = list(self.node_tuples())
        basis = [TriangleBasisFunction(node_tuples[i][:2]) 
                for i in self.face_indices()[0]]
        face_vandermonde = generic_vandermonde(
                [unodes[i][:2] for i in self.face_indices()[0]],
                [basis[i] for i in self.face_indices()[0]],
                )

        return 1/(face_vandermonde*face_vandermonde.T)

    def shuffle_face_indices_to_match(self, face_1_vertices, face_2_vertices, face_2_indices):
        (a,b,c) = face_1_vertices
        f2_tuple = face_2_vertices

        try:
            idx_map = self._shuffle_face_idx_map
        except AttributeError:
            idx_map = self._shuffle_face_idx_map = {}
            idx = 0
            for j in range(0, self.order+1):
                for i in range(0, self.order+1-j):
                    self._shuffle_face_idx_map[i,j] = idx
                    idx += 1

        # flip and shift_left generate S_3
        def flip(indices):
            """Flip the indices along the unit hypotenuse."""
            result = []
            for j in range(0, self.order+1):
                for i in range(0, self.order+1-j):
                    result.append(indices[idx_map[j,i]])
            return result

        def shift_left(indices):
            """Rotate all edges to the left."""
            result = len(indices)*[0]
            idx = 0
            for j in range(0, self.order+1):
                for i in range(0, self.order+1-j):
                    result[idx_map[j, self.order-i-j]] = indices[idx]
                    idx += 1
            return result

        # yay, enumerate S_3 by hand
        if f2_tuple == (a,b,c):
            return face_2_indices
        elif f2_tuple == (a,c,b):
            return flip(face_2_indices)
        elif f2_tuple == (b,c,a):
            # (b,c,a) -sl-> (c,a,b) -sl-> (a,b,c)
            return shift_left(shift_left(face_2_indices))
        elif f2_tuple == (b,a,c):
            # (b,a,c) -sl-> (a,c,b) -fl-> (a,b,c)
            return flip(shift_left(face_2_indices))
        elif f2_tuple == (c,a,b):
            # (c,a,b) -sl-> (a,b,c)
            return shift_left(face_2_indices)
        elif f2_tuple == (c,b,a):
            # (c,b,a) -fl-> (c,a,b) -sl-> (a,b,c)
            return shift_left(flip(face_2_indices))
        else:
            raise ValueError, "face_2_vertices is not a permutation of face_1_vertices"

    # time step scaling -------------------------------------------------------
    def dt_non_geometric_factor(self):
        unodes = self.unit_nodes()
        return 2/3*min(
                min(comp.norm_2(unodes[fvi+1]-unodes[fvi])
                    for fvi in range(len(face_indices)-1))
                for face_indices in self.face_indices())

    def dt_geometric_factor(self, vertices, el):
        return abs(el.map.jacobian)/max(abs(fj) for fj in el.face_jacobians)


