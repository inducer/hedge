"""Local function space representation."""

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
import numpy.linalg as la
import pyublas
from hedge.tools import AffineMap
import hedge._internal
from math import sqrt, sin, cos, exp, pi
from pytools import memoize
import hedge.mesh




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

        # Find lgl and equidistant interpolation points
        r_lgl = legendre_gauss_lobatto_points(N)
        r_eq  = numpy.linspace(-1,1,N+1)

        self.int_f = newton_interpolation_function(r_eq, r_lgl - r_eq)

    def __call__(self, x):
        if abs(x) > 1-1e-10:
            return 0
        else:
            return self.int_f(x)/(1-x**2)




class FaceVertexMismatch(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)




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
        face_vertex_indices = cls.geometry \
                .face_vertices(all_vertex_indices)
        faces_vertices = cls.geometry.face_vertices(vertices)

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




TriangleBasisFunction = hedge._internal.TriangleBasisFunction
GradTriangleBasisFunction = hedge._internal.GradTriangleBasisFunction
TetrahedronBasisFunction = hedge._internal.TetrahedronBasisFunction
GradTetrahedronBasisFunction = hedge._internal.GradTetrahedronBasisFunction




class Element(object):
    def basis_functions(self):
        """Get a sequence of functions that form a basis of the approximation space."""
        raise NotImplementedError

    def grad_basis_functions(self):
        """Get the gradient functions of the basis_functions(), in the same order."""
        raise NotImplementedError

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
        return numpy.dot(v, v.T)

    @memoize
    def mass_matrix(self):
        """Return the mass matrix of the unit element with respect 
        to the nodal coefficients. Multiply by the Jacobian to obtain
        the global mass matrix.
        """

        return numpy.asarray(la.inv(self.inverse_mass_matrix()), order="C")

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

        from hedge.tools import leftsolve
        # see doc/hedge-notes.tm
        v = self.vandermonde()
        return [leftsolve(v, vdiff) for vdiff in self.grad_vandermonde()]




class SimplicialElement(Element):
    # numbering ---------------------------------------------------------------
    def node_count(self):
        """Return the number of interpolation nodes in this element."""
        d = self.dimensions
        o = self.order
        from operator import mul
        from pytools import factorial
        return int(reduce(mul, (o+1+i for i in range(d)))/factorial(d))

    @memoize
    def vertex_indices(self):
        """Return the list of the vertices' node indices."""
        from pytools import wandering_element

        result = []

        node_tup_to_idx = dict(
                (ituple, idx) 
                for idx, ituple in enumerate(self.node_tuples()))

        vertex_tuples = [self.dimensions * (0,)] \
                + list(wandering_element(self.dimensions, wanderer=self.order))

        return [node_tup_to_idx[vt] for vt in vertex_tuples]

    @memoize
    def face_indices(self):
        """Return a list of face index lists. Each face index list contains
        the local node numbers of the nodes on that face.
        """

        node_tup_to_idx = dict(
                (ituple, idx) 
                for idx, ituple in enumerate(self.node_tuples()))

        from pytools import generate_nonnegative_integer_tuples_summing_to_at_most

        enum_order_nodes_gen = generate_nonnegative_integer_tuples_summing_to_at_most(
                    self.order, self.dimensions)

        faces = [[] for i in range(self.dimensions+1)]

        for node_tup in enum_order_nodes_gen:
            for face_idx in self.faces_for_node_tuple(node_tup):
                faces[face_idx].append(node_tup_to_idx[node_tup])

        return [tuple(fi) for fi in faces]

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

    # basis functions ---------------------------------------------------------
    def generate_mode_identifiers(self):
        """Generate a hashable objects identifying each basis function, in order.

        The output from this function is required to be in the same order
        as that of L{basis_functions} and L{grad_basis_functions}, and thereby
        also from L{vandermonde}.
        """
        from pytools import generate_nonnegative_integer_tuples_summing_to_at_most

        return generate_nonnegative_integer_tuples_summing_to_at_most(
                self.order, self.dimensions)

    # time step scaling -------------------------------------------------------
    def dt_non_geometric_factor(self):
        unodes = self.unit_nodes()
        vertex_indices = self.vertex_indices()

        return 2/3*\
                min(min(min(
                    la.norm(unodes[face_node_index]-unodes[vertex_index])
                    for vertex_index in vertex_indices
                    if vertex_index != face_node_index)
                    for face_node_index in face_indices)
                    for face_indices in self.face_indices())





class IntervalElementBase(SimplicialElement):
    dimensions = 1
    has_local_jacobians = False
    geometry = hedge.mesh.Interval

    # numbering ---------------------------------------------------------------
    @memoize
    def node_tuples(self):
        """Generate tuples enumerating the node indices present
        in this element. Each tuple has a length equal to the dimension
        of the element. The tuples constituents are non-negative integers
        whose sum is less than or equal to the order of the element.
        
        The order in which these nodes are generated dictates the local 
        node numbering.
        """
        return [(i,) for i in range(self.order+1)]

    def faces_for_node_tuple(self, node_idx):
        """Return the list of face indices of faces on which the node 
        represented by C{node_idx} lies.
        """

        if node_idx == (0,):
            return [0]
        elif node_idx == (self.order,):
            return [1]
        else:
            return []

    # node wrangling ----------------------------------------------------------
    @memoize
    def get_submesh_indices(self):
        """Return a list of tuples of indices into the node list that
        generate a tesselation of the reference element."""

        return [(i,i+1) for i in range(self.order)]

    # face operations ---------------------------------------------------------
    @memoize
    def face_mass_matrix(self):
        return numpy.array([[1]], dtype=float)

    @staticmethod
    def get_face_index_shuffle_to_match(face_1_vertices, face_2_vertices):
        if set(face_1_vertices) != set(face_2_vertices):
            raise FaceVertexMismatch("face vertices do not match")

        class IntervalFaceIndexShuffle:
            def __hash__(self):
                return 0x3472477

            def __eq__(self, other):
                return True

            def __call__(self, indices):
                return indices

        return IntervalFaceIndexShuffle()





class IntervalElement(IntervalElementBase):
    """An arbitrary-order polynomial finite interval element.

    Coordinate systems used:
    ========================

    unit coordinates (r)::

    ---[--------0--------]--->
       -1                1
    """

    def __init__(self, order):
        self.order = order

    # node wrangling ----------------------------------------------------------
    def nodes(self):
        """Generate warped nodes in unit coordinates (r,)."""

        from hedge.quadrature import legendre_gauss_lobatto_points
        return [numpy.array([x]) for x in legendre_gauss_lobatto_points(self.order)]

    equilateral_nodes = nodes
    unit_nodes = nodes

    # basis functions ---------------------------------------------------------
    @memoize
    def basis_functions(self):
        """Get a sequence of functions that form a basis of the approximation space.

        The approximation space is spanned by the polynomials:::

          r**i for i <= N
        """
        from hedge.polynomial import LegendreFunction

        class VectorLF:
            def __init__(self, n):
                self.lf = LegendreFunction(n)

            def __call__(self, x):
                return self.lf(x[0])

        return [VectorLF(idx[0]) for idx in self.generate_mode_identifiers()]

    def grad_basis_functions(self):
        """Get the gradient functions of the basis_functions(), in the same order."""
        from hedge.polynomial import DiffLegendreFunction

        class DiffVectorLF:
            def __init__(self, n):
                self.dlf = DiffLegendreFunction(n)

            def __call__(self, x):
                return numpy.array([self.dlf(x[0])])

        return [DiffVectorLF(idx[0]) 
            for idx in self.generate_mode_identifiers()]

    # time step scaling -------------------------------------------------------
    def dt_non_geometric_factor(self):
        unodes = self.unit_nodes()
        return la.norm(unodes[0] - unodes[1])

    def dt_geometric_factor(self, vertices, el):
        return abs(el.map.jacobian)




class TriangularElement(SimplicialElement):
    """An arbitrary-order triangular finite element.

    Coordinate systems used:
    ========================

    unit coordinates (r,s)::

    C
    |\\
    | \\
    |  O
    |   \\
    |    \\
    A-----B

    Points in unit coordinates::

        O = (0,0)
        A = (-1,-1)
        B = (1,-1)
        C = (-1,1)

    equilateral coordinates (x,y)::

            C
           / \\
          /   \\
         /     \\
        /   O   \\
       /         \\
      A-----------B 

    Points in equilateral coordinates::

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
    geometry = hedge.mesh.Triangle

    def __init__(self, order):
        self.order = order

    # numbering ---------------------------------------------------------------
    @memoize
    def node_tuples(self):
        """Generate tuples enumerating the node indices present
        in this element. Each tuple has a length equal to the dimension
        of the element. The tuples constituents are non-negative integers
        whose sum is less than or equal to the order of the element.
        
        The order in which these nodes are generated dictates the local 
        node numbering.
        """
        from pytools import generate_nonnegative_integer_tuples_summing_to_at_most
        node_tups = list(generate_nonnegative_integer_tuples_summing_to_at_most(
                self.order, self.dimensions))

        faces_to_nodes = {}
        for node_tup in node_tups:
            faces_to_nodes.setdefault(
                    frozenset(self.faces_for_node_tuple(node_tup)),
                    []).append(node_tup)

        result = []
        def add_face_nodes(faces):
            result.extend(faces_to_nodes.get(frozenset(faces), []))

        add_face_nodes([])
        add_face_nodes([0])
        add_face_nodes([0,1])
        add_face_nodes([1])
        add_face_nodes([1,2])
        add_face_nodes([2])
        add_face_nodes([0,2])

        assert set(result) == set(node_tups)
        assert len(result) == len(node_tups)

        return result

    def faces_for_node_tuple(self, node_tuple):
        """Return the list of face indices of faces on which the node 
        represented by C{node_tuple} lies.
        """
        m, n = node_tuple

        result = []
        if n == 0:
            result.append(0)
        if n+m == self.order:
            result.append(1)
        if m == 0:
            result.append(2)
        return result

    # node wrangling ----------------------------------------------------------
    @staticmethod
    def barycentric_to_equilateral((lambda1, lambda2, lambda3)):
        """Return the equilateral (x,y) coordinate corresponding
        to the barycentric coordinates (lambda1..lambdaN)."""

        # reflects vertices in equilateral coordinates
        return numpy.array([
            (-lambda1  +lambda2            ),
            (-lambda1  -lambda2  +2*lambda3)/sqrt(3.0)])

    # see doc/hedge-notes.tm
    equilateral_to_unit = AffineMap(
            numpy.array([[1,-1/sqrt(3)], [0,2/sqrt(3)]]),
                numpy.array([-1/3,-1/3]))

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
    def get_submesh_indices(self):
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
    @memoize
    def basis_functions(self):
        """Get a sequence of functions that form a basis of the approximation space.

        The approximation space is spanned by the polynomials:::

          r**i * s**j for i+j <= N
        """
        return [TriangleBasisFunction(*idx) for idx in 
                self.generate_mode_identifiers()]

    def grad_basis_functions(self):
        """Get the gradient functions of the basis_functions(), in the same order."""
        return [GradTriangleBasisFunction(*idx) for idx in 
                self.generate_mode_identifiers()]

    # face operations ---------------------------------------------------------
    @memoize
    def face_mass_matrix(self):
        from hedge.polynomial import legendre_vandermonde
        unodes = self.unit_nodes()
        face_vandermonde = legendre_vandermonde(
                [unodes[i][0] for i in self.face_indices()[0]],
                self.order)

        return numpy.asarray(
                la.inv(
                    numpy.dot(face_vandermonde, face_vandermonde.T)),
                order="C")

    @staticmethod
    def get_face_index_shuffle_to_match(face_1_vertices, face_2_vertices):
        if set(face_1_vertices) != set(face_2_vertices):
            raise FaceVertexMismatch("face vertices do not match")

        class TriangleFaceIndexShuffle:
            def __init__(self, operations):
                self.operations = operations

            def __hash__(self):
                return hash(self.operations)

            def __eq__(self, other):
                return self.operations == other.operations

            def __call__(self, indices):
                for op in self.operations:
                    if op == "flip":
                        indices = indices[::-1]
                    else:
                        raise RuntimeError, "invalid operation"
                return indices


        if face_1_vertices != face_2_vertices:
            assert face_1_vertices[::-1] == face_2_vertices
            return TriangleFaceIndexShuffle(("flip",))
        else:
            return TriangleFaceIndexShuffle(())

    # time step scaling -------------------------------------------------------
    def dt_geometric_factor(self, vertices, el):
        area = abs(2*el.map.jacobian)
        semiperimeter = sum(la.norm(vertices[vi1]-vertices[vi2]) 
                for vi1, vi2 in [(0,1), (1,2), (2,0)])/2
        return area/semiperimeter




class TetrahedralElement(SimplicialElement):
    """An arbitrary-order tetrahedral finite element.

    Coordinate systems used:
    ========================

    unit coordinates (r,s,t)::

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
    t L 
      
    (squint, and it might start making sense...)

    Points in unit coordinates::

        O=( 0, 0, 0)
        A=(-1,-1,-1)
        B=(+1,-1,-1)
        C=(-1,+1,-1)
        D=(-1,-1,+1)

    Points in equilateral coordinates (x,y,z)::

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
    geometry = hedge.mesh.Tetrahedron

    def __init__(self, order):
        self.order = order

    # numbering ---------------------------------------------------------------
    @memoize
    def node_tuples(self):
        """Generate tuples enumerating the node indices present
        in this element. Each tuple has a length equal to the dimension
        of the element. The tuples constituents are non-negative integers
        whose sum is less than or equal to the order of the element.
        
        The order in which these nodes are generated dictates the local 
        node numbering.
        """
        from pytools import generate_nonnegative_integer_tuples_summing_to_at_most
        node_tups = list(generate_nonnegative_integer_tuples_summing_to_at_most(
                self.order, self.dimensions))

        if False:
            # hand-tuned node order
            faces_to_nodes = {}
            for node_tup in node_tups:
                faces_to_nodes.setdefault(
                        frozenset(self.faces_for_node_tuple(node_tup)),
                        []).append(node_tup)

            result = []
            def add_face_nodes(faces):
                result.extend(faces_to_nodes.get(frozenset(faces), []))

            add_face_nodes([0,3])
            add_face_nodes([0])
            add_face_nodes([0,2])
            add_face_nodes([0,1])
            add_face_nodes([0,1,2])
            add_face_nodes([0,1,3])
            add_face_nodes([0,2,3])
            add_face_nodes([1])
            add_face_nodes([1,2])
            add_face_nodes([1,2,3])
            add_face_nodes([1,3])
            add_face_nodes([2])
            add_face_nodes([2,3])
            add_face_nodes([3])
            add_face_nodes([])

            assert set(result) == set(node_tups)
            assert len(result) == len(node_tups)

        if True:
            # average-sort heuristic node order
            from pytools import average

            def order_number_for_node_tuple(nt):
                faces = self.faces_for_node_tuple(nt)
                if not faces:
                    return -1
                elif len(faces) >= 3:
                    return 1000
                else:
                    return average(faces)

            def cmp_node_tuples(nt1, nt2):
                return cmp(
                        order_number_for_node_tuple(nt1), 
                        order_number_for_node_tuple(nt2))

            result = node_tups
            #result.sort(cmp_node_tuples)

        for i, nt in enumerate(result):
            fnt = self.faces_for_node_tuple(nt)
            #print i, nt, fnt

        return result

    def faces_for_node_tuple(self, node_tuple):
        """Return the list of face indices of faces on which the node 
        represented by C{node_tuple} lies.
        """
        m,n,o = node_tuple
        result = []

        if o == 0:
            result.append(0)
        if n == 0:
            result.append(1)
        if m == 0:
            result.append(2)
        if n+m+o == self.order:
            result.append(3)

        return result

    # node wrangling ----------------------------------------------------------
    @staticmethod
    def barycentric_to_equilateral((lambda1, lambda2, lambda3, lambda4)):
        """Return the equilateral (x,y) coordinate corresponding
        to the barycentric coordinates (lambda1..lambdaN)."""

        # reflects vertices in equilateral coordinates
        return numpy.array([
            (-lambda1  +lambda2                        ),
            (-lambda1  -lambda2  +2*lambda3            )/sqrt(3.0),
            (-lambda1  -lambda2  -  lambda3  +3*lambda4)/sqrt(6.0),
            ])

    # see doc/hedge-notes.tm
    equilateral_to_unit = AffineMap(
            numpy.array([
                [1,-1/sqrt(3),-1/sqrt(6)], 
                [0, 2/sqrt(3),-1/sqrt(6)],
                [0,         0,   sqrt(6)/2]
                ]),
                numpy.array([-1/2,-1/2,-1/2]))

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
        face_vertex_indices = self.geometry \
                .face_vertices(all_vertex_indices)
        faces_vertices = self.geometry \
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
            assert abs(numpy.dot(directions[0],directions[1])) < 1e-16

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
    def get_submesh_indices(self):
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
            try_add_tet((0,0,0), (1,0,0), (0,1,0), (0,0,1))
            try_add_tet((1,0,1), (1,0,0), (0,0,1), (0,1,0))
            try_add_tet((1,0,1), (0,1,1), (0,1,0), (0,0,1))

            try_add_tet((1,0,0), (0,1,0), (1,0,1), (1,1,0))
            try_add_tet((0,1,1), (0,1,0), (1,1,0), (1,0,1))
            try_add_tet((0,1,1), (1,1,1), (1,0,1), (1,1,0))

            # negatively oriented
            #try_add_tet((0,0,0), (1,0,0), (0,0,1), (0,1,0))
            #try_add_tet((1,0,1), (1,0,0), (0,1,0), (0,0,1))
            #try_add_tet((1,0,1), (0,1,1), (0,0,1), (0,1,0))

            #try_add_tet((1,0,0), (0,1,0), (1,1,0), (1,0,1))
            #try_add_tet((0,1,1), (0,1,0), (1,0,1), (1,1,0))
            #try_add_tet((0,1,1), (1,1,1), (1,1,0), (1,0,1))
        return result

    # basis functions ---------------------------------------------------------
    @memoize
    def basis_functions(self):
        """Get a sequence of functions that form a basis of the approximation space.

        The approximation space is spanned by the polynomials::

          r**i * s**j * t**k  for  i+j+k <= order
        """
        return [TetrahedronBasisFunction(*idx) for idx in 
                self.generate_mode_identifiers()]

    def grad_basis_functions(self):
        """Get the (r,s,...) gradient functions of the basis_functions(),
        in the same order.
        """
        return [GradTetrahedronBasisFunction(*idx) for idx in 
                self.generate_mode_identifiers()]

    # face operations ---------------------------------------------------------
    @memoize
    def face_mass_matrix(self):
        from hedge.polynomial import generic_vandermonde

        from pytools import generate_nonnegative_integer_tuples_summing_to_at_most

        basis = [TriangleBasisFunction(*node_tup) 
                for node_tup in 
                generate_nonnegative_integer_tuples_summing_to_at_most(
                    self.order, self.dimensions-1)]

        unodes = self.unit_nodes()
        face_indices = self.face_indices()

        face_vandermonde = generic_vandermonde(
                [unodes[i][:2] for i in face_indices[0]],
                basis)

        return numpy.asarray(
                la.inv(
                    numpy.dot(face_vandermonde, face_vandermonde.T)),
                order="C")

    def get_face_index_shuffle_to_match(self, face_1_vertices, face_2_vertices):
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

        order = self.order

        class TetrahedronFaceIndexShuffle:
            def __init__(self, operations):
                self.operations = operations

            def __hash__(self):
                return hash(self.operations)

            def __eq__(self, other):
                return self.operations == other.operations

            def __call__(self, indices):
                for op in self.operations:
                    if op == "flip":
                        indices = self.flip(indices)
                    elif op == "shift_left":
                        indices = self.shift_left(indices)
                    else:
                        raise RuntimeError, "invalid operation"
                return indices

            # flip and shift_left generate S_3
            def flip(self, indices):
                """Flip the indices along the unit hypotenuse."""
                result = []
                for j in range(0, order+1):
                    for i in range(0, order+1-j):
                        result.append(indices[idx_map[j,i]])
                return result

            def shift_left(self, indices):
                """Rotate all edges to the left."""
                result = len(indices)*[0]
                idx = 0
                for j in range(0, order+1):
                    for i in range(0, order+1-j):
                        result[idx_map[j, order-i-j]] = indices[idx]
                        idx += 1
                return result

        # yay, enumerate S_3 by hand
        if f2_tuple == (a,b,c):
            #return face_2_indices
            return TetrahedronFaceIndexShuffle(())
        elif f2_tuple == (a,c,b):
            #return flip(face_2_indices)
            return TetrahedronFaceIndexShuffle(("flip",))
        elif f2_tuple == (b,c,a):
            # (b,c,a) -sl-> (c,a,b) -sl-> (a,b,c)
            #return shift_left(shift_left(face_2_indices))
            return TetrahedronFaceIndexShuffle(("shift_left", "shift_left"))
        elif f2_tuple == (b,a,c):
            # (b,a,c) -sl-> (a,c,b) -fl-> (a,b,c)
            #return flip(shift_left(face_2_indices))
            return TetrahedronFaceIndexShuffle(("shift_left", "flip"))
        elif f2_tuple == (c,a,b):
            # (c,a,b) -sl-> (a,b,c)
            #return shift_left(face_2_indices)
            return TetrahedronFaceIndexShuffle(("shift_left",))
        elif f2_tuple == (c,b,a):
            # (c,b,a) -fl-> (c,a,b) -sl-> (a,b,c)
            #return shift_left(flip(face_2_indices))
            return TetrahedronFaceIndexShuffle(("flip", "shift_left"))
        else:
            raise FaceVertexMismatch("face vertices do not match")

    # time step scaling -------------------------------------------------------
    def dt_geometric_factor(self, vertices, el):
        return abs(el.map.jacobian)/max(abs(fj) for fj in el.face_jacobians)


