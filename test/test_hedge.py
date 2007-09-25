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
import unittest




class Monomial:
    def __init__(self, exponents, factor=1):
        import pylinear.array as num

        self.exponents = exponents
        self.ones = num.ones((len(self.exponents),))
        self.factor = factor

    def __call__(self, x):
        from operator import mul

        eps = 1e-15
        x = (x+self.ones)/2
        for xi in x:
            assert -eps <= xi <= 1+eps
        return self.factor* \
                reduce(mul, (x[i]**alpha 
                    for i, alpha in enumerate(self.exponents)))

    def theoretical_integral(self):
        from pytools import factorial
        from operator import mul

        return (self.factor*2**len(self.exponents)*
            reduce(mul, (factorial(alpha) for alpha in self.exponents))
            /
            factorial(len(self.exponents)+sum(self.exponents)))

    def diff(self, coordinate):
        diff_exp = list(self.exponents)
        orig_exp = diff_exp[coordinate]
        if orig_exp == 0:
            return Monomial(diff_exp, 0)
        diff_exp[coordinate] = orig_exp-1
        return Monomial(diff_exp, self.factor*orig_exp)




class TestHedge(unittest.TestCase):
    def test_face_vertex_order(self):
        """Verify that face_indices() emits face vertex indices in the right order"""
        from hedge.element import TriangularElement, TetrahedralElement

        for el in [TriangularElement(5), TetrahedralElement(5)]:
            vertex_indices = el.vertex_indices()
            for fn, (face_vertices, face_indices) in enumerate(zip(
                    el.geometry_class.face_vertices(vertex_indices), 
                    el.face_indices())):
                face_vertices_i = 0
                for fi in face_indices:
                    if fi == face_vertices[face_vertices_i]:
                        face_vertices_i += 1

                self.assert_(face_vertices_i == len(face_vertices))

    # -------------------------------------------------------------------------
    def test_newton_interpolation(self):
        """Verify Newton interpolation"""
        from hedge.interpolation import newton_interpolation_function
        
        x = [-1.5, -0.75, 0, 0.75, 1.5]
        y = [-14.1014, -0.931596, 0, 0.931596, 14.1014]
        nf = newton_interpolation_function(x, y)

        errors = [abs(yi-nf(xi)) for xi, yi in zip(x, y)]
        #print errors
        self.assert_(sum(errors) < 1e-14)
    # -------------------------------------------------------------------------
    def test_orthonormality_jacobi_1d(self):
        """Verify that the Jacobi polymials are orthogonal in 1D"""
        from hedge.polynomial import JacobiFunction
        from hedge.quadrature import LegendreGaussQuadrature

        max_n = 10
        int = LegendreGaussQuadrature(4*max_n) # overkill...

        class WeightFunction:
            def __init__(self, alpha, beta):
                self.alpha = alpha
                self.beta = beta

            def __call__(self, x):
                return (1-x)**self.alpha * (1+x)**self.beta

        for alpha, beta, ebound in [
                (0, 0, 5e-14), 
                (1, 0, 4e-14), 
                (3, 2, 3e-14), 
                (0, 2, 3e-13), 
                (5, 0, 3e-13), 
                (3, 4, 1e-14)
                ]:
            jac_f = [JacobiFunction(alpha, beta, n) for n in range(max_n)]
            wf = WeightFunction(alpha, beta)
            maxerr = 0

            for i, fi in enumerate(jac_f):
                for j, fj in enumerate(jac_f):
                    result = int(lambda x: wf(x)*fi(x)*fj(x))

                    if i == j:
                        true_result = 1
                    else:
                        true_result = 0
                    err = abs(result-true_result)
                    maxerr = max(maxerr, err)
                    if abs(result-true_result) > ebound:
                        print "bad", alpha, beta, i, j, abs(result-true_result)
                    self.assert_(abs(result-true_result) < ebound)
            #print alpha, beta, maxerr
    # -------------------------------------------------------------------------
    def test_transformed_quadrature(self):
        """Test 1D quadrature on arbitrary intervals"""
        from math import exp, sqrt, pi

        def gaussian_density(x, mu, sigma):
            return 1/(sigma*sqrt(2*pi))*exp(-(x-mu)**2/(2*sigma**2))

        from hedge.quadrature import LegendreGaussQuadrature, TransformedQuadrature

        mu = 17
        sigma = 12
        tq = TransformedQuadrature(LegendreGaussQuadrature(20), mu-6*sigma, mu+6*sigma)
        
        result = tq(lambda x: gaussian_density(x, mu, sigma))
        self.assert_(abs(result - 1) < 1e-9)
    # -------------------------------------------------------------------------
    def test_warp(self):
        """Check some assumptions on the node warp factor calculator"""
        n = 17
        from hedge.element import WarpFactorCalculator
        wfc = WarpFactorCalculator(n)

        self.assert_(abs(wfc.int_f(-1)) < 1e-15)
        self.assert_(abs(wfc.int_f(1)) < 2e-15)

        from hedge.quadrature import LegendreGaussQuadrature

        lgq = LegendreGaussQuadrature(n)
        #print abs(lgq(wfc))
        self.assert_(abs(lgq(wfc)) < 4e-14)
    # -------------------------------------------------------------------------
    def test_simp_nodes(self):
        """Verify basic assumptions on simplex interpolation nodes"""
        from hedge.element import TriangularElement, TetrahedralElement

        triorder = 8
        tri = TriangularElement(triorder)
        els = [tri, TriangularElement(17), TetrahedralElement(13)]

        for el in els:
            eps = 1e-10

            unodes = list(el.unit_nodes())
            self.assert_(len(unodes) == el.node_count())
            for ux in unodes:
                for uc in ux:
                    self.assert_(uc >= -1-eps)
                self.assert_(sum(ux) <= 1+eps)

            equnodes = list(el.equidistant_unit_nodes())
            self.assert_(len(equnodes) == el.node_count())
            for ux in equnodes:
                for uc in ux:
                    self.assert_(uc >= -1-eps)
                self.assert_(sum(ux) <= 1+eps)

            for indices in el.node_tuples():
                for index in indices:
                    self.assert_(index >= 0)
                self.assert_(sum(indices) <= el.order)
    # -------------------------------------------------------------------------
    def test_tri_nodes_against_known_values(self):
        """Check triangle nodes against a previous implementation"""
        from hedge.element import TriangularElement, TetrahedralElement

        triorder = 8
        tri = TriangularElement(triorder)

        def tri_equilateral_nodes_reference(self):
            # This is the old, more explicit, less general way of computing
            # the triangle nodes. Below, we compare its results with that of the
            # new routine.

            alpha_opt = [0.0000, 0.0000, 1.4152, 0.1001, 0.2751, 0.9800, 1.0999,
                    1.2832, 1.3648, 1.4773, 1.4959, 1.5743, 1.5770, 1.6223, 1.6258]
                      
            try:
                alpha = alpha_opt[self.order-1]
            except IndexError:
                alpha = 5/3

            from hedge.element import WarpFactorCalculator
            import pylinear.array as num
            from math import sin, cos, pi

            warp = WarpFactorCalculator(self.order)

            edge1dir = num.array([1,0])
            edge2dir = num.array([cos(2*pi/3), sin(2*pi/3)])
            edge3dir = num.array([cos(4*pi/3), sin(4*pi/3)])

            for bary in self.equidistant_barycentric_nodes():
                lambda1, lambda2, lambda3 = bary

                # find equidistant (x,y) coordinates in equilateral triangle
                point = self.barycentric_to_equilateral(bary)

                # compute blend factors
                blend1 = 4*lambda1*lambda2 # nonzero on AB
                blend2 = 4*lambda3*lambda2 # nonzero on BC
                blend3 = 4*lambda3*lambda1 # nonzero on AC

                # calculate amount of warp for each node, for each edge
                warp1 = blend1*warp(lambda2 - lambda1)*(1 + (alpha*lambda3)**2)
                warp2 = blend2*warp(lambda3 - lambda2)*(1 + (alpha*lambda1)**2)
                warp3 = blend3*warp(lambda1 - lambda3)*(1 + (alpha*lambda2)**2)

                # return warped point
                yield point + warp1*edge1dir + warp2*edge2dir + warp3*edge3dir

        if False:
            outf = open("trinodes1.dat", "w")
            for ux in tri.equilateral_nodes():
                outf.write("%g\t%g\n" % tuple(ux))
            outf = open("trinodes2.dat", "w")
            for ux in tri_equilateral_nodes_reference(tri):
                outf.write("%g\t%g\n" % tuple(ux))

        from pylinear.computation import norm_2
        for n1, n2 in zip(tri.equilateral_nodes(), 
                tri_equilateral_nodes_reference(tri)):
            self.assert_(norm_2(n1-n2) < 3e-15)

        def node_indices_2(order):
            for n in range(0, order+1):
                 for m in range(0, order+1-n):
                     yield m,n

        self.assert_(set(tri.node_tuples()) == set(node_indices_2(triorder)))
    # -------------------------------------------------------------------------
    def test_simp_basis_grad(self):
        """Do a simplistic FD-style check on the differentiation matrix"""
        from itertools import izip
        from hedge.element import TriangularElement, TetrahedralElement
        from random import uniform
        import pylinear.array as num
        import pylinear.computation as comp

        els = [(1, TriangularElement(8)), (3,TetrahedralElement(7))]
        for err_factor, el in els:
            d = el.dimensions
            for i_bf, (bf, gradbf) in \
                    enumerate(izip(el.basis_functions(), el.grad_basis_functions())):
                for i in range(10):
                    base = -0.95
                    remaining = 1.90
                    r = num.zeros((d,))
                    for i in range(d):
                        rn = uniform(0, remaining)
                        r[i] = base+rn
                        remaining -= rn

                    from pytools import wandering_element
                    h = 1e-4
                    gradbf_v = num.array(gradbf(r))
                    approx_gradbf_v = num.array([
                        (bf(r+h*dir) - bf(r-h*dir))/(2*h)
                        for dir in [num.array(dir) for dir in wandering_element(d)]
                        ])
                    err = comp.norm_infinity(approx_gradbf_v-gradbf_v)
                    #print el.dimensions, el.order, i_bf, err
                    self.assert_(err < err_factor*h)
    # -------------------------------------------------------------------------
    def test_tri_face_node_distribution(self):
        """Test whether the nodes on the faces of the triangle are distributed 
        according to the same proportions on each face.

        If this is not the case, then reusing the same face mass matrix
        for each face would be invalid.
        """

        from hedge.element import TriangularElement
        import pylinear.array as num
        import pylinear.computation as comp

        tri = TriangularElement(8)
        unodes = tri.unit_nodes()
        projected_face_points = []
        for face_i in tri.face_indices():
            start = unodes[face_i[0]]
            end = unodes[face_i[-1]]
            dir = end-start
            dir /= comp.norm_2_squared(dir)
            pfp = num.array([dir*(unodes[i]-start) for i in face_i])
            projected_face_points.append(pfp)

        first_points =  projected_face_points[0]
        for points in projected_face_points[1:]:
            self.assert_(comp.norm_infinity(points-first_points) < 1e-15)
    # -------------------------------------------------------------------------
    def test_simp_face_normals_and_jacobians(self):
        """Check computed face normals and face jacobians on simplicial elements
        """
        from hedge.element import TriangularElement, TetrahedralElement
        from hedge.mesh import Triangle
        from hedge.tools import AffineMap
        import pylinear.array as num
        import pylinear.computation as comp
        from pylinear.randomized import make_random_vector

        for el in [
                TetrahedralElement(1), 
                TriangularElement(4), 
                ]:
            for i in range(50):
                geo = el.geometry_class

                vertices = [make_random_vector(el.dimensions, num.Float) 
                        for vi in range(el.dimensions+1)]
                #array = num.array
                #vertices = [array([-1, -1.0, -1.0]), array([1, -1.0, -1.0]), array([-1.0, 1, -1.0]), array([-1.0, -1.0, 1.0])]
                map = geo.get_map_unit_to_global(vertices)

                unodes = el.unit_nodes()
                nodes = [map(v) for v in unodes]

                all_vertex_indices = range(el.dimensions+1)

                for face_i, fvi, normal, jac in \
                        zip(el.face_indices(), 
                                geo.face_vertices(all_vertex_indices),
                                *geo.face_normals_and_jacobians(map)):
                    mapped_corners = [vertices[i] for i in fvi]
                    mapped_face_basis = [mc-mapped_corners[0] for mc in mapped_corners[1:]]

                    # face vertices must be among all face nodes
                    close_nodes = 0
                    for fi in face_i:
                        face_node = nodes[fi]
                        for mc in mapped_corners:
                            if comp.norm_2(mc-face_node) < 1e-13:
                                close_nodes += 1

                    self.assert_(close_nodes == len(mapped_corners))

                    opp_node = (set(all_vertex_indices) - set(fvi)).__iter__().next()
                    mapped_opposite = vertices[opp_node]

                    if el.dimensions == 2:
                        true_jac = comp.norm_2(mapped_corners[1]-mapped_corners[0])/2
                    elif el.dimensions == 3:
                        mapped_face_projection = num.array(comp.orthonormalize(mapped_face_basis))
                        projected_corners = [num.zeros((2,))] + [mapped_face_projection*v 
                                for v in mapped_face_basis]
                        true_jac = abs(Triangle
                                .get_map_unit_to_global(projected_corners)
                                .jacobian)
                    else:
                        assert False

                    #print abs(true_jac-jac)/true_jac
                    #print "aft, bef", comp.norm_2(mapped_end-mapped_start),comp.norm_2(end-start)

                    self.assert_(abs(true_jac - jac)/true_jac < 1e-13)
                    self.assert_(abs(comp.norm_2(normal) - 1) < 1e-13)
                    for mfbv in mapped_face_basis:
                        self.assert_(abs(normal*mfbv) < 1e-13)

                    for mc in mapped_corners:
                        self.assert_((mapped_opposite-mc)*normal < 0)
    # -------------------------------------------------------------------------
    def test_tri_map(self):
        """Verify that the mapping and node-building operations maintain triangle vertices"""
        from hedge.element import TriangularElement
        import pylinear.array as num
        import pylinear.computation as comp
        from pylinear.randomized import \
                make_random_vector

        n = 8
        tri = TriangularElement(n)

        node_dict = dict((ituple, idx) for idx, ituple in enumerate(tri.node_tuples()))
        corner_indices = [node_dict[0,0], node_dict[n,0], node_dict[0,n]]
        unodes = tri.unit_nodes()
        corners = [unodes[i] for i in corner_indices]

        for i in range(10):
            vertices = [make_random_vector(2, num.Float) for vi in range(3)]
            map = tri.geometry_class.get_map_unit_to_global(vertices)
            global_corners = [map(pt) for pt in corners]
            for gc, v in zip(global_corners, vertices):
                self.assert_(comp.norm_2(gc-v) < 1e-12)
    # -------------------------------------------------------------------------
    def test_tri_map_jacobian_and_mass_matrix(self):
        """Verify whether tri map jacobians recover known values of triangle area"""
        from hedge.element import TriangularElement
        import pylinear.array as num
        import pylinear.computation as comp
        from pylinear.randomized import make_random_vector
        from math import sqrt, exp, pi

        edata = TriangularElement(9)
        ones = num.ones((edata.node_count(),))
        unit_tri_area = 2
        self.assert_(abs(ones*(edata.mass_matrix()*ones)-unit_tri_area) < 1e-11)

        for i in range(10):
            vertices = [make_random_vector(2, num.Float) for vi in range(3)]
            map = edata.geometry_class.get_map_unit_to_global(vertices)
            mat = num.zeros((2,2))
            mat[:,0] = (vertices[1] - vertices[0])
            mat[:,1] = (vertices[2] - vertices[0])
            tri_area = abs(comp.determinant(mat)/2)
            tri_area_2 = abs(unit_tri_area*map.jacobian)
            self.assert_(abs(tri_area - tri_area_2)/tri_area < 1e-15)
    # -------------------------------------------------------------------------
    def no_test_tri_mass_mat_gauss(self):
        """Check the integral of a Gaussian on a disk using the mass matrix"""

        # This is a bad test, since it's never exact. The Gaussian has infinite support,
        # and this *does* matter numerically.

        from hedge.mesh import make_disk_mesh
        from hedge.element import TriangularElement
        from hedge.discretization import Discretization
        from math import sqrt, exp, pi

        sigma_squared = 1/219.3

        mesh = make_disk_mesh()
        discr = Discretization(make_disk_mesh(), TriangularElement(4))
        f = discr.interpolate_volume_function(lambda x: exp(-x*x/(2*sigma_squared)))
        ones = discr.interpolate_volume_function(lambda x: 1)

        #discr.visualize_vtk("gaussian.vtk", [("f", f)])
        num_integral_1 = ones * (discr.mass_operator * f)
        num_integral_2 = f * (discr.mass_operator * ones)
        dim = 2
        true_integral = (2*pi)**(dim/2)*sqrt(sigma_squared)**dim
        err_1 = abs(num_integral_1-true_integral)
        err_2 = abs(num_integral_2-true_integral)
        self.assert_(err_1 < 1e-11)
        self.assert_(err_2 < 1e-11)
    # -------------------------------------------------------------------------
    def test_tri_mass_mat_trig(self):
        """Check the integral of some trig functions on a square using the mass matrix"""

        from hedge.mesh import make_square_mesh
        from hedge.element import TriangularElement
        from hedge.discretization import Discretization
        import pylinear.computation as comp
        from math import sqrt, pi, cos, sin

        mesh = make_square_mesh(a=-pi, b=pi, max_area=(2*pi/10)**2/2)
        discr = Discretization(mesh, TriangularElement(8))
        f = discr.interpolate_volume_function(lambda x: cos(x[0])**2*sin(x[1])**2)
        ones = discr.interpolate_volume_function(lambda x: 1)

        #discr.visualize_vtk("trig.vtk", [("f", f)])
        num_integral_1 = ones * (discr.mass_operator * f)
        num_integral_2 = f * (discr.mass_operator * ones)
        true_integral = pi**2
        err_1 = abs(num_integral_1-true_integral)
        err_2 = abs(num_integral_2-true_integral)
        #print err_1, err_2
        self.assert_(err_1 < 1e-10)
        self.assert_(err_2 < 1e-10)
    # -------------------------------------------------------------------------
    def test_tri_diff_mat(self):
        """Check differentiation matrix along the coordinate axes on a disk
        
        Uses sines as the function to differentiate.
        """
        import pylinear.computation as comp
        from hedge.mesh import make_disk_mesh
        from hedge.element import TriangularElement
        from hedge.discretization import Discretization
        from math import sin, cos, sqrt

        for coord in [0, 1]:
            mesh = make_disk_mesh()
            discr = Discretization(make_disk_mesh(), TriangularElement(4))
            f = discr.interpolate_volume_function(lambda x: sin(3*x[coord]))
            df = discr.interpolate_volume_function(lambda x: 3*cos(3*x[coord]))

            df_num = discr.nabla[coord] * f
            error = df_num - df
            #discr.visualize_vtk("diff-err.vtk",
                    #[("f", f), ("df", df), ("df_num", df_num), ("error", error)])

            linf_error = comp.norm_infinity(df_num-df)
            #print linf_error
            self.assert_(linf_error < 3e-5)
    # -------------------------------------------------------------------------
    def test_2d_gauss_theorem(self):
        """Verify Gauss's theorem explicitly on a mesh"""

        from hedge.element import TriangularElement
        from hedge.tools import AffineMap
        from hedge.mesh import make_disk_mesh
        from hedge.flux import Flux
        from hedge.discretization import Discretization, pair_with_boundary
        import pylinear.array as num
        import pylinear.computation as comp
        from pylinear.randomized import make_random_vector
        from math import sin, cos, sqrt, exp, pi

        class NormalFlux(Flux):
            def __init__(self, coordinate):
                Flux.__init__(self)
                self.coordinate = coordinate

            def __call__(self, int_face, ext_face):
                return int_face.normal[self.coordinate]

        class ZeroFlux(Flux):
            def __call__(self, int_face, ext_face):
                return 0

        one_sided_x = (NormalFlux(0), ZeroFlux())
        one_sided_y = (NormalFlux(1), ZeroFlux())

        def f1(x):
            return sin(3*x[0])+cos(3*x[1])
        def f2(x):
            return sin(2*x[0])+cos(x[1])

        edata = TriangularElement(2)

        discr = Discretization(make_disk_mesh(), edata)
        ones = discr.interpolate_volume_function(lambda x: 1)
        face_zeros = discr.boundary_zeros()

        f1_v = discr.interpolate_volume_function(f1)
        f2_v = discr.interpolate_volume_function(f2)

        f1_f = discr.interpolate_boundary_function(f1)
        f2_f = discr.interpolate_boundary_function(f2)

        dx_v = discr.nabla[0] * f1_v
        dy_v = discr.nabla[1] * f2_v

        int_div = \
                ones*(discr.mass_operator*dx_v) + \
                ones*(discr.mass_operator*dy_v)

        boundary_int = (
                discr.get_flux_operator(one_sided_x)*pair_with_boundary(f1_v, face_zeros) +
                discr.get_flux_operator(one_sided_y)*pair_with_boundary(f2_v, face_zeros)
                )*ones

        #print abs(boundary_int-int_div)
        self.assert_(abs(boundary_int-int_div) < 5e-15)
    # -------------------------------------------------------------------------
    def test_simp_cubature(self):
        """Check that Grundmann-Moeller cubature works as advertised"""
        from pytools import generate_nonnegative_integer_tuples_summing_to_at_most
        from hedge.quadrature import SimplexCubature

        for dim in range(2,3+1):
            for s in range(3+1):
                cub = SimplexCubature(s, dim)
                for comb in generate_nonnegative_integer_tuples_summing_to_at_most(
                        2*s+1, dim):
                    f = Monomial(comb)
                    i_f = cub(f)
                    err = abs(i_f - f.theoretical_integral())
                    self.assert_(err < 2e-15)
    # -------------------------------------------------------------------------
    def test_simp_mass_and_diff_matrices_by_monomial(self):
        """Verify simplicial mass and differentiation matrices using monomials"""

        from hedge.element import TriangularElement, TetrahedralElement
        from pytools import generate_nonnegative_integer_tuples_summing_to_at_most

        import pylinear.array as num
        import pylinear.computation as comp
        from pylinear.randomized import make_random_vector
        from operator import add, mul

        thresh = 1e-14

        for el in [
                TriangularElement(3),
                TetrahedralElement(5),
                ]:
            for comb in generate_nonnegative_integer_tuples_summing_to_at_most(
                    el.order, el.dimensions):
                ones = num.ones((el.node_count(),))
                unodes = el.unit_nodes()
                f = Monomial(comb)
                f_n = num.array([f(x) for x in unodes])
                int_f_n = ones*el.mass_matrix()*f_n
                int_f = f.theoretical_integral()
                err = abs(int_f - int_f_n)
                if err > thresh:
                    print "bad", el, comb, int_f, int_f_n, err
                self.assert_(err < thresh)

                dmats = el.differentiation_matrices()
                for i in range(el.dimensions):
                    df = f.diff(i)
                    df = num.array([df(x) for x in unodes])/2
                    df_n = dmats[i]*f_n
                    err = comp.norm_infinity(df - df_n)
                    if err > thresh:
                        print "bad-diff", comb, i, err
                    self.assert_(err < thresh)
    # -------------------------------------------------------------------------
    def test_simp_gauss_theorem(self):
        """Verify Gauss's theorem explicitly on simplicial elements"""

        from hedge.element import TriangularElement, TetrahedralElement
        from hedge.tools import AffineMap
        import pylinear.array as num
        import pylinear.computation as comp
        from pylinear.randomized import make_random_vector
        from operator import add
        from math import sin, cos, sqrt, exp, pi

        def f1_2d(x):
            return sin(3*x[0])+cos(3*x[1])
        def f2_2d(x):
            return sin(2*x[0])+cos(x[1])

        def f1_3d(x):
            return sin(3*x[0])+cos(3*x[1])+sin(1.9*x[2])
        def f2_3d(x):
            return sin(1.2*x[0])+cos(2.1*x[1])-cos(1.5*x[2])
        def f3_3d(x):
            return 5*sin(-0.2*x[0])-3*cos(x[1])+cos(x[2])

        #def f1_3d(x):
            #return 1
        #def f2_3d(x):
            #return 0
        #def f3_3d(x):
            #return 0

        def d(imap, coordinate, field):
            col = imap.matrix[:, coordinate]
            matrices = el.differentiation_matrices()
            return reduce(add, (dmat*coeff*field
                        for dmat, coeff in zip(matrices, col)))

        array = num.array

        triangles = [
                [array([-7.1687642250744492, 0.63058995062684642]), array([9.9744219044921199, 6.6530989283689781]), array([12.269380138171147, -17.529689194536481])],
                [array([-3.1285787297852634, -16.579403405465403]), array([-5.2882160938912515, -6.2209234150214137]), array([11.251223490342774, 4.6571427341871727])],
                [array([4.7407472917152553, -18.406868078408063]), array([1.8224524488556271, 11.551374404003361]), array([2.523148394963088, 1.632574414790982])],
                [array([-11.523714017493292, -14.2820557378961]), array([-0.44311816855771136, 19.572194735728861]), array([5.2855990566779445, -9.8743423935894388])],
                [array([1.113949150102217, -3.2255502625302639]), array([-13.028732972681315, 2.1525752429773379]), array([-2.3929000970202705, 6.2884649052982198])],
                [array([-8.0878061368549741, -14.604092423350167]), array([4.5339922477199996, 8.3770287646932022]), array([-5.2180549365480156, -1.9930760171433717])],
                [array([-1.9047012017294165, -3.6517882549544485]), array([3.1461902282192784, 5.7397849191668229]), array([-11.072761256907262, -8.3758508470287509])],
                [array([8.6609581113102934, 9.1121629958018566]), array([3.8230948675835497, -14.004679313330751]), array([10.975623610855521, 1.6267418698764553])],
                [array([13.959685276941629, -12.201892555481464]), array([-7.8057604576925499, -3.5283871457281757]), array([-0.41961743047735317, -3.2615635891671872])],
                [array([-9.8469907360335078, 6.0635407355366242]), array([7.8727080309703439, 7.634505157189091]), array([-2.7723038834027118, 8.5441656500931789])],
                ]
        tets = [
                #[make_random_vector(3, num.Float) for i in range(4)]
                #for j in range(10)
                [array([-0.087835976182812386, 8.4241880323369127, 2.6557808710807933]), array([5.4875560966799677, -7.5530368326122499, 8.4703868377747877]), array([-8.4888098806626751, 1.8369058453192324, -6.9041468708803713]), array([17.327527449499168, -9.0586108433594319, 5.7459746913914636])],
                [array([16.993689961344593, -12.116305360441197, -12.711045554409088]), array([-2.0324332275643817, -5.0524187595904335, 5.9257028535230383]), array([6.4221144729287687, -7.2496949199427245, -1.1590391996379827]), array([-5.7529432953399171, -6.9587987820990262, 3.7223773892240426])],
                [array([-0.4423263927732628, -1.6306971591009138, -1.2554069824001064]), array([-9.1171749892163785, 14.232868970928301, 4.6548620163014505]), array([16.554360867165595, -2.1451702825571202, -1.9050837421951314]), array([-8.7455417971698139, 19.016251630886945, -15.137691422305545])],
                [array([-1.9251811954429843, -4.5369007736338665, 9.2675942450331963]), array([-13.586778017089083, -3.6666239130220553, -14.095112617514117]), array([-15.014799506040006, -3.4363888726140681, -0.85237426213437206]), array([6.3854978041452597, 13.293981904633554, -7.8432774486183146])],
                [array([-6.761839340374304, 14.864784704377955, 1.574274771089831]), array([-0.1823468063801317, -21.892423945260102, 11.565172070570537]), array([-0.14658389181168049, 13.07241603902848, 7.2652184007323042]), array([-20.35574011769496, 14.816503793175773, -7.2800214849607254])],
                [array([23.294362873156878, 13.644282203469114, 10.383738204469243]), array([-19.792088993452555, 0.4209925297886693, -7.3933945447515388]), array([-2.832898385995708, -1.6480401382241885, -6.2689214950820924]), array([-0.081772347748623617, -3.3803599922239673, -19.614368674546114])],
                [array([0.43913744703796659, -16.473036116412242, -0.8653853759721295]), array([-7.3270752283484297, -0.97723025169973787, 2.1330514627504464]), array([3.8730334021748307, -9.0983850278388143, 3.3578300089831501]), array([18.639504439820936, 20.594835769217696, -10.666261239487298])],
                [array([-12.786230591302058, -9.2931510923111169, -2.1598642469378935]), array([-4.0458439207057459, -9.0298998997705144, -0.11666215074316685]), array([7.5023999981398424, 4.8603369473110583, 2.1813627427875013]), array([2.9579841500551272, -22.563123335973565, 10.335559822513606])],
                [array([-7.7732699602949893, 15.816977096296963, -6.8826683632918728]), array([7.6233333630240256, -9.3309869383569026, 0.50189282953625991]), array([-11.272342858699034, 1.089016041114454, -6.0359393299451476]), array([-6.4746449930954348, -0.026130504314747997, -2.2786267101817677])],
                [array([-18.243993907118757, 5.0646875774948974, -9.2110046334596856]), array([-8.1550804560957264, -3.1021806460634913, 7.5622831439916105]), array([19.460768761970783, 17.494565076685859, 16.295621155355697]), array([4.6186236213250131, -1.3869183721072562, -0.2159066724152843])],
                ]

        for el_geoms, el, f in [
                (triangles, TriangularElement(9), (f1_2d, f2_2d)),
                (tets, TetrahedralElement(1), (f1_3d, f2_3d, f3_3d)),
                ]:
            for vertices in el_geoms:
                ones = num.ones((el.node_count(),))
                face_ones = num.ones((len(el.face_indices()[0]),))

                map = el.geometry_class.get_map_unit_to_global(vertices)
                imap = map.inverted()

                mapped_points = [map(node) for node in el.unit_nodes()]

                f_n = [num.array([fi(x) for x in mapped_points])
                        for fi in f]
                df_n = [d(imap, i, f_n[i]) for i, fi_n in enumerate(f_n)]

                int_div_f = abs(map.jacobian)*sum(
                        ones*el.mass_matrix()*dfi_n for dfi_n in df_n)

                if False:
                    boundary_comp = [
                            array([
                                fjac * face_ones * el.face_mass_matrix() 
                                * num.take(fi_n, face_indices) * n_coord
                                for fi_n, n_coord in zip(f_n, n)])
                            for face_indices, n, fjac
                            in zip(el.face_indices(), *el.face_normals_and_jacobians(map))
                            ]

                boundary_sum = sum(
                        sum(
                            fjac * face_ones * el.face_mass_matrix() 
                            * num.take(fi_n, face_indices) * n_coord
                            for fi_n, n_coord in zip(f_n, n))
                        for face_indices, n, fjac
                        in zip(el.face_indices(), 
                            *el.geometry_class.face_normals_and_jacobians(map))
                        )

                #print el.face_normals_and_jacobians(map)[1]
                #print 'mp', [mapped_points[fi] for fi in el.face_indices()[2]]
                #print num.take(f_n[0], el.face_indices()[2])
                #print 'bc', boundary_comp
                #print 'bs', boundary_sum
                #print 'idiv', int_div_f
                #print abs(boundary_sum-int_div_f)
                self.assert_(abs(boundary_sum-int_div_f) < 1e-12)
    # -------------------------------------------------------------------------
    def test_simp_orthogonality(self):
        """Test orthogonality of simplicial bases using Grundmann-Moeller cubature"""
        from hedge.quadrature import SimplexCubature
        from hedge.element import TriangularElement, TetrahedralElement

        for order, ebound in [
                (1, 2e-15),
                (2, 5e-15),
                (3, 1e-14),
                #(4, 3e-14),
                #(7, 3e-14),
                #(9, 2e-13),
                ]:
            for ldis in [TriangularElement(order), TetrahedralElement(order)]:
                cub = SimplexCubature(order, ldis.dimensions)
                basis = ldis.basis_functions()

                maxerr = 0
                for i, f in enumerate(basis):
                    for j, g in enumerate(basis):
                        if i == j:
                            true_result = 1
                        else:
                            true_result = 0
                        result = cub(lambda x: f(x)*g(x))
                        err = abs(result-true_result)
                        maxerr = max(maxerr, err)
                        if err > ebound:
                            print "bad", order,i,j, err
                        self.assert_(err < ebound)
                #print order, maxerr
    # -------------------------------------------------------------------------
    def test_1d_mass_matrix_vs_quadrature(self):
        """Check that a 1D mass matrix for Legendre-Gauss points gives the right weights"""
        from hedge.quadrature import LegendreGaussQuadrature
        from hedge.polynomial import legendre_vandermonde
        import pylinear.array as num
        import pylinear.computation as comp

        for n in range(13):
            lgq = LegendreGaussQuadrature(n)
            vdm = legendre_vandermonde(lgq.points, n)
            mass_mat = 1/(vdm*vdm.T)
            ones = num.ones((mass_mat.shape[0],))
            self.assert_(comp.norm_infinity(
                    ((vdm*vdm.T) <<num.solve>> ones)
                    -
                    num.array(lgq.weights)) < 2e-14)
    # -------------------------------------------------------------------------
    def test_mapping_differences_tri(self):
        """Check that triangle interpolation is independent of mapping to reference
        """
        from hedge.element import TriangularElement
        import pylinear.array as num
        import pylinear.computation as comp
        from pylinear.randomized import make_random_vector
        from random import random
        from pytools import generate_permutations

        def shift(list):
            return list[1:] + [list[0]]

        class LinearCombinationOfFunctions:
            def __init__(self, coefficients, functions, premap):
                self.coefficients = coefficients
                self.functions = functions
                self.premap = premap

            def __call__(self, x):
                return sum(coeff*f(self.premap(x)) for coeff, f in 
                        zip(self.coefficients, self.functions))

        def random_barycentric_coordinates(dim):
            remain = 1
            coords = []
            for i in range(dim):
                coords.append(random() * remain)
                remain -= coords[-1]
            coords.append(remain)
            return coords

        tri = TriangularElement(5)

        for trial_number in range(10):
            vertices = [make_random_vector(2, num.Float) for vi in range(3)]
            map = tri.geometry_class.get_map_unit_to_global(vertices)
            nodes = [map(node) for node in tri.unit_nodes()]
            node_values = num.array([random() for node in nodes])

            functions = []
            for pvertices in generate_permutations(vertices):
                pmap = tri.geometry_class.get_map_unit_to_global(pvertices)
                pnodes = [pmap(node) for node in tri.unit_nodes()]

                # map from pnode# to node#
                nodematch = {}
                for pi, pn in enumerate(pnodes):
                    for i, n in enumerate(nodes):
                        if comp.norm_2(n - pn) < 1e-13:
                            nodematch[pi] = i
                            break

                pnode_values = num.array([node_values[nodematch[pi]] 
                        for pi in range(len(nodes))])

                interp_f = LinearCombinationOfFunctions(
                        tri.vandermonde() <<num.solve>> pnode_values,
                        tri.basis_functions(),
                        pmap.inverted())

                # verify interpolation property
                #for n, nv in zip(pnodes, pnode_values):
                    #self.assert_(abs(interp_f(n) - nv) < 1e-13)

                functions.append(interp_f)

            for subtrial_number in range(15):
                pt_in_element = sum(
                        coeff*vertex
                        for coeff, vertex in zip(
                            random_barycentric_coordinates(2),
                            vertices))
                f_values = [f(pt_in_element) for f in functions]
                avg = sum(f_values) / len(f_values)
                err = [abs(fv-avg) for fv in f_values]
                self.assert_(max(err) < 5e-13)
    # -------------------------------------------------------------------------
    def test_interior_fluxes_tri(self):
        """Check triangle surface integrals computed using interior fluxes
        against their known values.
        """

        from math import pi, sin, cos

        def round_trip_connect(start, end):
            for i in range(start, end):
                yield i, i+1
            yield end, start

        a = -pi
        b = pi
        points = [
                (a,0), (b,0), 
                (a,-1), (b,-1),
                (a,1), (b,1)
                ]
                
        import meshpy.triangle as triangle

        mesh_info = triangle.MeshInfo()
        mesh_info.set_points(points)
        mesh_info.set_facets(
                [(0,1),(1,3),(3,2),(2,0),(0,4),(4,5),(1,5)]
                )

        mesh_info.regions.resize(2)
        mesh_info.regions[0] = [
                0,-0.5, # coordinate
                1, # lower element tag
                0.1, # max area
                ]
        mesh_info.regions[1] = [
                0,0.5, # coordinate
                2, # upper element tag
                0.01, # max area
                ]

        generated_mesh = triangle.build(mesh_info, 
                attributes=True,
                volume_constraints=True)
        #triangle.write_gnuplot_mesh("mesh.dat", generated_mesh)

        from hedge.mesh import ConformalMesh

        def element_tagger(el):
            if generated_mesh.element_attributes[el.id] == 1:
                return ["upper"]
            else:
                return ["lower"]

        mesh = ConformalMesh(
                generated_mesh.points,
                generated_mesh.elements,
                element_tagger=element_tagger)

        from hedge.element import TriangularElement
        from hedge.discretization import Discretization
        discr = Discretization(mesh, TriangularElement(4))

        u_i = discr.interpolate_tag_volume_function(
                lambda x: sin(x[0]-x[1]),
                "lower")
        u_o = discr.interpolate_tag_volume_function(
                lambda x: cos(x[0]-x[1]),
                "upper")
        u = u_i + u_o

        #discr.visualize_vtk("dual.vtk", [("u", u)])

        from hedge.flux import make_normal, FluxScalarPlaceholder
        fluxu = FluxScalarPlaceholder()
        res = discr.get_flux_operator(
                (fluxu.int - fluxu.ext)*make_normal(discr.dimensions)[1]) * u

        ones = discr.interpolate_volume_function(lambda x: 1)
        self.assert_(abs(res*ones) < 5e-14)
    # -------------------------------------------------------------------------
    def test_interior_fluxes_tet(self):
        """Check tetrahedron surface integrals computed using interior fluxes
        against their known values.
        """

        import meshpy.tet as tet
        from math import pi, sin, cos

        mesh_info = tet.MeshInfo()

        # construct a two-box extrusion of this base
        base = [(-pi,-pi,0), (pi,-pi,0), (pi,pi,0), (-pi,pi,0)]

        # first, the nodes
        mesh_info.set_points(
                base
                +[(x,y,z+pi) for x,y,z in base]
                +[(x,y,z+pi+1) for x,y,z in base]
                )

        # next, the facets

        # vertex indices for a box missing the -z face
        box_without_minus_z = [ 
            [4,5,6,7],
            [0,4,5,1],
            [1,5,6,2],
            [2,6,7,3],
            [3,7,4,0],
            ]

        def add_to_all_vertex_indices(facets, increment):
            return [[pt+increment for pt in facet] for facet in facets]

        mesh_info.set_facets(
            [[0,1,2,3]] # base
            +box_without_minus_z # first box
            +add_to_all_vertex_indices(box_without_minus_z, 4) # second box
            )

        # set the volume properties -- this is where the tet size constraints are
        mesh_info.regions.resize(2)
        mesh_info.regions[0] = [0,0,pi/2, # point in volume -> first box
                0, # region tag (user-defined number)
                0.5, # max tet volume in region
                ]
        mesh_info.regions[1] = [0,0,pi+0.5, # point in volume -> second box
                1, # region tag (user-defined number, arbitrary)
                0.1, # max tet volume in region
                ]

        generated_mesh = tet.build(mesh_info, attributes=True, volume_constraints=True)
        #mesh.write_vtk("sandwich-mesh.vtk")

        from hedge.mesh import ConformalMesh

        def element_tagger(el):
            if generated_mesh.element_attributes[el.id] == 1:
                return ["upper"]
            else:
                return ["lower"]

        mesh = ConformalMesh(
                generated_mesh.points,
                generated_mesh.elements,
                element_tagger=element_tagger)

        from hedge.element import TetrahedralElement
        from hedge.discretization import Discretization
        discr = Discretization(mesh, TetrahedralElement(4))

        u_l = discr.interpolate_tag_volume_function(
                lambda x: sin(x[0]-x[1]+x[2]),
                "lower")
        u_u = discr.interpolate_tag_volume_function(
                lambda x: cos(x[0]-x[1]+x[2]),
                "upper")
        u = u_l + u_u

        # visualize the produced field
        #from hedge.visualization import SiloVisualizer
        #vis = SiloVisualizer(discr)
        #vis("sandwich.silo", [("u_l", u_l), ("u_u", u_u)], expressions=[("u", "u_l+u_u")],
                #write_coarse_mesh=True)

        # make sure the surface integral of the difference 
        # between top and bottom is zero
        from hedge.flux import make_normal, FluxScalarPlaceholder
        fluxu = FluxScalarPlaceholder()
        res = discr.get_flux_operator(
                (fluxu.int - fluxu.ext)*make_normal(discr.dimensions)[1]) * u
        ones = discr.interpolate_volume_function(lambda x: 1)
        self.assert_(abs(res*ones) < 5e-14)
    # -------------------------------------------------------------------------
    def test_symmetry_preservation_2d(self):
        """Test whether we preserve symmetry in a symmetric 2D advection problem"""

        import pylinear.array as num

        def make_mesh():
            from hedge.mesh import ConformalMesh
            array = num.array

            #
            #    1---8---2
            #    |7 /|\ 1|
            #    | / | \ |
            #    |/ 6|0 \|
            #    5---4---7
            #    |\ 5|3 /|
            #    | \ | / |
            #    |4 \|/ 2|
            #    0---6---3
            #
            points = [
                    array([-0.5, -0.5]), 
                    array([-0.5, 0.5]), 
                    array([0.5, 0.5]), 
                    array([0.5, -0.5]), 
                    array([0.0, 0.0]), 
                    array([-0.5, 0.0]), 
                    array([0.0, -0.5]), 
                    array([0.5, 0.0]), 
                    array([0.0, 0.5])]

            elements = [
                    [8,7,4],
                    [8,7,2],
                    [6,7,3],
                    [7,4,6],
                    [5,6,0],
                    [5,6,4],
                    [5,8,4],
                    [1,5,8],
                    ]

            def boundary_tagger(vertices, el, face_nr):
                if el.face_normals[face_nr] * a > 0:
                    return ["inflow"]
                else:
                    return ["outflow"]

            return ConformalMesh(points, elements, boundary_tagger)

        from hedge.discretization import \
                Discretization, SymmetryMap, pair_with_boundary
        from hedge.element import TriangularElement
        from hedge.flux import make_normal, FluxScalarPlaceholder
        from hedge.timestep import RK4TimeStepper
        from hedge.mesh import REORDER_NONE
        from hedge.tools import dot
        from math import sqrt

        a = num.array([1,0])

        mesh = make_mesh()
        discr = Discretization(mesh, TriangularElement(4), 
                reorder=REORDER_NONE)

        def f(x):
            if x < 0.5: return 0
            else: return (x-0.5)

        def u_analytic(t, x):
            return f(a*x+t)

        u = discr.interpolate_volume_function(lambda x: u_analytic(0, x))
        dt = 1e-2
        nsteps = int(1/dt)

        from hedge.tools import dot
        nabla = discr.nabla

        def rhs_strong(t, u):
            bc = discr.interpolate_boundary_function(
                    lambda x: u_analytic(t, x),
                    "inflow")

            rhsint = dot(a, nabla*u)
            rhsflux = flux_op * u
            rhsbdry = flux_op * pair_with_boundary(u, bc, "inflow")

            return rhsint-discr.inverse_mass_operator*(rhsflux+rhsbdry)

        sym_map = SymmetryMap(discr, 
                lambda x: num.array([x[0], -x[1]]),
                {0:3, 2:1, 5:6, 7:4})

        normal = make_normal(discr.dimensions)
        fluxu = FluxScalarPlaceholder(0)
        for flux_name, flux in [
                ("lax-friedrichs",
                    dot(normal, a) * (fluxu.int - fluxu.avg)
                    + 0.5 *(fluxu.int -fluxu.ext)),
                ("central",
                    dot(normal, a) * (fluxu.int - fluxu.avg)),
                ]:
            stepper = RK4TimeStepper()
            flux_op = discr.get_flux_operator(flux)
            for step in range(nsteps):
                u = stepper(u, step*dt, dt, rhs_strong)
                sym_error_u = u-sym_map(u)
                sym_error_u_l2 = sqrt(sym_error_u*(discr.mass_operator*sym_error_u))
                self.assert_(sym_error_u_l2 < 1e-13)
    # -------------------------------------------------------------------------
    def test_convergence_advec_2d(self):
        """Test whether 2D advection actually converges"""

        import pylinear.array as num
        from hedge.mesh import make_disk_mesh
        from hedge.discretization import Discretization, pair_with_boundary
        from hedge.element import TriangularElement
        from hedge.timestep import RK4TimeStepper
        from hedge.tools import EOCRecorder, dot
        from hedge.flux import make_normal, FluxScalarPlaceholder
        from math import sin, pi, sqrt

        a = num.array([1,0])

        def u_analytic(t, x):
            return sin(a*x+t)

        def boundary_tagger(vertices, el, face_nr):
            if el.face_normals[face_nr] * a > 0:
                return ["inflow"]
            else:
                return ["outflow"]

        mesh = make_disk_mesh(r=pi, boundary_tagger=boundary_tagger, max_area=0.5)

        from hedge.tools import dot

        normal = make_normal(2)
        fluxu = FluxScalarPlaceholder(0)
        for flux_name, flux in [
                ("lax-friedrichs",
                    dot(normal, a) * (fluxu.int - fluxu.avg)
                    + 0.5 *(fluxu.int -fluxu.ext)),
                ("central",
                    dot(normal, a) * (fluxu.int - fluxu.avg)),
                ]:

            eoc_rec = EOCRecorder()

            for order in [1,2,3,4,5,6]:
                discr = Discretization(mesh, TriangularElement(order))
                nabla = discr.nabla
                flux_op = discr.get_flux_operator(flux)

                u = discr.interpolate_volume_function(lambda x: u_analytic(0, x))
                dt = 1e-2
                nsteps = int(0.1/dt)

                def rhs_strong(t, u):
                    bc = discr.interpolate_boundary_function(
                            lambda x: u_analytic(t, x),
                            "inflow")

                    rhsint = dot(a, nabla*u)
                    rhsflux = flux_op * u
                    rhsbdry = flux_op * pair_with_boundary(u, bc, "inflow")

                    return rhsint-discr.inverse_mass_operator*(rhsflux+rhsbdry)

                stepper = RK4TimeStepper()
                for step in range(nsteps):
                    u = stepper(u, step*dt, dt, rhs_strong)

                u_true = discr.interpolate_volume_function(
                        lambda x: u_analytic(nsteps*dt, x))
                error = u-u_true
                error_l2 = sqrt(error*(discr.mass_operator*error))
                eoc_rec.add_data_point(order, error_l2)
            self.assert_(eoc_rec.estimate_order_of_convergence()[0,1] > 7)
            #print "%s\n%s\n" % (flux_name.upper(), "-" * len(flux_name))
            #print eoc_rec.pretty_print(abscissa_label="Poly. Order", 
                    #error_label="L2 Error")
    # -------------------------------------------------------------------------
    def test_all_periodic_no_boundary(self):
        """Test that an all-periodic brick has no boundary."""
        from hedge.mesh import make_box_mesh

        mesh = make_box_mesh(periodicity=(True,True,True))

        def count(iterable):
            result = 0
            for i in iterable:
                result += 1
            return result

        self.assert_(count(mesh.tag_to_boundary[None]) == 0)



                




if __name__ == '__main__':
    unittest.main()
