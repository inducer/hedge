from __future__ import division
import unittest




class TestHedge(unittest.TestCase):
    def test_newton_interpolation(self):
        from hedge.interpolation import newton_interpolation_function
        
        x = [-1.5, -0.75, 0, 0.75, 1.5]
        y = [-14.1014, -0.931596, 0, 0.931596, 14.1014]
        nf = newton_interpolation_function(x, y)

        errors = [abs(yi-nf(xi)) for xi, yi in zip(x, y)]
        self.assert_(sum(errors) < 1e-10)
    # -------------------------------------------------------------------------
    def test_orthonormality_1d(self):
        n = 10

        from hedge.polynomial import legendre_function
        from hedge.quadrature import LegendreGaussQuadrature

        leg_f = [legendre_function(i) for i in range(n)]

        lgq = LegendreGaussQuadrature(n)

        for i, fi in enumerate(leg_f):
            for j, fj in enumerate(leg_f):
                result = lgq(lambda x: fi(x)*fj(x))
                if fi == fj:
                    self.assert_(abs(result-1) < 1e-9)
                else:
                    self.assert_(abs(result) < 1e-9)
    # -------------------------------------------------------------------------
    def test_transformed_quadrature(self):
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
        n = 17
        from hedge.element import WarpFactorCalculator
        wfc = WarpFactorCalculator(n)

        self.assert_(abs(wfc.int_f(-1)) < 1e-10)
        self.assert_(abs(wfc.int_f(1)) < 1e-10)

        from hedge.quadrature import LegendreGaussQuadrature

        lgq = LegendreGaussQuadrature(n)
        self.assert_(abs(lgq(wfc)) < 1e-10)
    # -------------------------------------------------------------------------
    def test_tri_nodes(self):
        from hedge.element import TriangularElement

        n = 17
        tri = TriangularElement(n)
        unodes = list(tri.unit_nodes())
        self.assert_(len(unodes) == tri.node_count())

        eps = 1e-10
        for ux in unodes:
            self.assert_(ux[0] >= -1-eps)
            self.assert_(ux[1] >= -1-eps)
            self.assert_(ux[0]+ux[1] <= 1+eps)

        for i, j in tri.node_indices():
            self.assert_(i >= 0)
            self.assert_(j >= 0)
            self.assert_(i+j <= n)
    # -------------------------------------------------------------------------
    def test_tri_basis_grad(self):
        from itertools import izip
        from hedge.element import TriangularElement
        from random import uniform
        import pylinear.array as num
        import pylinear.computation as comp

        tri = TriangularElement(8)
        for bf, gradbf in izip(tri.basis_functions(), tri.grad_basis_functions()):
            for i in range(10):
                r = uniform(-0.95, 0.95)
                s = uniform(-0.95, -r-0.05)

                h = 1e-4
                gradbf_v = num.array(gradbf((r,s)))
                approx_gradbf_v = num.array([
                    (bf((r+h,s)) - bf((r-h,s)))/(2*h),
                    (bf((r,s+h)) - bf((r,s-h)))/(2*h)
                    ])
                self.assert_(comp.norm_infinity(approx_gradbf_v-gradbf_v) < h)
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
    def test_tri_face_normals_and_jacobians(self):
        """Check computed face normals and face jacobians
        """
        from hedge.element import TriangularElement
        from hedge.tools import AffineMap
        import pylinear.array as num
        import pylinear.computation as comp
        from pylinear.randomized import make_random_vector

        tri = TriangularElement(8)

        for i in range(10):
            vertices = [make_random_vector(2, num.Float) for vi in range(3)]
            map = tri.get_map_unit_to_global(vertices)

            unodes = tri.unit_nodes()
            nodes = [map(v) for v in unodes]
            normals, jacobians = tri.face_normals_and_jacobians(map)

            for face_i, normal, jac in zip(tri.face_indices(), normals, jacobians):
                mapped_start = nodes[face_i[0]]
                mapped_end = nodes[face_i[-1]]
                mapped_dir = mapped_end-mapped_start
                start = unodes[face_i[0]]
                end = unodes[face_i[-1]]
                true_jac = comp.norm_2(mapped_end-mapped_start)/2

                #print abs(true_jac-jac)/true_jac
                #print "aft, bef", comp.norm_2(mapped_end-mapped_start),comp.norm_2(end-start)

                self.assert_(abs(true_jac - jac)/true_jac < 1e-13)
                self.assert_(abs(comp.norm_2(normal) - 1) < 1e-13)
                self.assert_(abs(normal*mapped_dir) < 1e-13)
    # -------------------------------------------------------------------------
    def test_tri_map(self):
        from hedge.element import TriangularElement
        import pylinear.array as num
        import pylinear.computation as comp
        from pylinear.randomized import \
                make_random_vector

        n = 8
        tri = TriangularElement(n)

        node_dict = dict((ituple, idx) for idx, ituple in enumerate(tri.node_indices()))
        corner_indices = [node_dict[0,0], node_dict[n,0], node_dict[0,n]]
        unodes = tri.unit_nodes()
        corners = [unodes[i] for i in corner_indices]

        for i in range(10):
            vertices = [make_random_vector(2, num.Float) for vi in range(3)]
            map = tri.get_map_unit_to_global(vertices)
            global_corners = [map(pt) for pt in corners]
            for gc, v in zip(global_corners, vertices):
                self.assert_(comp.norm_2(gc-v) < 1e-12)
    # -------------------------------------------------------------------------
    def test_tri_map_jacobian_and_mass_matrix(self):
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
            map = edata.get_map_unit_to_global(vertices)
            mat = num.zeros((2,2))
            mat[:,0] = (vertices[1] - vertices[0])
            mat[:,1] = (vertices[2] - vertices[0])
            tri_area = abs(comp.determinant(mat)/2)
            tri_area_2 = abs(unit_tri_area*map.jacobian)
            self.assert_(abs(tri_area - tri_area_2)/tri_area < 1e-10)
    # -------------------------------------------------------------------------
    def test_tri_mass_mat(self):
        """Check the integral of a Gaussian on a disk using the mass matrix"""
        from hedge.mesh import make_disk_mesh
        from hedge.element import TriangularElement
        from hedge.discretization import Discretization
        from math import sqrt, exp, pi

        sigma_squared = 1/219.3

        mesh = make_disk_mesh()
        discr = Discretization(make_disk_mesh(), TriangularElement(9))
        f = discr.interpolate_volume_function(lambda x: exp(-x*x/(2*sigma_squared)))
        ones = discr.interpolate_volume_function(lambda x: 1)

        #discr.visualize_field("gaussian.vtk", [("f", f)])
        num_integral_1 = ones * discr.apply_mass_matrix(f)
        num_integral_2 = f * discr.apply_mass_matrix(ones)
        dim = 2
        true_integral = (2*pi)**(dim/2)*sqrt(sigma_squared)**dim
        err_1 = abs(num_integral_1-true_integral)
        err_2 = abs(num_integral_2-true_integral)
        self.assert_(err_1 < 1e-11)
        self.assert_(err_2 < 1e-11)
    # -------------------------------------------------------------------------
    def test_tri_diff_mat(self):
        """Check differentiation matrix along the coordinate axes on a disk.
        
        Uses sines as the function to differentiate.
        """
        from hedge.mesh import make_disk_mesh
        from hedge.element import TriangularElement
        from hedge.discretization import Discretization
        from math import sin, cos, sqrt

        for coord in [0, 1]:
            mesh = make_disk_mesh()
            discr = Discretization(make_disk_mesh(), TriangularElement(9))
            f = discr.interpolate_volume_function(lambda x: sin(3*x[coord]))
            df = discr.interpolate_volume_function(lambda x: 3*cos(3*x[coord]))

            df_num = discr.differentiate(coord, f)
            error = df_num - df
            #discr.visualize_field("diff-err.vtk",
                    #[("f", f), ("df", df), ("df_num", df_num), ("error", error)])

            l2_error = sqrt(error * discr.apply_mass_matrix(error))
            #print l2_error, len(mesh.elements)
            self.assert_(l2_error < 1e-4)
    # -------------------------------------------------------------------------
    def test_tri_gauss_theorem(self):
        """Verify Gauss's theorem explicitly on a couple of elements 
        in random orientation."""

        from hedge.element import TriangularElement
        from hedge.tools import AffineMap
        import pylinear.array as num
        import pylinear.computation as comp
        from pylinear.randomized import make_random_vector
        from operator import add
        from math import sin, cos, sqrt, exp, pi

        edata = TriangularElement(9)
        ones = num.ones((edata.node_count(),))
        face_ones = num.ones((len(edata.face_indices()[0]),))

        def f1(x):
            return sin(3*x[0])+cos(3*x[1])
        def f2(x):
            return sin(2*x[0])+cos(x[1])

        def d(imap, coordinate, field):
            col = imap.matrix[:, coordinate]
            matrices = edata.differentiation_matrices()
            return reduce(add, (dmat*coeff*field
                        for dmat, coeff in zip(matrices, col)))

        for i in range(10):
            na = num.array
            #vertices = [na([0,0]), na([1,0]), na([0,1])]
            vertices = [make_random_vector(2, num.Float) for vi in range(3)]
            map = edata.get_map_unit_to_global(vertices)
            imap = map.inverted()

            mapped_points = [map(node) for node in edata.unit_nodes()]
            f1_n = num.array([f1(x) for x in mapped_points])
            f2_n = num.array([f2(x) for x in mapped_points])

            dx_n = d(imap, 0, f1_n)
            dy_n = d(imap, 1, f2_n)

            int_div_f = abs(map.jacobian)*(
                    ones*edata.mass_matrix()*dx_n +
                    ones*edata.mass_matrix()*dy_n
                    )

            normals, jacobians = edata.face_normals_and_jacobians(map)
            boundary_sum = sum(
                    sum(
                        fjac * face_ones * edata.face_mass_matrix() 
                        * num.take(f_n, face_indices) * n_coord
                        for f_n, n_coord in zip([f1_n, f2_n], n))
                    for face_indices, n, fjac
                    in zip(edata.face_indices(), normals, jacobians))
            self.assert_(abs(boundary_sum-int_div_f) < 1e-7)

               



if __name__ == '__main__':
    unittest.main()
