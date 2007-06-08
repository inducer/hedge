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
    def test_orthonormality_jacobi_1d(self):
        from hedge.polynomial import jacobi_function
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
                (0, 0, 1e-9), 
                (1, 0, 1e-9), 
                (3, 2, 1e-10), 
                (0, 2, 1e-9), 
                (5, 0, 1e-8), 
                (3, 4, 1e-9)
                ]:
            jac_f = [jacobi_function(alpha, beta, n) for n in range(max_n)]
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

        #discr.visualize_field("gaussian.vtk", [("f", f)])
        num_integral_1 = ones * discr.apply_mass_matrix(f)
        num_integral_2 = f * discr.apply_mass_matrix(ones)
        dim = 2
        true_integral = (2*pi)**(dim/2)*sqrt(sigma_squared)**dim
        err_1 = abs(num_integral_1-true_integral)
        err_2 = abs(num_integral_2-true_integral)
        print err_1
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
        discr = Discretization(mesh, TriangularElement(9))
        f = discr.interpolate_volume_function(lambda x: cos(x[0])**2*sin(x[1])**2)
        ones = discr.interpolate_volume_function(lambda x: 1)

        discr.visualize_vtk("trig.vtk", [("f", f)])
        num_integral_1 = ones * discr.apply_mass_matrix(f)
        num_integral_2 = f * discr.apply_mass_matrix(ones)
        true_integral = pi**2
        err_1 = abs(num_integral_1-true_integral)
        err_2 = abs(num_integral_2-true_integral)
        self.assert_(err_1 < 1e-10)
        self.assert_(err_2 < 1e-10)
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
            discr = Discretization(make_disk_mesh(), TriangularElement(4))
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
    def test_2d_gauss_theorem(self):
        """Verify Gauss's theorem explicitly on a couple of elements 
        in random orientation."""

        from hedge.element import TriangularElement
        from hedge.tools import AffineMap
        from hedge.mesh import make_disk_mesh
        from hedge.discretization import Discretization
        import pylinear.array as num
        import pylinear.computation as comp
        from pylinear.randomized import make_random_vector
        from math import sin, cos, sqrt, exp, pi

        class OneSidedFlux:
            def __init__(self, coordinate):
                self.coordinate = coordinate
            def local_coeff(self, normal):
                return normal[self.coordinate]
            def neighbor_coeff(self, normal):
                return 0

        one_sided_x = OneSidedFlux(0)
        one_sided_y = OneSidedFlux(1)

        def f1(x):
            return sin(3*x[0])+cos(3*x[1])
        def f2(x):
            return sin(2*x[0])+cos(x[1])

        edata = TriangularElement(9)

        discr = Discretization(make_disk_mesh(), edata)
        ones = discr.interpolate_volume_function(lambda x: 1)
        face_zeros = discr.boundary_zeros("boundary")
        face_ones = discr.interpolate_boundary_function(
                "boundary", lambda x: 1)

        mapped_points = [map(node) for node in edata.unit_nodes()]
        f1_v = discr.interpolate_volume_function(f1)
        f2_v = discr.interpolate_volume_function(f2)

        f1_f = discr.interpolate_boundary_function("boundary", f1)
        f2_f = discr.interpolate_boundary_function("boundary", f2)

        dx_v = discr.differentiate(0, f1_v)
        dy_v = discr.differentiate(1, f2_v)

        int_div = \
                ones*discr.apply_mass_matrix(dx_v) + \
                ones*discr.apply_mass_matrix(dy_v)

        boundary_int = (
                discr.lift_boundary_flux("boundary",
                    one_sided_x, f1_v, face_zeros) +
                discr.lift_boundary_flux("boundary",
                    one_sided_y, f2_v, face_zeros)
                )*ones

        print abs(boundary_int-int_div)
        self.assert_(abs(boundary_int-int_div) < 1e-11)

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
    # -------------------------------------------------------------------------
    def test_cubature(self):
        """Test the integrity of the cubature data."""

        from hedge.cubature import integrate_on_tetrahedron, TetrahedronCubatureData

        for i in range(len(TetrahedronCubatureData)):
            self.assert_(abs(integrate_on_tetrahedron(i+1, lambda x: 1)-2) < 1e-14)
    # -------------------------------------------------------------------------
    def test_tri_orthogonality(self):
        """Test the integrity of the cubature data."""

        from hedge.cubature import integrate_on_tetrahedron, TetrahedronCubatureData
        from hedge.element import TriangularElement

        for order, ebound in [
                (3, 1e-11),
                (4, 1e-11),
                (7, 1e-10),
                (9, 1e-8),
                ]:
            edata = TriangularElement(order)
            basis = edata.basis_functions()

            maxerr = 0
            for i, f in enumerate(basis):
                for j, g in enumerate(basis):
                    if i == j:
                        true_result = 1
                    else:
                        true_result = 0
                    result = integrate_on_tetrahedron(2*order, lambda x: f(x)*g(x))
                    err = abs(result-true_result)
                    maxerr = max(maxerr, err)
                    if err > ebound:
                        print "bad", order,i,j, err
                    self.assert_(err < ebound)
            #print order, maxerr
    # -------------------------------------------------------------------------
    def test_1d_mass_matrix_vs_quadrature(self):
        pass






               



if __name__ == '__main__':
    unittest.main()
