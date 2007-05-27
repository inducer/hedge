import unittest




class TestHedge(unittest.TestCase):
    def test_newton_interpolation(self):
        from hedge.interpolation import newton_interpolation_function
        
        x = [-1.5, -0.75, 0, 0.75, 1.5]
        y = [-14.1014, -0.931596, 0, 0.931596, 14.1014]
        nf = newton_interpolation_function(x, y)

        errors = [abs(yi-nf(xi)) for xi, yi in zip(x, y)]
        self.assert_(sum(errors) < 1e-10)

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

    def test_warp(self):
        n = 17
        from hedge.element import WarpFactorCalculator
        wfc = WarpFactorCalculator(n)

        self.assert_(abs(wfc.int_f(-1)) < 1e-10)
        self.assert_(abs(wfc.int_f(1)) < 1e-10)

        from hedge.quadrature import LegendreGaussQuadrature

        lgq = LegendreGaussQuadrature(n)
        self.assert_(abs(lgq(wfc)) < 1e-10)

    def test_tri_nodes(self):
        from hedge.element import Triangle

        n = 17
        tri = Triangle(n)
        unodes = list(tri.unit_nodes())
        self.assert_(len(unodes) == (n+1)*(n+2)/2)

        eps = 1e-10
        for ux in unodes:
            self.assert_(ux[0] >= -1-eps)
            self.assert_(ux[1] >= -1-eps)
            self.assert_(ux[0]+ux[1] <= 1+eps)

    def test_tri_basis_grad(self):
        from itertools import izip
        from hedge.element import Triangle
        from random import uniform
        import pylinear.array as num
        import pylinear.computation as comp

        tri = Triangle(8)
        for bf, gradbf in izip(tri.basis_functions(), tri.grad_basis_functions()):
            for i in range(10):
                r = uniform(-0.95, 0.95)
                s = uniform(-0.95, -r-0.05)

                h = 1e-4
                gradbf_v = gradbf((r,s))
                approx_gradbf_v = num.array([
                    (bf((r+h,s)) - bf((r-h,s)))/(2*h),
                    (bf((r,s+h)) - bf((r,s-h)))/(2*h)
                    ])
                self.assert_(comp.norm_infinity(approx_gradbf_v-gradbf_v) < h)








if __name__ == '__main__':
    unittest.main()
