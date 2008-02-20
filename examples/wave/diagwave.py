from __future__ import division

import pylinear.array as num
import pylinear.computation as comp
import hedge.tools
import hedge.mesh
import hedge.data




class Diagonalized1DWaveOperator:
    def __init__(self, c, discr, source_f=None, 
            flux_type="upwind",
            dirichlet_tag=hedge.mesh.TAG_ALL,
            neumann_tag=hedge.mesh.TAG_NONE,
            radiation_tag=hedge.mesh.TAG_NONE):
        self.c = c
        self.discr = discr
        self.source_f = source_f

        assert c > 0, "wave speed has to be positive"

        self.dirichlet_tag = dirichlet_tag
        self.neumann_tag = neumann_tag
        self.radiation_tag = radiation_tag

        from hedge.mesh import check_bc_coverage
        check_bc_coverage(discr.mesh, [
            dirichlet_tag,
            neumann_tag,
            radiation_tag])

        from hedge.flux import FluxVectorPlaceholder, make_normal

        dim = discr.dimensions
        s = FluxVectorPlaceholder(2)
        normal = make_normal(dim)

        from pytools.arithmetic_container import join_fields
        from hedge.tools import dot

        coeff_sign = self.coeff_sign = join_fields(-1, 1)
        flux_weak = s.avg*normal[0]*coeff_sign
        flux_strong = coeff_sign * s.int * normal[0] - flux_weak

        self.flux = discr.get_flux_operator(self.c*flux_strong)

        self.nabla = discr.nabla
        self.mass = discr.mass_operator
        self.m_inv = discr.inverse_mass_operator

        self.radiation_normals = discr.boundary_normals(self.radiation_tag)
        from pytools.arithmetic_container import outer_product, ArithmeticListMatrix
        self.radn_outer_radn = outer_product(
                self.radiation_normals, 
                self.radiation_normals,
                mult_op=num.multiply)

        from math import sqrt
        v = num.array([[1,1],[1,-1]])/sqrt(2)
        self.V = ArithmeticListMatrix(v)
        self.Vt = ArithmeticListMatrix(v.T)

    def rhs(self, t, w):
        from hedge.discretization import pair_with_boundary, cache_diff_results
        from pytools.arithmetic_container import join_fields
        from hedge.tools import dot

        s = self.Vt*w

        from pytools.arithmetic_container import work_with_arithmetic_containers
        ac_multiply = work_with_arithmetic_containers(num.multiply)

        rad_s = self.discr.boundarize_volume_field(s, self.radiation_tag)
        rad_n = self.radiation_normals
        ind_right = (rad_n[0]+1)/2
        ind_left = -(rad_n[0]-1)/2
        rad_bc = join_fields(
                num.multiply(ind_left, rad_s[0]),
                num.multiply(ind_right, rad_s[1])
                )
        self.rad_bc = rad_bc
        rhs = (-self.c*self.coeff_sign*(self.nabla[0]*cache_diff_results(s))
                + self.m_inv * (
                self.flux*s 
                + self.flux * pair_with_boundary(s, rad_bc, self.radiation_tag)
                ))

        w_rhs = self.V*rhs 
        if self.source_f is not None:
            w_rhs[0] += self.source_f(t)

        return w_rhs

    def max_eigenvalue(self):
        return self.c





