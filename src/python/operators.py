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




class MaxwellOperator:
    """A 3D Maxwell operator with PEC boundaries.

    Field order is [Ex Ey Ez Hx Hy Hz].
    """

    def __init__(self, discr, epsilon, mu, upwind_alpha=1, pec_tag=None,
            direct_flux=True):
        from hedge.flux import make_normal, FluxVectorPlaceholder
        from hedge.discretization import pair_with_boundary, check_bc_coverage
        from math import sqrt
        from pytools.arithmetic_container import join_fields
        from hedge.tools import cross

        self.discr = discr

        self.epsilon = epsilon
        self.mu = mu

        self.pec_tag = pec_tag

        check_bc_coverage(discr, [pec_tag])

        dim = discr.dimensions
        normal = make_normal(dim)
        w = FluxVectorPlaceholder(6)
        e = w[0:3]
        h = w[3:6]

        Z = sqrt(mu/epsilon)
        Y = 1/Z

        self.flux = discr.get_flux_operator(join_fields(
                # flux e
                1/epsilon*(
                    1/2*cross(normal, h.int-h.ext)
                    -upwind_alpha/(2*Z)*cross(normal, cross(normal, e.int-e.ext))
                    ),
                # flux h
                1/mu*(
                    -1/2*cross(normal, e.int-e.ext)
                    -upwind_alpha/(2*Y)*cross(normal, cross(normal, h.int-h.ext))
                    ),
                ), 
                direct=direct_flux)

        self.nabla = discr.nabla
        self.m_inv = discr.inverse_mass_operator


    def rhs(self, t, w):
        from hedge.tools import cross
        from hedge.discretization import pair_with_boundary

        e = w[0:3]
        h = w[3:6]

        def curl(field):
            return cross(self.nabla, field)

        bc = join_fields(
                -self.discr.boundarize_volume_field(e, self.pec_tag),
                self.discr.boundarize_volume_field(h, self.pec_tag)
                )

        bpair = pair_with_boundary(w, bc, self.pec_tag)

        return (
                join_fields(
                    1/self.epsilon * curl(h),
                    - 1/self.mu * curl(e),
                    )
                - self.m_inv*(
                    self.flux * w
                    +self.flux * pair_with_boundary(w, bc, self.pec_tag)
                    )
                )




class StrongLaplacianOperatorBase:
    def __init__(self, discr, coeff=lambda x: 1, 
            dirichlet_bc=lambda x, t: 0, dirichlet_tag="dirichlet",
            neumann_bc=lambda x, t: 0, neumann_tag="neumann",
            ldg=True):
        self.discr = discr

        
        fs = self.get_strong_flux_set(ldg)

        self.flux_u = discr.get_flux_operator(fs.flux_u)
        self.flux_v = discr.get_flux_operator(fs.flux_v)
        self.flux_u_bdry = discr.get_flux_operator(fs.flux_u_bdry)
        self.flux_v_bdry = discr.get_flux_operator(fs.flux_v_bdry)

        self.nabla = discr.nabla
        self.stiff = discr.stiffness_operator
        self.mass = discr.mass_operator
        self.m_inv = discr.inverse_mass_operator

        from math import sqrt
        from hedge.tools import coefficient_to_matrix
        from hedge.discretization import check_bc_coverage

        check_bc_coverage(discr, [dirichlet_tag, neumann_tag])

        self.coeff_func = coeff
        self.sqrt_coeff = coefficient_to_matrix(discr, lambda x: sqrt(coeff(x)))
        self.dirichlet_bc_func = dirichlet_bc
        self.dirichlet_tag = dirichlet_tag
        self.neumann_bc_func = neumann_bc
        self.neumann_tag = neumann_tag

        self.neumann_normals = discr.boundary_normals(self.neumann_tag)

    def get_weak_flux_set(self, ldg):
        class FluxSet: pass
        fs = FluxSet()

        from hedge.flux import FluxVectorPlaceholder, FluxScalarPlaceholder, make_normal

        # note here:
        # local elliptic DG is unlike the other kids in that the computation
        # of the flux of u depends *only* on u, whereas the computation of
        # the flux of v (yielding the final right hand side) may also depend
        # on u. That's why we use the layout [u,v], where v is simply omitted
        # for the u flux computation.

        dim = self.discr.dimensions
        vec = FluxVectorPlaceholder(1+dim)
        fs.u = u = vec[0]
        fs.v = v = vec[1:]
        normal = fs.normal = make_normal(dim)

        from hedge.tools import dot

        # central
        fs.flux_v = dot(v.avg, normal)
        fs.flux_u = u.avg*normal
        fs.flux_v_bdry = fs.flux_v
        fs.flux_u_bdry = fs.flux_u

        if ldg:
            from pytools.arithmetic_container import ArithmeticList 
            ldg_beta = ArithmeticList([1]*dim)

            fs.flux_v = fs.flux_v + dot((v.int-v.ext)*0.5, ldg_beta)
            fs.flux_u = fs.flux_u -(u.int-u.ext)*0.5*ldg_beta

        return fs

    def get_strong_flux_set(self, ldg):
        from hedge.tools import dot

        fs = self.get_weak_flux_set(ldg)

        u = fs.u
        v = fs.v
        normal = fs.normal

        fs.flux_u = u.int*normal - fs.flux_u
        fs.flux_v = dot(v.int, normal) - fs.flux_v
        fs.flux_u_bdry = u.int*normal - fs.flux_u_bdry
        fs.flux_v_bdry = dot(v.int, normal) - fs.flux_v_bdry

        return fs

    def v(self, t, u):
        from hedge.discretization import pair_with_boundary
        from math import sqrt

        dtag = self.dirichlet_tag
        ntag = self.neumann_tag


    def rhs(self, t, u):
        from hedge.discretization import pair_with_boundary
        from math import sqrt
        from hedge.tools import dot
        from pytools.arithmetic_container import \
                work_with_arithmetic_containers, \
                join_fields

        def neumann_bc_func(x):
            return sqrt(self.coeff_func(x))*self.neumann_bc_func(t, x)

        dtag = self.dirichlet_tag
        ntag = self.neumann_tag

        sqrt_coeff_u = self.sqrt_coeff * u

        dirichlet_bc_u = self.dirichlet_bc_u(t, sqrt_coeff_u)
        neumann_bc_u = self.discr.boundarize_volume_field(sqrt_coeff_u, ntag)

        v = self.m_inv * (
                self.sqrt_coeff*(self.stiff * u)
                - self.flux_u*sqrt_coeff_u
                - self.flux_u_bdry*pair_with_boundary(sqrt_coeff_u, dirichlet_bc_u, dtag)
                - self.flux_u_bdry*pair_with_boundary(sqrt_coeff_u, neumann_bc_u, ntag)
                )
        sqrt_coeff_v = self.sqrt_coeff * v

        ac_multiply = work_with_arithmetic_containers(num.multiply)

        dirichlet_bc_v = self.discr.boundarize_volume_field(sqrt_coeff_v, dtag)

        neumann_bc_v = (
                -self.discr.boundarize_volume_field(sqrt_coeff_v, ntag)
                +
                2*ac_multiply(self.neumann_normals,
                self.discr.interpolate_boundary_function(neumann_bc_func, ntag))
                )

        w = join_fields(sqrt_coeff_u, sqrt_coeff_v)
        dirichlet_bc_w = join_fields(dirichlet_bc_u, dirichlet_bc_v)
        neumann_bc_w = join_fields(neumann_bc_u, neumann_bc_v)

        return  (
                dot(self.stiff, self.sqrt_coeff*v)
                - self.flux_v * w
                - self.flux_v_bdry * pair_with_boundary(w, dirichlet_bc_w, dtag)
                - self.flux_v_bdry * pair_with_boundary(w, neumann_bc_w, ntag)
                )





class StrongPoissonOperator(StrongLaplacianOperatorBase):
    def dirichlet_bc_u(self, t, sqrt_coeff_u):
        from math import sqrt

        def dir_bc_func(x):
            return sqrt(self.coeff_func(x))*self.dirichlet_bc_func(t, x)

        dtag = self.dirichlet_tag
        return self.discr.interpolate_boundary_function(dir_bc_func, dtag)

    def get_weak_flux_set(self, ldg):
        fs = StrongLaplacianOperatorBase.get_weak_flux_set(self, ldg)

        from hedge.flux import PenaltyTerm

        # apply stabilisation
        u = fs.u
        stab_term = 1 * PenaltyTerm() * (u.int - u.ext)

        fs.flux_v -= stab_term
        fs.flux_v_bdry -= stab_term

        return fs





class StrongHeatOperator(StrongLaplacianOperatorBase):
    def dirichlet_bc_u(self, t, sqrt_coeff_u):
        from math import sqrt

        def dir_bc_func(x):
            return sqrt(self.coeff_func(x))*self.dirichlet_bc_func(t, x)

        dtag = self.dirichlet_tag
        return (
                -self.discr.boundarize_volume_field(sqrt_coeff_u, dtag)
                +2*self.discr.interpolate_boundary_function(dir_bc_func, dtag))

    def rhs(self, t, u):
        return self.m_inv * StrongLaplacianOperatorBase.rhs(self, t, u)
