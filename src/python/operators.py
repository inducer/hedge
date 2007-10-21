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
import pylinear.operator as operator
import hedge.mesh




class MaxwellOperator:
    """A 3D Maxwell operator with PEC boundaries.

    Field order is [Ex Ey Ez Hx Hy Hz].
    """

    def __init__(self, discr, epsilon, mu, upwind_alpha=1, 
            pec_tag=hedge.mesh.TAG_ALL, direct_flux=True):
        from hedge.flux import make_normal, FluxVectorPlaceholder
        from hedge.mesh import check_bc_coverage
        from hedge.discretization import pair_with_boundary
        from math import sqrt
        from pytools.arithmetic_container import join_fields
        from hedge.tools import cross

        self.discr = discr

        self.epsilon = epsilon
        self.mu = mu

        self.pec_tag = pec_tag

        check_bc_coverage(discr.mesh, [pec_tag])

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
        self.flux_u_dbdry = discr.get_flux_operator(fs.flux_u_dbdry)
        self.flux_v_dbdry = discr.get_flux_operator(fs.flux_v_dbdry)
        self.flux_u_nbdry = discr.get_flux_operator(fs.flux_u_nbdry)
        self.flux_v_nbdry = discr.get_flux_operator(fs.flux_v_nbdry)

        self.nabla = discr.nabla
        self.stiff = discr.stiffness_operator
        self.mass = discr.mass_operator
        self.m_inv = discr.inverse_mass_operator

        from math import sqrt
        from hedge.tools import coefficient_to_matrix
        from hedge.mesh import check_bc_coverage

        check_bc_coverage(discr.mesh, [dirichlet_tag, neumann_tag])

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

        # local DG is unlike the other kids in that the computation of the flux
        # of u depends *only* on u, whereas the computation of the flux of v
        # (yielding the final right hand side) may also depend on u. That's why
        # we use the layout [u,v], where v is simply omitted for the u flux
        # computation.

        dim = self.discr.dimensions
        vec = FluxVectorPlaceholder(1+dim)
        fs.u = u = vec[0]
        fs.v = v = vec[1:]
        normal = fs.normal = make_normal(dim)

        from hedge.tools import dot

        # central
        fs.flux_u = u.avg*normal
        fs.flux_v = dot(v.avg, normal)

        # dbdry is "dirichlet boundary"
        # nbdry is "neumann boundary"
        fs.flux_u_dbdry = fs.flux_u
        fs.flux_u_nbdry = fs.flux_u

        fs.flux_v_dbdry = fs.flux_v
        fs.flux_v_nbdry = fs.flux_v

        if ldg:
            from pytools.arithmetic_container import ArithmeticList 
            ldg_beta = ArithmeticList([1]*dim)

            fs.flux_u = fs.flux_u - (u.int-u.ext)*0.5*ldg_beta
            fs.flux_v = fs.flux_v + dot((v.int-v.ext)*0.5, ldg_beta)

        return fs

    def get_strong_flux_set(self, ldg):
        from hedge.tools import dot

        fs = self.get_weak_flux_set(ldg)

        u = fs.u
        v = fs.v
        normal = fs.normal

        fs.flux_u = u.int*normal - fs.flux_u
        fs.flux_v = dot(v.int, normal) - fs.flux_v
        fs.flux_u_dbdry = u.int*normal - fs.flux_u_dbdry
        fs.flux_v_dbdry = dot(v.int, normal) - fs.flux_v_dbdry
        fs.flux_u_nbdry = u.int*normal - fs.flux_u_nbdry
        fs.flux_v_nbdry = dot(v.int, normal) - fs.flux_v_nbdry

        return fs






class StrongPoissonOperator(StrongLaplacianOperatorBase, operator.Operator(num.Float64)):
    """Implements LDG according to

    P. Castillo et al., 
    Local discontinuous Galerkin methods for elliptic problems", 
    Communications in Numerical Methods in Engineering 18, no. 1 (2002): 69-75.
    """
    def __init__(self, *args, **kwargs):
        operator.Operator(num.Float64).__init__(self)
        StrongLaplacianOperatorBase.__init__(self, *args, **kwargs)

        self.dirichlet_zeros = self.discr.boundary_zeros(self.dirichlet_tag)
        self.neumann_zeros = self.discr.boundary_zeros(self.neumann_tag)

    # pylinear operator infrastructure ----------------------------------------
    def size1(self):
        return len(self.discr)

    def size2(self):
        return len(self.discr)

    def apply(self, before, after):
        after[:] = self.op(before)

    # boundary conditions -----------------------------------------------------
    def dirichlet_bc_u(self):
        from math import sqrt

        def dir_bc_func(x):
            return sqrt(self.coeff_func(x))*self.dirichlet_bc_func(0, x)

        return self.discr.interpolate_boundary_function(dir_bc_func, self.dirichlet_tag)

    def dirichlet_bc_v(self, sqrt_coeff_v):
        return self.discr.boundarize_volume_field(sqrt_coeff_v, self.dirichlet_tag)

    def neumann_bc_u(self, sqrt_coeff_u):
        return self.discr.boundarize_volume_field(sqrt_coeff_u, self.neumann_tag)

    def neumann_bc_v(self):
        from pytools.arithmetic_container import work_with_arithmetic_containers
        from math import sqrt

        def neumann_bc_func(x):
            return sqrt(self.coeff_func(x))*self.neumann_bc_func(0, x)

        ac_multiply = work_with_arithmetic_containers(num.multiply)

        return ac_multiply(self.neumann_normals,
                self.discr.interpolate_boundary_function(
                    neumann_bc_func, self.neumann_tag))

    # fluxes ------------------------------------------------------------------
    def get_weak_flux_set(self, ldg):
        fs = StrongLaplacianOperatorBase.get_weak_flux_set(self, ldg)

        from hedge.flux import PenaltyTerm
        from hedge.tools import dot

        # apply stabilisation
        u = fs.u
        v = fs.v
        normal = fs.normal

        fs.flux_v -= PenaltyTerm() * (u.int - u.ext)

        # boundary fluxes

        fs.flux_u_dbdry = fs.normal * u.ext
        fs.flux_v_dbdry = dot(v.int, normal) - PenaltyTerm()*(u.int - u.ext)

        fs.flux_u_nbdry = fs.normal * u.int
        fs.flux_v_nbdry = dot(fs.normal, v.ext)

        return fs

    # operator application, rhs prep ------------------------------------------
    def op(self, u):
        from hedge.discretization import pair_with_boundary
        from math import sqrt
        from hedge.tools import dot
        from pytools.arithmetic_container import join_fields, ArithmeticList

        dim = self.discr.dimensions

        dtag = self.dirichlet_tag
        ntag = self.neumann_tag

        sqrt_coeff_u = self.sqrt_coeff * u

        v = self.m_inv * (
                self.sqrt_coeff*(self.stiff * u)
                - self.flux_u*sqrt_coeff_u
                - self.flux_u_dbdry*pair_with_boundary(sqrt_coeff_u, self.dirichlet_zeros, dtag)
                - self.flux_u_nbdry*pair_with_boundary(sqrt_coeff_u, self.neumann_zeros, ntag)
                )
        sqrt_coeff_v = self.sqrt_coeff * v

        dirichlet_bc_v = self.dirichlet_bc_v(sqrt_coeff_v)
        neumann_bc_v = ArithmeticList(self.neumann_zeros for i in range(dim))

        w = join_fields(sqrt_coeff_u, sqrt_coeff_v)
        dirichlet_bc_w = join_fields(self.dirichlet_zeros, dirichlet_bc_v)
        neumann_bc_w = join_fields(self.neumann_bc_u(sqrt_coeff_u), neumann_bc_v)

        return (
                dot(self.stiff, sqrt_coeff_v)
                - self.flux_v * w
                - self.flux_v_dbdry * pair_with_boundary(w, dirichlet_bc_w, dtag)
                - self.flux_v_nbdry * pair_with_boundary(w, neumann_bc_w, ntag)
                )

    def prepare_rhs(self, rhs):
        """Perform the rhs(*) function in the class description, i.e.
        return a right hand side for the linear system op(u)=rhs(f).
        
        In matrix form, LDG looks like this:
        
        Mv = Cu + g
        Mf = Av + Bu + h

        where v is the auxiliary vector, u is the argument of the operator, f
        is the result of the operator and g and h are inhom boundary data, and
        A,B,C are some operator+lifting matrices

        M f = A Minv(Cu + g) + Bu + h

        so the linear system looks like

        M f = A Minv Cu + A Minv g + Bu + h
        M f - A Minv g - h = (A Minv C + B)u

        So the right hand side we're putting together here is really

        M f - A Minv g - h
        """

        from pytools.arithmetic_container import ArithmeticList 
        from hedge.discretization import pair_with_boundary
        from hedge.tools import dot
        from pytools.arithmetic_container import join_fields

        dim = self.discr.dimensions

        dtag = self.dirichlet_tag
        ntag = self.neumann_tag

        vol_zeros = self.discr.volume_zeros()
        dirichlet_bc_u = self.dirichlet_bc_u()
        vpart = self.m_inv * (
                -(self.flux_u_dbdry*pair_with_boundary(vol_zeros, dirichlet_bc_u, dtag))
                )
        sqrt_coeff_v = self.sqrt_coeff * vpart

        dirichlet_bc_v = ArithmeticList(self.dirichlet_zeros for i in range(dim))
        neumann_bc_v = self.neumann_bc_v()

        w = join_fields(vol_zeros, sqrt_coeff_v)
        dirichlet_bc_w = join_fields(dirichlet_bc_u, dirichlet_bc_v)
        neumann_bc_w = join_fields(self.neumann_zeros, neumann_bc_v)

        return self.discr.mass_operator * rhs - (
                dot(self.stiff, sqrt_coeff_v)
                - self.flux_v * w
                - self.flux_v_dbdry * pair_with_boundary(w, dirichlet_bc_w, dtag)
                - self.flux_v_nbdry * pair_with_boundary(w, neumann_bc_w, ntag)
                )




class StrongHeatOperator(StrongLaplacianOperatorBase):
    def dirichlet_bc_u(self, t, sqrt_coeff_u):
        from math import sqrt

        def dir_bc_func(x):
            return sqrt(self.coeff_func(x))*self.dirichlet_bc_func(t, x)

        dtag = self.dirichlet_tag
        return (
                -self.discr.boundarize_volume_field(sqrt_coeff_u, dtag)
                +2*self.discr.interpolate_boundary_function(dir_bc_func, dtag))

    def dirichlet_bc_v(self, t, sqrt_coeff_v):
        return self.discr.boundarize_volume_field(
                sqrt_coeff_v, self.dirichlet_tag)

    def neumann_bc_u(self, t, sqrt_coeff_u):
        return self.discr.boundarize_volume_field(
                sqrt_coeff_u, self.neumann_tag)

    def neumann_bc_v(self, t, sqrt_coeff_v):
        from pytools.arithmetic_container import work_with_arithmetic_containers
        from math import sqrt

        def neumann_bc_func(x):
            return sqrt(self.coeff_func(x))*self.neumann_bc_func(t, x)

        ntag = self.neumann_tag

        ac_multiply = work_with_arithmetic_containers(num.multiply)

        return (
                -self.discr.boundarize_volume_field(sqrt_coeff_v, ntag)
                +
                2*ac_multiply(self.neumann_normals,
                self.discr.interpolate_boundary_function(neumann_bc_func, ntag))
                )

    def rhs(self, t, u):
        from hedge.discretization import pair_with_boundary
        from math import sqrt
        from hedge.tools import dot
        from pytools.arithmetic_container import join_fields

        dtag = self.dirichlet_tag
        ntag = self.neumann_tag

        sqrt_coeff_u = self.sqrt_coeff * u

        dirichlet_bc_u = self.dirichlet_bc_u(t, sqrt_coeff_u)
        neumann_bc_u = self.neumann_bc_u(t, sqrt_coeff_u)

        v = self.m_inv * (
                self.sqrt_coeff*(self.stiff * u)
                - self.flux_u*sqrt_coeff_u
                - self.flux_u_dbdry*pair_with_boundary(sqrt_coeff_u, dirichlet_bc_u, dtag)
                - self.flux_u_nbdry*pair_with_boundary(sqrt_coeff_u, neumann_bc_u, ntag)
                )
        sqrt_coeff_v = self.sqrt_coeff * v

        dirichlet_bc_v = self.dirichlet_bc_v(t, sqrt_coeff_v)
        neumann_bc_v = self.neumann_bc_v(t, sqrt_coeff_v)

        w = join_fields(sqrt_coeff_u, sqrt_coeff_v)
        dirichlet_bc_w = join_fields(dirichlet_bc_u, dirichlet_bc_v)
        neumann_bc_w = join_fields(neumann_bc_u, neumann_bc_v)

        return self.m_inv * (
                dot(self.stiff, self.sqrt_coeff*v)
                - self.flux_v * w
                - self.flux_v_dbdry * pair_with_boundary(w, dirichlet_bc_w, dtag)
                - self.flux_v_nbdry * pair_with_boundary(w, neumann_bc_w, ntag)
                )
