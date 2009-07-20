# -*- coding: utf8 -*-
"""Canned operators for several PDEs, such as Maxwell's, heat, Poisson, etc."""

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




# THIS FILE HAS GROWN TOO LARGE AND IS BEING DEPRECATED. DON'T ADD STUFF HERE.
# ADD IT TO A NEW OR EXISTING FILE UNDER hedge.models INSTEAD.





import numpy
import numpy.linalg as la
import hedge.tools
import hedge.mesh
import hedge.data
from pytools import memoize_method
from hedge.models import Operator, TimeDependentOperator




class StrongWaveOperator:
    """This operator discretizes the Wave equation S{part}tt u = c^2 S{Delta} u.

    To be precise, we discretize the hyperbolic system

      * S{part}t u - c div v = 0
      * S{part}t v - c grad u = 0

    The sign of M{v} determines whether we discretize the forward or the
    backward wave equation.

    c is assumed to be constant across all space.
    """

    def __init__(self, c, dimensions, source_f=None,
            flux_type="upwind",
            dirichlet_tag=hedge.mesh.TAG_ALL,
            neumann_tag=hedge.mesh.TAG_NONE,
            radiation_tag=hedge.mesh.TAG_NONE):
        assert isinstance(dimensions, int)

        self.c = c
        self.dimensions = dimensions
        self.source_f = source_f

        if self.c > 0:
            self.sign = 1
        else:
            self.sign = -1

        self.dirichlet_tag = dirichlet_tag
        self.neumann_tag = neumann_tag
        self.radiation_tag = radiation_tag

        self.flux_type = flux_type

    def flux(self):
        from hedge.flux import FluxVectorPlaceholder, make_normal

        dim = self.dimensions
        w = FluxVectorPlaceholder(1+dim)
        u = w[0]
        v = w[1:]
        normal = make_normal(dim)

        from hedge.tools import join_fields
        flux_weak = join_fields(
                numpy.dot(v.avg, normal),
                u.avg * normal)

        if self.flux_type == "central":
            pass
        elif self.flux_type == "upwind":
            # see doc/notes/hedge-notes.tm
            flux_weak -= self.sign*join_fields(
                    0.5*(u.int-u.ext),
                    0.5*(normal * numpy.dot(normal, v.int-v.ext)))
        else:
            raise ValueError, "invalid flux type '%s'" % self.flux_type

        flux_strong = join_fields(
                numpy.dot(v.int, normal),
                u.int * normal) - flux_weak

        return -self.c*flux_strong

    def op_template(self):
        from hedge.optemplate import \
                make_vector_field, \
                pair_with_boundary, \
                get_flux_operator, \
                make_nabla, \
                InverseMassOperator, \
                BoundarizeOperator

        d = self.dimensions

        w = make_vector_field("w", d+1)
        u = w[0]
        v = w[1:]

        # boundary conditions -------------------------------------------------
        from hedge.tools import join_fields


        # dirichlet BC's ------------------------------------------------------
        dir_u = BoundarizeOperator(self.dirichlet_tag) * u
        dir_v = BoundarizeOperator(self.dirichlet_tag) * v
        dir_bc = join_fields(-dir_u, dir_v)

        # neumann BC's --------------------------------------------------------
        neu_u = BoundarizeOperator(self.neumann_tag) * u
        neu_v = BoundarizeOperator(self.neumann_tag) * v
        neu_bc = join_fields(neu_u, -neu_v)

        # radiation BC's ------------------------------------------------------
        from hedge.optemplate import make_normal
        rad_normal = make_normal(self.radiation_tag, d)

        rad_u = BoundarizeOperator(self.radiation_tag) * u
        rad_v = BoundarizeOperator(self.radiation_tag) * v

        rad_bc = join_fields(
                0.5*(rad_u - self.sign*numpy.dot(rad_normal, rad_v)),
                0.5*rad_normal*(numpy.dot(rad_normal, rad_v) - self.sign*rad_u)
                )

        # entire operator -----------------------------------------------------
        nabla = make_nabla(d)
        flux_op = get_flux_operator(self.flux())

        from hedge.tools import join_fields
        return (
                - join_fields(
                    -self.c*numpy.dot(nabla, v),
                    -self.c*(nabla*u)
                    )
                +
                InverseMassOperator() * (
                    flux_op*w
                    + flux_op * pair_with_boundary(w, dir_bc, self.dirichlet_tag)
                    + flux_op * pair_with_boundary(w, neu_bc, self.neumann_tag)
                    + flux_op * pair_with_boundary(w, rad_bc, self.radiation_tag)
                    ))


    def bind(self, discr):
        from hedge.mesh import check_bc_coverage
        check_bc_coverage(discr.mesh, [
            self.dirichlet_tag,
            self.neumann_tag,
            self.radiation_tag])

        compiled_op_template = discr.compile(self.op_template())

        def rhs(t, w):
            rhs = compiled_op_template(w=w)

            if self.source_f is not None:
                rhs[0] += self.source_f(t)

            return rhs

        return rhs

    def max_eigenvalue(self):
        return abs(self.c)




class VariableVelocityStrongWaveOperator:
    """This operator discretizes the Wave equation S{part}tt u = c^2 S{Delta} u.

    To be precise, we discretize the hyperbolic system

      * S{part}t u - c div v = 0
      * S{part}t v - c grad u = 0
    """

    def __init__(self, c, dimensions, source=None,
            flux_type="upwind",
            dirichlet_tag=hedge.mesh.TAG_ALL,
            neumann_tag=hedge.mesh.TAG_NONE,
            radiation_tag=hedge.mesh.TAG_NONE,
            time_sign=1):
        """`c` is assumed to be positive and conforms to the
        `hedge.data.ITimeDependentGivenFunction` interface.

        `source` also conforms to the
        `hedge.data.ITimeDependentGivenFunction` interface.
        """
        assert isinstance(dimensions, int)

        self.c = c
        self.time_sign = time_sign
        self.dimensions = dimensions
        self.source = source

        self.dirichlet_tag = dirichlet_tag
        self.neumann_tag = neumann_tag
        self.radiation_tag = radiation_tag

        self.flux_type = flux_type

    def flux(self):
        from hedge.flux import FluxVectorPlaceholder, make_normal

        dim = self.dimensions
        w = FluxVectorPlaceholder(2+dim)
        c = w[0]
        u = w[1]
        v = w[2:]
        normal = make_normal(dim)

        from hedge.tools import join_fields
        flux = self.time_sign*1/2*join_fields(
                c.ext * numpy.dot(v.ext, normal)
                - c.int * numpy.dot(v.int, normal),
                normal*(c.ext*u.ext - c.int*u.int))

        if self.flux_type == "central":
            pass
        elif self.flux_type == "upwind":
            flux += join_fields(
                    c.ext*u.ext - c.int*u.int,
                    c.ext*normal*numpy.dot(normal, v.ext)
                    - c.int*normal*numpy.dot(normal, v.int)
                    )
        else:
            raise ValueError, "invalid flux type '%s'" % self.flux_type

        return flux

    def op_template(self):
        from hedge.optemplate import \
                Field, \
                make_vector_field, \
                pair_with_boundary, \
                get_flux_operator, \
                make_nabla, \
                InverseMassOperator, \
                BoundarizeOperator

        d = self.dimensions

        w = make_vector_field("w", d+1)
        u = w[0]
        v = w[1:]

        from hedge.tools import join_fields
        c = Field("c")
        flux_w = join_fields(c, w)

        # boundary conditions -------------------------------------------------
        from hedge.flux import make_normal
        normal = make_normal(d)

        from hedge.tools import join_fields
        # dirichlet BC's ------------------------------------------------------
        dir_c = BoundarizeOperator(self.dirichlet_tag) * c
        dir_u = BoundarizeOperator(self.dirichlet_tag) * u
        dir_v = BoundarizeOperator(self.dirichlet_tag) * v

        dir_bc = join_fields(dir_c, -dir_u, dir_v)

        # neumann BC's --------------------------------------------------------
        neu_c = BoundarizeOperator(self.neumann_tag) * c
        neu_u = BoundarizeOperator(self.neumann_tag) * u
        neu_v = BoundarizeOperator(self.neumann_tag) * v

        neu_bc = join_fields(neu_c, neu_u, -neu_v)

        # radiation BC's ------------------------------------------------------
        from hedge.optemplate import make_normal
        rad_normal = make_normal(self.radiation_tag, d)

        rad_c = BoundarizeOperator(self.radiation_tag) * c
        rad_u = BoundarizeOperator(self.radiation_tag) * u
        rad_v = BoundarizeOperator(self.radiation_tag) * v

        rad_bc = join_fields(
                rad_c,
                0.5*(rad_u - self.time_sign*numpy.dot(rad_normal, rad_v)),
                0.5*rad_normal*(numpy.dot(rad_normal, rad_v) - self.time_sign*rad_u)
                )

        # entire operator -----------------------------------------------------
        nabla = make_nabla(d)
        flux_op = get_flux_operator(self.flux())

        return (
                - join_fields(
                    -numpy.dot(nabla, self.time_sign*c*v),
                    -(nabla*(self.time_sign*c*u))
                    )
                +
                InverseMassOperator() * (
                    flux_op*flux_w
                    + flux_op * pair_with_boundary(flux_w, dir_bc, self.dirichlet_tag)
                    + flux_op * pair_with_boundary(flux_w, neu_bc, self.neumann_tag)
                    + flux_op * pair_with_boundary(flux_w, rad_bc, self.radiation_tag)
                    ))


    def bind(self, discr):
        from hedge.mesh import check_bc_coverage
        check_bc_coverage(discr.mesh, [
            self.dirichlet_tag,
            self.neumann_tag,
            self.radiation_tag])

        compiled_op_template = discr.compile(self.op_template())

        def rhs(t, w):
            rhs = compiled_op_template(w=w,
                    c=self.c.volume_interpolant(t, discr))

            if self.source is not None:
                rhs[0] += self.source.volume_interpolant(t, discr)

            return rhs

        return rhs

    #def max_eigenvalue(self):
        #return abs(self.c)




class GasDynamicsOperatorBase(TimeDependentOperator):
    """Basis for Euler and Navier-Stokes Operator.
    Field order is [rho E rho_u_x rho_u_y ...].
    """
    def __init__(self, dimensions, gamma, bc):
        self.dimensions = dimensions
        self.gamma = gamma
        self.bc = bc

    def rho(self, q):
        return q[0]

    def e(self, q):
        return q[1]

    def rho_u(self, q):
        return q[2:2+self.dimensions]

    def u(self, q):
        from hedge.tools import make_obj_array
        return make_obj_array([
                rho_u_i/self.rho(q)
                for rho_u_i in self.rho_u(q)])

    def bind(self, discr):
        from hedge.mesh import TAG_ALL
        bound_op = discr.compile(self.op_template())

        def wrap(t, q):
            opt_result = bound_op(
                    q=q, 
                    bc_q=self.bc.boundary_interpolant(t, discr, TAG_ALL))
            max_speed = opt_result[-1]
            ode_rhs = opt_result[:-1]
	    return ode_rhs, discr.nodewise_max(max_speed)

        return wrap




class EulerOperator(GasDynamicsOperatorBase):
    """An nD Euler operator.

    see JSH, TW: Nodal Discontinuous Galerkin Methods p.206

    dq/dt + dF/dx + dG/dy = 0

    where e.g. in 2D

    q = (rho, rho_u_x, rho_u_y, E)
    F = (rho_u_x, rho_u_x^2 + p, rho_u_x * rho_u_y / rho, u_x * (E + p))
    G = (rho_u_y, rho_u_x * rho_u_y / rho, rho_u_y^2 + p, u_y * (E + p))

    Field order is [rho E rho_u_x rho_u_y ...].
    """

    def op_template(self):
        from hedge.optemplate import make_vector_field, \
                make_common_subexpression as cse

        def u(q):
            return cse(self.u(q))

        def p(q):
            return cse((self.gamma-1)*(self.e(q) - 0.5*numpy.dot(self.rho_u(q), u(q))))

        def flux(q):
            from pytools import delta
            from hedge.tools import make_obj_array, join_fields
            return [ # one entry for each flux direction
                    cse(join_fields(
                        # flux rho
                        self.rho_u(q)[i],

                        # flux E
                        cse(self.e(q)+p(q))*u(q)[i],

                        # flux rho_u
                        make_obj_array([
                            self.rho_u(q)[i]*self.u(q)[j] + delta(i,j) * p(q)
                            for j in range(self.dimensions)
                            ])
                        ))
                    for i in range(self.dimensions)]

        from hedge.optemplate import make_nabla, InverseMassOperator, \
                ElementwiseMaxOperator

        from pymbolic import var
        sqrt = var("sqrt")

        state = make_vector_field("q", self.dimensions+2)
        bc_state = make_vector_field("bc_q", self.dimensions+2)

        c = cse(sqrt(self.gamma*p(state)/self.rho(state)))

        speed = sqrt(numpy.dot(u(state), u(state))) + c

        from hedge.tools import make_lax_friedrichs_flux, join_fields
        from hedge.mesh import TAG_ALL

        flux_state = flux(state)

        return join_fields(
                (- numpy.dot(make_nabla(self.dimensions), flux_state)
                    + InverseMassOperator()*make_lax_friedrichs_flux(
                        wave_speed=
			ElementwiseMaxOperator()*
			c,
                        state=state, fluxes=flux_state,
                        bdry_tags_states_and_fluxes=[
                            (TAG_ALL, bc_state, flux(bc_state))
                            ],
                        strong=True
                        )),
                    speed)




class NavierStokesOperator(GasDynamicsOperatorBase):
    """An nD Navier-Stokes operator.

    see JSH, TW: Nodal Discontinuous Galerkin Methods p.320

    dq/dt = d/dx * (-F + tau_:1) + d/dy * (-G + tau_:2)

    where e.g. in 2D

    q = (rho, rho_u_x, rho_u_y, E)
    F = (rho_u_x, rho_u_x^2 + p, rho_u_x * rho_u_y / rho, u_x * (E + p))
    G = (rho_u_y, rho_u_x * rho_u_y / rho, rho_u_y^2 + p, u_y * (E + p))
    
    tau_11 = mu * (2 * du/dx - 2/3 * (du/dx + dv/dy))
    tau_12 = mu * (du/dy + dv/dx)
    tau_21 = tau_12
    tau_22 = mu * (2 * dv/dy - 2/3 * (du/dx + dv/dy))
    tau_31 = u * tau_11 + v * tau_12
    tau_32 = u * tau_21 + v * tau_22

    Field order is [rho E rho_u_x rho_u_y ...].
    """

    def __init__(self, dimensions, gamma, mu, bc):
        GasDynamicsOperatorBase.__init__(self, dimensions, gamma, bc)
        self.mu = mu

    def op_template(self):
        from hedge.optemplate import make_vector_field, \
                make_common_subexpression as cse

        def u(q):
            return cse(self.u(q))

        def p(q):
            return cse((self.gamma-1)*(self.e(q) - 
                       0.5*numpy.dot(self.rho_u(q), u(q))))

        def make_gradient(flux_func, bdry_tags_and_states):

            dimensions = self.dimensions
            d = len(flux_func)

            from hedge.optemplate import make_nabla
            nabla = make_nabla(dimensions)

            nabla_func = numpy.zeros((d, dimensions), dtype=object)
            for i in range(dimensions):
                nabla_func[:,i] = nabla[i] * flux_func

            from hedge.flux import make_normal, FluxVectorPlaceholder
            normal = make_normal(d)
            fluxes_ph = FluxVectorPlaceholder(d)

            flux = numpy.zeros((d, dimensions), dtype=object)
            for i in range(dimensions):
                flux[:,i] = 0.5 * normal[i] * (fluxes_ph.ext - fluxes_ph.int)

            from hedge.optemplate import get_flux_operator
            flux_op = numpy.zeros((dimensions), dtype=object)
            for i in range(dimensions):
                flux_op[i] = get_flux_operator(flux[:,i])

            from hedge.optemplate import pair_with_boundary
            flux_part = numpy.zeros((d, dimensions), dtype=object)
            for i in range(dimensions):
                flux_part[:,i] = (flux_op[i]*flux_func
                                + sum(
                                flux_op[i]*
                                pair_with_boundary(flux_func, ext_state, tag)
                                for tag, ext_state in bdry_tags_and_states))

            from hedge.optemplate import InverseMassOperator
            return (nabla_func - InverseMassOperator() * flux_part)

        def tau(q):
            from hedge.optemplate import make_nabla
            from pytools import delta

            mu = self.mu
            dimensions = self.dimensions

            dq = make_gradient(flux_func=q,
                            bdry_tags_and_states=[(TAG_ALL, bc_state)])

            du = numpy.zeros((dimensions, dimensions), dtype=object)
            for i in range(dimensions):
                for j in range(dimensions):
                    du[i,j] = (dq[i+2,j] - u(q)[i] * dq[0,j]) / self.rho(q)

            tau = numpy.zeros((dimensions+1, dimensions), dtype=object)
            for i in range(dimensions):
                for j in range(dimensions):
                    tau[i,j] = mu * (du[i,j] + du[j,i] -
                               2/3 * delta(i,j) * (du[0,0] + du[1,1]))
            for j in range(dimensions):
                tau[dimensions,j] = numpy.dot(u(q), tau[j])
            return tau

        def flux(q):
            from pytools import delta
            from hedge.tools import make_obj_array, join_fields

            return [ # one entry for each flux direction
                    cse(join_fields(
                        # flux rho
                        self.rho_u(q)[i],

                        # flux E
                        cse(self.e(q)+p(q))*u(q)[i] - 
                        cse(tau(q)[self.dimensions,i]),

                        # flux rho_u
                        make_obj_array([
                            self.rho_u(q)[i]*self.u(q)[j] + delta(i,j) * p(q) -
                            cse(tau(q)[i,j])
                            for j in range(self.dimensions)
                            ])
                        ))
                    for i in range(self.dimensions)]

        def bdry_flux(q_bdry, q_vol):
            from pytools import delta
            from hedge.tools import make_obj_array, join_fields
            from hedge.optemplate import BoundarizeOperator
            return [ # one entry for each flux direction
                    cse(join_fields(
                        # flux rho
                        self.rho_u(q_bdry)[i],

                        # flux E
                        cse(self.e(q_bdry)+p(q_bdry))*u(q_bdry)[i] - 
                        BoundarizeOperator(TAG_ALL)*cse(tau(q_vol)[self.dimensions,i]),

                        # flux rho_u
                        make_obj_array([
                            self.rho_u(q_bdry)[i]*self.u(q_bdry)[j] + 
                            delta(i,j) * p(q_bdry) - 
                            BoundarizeOperator(TAG_ALL)*cse(tau(q_vol)[i,j])
                            for j in range(self.dimensions)
                            ])
                        ))
                    for i in range(self.dimensions)]


        from hedge.optemplate import make_nabla, InverseMassOperator, \
                ElementwiseMaxOperator

        from pymbolic import var
        sqrt = var("sqrt")

        state = make_vector_field("q", self.dimensions+2)
        bc_state = make_vector_field("bc_q", self.dimensions+2)

        c = cse(sqrt(self.gamma*p(state)/self.rho(state)))

        speed = sqrt(numpy.dot(u(state), u(state))) + c

        from hedge.tools import make_lax_friedrichs_flux, join_fields
        from hedge.mesh import TAG_ALL

        flux_state = flux(state)

        return join_fields(
                (- numpy.dot(make_nabla(self.dimensions), flux_state)
                 + InverseMassOperator()*make_lax_friedrichs_flux(
                        wave_speed=cse(ElementwiseMaxOperator()*c),
                        state=state, fluxes=flux_state,
                        bdry_tags_states_and_fluxes=[(TAG_ALL, bc_state,
                        bdry_flux(bc_state, state))],
                        strong=True
                        )),
                 speed)
