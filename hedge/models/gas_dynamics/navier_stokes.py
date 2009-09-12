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




import numpy
import numpy.linalg as la
import hedge.tools
import hedge.mesh
import hedge.data
from pytools import memoize_method
from hedge.models import Operator, TimeDependentOperator
from hedge.models.gas_dynamics import GasDynamicsOperatorBase








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
                flux[:,i] = 0.5 * normal[i] * (fluxes_ph.int - fluxes_ph.ext)

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

        def artificial_viscosity(q):
            dq = make_gradient(flux_func=q,
                            bdry_tags_and_states=[(TAG_ALL, bc_state)])
            sensor_value = 0
            art_vis = sensor_value * dq
            return art_vis

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
                        )
                        - artificial_viscosity(q)[:,i]
                        )
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
                        )
                        - BoundarizeOperator(TAG_ALL)*artificial_viscosity(q_vol)[:,i]
                        )
                    for i in range(self.dimensions)]


        from hedge.optemplate import make_nabla, InverseMassOperator, \
                ElementwiseMaxOperator

        from pymbolic import var
        sqrt = var("sqrt")

        state = make_vector_field("q", self.dimensions+2)
        bc_state = make_vector_field("bc_q", self.dimensions+2)

        c = cse(sqrt(self.gamma*p(state)/self.rho(state)))

        speed = cse(sqrt(numpy.dot(u(state), u(state)))) + c

        from hedge.tools import make_lax_friedrichs_flux, join_fields
        from hedge.mesh import TAG_ALL

        flux_state = flux(state)

        return join_fields(
                (- numpy.dot(make_nabla(self.dimensions), flux_state)
                 + InverseMassOperator()*make_lax_friedrichs_flux(
                        wave_speed=cse(ElementwiseMaxOperator()*speed),
                        state=state, fluxes=flux_state,
                        bdry_tags_states_and_fluxes=[(TAG_ALL, bc_state,
                        bdry_flux(bc_state, state))],
                        strong=True
                        )),
                 speed)




class NavierStokesWithHeatOperator(GasDynamicsOperatorBase):
    """An nD Navier-Stokes operator with heat flux.

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

    q = -k * nabla * T
    k = c_p * mu / Pr

    Field order is [rho E rho_u_x rho_u_y ...].
    """

    def __init__(self, dimensions, discr, gamma,
            prandtl, spec_gas_const, 
            bc_q_in, bc_q_out, bc_q_noslip,
            inflow_tag="inflow",
            outflow_tag="outflow",
            noslip_tag="noslip"):

        GasDynamicsOperatorBase.__init__(self, dimensions, gamma, 
                bc_q_in, bc_q_out, bc_q_noslip,
                inflow_tag, outflow_tag, noslip_tag)
        self.prandtl = prandtl
        self.spec_gas_const = spec_gas_const
        self.discr = discr

    def op_template(self):
        from hedge.optemplate import make_vector_field, \
                make_common_subexpression as cse

        c_p = self.gamma / (self.gamma - 1) * self.spec_gas_const
        c_v = c_p - self.spec_gas_const

        AXES = ["x", "y", "z", "w"]

        def u(q):
            return cse(self.u(q), "u")

        def rho(q):
            return cse(self.rho(q), "rho")

        def rho_u(q):
            return cse(self.rho_u(q), "rho_u")

        def p(q):
            return cse((self.gamma-1)*(self.e(q) - 
                       0.5*numpy.dot(self.rho_u(q), u(q))), "p")

        def t(q):
            return cse(
                    (self.e(q)/self.rho(q) - 0.5 * numpy.dot(u(q),u(q))) / c_v,
                    "t")

        def mu(q):
            #Sutherland's law with t_ref = 1K and t_inf = 280K
            #mu_inf = 1.735e-5
            #mu_inf = 0.
            mu_inf = (0.1 * self.gamma ** 0.5) / 100
            t_s = 110.4
            #return cse(mu_inf * t(q) ** 1.5 * (1 + t_s) / (t(q) + t_s))
            return mu_inf

        def heat_flux(q):
            from hedge.tools import make_obj_array
            from hedge.optemplate import make_nabla

            nabla = make_nabla(self.dimensions)
            k = c_p * mu(q) / self.prandtl

            result = numpy.empty((self.dimensions,1), dtype=object)
            for i in range(len(result)):
                result[i] = -k * nabla[i] * t(q)
            return result

        def make_gradient(q):

            dimensions = self.dimensions
            d = len(q)

            from hedge.optemplate import make_nabla
            nabla = make_nabla(dimensions)

            nabla_func = numpy.zeros((d, dimensions), dtype=object)
            for i in range(dimensions):
                nabla_func[:,i] = nabla[i] * q

            from hedge.flux import make_normal, FluxVectorPlaceholder
            normal = make_normal(d)
            fluxes_ph = FluxVectorPlaceholder(d)

            flux = numpy.zeros((d, dimensions), dtype=object)
            for i in range(dimensions):
                flux[:,i] = 0.5 * normal[i] * (fluxes_ph.int - fluxes_ph.ext)

            from hedge.optemplate import get_flux_operator
            flux_op = numpy.zeros((dimensions), dtype=object)
            for i in range(dimensions):
                flux_op[i] = get_flux_operator(flux[:,i])

            from hedge.optemplate import BoundaryPair
            flux_part = numpy.zeros((d, dimensions), dtype=object)
            for i in range(dimensions):
                flux_part[:,i] = (flux_op[i]*q
                                + sum(
                                flux_op[i]*
                                BoundaryPair(q, bc, tag)
                                for tag, bc in all_tags_and_bcs)
                                )

            from hedge.optemplate import InverseMassOperator
            return (nabla_func - InverseMassOperator() * flux_part)

        def tau(q):
            from hedge.optemplate import make_nabla
            from pytools import delta

            dimensions = self.dimensions

            dq = make_gradient(q)

            du = numpy.zeros((dimensions, dimensions), dtype=object)
            for i in range(dimensions):
                for j in range(dimensions):
                    du[i,j] = cse(
                            (dq[i+2,j] - u(q)[i] * dq[0,j]) / self.rho(q),
                            "du%d_d%s" % (i, AXES[j]))

            tau = numpy.zeros((dimensions+1, dimensions), dtype=object)
            for i in range(dimensions):
                for j in range(dimensions):
                    tau[i,j] = cse(mu(q) * (du[i,j] + du[j,i] -
                               2/3 * delta(i,j) * (du[0,0] + du[1,1])),
                               "tau_%d%d" % (i, j))

            for j in range(dimensions):
                tau[dimensions,j] = cse(numpy.dot(u(q), tau[j]),
                                    "tau_%d%d" % (dimensions, j))
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
                        tau(q)[self.dimensions,i] #+ 
                        #heat_flux(q)[i]
                        ,

                        # flux rho_u
                        make_obj_array([
                            self.rho_u(q)[i]*self.u(q)[j] + delta(i,j) * p(q) -
                            tau(q)[i,j]
                            for j in range(self.dimensions)
                            ])
                        ), "%s_flux" % AXES[i])
                    for i in range(self.dimensions)]

        def bdry_flux(q_bdry, q_vol, tag):
            from pytools import delta
            from hedge.tools import make_obj_array, join_fields
            from hedge.optemplate import BoundarizeOperator
            return [ # one entry for each flux direction
                    cse(join_fields(
                        # flux rho
                        self.rho_u(q_bdry)[i],

                        # flux E
                        cse(self.e(q_bdry)+p(q_bdry))*u(q_bdry)[i] - 
                        BoundarizeOperator(tag)(
                            tau(q_vol)[self.dimensions,i] #+ 
                            #heat_flux(q_vol)[i]
                            ),

                        # flux rho_u
                        make_obj_array([
                            self.rho_u(q_bdry)[i]*self.u(q_bdry)[j] + 
                            delta(i,j) * p(q_bdry) - 
                            BoundarizeOperator(tag)(tau(q_vol)[i,j])
                            for j in range(self.dimensions)
                            ])
                        ), "%s_bflux" % AXES[i])
                    for i in range(self.dimensions)]

        from pymbolic import var
        sqrt = var("sqrt")

        state = make_vector_field("q", self.dimensions+2)

        c = cse(sqrt(self.gamma*p(state)/self.rho(state)), "c")

        speed = cse(sqrt(numpy.dot(u(state), u(state))), "norm_u") + c

        from hedge.tools import make_obj_array, join_fields
        from hedge.optemplate import BoundarizeOperator

        # boundary conditions -------------------------------------------------

        def outflow_state(state):
            from hedge.optemplate import make_normal
            normal = make_normal(self.outflow_tag, self.dimensions)

            state0 = make_vector_field("bc_q_out", self.dimensions+2)

            rho0 = rho(state0)
            drhom = BoundarizeOperator(self.outflow_tag)(rho(state)) - rho0
            dumvec = BoundarizeOperator(self.outflow_tag)(u(state)) - u(state0)
            dpm = BoundarizeOperator(self.outflow_tag)(p(state)) - p(state0)
            c = BoundarizeOperator(self.outflow_tag)(
                (self.gamma * p(state) / rho(state))**0.5)

            prims = join_fields(
                    drhom + numpy.dot(normal, dumvec)*rho0/(2*c) -
                        dpm/(2*c*c) + rho0,
                    c*rho0*numpy.dot(normal, dumvec)/2 + dpm/2 + p(state0),
                    dumvec - normal*numpy.dot(normal, dumvec)/2 +
                        dpm*normal/(2*c*rho0) + u(state0))

            cons = join_fields(
                   prims[0],
                   prims[1] / (self.gamma - 1) + prims[0] / 2 *
                       numpy.dot(prims[2: 2+self.dimensions], 
                       prims[2: 2+self.dimensions]),
                   prims[0] * prims[2: 2+self.dimensions])
            return cons

        def inflow_state(state):
            from hedge.optemplate import make_normal
            normal = make_normal(self.inflow_tag, self.dimensions)

            state0 = make_vector_field("bc_q_in", self.dimensions+2)

            rho0 = rho(state0)
            drhom = BoundarizeOperator(self.inflow_tag)(rho(state)) - rho0
            dumvec = BoundarizeOperator(self.inflow_tag)(u(state)) - u(state0)
            dpm = BoundarizeOperator(self.inflow_tag)(p(state)) - p(state0)
            c = BoundarizeOperator(self.inflow_tag)(
                (self.gamma * p(state) / rho(state))**0.5)

            prims = join_fields(
                    numpy.dot(normal, dumvec)*rho0/(2*c) + dpm/(2*c*c) + rho0,
                    c*rho0*numpy.dot(normal, dumvec)/2 + dpm/2 + p(state0),
                    normal*numpy.dot(normal, dumvec)/2 + dpm*normal/(2*c*rho0)
                    + u(state0))

            cons = join_fields(
                   prims[0],
                   prims[1] / (self.gamma - 1) + prims[0] / 2 *
                       numpy.dot(prims[2: 2+self.dimensions], 
                       prims[2: 2+self.dimensions]),
                   prims[0] * prims[2: 2+self.dimensions])
            return cons

        def noslip_state(state):
            from hedge.optemplate import make_normal
            normal = make_normal(self.noslip_tag, self.dimensions)

            state0 = make_vector_field("bc_q_noslip", self.dimensions+2)

            rho0 = rho(state0)
            drhom = BoundarizeOperator(self.noslip_tag)(rho(state)) - rho0
            dumvec = BoundarizeOperator(self.noslip_tag)(u(state)) - u(state0)
            dpm = BoundarizeOperator(self.noslip_tag)(p(state)) - p(state0)
            c = BoundarizeOperator(self.noslip_tag)(
                (self.gamma * p(state) / rho(state))**0.5)

            prims = join_fields(
                    numpy.dot(normal, dumvec)*rho0/(2*c) + dpm/(2*c*c) + rho0,
                    c*rho0*numpy.dot(normal, dumvec)/2 + dpm/2 + p(state0),
                    [0]*self.dimensions)
                    #normal*numpy.dot(normal, dumvec)/2 + dpm*normal/(2*c*rho0)
                    #+u(state0))

            cons = join_fields(
                   prims[0],
                   prims[1] / (self.gamma - 1) + prims[0] / 2 *
                       numpy.dot(prims[2: 2+self.dimensions], 
                       prims[2: 2+self.dimensions]),
                   prims[0] * prims[2: 2+self.dimensions])
            return cons

        all_tags_and_bcs = [
                (self.outflow_tag, outflow_state(state)),
                (self.inflow_tag, inflow_state(state)),
                (self.noslip_tag, noslip_state(state))
                    ]

        flux_state = flux(state)

        from hedge.tools import make_lax_friedrichs_flux, join_fields
        from hedge.optemplate import make_nabla, InverseMassOperator, \
                ElementwiseMaxOperator

        # operator assembly ---------------------------------------------------
        return join_fields(
                (- numpy.dot(make_nabla(self.dimensions), flux_state)
                 + InverseMassOperator()*make_lax_friedrichs_flux(
                        wave_speed=cse(ElementwiseMaxOperator()*speed, "emax_c"),
                        state=state, fluxes=flux_state,
                        bdry_tags_states_and_fluxes=[
                            (tag, bc, bdry_flux(bc, state, tag))
                            for tag, bc in all_tags_and_bcs
                            ],
                        strong=True
                        )),
                 speed)
