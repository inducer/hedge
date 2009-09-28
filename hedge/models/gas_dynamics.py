# -*- coding: utf8 -*-
"""Operator for compressible Navier-Stokes and Euler equations."""

from __future__ import division

__copyright__ = "Copyright (C) 2007 Hendrik Riedmann, Andreas Kloeckner"

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
from pytools import Record




class GasDynamicsOperator(TimeDependentOperator):
    """An nD Navier-Stokes and Euler operator.

    see JSH, TW: Nodal Discontinuous Galerkin Methods p.320 and p.206

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
    
    For Euler: mu = 0
    
    For the heat flux:

    q = -k * nabla * T
    k = c_p * mu / Pr

    Field order is [rho E rho_u_x rho_u_y ...].
    """
    def __init__(self, dimensions,
            gamma, bc_inflow, bc_outflow, bc_noslip,
            prandtl=0.0, spec_gas_const=0.0, mu=0.0,
            inflow_tag="inflow",
            outflow_tag="outflow",
            noslip_tag="noslip",
            source=None,
            euler=False):
        """
        :param source: should implement 
        :class:`hedge.data.IFieldDependentGivenFunction`
        or be None.
        """

        self.dimensions = dimensions
        
        self.gamma = gamma
        self.prandtl = prandtl
        self.spec_gas_const = spec_gas_const
        self.mu = mu

        self.bc_inflow = bc_inflow
        self.bc_outflow = bc_outflow
        self.bc_noslip = bc_noslip

        self.inflow_tag = inflow_tag
        self.outflow_tag = outflow_tag
        self.noslip_tag = noslip_tag

        self.source = source

        self.euler = euler
        
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
            mu = self.mu
            if self.euler == True:
                assert mu == 0.
            elif mu == "sutherland":
                # Sutherland's law: !!!not tested!!!
                t_s = 110.4
                mu_inf = 1.735e-5
                mu = cse(mu_inf * t(q) ** 1.5 * (1 + t_s) / (t(q) + t_s))
            return mu

        def heat_flux(q):
            # !!!not tested!!!
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
                               2/3 * delta(i,j) * numpy.trace(du)),
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

        class BCInfo(Record): pass

        def make_bc_info(bc_name, tag, state, set_velocity_to_zero=False):
            if set_velocity_to_zero:
                state0 = join_fields(make_vector_field(bc_name, 2), [0]*self.dimensions)
            else:
                state0 = make_vector_field(bc_name, self.dimensions+2)

            rho0 = rho(state0)
            p0 = p(state0)
            u0 = u(state0)
            c0 = (self.gamma * p0 / rho0)**0.5

            bdrize_op = BoundarizeOperator(tag)
            return BCInfo(
                rho0=rho0, p0=p0, u0=u0, c0=c0,

                # notation: suffix "m" for "minus", i.e. "interior"
                drhom=cse(bdrize_op(rho(state)) - rho0, "drhom"),
                dumvec=cse(bdrize_op(u(state)) - u0, "dumvec"),
                dpm=cse(bdrize_op(p(state)) - p0, "dpm"))

        def primitive_to_conservative(prims):
            rho = prims[0]
            p = prims[1]
            u = prims[2:]
            return join_fields(
                   rho,
                   cse(p / (self.gamma - 1) + rho / 2 * numpy.dot(u, u), "e"),
                   rho * u)

        def outflow_state(state):
            from hedge.optemplate import make_normal
            normal = make_normal(self.outflow_tag, self.dimensions)
            bc = make_bc_info("bc_q_out", self.outflow_tag, state)

            # see hedge/doc/maxima/euler.mac
            return primitive_to_conservative(join_fields(
                # bc rho
                cse(bc.rho0
                + bc.drhom + numpy.dot(normal, bc.dumvec)*bc.rho0/(2*bc.c0)
                - bc.dpm/(2*bc.c0*bc.c0), "bc_rho_outflow"),

                # bc p
                cse(bc.p0
                + bc.c0*bc.rho0*numpy.dot(normal, bc.dumvec)/2 + bc.dpm/2, "bc_p_outflow"),

                # bc u
                cse(bc.u0
                + bc.dumvec - normal*numpy.dot(normal, bc.dumvec)/2
                + bc.dpm*normal/(2*bc.c0*bc.rho0), "bc_u_outflow")))

        def inflow_state_inner(normal, bc, name):
            # see hedge/doc/maxima/euler.mac
            return primitive_to_conservative(join_fields(
                # bc rho
                cse(bc.rho0
                + numpy.dot(normal, bc.dumvec)*bc.rho0/(2*bc.c0) + bc.dpm/(2*bc.c0*bc.c0), "bc_rho_"+name),

                # bc p
                cse(bc.p0
                + bc.c0*bc.rho0*numpy.dot(normal, bc.dumvec)/2 + bc.dpm/2, "bc_p_"+name),

                # bc u
                cse(bc.u0
                + normal*numpy.dot(normal, bc.dumvec)/2 + bc.dpm*normal/(2*bc.c0*bc.rho0), "bc_u_"+name)))

        def inflow_state(state):
            from hedge.optemplate import make_normal
            normal = make_normal(self.inflow_tag, self.dimensions)
            bc = make_bc_info("bc_q_in", self.inflow_tag, state)
            return inflow_state_inner(normal, bc, "inflow")

        def noslip_state(state):
            from hedge.optemplate import make_normal
            normal = make_normal(self.noslip_tag, self.dimensions)
            bc = make_bc_info("bc_q_noslip", self.noslip_tag, state,
                    set_velocity_to_zero=True)
            return inflow_state_inner(normal, bc, "noslip")

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
        result = join_fields(
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

        if self.source is not None:
            #need extra slot for speed, will set to zero in source class
            source_ph = make_vector_field("source_vect", self.dimensions+2+1)
            result = join_fields(result + source_ph)
        
        return result

    def bind(self, discr):
        bound_op = discr.compile(self.op_template())

        from hedge.mesh import check_bc_coverage
        check_bc_coverage(discr.mesh, [
            self.inflow_tag,
            self.outflow_tag,
            self.noslip_tag,
            ])

        def rhs(t, q):
            extra_kwargs = {}
            if self.source is not None:
                extra_kwargs["source_vect"] = self.source.volume_interpolant(
                        t, q, discr)

            opt_result = bound_op(q=q,
                    bc_q_in=self.bc_inflow.boundary_interpolant(
                        t, discr, self.inflow_tag),
                    bc_q_out=self.bc_inflow.boundary_interpolant(
                        t, discr, self.outflow_tag),
                    bc_q_noslip=self.bc_inflow.boundary_interpolant(
                        t, discr, self.noslip_tag),
                    **extra_kwargs
                    )

            max_speed = opt_result[-1]
            ode_rhs = opt_result[:-1]
	    return ode_rhs, discr.nodewise_max(max_speed)

        return rhs
       


class SlopeLimiter1NEuler:
    def __init__(self, discr,gamma,dimensions,op):
        """Construct a limiter from Jan's book page 225
        """
        self.discr = discr
        self.gamma=gamma
        self.dimensions=dimensions
        self.op=op

        #AVE*colVect=average of colVect
        self.AVE_map = {}

        for eg in discr.element_groups:
            ldis = eg.local_discretization
            node_count = ldis.node_count()


            # build AVE matrix
            massMatrix = ldis.mass_matrix()
            #compute area of the element
            self.standard_el_vol= numpy.sum(numpy.dot(massMatrix,numpy.ones(massMatrix.shape[0])))
            
            from numpy import size, zeros, sum
            AVEt = sum(massMatrix,0)
            AVEt = AVEt/self.standard_el_vol
            AVE = zeros((size(AVEt),size(AVEt)))
            for ii in range(0,size(AVEt)):
                AVE[ii]=AVEt
            self.AVE_map[eg] = AVE

    def get_average(self,vec):

        from hedge.tools import log_shape
        from pytools import indices_in_shape
        from hedge._internal import perform_elwise_operator


        ls = log_shape(vec)
        result = self.discr.volume_zeros(ls)

        from pytools import indices_in_shape
        for i in indices_in_shape(ls):
            from hedge._internal import perform_elwise_operator
            for eg in self.discr.element_groups:
                perform_elwise_operator(eg.ranges, eg.ranges, 
                        self.AVE_map[eg], vec[i], result[i])
		
                return result

    def __call__(self, vec):

        #join fields 
        from hedge.tools import join_fields

        #get conserved fields
        rho=self.op.rho(vec)
        e=self.op.e(vec)
        rho_velocity=self.op.rho_u(vec)

        #get primative fields 
        #to do

        #reset field values to cell average
        rhoLim=self.get_average(rho)
        eLim=self.get_average(e)
        temp=join_fields([self.get_average(rho_vel)
                for rho_vel in rho_velocity])

        #should do for primative fields too

 
        return join_fields(rhoLim, eLim, temp)


