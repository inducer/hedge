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

    def __init__(self, dimensions, gamma,
            bc,
            inflow_tag="inflow",
            outflow_tag="outflow",
            no_slip_tag="no_slip"):

        GasDynamicsOperatorBase.__init__(self, dimensions, gamma, bc,
                inflow_tag, outflow_tag, no_slip_tag)


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
        #bc_state = make_vector_field("bc_q", self.dimensions+2)

        c = cse(sqrt(self.gamma*p(state)/self.rho(state)))

        speed = sqrt(numpy.dot(u(state), u(state))) + c

        from hedge.tools import make_lax_friedrichs_flux, join_fields
        from hedge.mesh import TAG_ALL

        flux_state = flux(state)

        from hedge.optemplate import BoundarizeOperator
        from hedge.tools import make_obj_array

        all_tags_and_bcs = [
                (self.outflow_tag, BoundarizeOperator(self.outflow_tag)(state)),
                (self.inflow_tag, make_vector_field("bc_q_in", self.dimensions+2)),
                (self.no_slip_tag, BoundarizeOperator(self.no_slip_tag)(join_fields(
                    # rho
                    state[0],
                    # energy
                    state[1],
                    # momenta
                    make_obj_array(numpy.zeros(self.dimensions))
                    )))
                    ]


        return join_fields(
                (- numpy.dot(make_nabla(self.dimensions), flux_state)
                    + InverseMassOperator()*make_lax_friedrichs_flux(
                        wave_speed=
			ElementwiseMaxOperator()*
			speed,
                        state=state, fluxes=flux_state,
                        bdry_tags_states_and_fluxes=[
                            (tag, bc_state, flux(bc_state))
                            for tag, bc_state in all_tags_and_bcs
                            ],
                        strong=True
                        )),
                    speed)




class SourcesEulerOperator(GasDynamicsOperatorBase):
    '''extension of EulerOperator (above) to include sources
       which can depend on the hydrodynamic fields as 
       well as other fields, referre to externalField'''

    def __init__(self, dimensions, gamma, bc, source):
        GasDynamicsOperatorBase.__init__(self, dimensions, gamma, bc)
        #self.dimensions = dimensions
        #self.gamma = gamma
        #self.bc = bc
        self.source=source

    def op_template(self):
        from hedge.optemplate import make_vector_field, \
                make_common_subexpression as cse

        def MakeSource(q,externalField):
            return cse(self.source.GetSource(q,externalField))

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
        externalFieldState = make_vector_field("externalField", self.dimensions)
        bc_state = make_vector_field("bc_q", self.dimensions+2)

        c = cse(sqrt(self.gamma*p(state)/self.rho(state)))

        speed = sqrt(numpy.dot(u(state), u(state))) + c

        from hedge.tools import make_lax_friedrichs_flux, join_fields
        from hedge.mesh import TAG_ALL, TAG_NONE
      
        return join_fields(
                (- numpy.dot(make_nabla(self.dimensions), flux(state))
                    + MakeSource(state,externalFieldState)
                    + InverseMassOperator()*make_lax_friedrichs_flux(
                        wave_speed=cse(ElementwiseMaxOperator()*speed),
                        state=state, fluxes=flux(state),
                        bdry_tags_states_and_fluxes=[
                            (TAG_ALL, bc_state, flux(bc_state))
                            ],
                        strong=True
                        )),
                    speed)

    def bind(self, discr):
        from hedge.mesh import TAG_ALL
        bound_op = discr.compile(self.op_template())
        def wrap(t, q, gradu):
            opt_result = bound_op(
                    q=q, 
                    bc_q=self.bc.boundary_interpolant(t, discr, TAG_ALL),
                    externalField=gradu)
            max_speed = opt_result[-1]
            ode_rhs = opt_result[:-1]
	    return ode_rhs, discr.nodewise_max(max_speed)
        return wrap

