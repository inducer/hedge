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





class GasDynamicsOperatorBase(TimeDependentOperator):
    """Basis for Euler and Navier-Stokes Operator.
    Field order is [rho E rho_u_x rho_u_y ...].
    """
    def __init__(self, dimensions, gamma,
            bc_inflow, bc_outflow, bc_noslip,
            inflow_tag="inflow",
            outflow_tag="outflow",
            noslip_tag="noslip"):

        self.dimensions = dimensions
        self.gamma = gamma

        self.bc_inflow = bc_inflow
        self.bc_outflow = bc_outflow
        self.bc_noslip = bc_noslip

        self.inflow_tag = inflow_tag
        self.outflow_tag = outflow_tag
        self.noslip_tag = noslip_tag

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
        bound_op = discr.compile(self.op_template())

        from hedge.mesh import check_bc_coverage
        check_bc_coverage(discr.mesh, [
            self.inflow_tag,
            self.outflow_tag,
            self.noslip_tag,
            ])

        def wrap(t, q):
            opt_result = bound_op(q=q,
                    bc_q_in=self.bc_inflow.boundary_interpolant(
                        t, discr, self.inflow_tag),
                    bc_q_out=self.bc_inflow.boundary_interpolant(
                        t, discr, self.outflow_tag),
                    bc_q_noslip=self.bc_inflow.boundary_interpolant(
                        t, discr, self.noslip_tag),
                    )
            max_speed = opt_result[-1]
            ode_rhs = opt_result[:-1]
	    return ode_rhs, discr.nodewise_max(max_speed)

        return wrap
