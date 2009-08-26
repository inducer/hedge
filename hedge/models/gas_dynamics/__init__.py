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

    def _scotts_bind(self, discr):
        from hedge.mesh import TAG_ALL, TAG_NONE
        from hedge.tools import join_fields
        bound_op = discr.compile(self.op_template())
        def wrap(t, q):
            temp1=discr.get_boundary('outflow').vol_indices
            temp= join_fields(q[0][temp1], q[1][temp1], q[2][temp1])
            opt_result = bound_op(
                    q=q, 
                    #bc_q=self.bc.boundary_interpolant(t, discr, TAG_ALL)
                    #bc_q=temp
                    bc_q=self.bc.boundary_interpolant(t, discr, 'inflow'),
                    bc_q_out=temp
                    )
            max_speed = opt_result[-1]
            ode_rhs = opt_result[:-1]
	    return ode_rhs, discr.nodewise_max(max_speed)
        return wrap





