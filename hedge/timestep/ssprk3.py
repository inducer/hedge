"""LSERK ODE timestepper."""

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
from pytools import memoize
from hedge.timestep.base import TimeStepper


_SSPRK3A = [(1.0,0.0,0.0),
        (.75,.25,0.0),
        (1.0/3.0,0.0,2.0/3.0)]

_SSPRK3B = [(1.0,0.0,0.0),
        (0.0,1.0/4.0,0.0),
        (0.0,0.0,2.0/3.0)]

_SSPRK3C = [0.0,1.0,0.5]





class SSPRK3TimeStepper(TimeStepper):
    '''A third-order strong stability preserving Runge-Kutta method

    See JSH, TW: Nodal Discontinuous Galerkin Methods p.158

    '''
   
    #from book would expect factor to be 1, this factor might not be optimal. See Ref. above
    dt_fudge_factor = 1/1.5

    def __init__(self, allow_jit=False, limit_stages=False, limiter=None):
        from pytools.log import IntervalTimer, EventCounter
        self.timer = IntervalTimer(
                "t_ssprk3", "Time spent doing algebra in SSPRK3")
        self.flop_counter = EventCounter(
                "n_flops_ssprk3", "Floating point operations performed in SSPRK3")

        self.coeffs = zip(_SSPRK3A, _SSPRK3B, _SSPRK3C)

        self.allow_jit = allow_jit
        self.limit_stages = limit_stages
        self.limiter = limiter

    def add_instrumentation(self, logmgr):
        logmgr.add_quantity(self.timer)
        logmgr.add_quantity(self.flop_counter)

    def __call__(self, y, t, dt, rhs):
        try:
            self.residual
        except AttributeError:
            from hedge.tools import join_fields
            self.residual = 0*rhs(t, y)
            from hedge.tools import count_dofs, has_data_in_numpy_arrays
            self.dof_count = count_dofs(self.residual)

            self.use_jit = self.allow_jit and has_data_in_numpy_arrays(
                    y, allow_objarray_levels=1)

        if self.use_jit:
            print 'not coded yet'
            #from hedge.tools import numpy_linear_comb

            #this_rhs = dt*rhs(t,y)
        else:
            sub_timer = self.timer.start_sub_timer()
            if self.limit_stages:
                v1 = y + dt*rhs(t, y)
                v1=self.limiter(v1)
                v2 = (3*y + v1 + dt*rhs(t+dt,v1))/4
                v2=self.limiter(v2)
                y = (y + 2*v2 + 2*dt*rhs(t+dt/2,v2))/3
                y=self.limiter(y)
            else:
                v1 = y + dt*rhs(t, y)
                v2 = (3*y + v1 + dt*rhs(t+dt,v1))/4
                y = (y + 2*v2 + 2*dt*rhs(t+dt/2,v2))/3


            sub_timer.stop().submit()
        

        # 5 is the number of flops above, *NOT* the number of stages,
        # which is already captured in len(self.coeffs)
        self.flop_counter.add(3*self.dof_count*5)

        return y
