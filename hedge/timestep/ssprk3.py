"""Strongly Stability Preserving third-order RK ODE timestepper."""

from __future__ import division

__copyright__ = "Copyright (C) 2009 Scott Field"

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


class SSPRK3TimeStepper(TimeStepper):
    """A third-order strong stability preserving Runge-Kutta method

    See JSH, TW: Nodal Discontinuous Galerkin Methods p.158

    :param limit_stages: bool indicating whether to limit after each stage.
    """
    dt_fudge_factor = 1

    def __init__(self, allow_jit=False, limit_stages=False, limiter=None):
        from pytools.log import IntervalTimer, EventCounter
        self.timer = IntervalTimer(
                "t_ssprk3", "Time spent doing algebra in SSPRK3")
        self.flop_counter = EventCounter(
                "n_flops_ssprk3", "Floating point operations performed in SSPRK3")

        self.allow_jit = allow_jit
        self.limit_stages = limit_stages
        self.limiter = limiter

    def get_stability_relevant_init_args(self):
        return ()

    def __getinitargs__(self):
        return (self.allow_jit, self.limit_stages, self.limiter)

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
            raise NotImplementedError

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

        self.flop_counter.add(3*self.dof_count*5)

        return y
