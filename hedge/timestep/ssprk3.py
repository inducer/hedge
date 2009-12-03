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
    """
    dt_fudge_factor = 1

    def __init__(self, dtype=numpy.float64, rcon=None, limiter=None):
        self.dtype = numpy.dtype(dtype)
        self.rcon = rcon
        from pytools import match_precision
        self.scalar_type = match_precision(
                numpy.dtype(numpy.float64), self.dtype).type

        if limiter is None:
            self.limiter = lambda x: x
        else:
            self.limiter = limiter

        # diagnostics init
        from pytools.log import IntervalTimer, EventCounter

        timer_factory = IntervalTimer
        if rcon is not None:
            timer_factory = rcon.make_timer

        self.timer = IntervalTimer(
                "t_ssprk3", "Time spent doing algebra in SSPRK3")
        self.flop_counter = EventCounter(
                "n_flops_ssprk3", "Floating point operations performed in SSPRK3")

    def get_stability_relevant_init_args(self):
        return ()

    def add_instrumentation(self, logmgr):
        logmgr.add_quantity(self.timer)
        logmgr.add_quantity(self.flop_counter)

    def __call__(self, y, t, dt, rhs):
        try:
            lc2 = self.linear_combiner_2
            lc3 = self.linear_combiner_3
        except AttributeError:
            from hedge.tools import count_dofs
            self.dof_count = count_dofs(y)

            from hedge.tools.linear_combination import make_linear_combiner
            lc2 = self.linear_combiner_2 = make_linear_combiner(
                    self.dtype, self.scalar_type, y, arg_count=2, rcon=self.rcon)
            lc3 = self.linear_combiner_3 = make_linear_combiner(
                    self.dtype, self.scalar_type, y, arg_count=3, rcon=self.rcon)

            from hedge.tools import count_dofs
            self.dof_count = count_dofs(0*rhs(t, y))

        sub_timer = self.timer.start_sub_timer()
        v1 = self.limiter(lc2((1, y), (dt, rhs(t, y))))
        v2 = self.limiter(lc3((3/4, y), (1/4, v1), (dt/4, rhs(t+dt,v1))))
        y = self.limiter(lc3((1/3, y), (2/3, v2), (2*dt/3, rhs(t+dt/2,v2))))
        sub_timer.stop().submit()

        self.flop_counter.add(3*self.dof_count*5)

        return y
