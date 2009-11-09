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




_RK4A = [0.0,
        -567301805773 /1357537059087,
        -2404267990393/2016746695238,
        -3550918686646/2091501179385,
        -1275806237668/ 842570457699,
        ]

_RK4B = [1432997174477/ 9575080441755,
        5161836677717 /13612068292357,
        1720146321549 / 2090206949498,
        3134564353537 / 4481467310338,
        2277821191437 /14882151754819,
        ]

_RK4C = [0.0,
        1432997174477/9575080441755,
        2526269341429/6820363962896,
        2006345519317/3224310063776,
        2802321613138/2924317926251,
        1,
        ]




class RK4TimeStepper(TimeStepper):
    '''A low storage fourth-order Runge-Kutta method

    See JSH, TW: Nodal Discontinuous Galerkin Methods p.64
    or 
    Carpenter, M.H., and Kennedy, C.A., Fourth-order-2N-storage 
    Runge-Kutta schemes, NASA Langley Tech Report TM 109112, 1994

    '''


    dt_fudge_factor = 1

    def __init__(self, allow_jit=True):
        from pytools.log import IntervalTimer, EventCounter
        self.timer = IntervalTimer(
                "t_rk4", "Time spent doing algebra in RK4")
        self.flop_counter = EventCounter(
                "n_flops_rk4", "Floating point operations performed in RK4")
        self.coeffs = zip(_RK4A, _RK4B, _RK4C)

        self.allow_jit = allow_jit

    def get_stability_relevant_init_args(self):
        return ()

    def __getinitargs__(self):
        return (self.allow_jit,)

    def add_instrumentation(self, logmgr):
        logmgr.add_quantity(self.timer)
        logmgr.add_quantity(self.flop_counter)

    def __call__(self, y, t, dt, rhs):
        try:
            self.residual
        except AttributeError:
            self.residual = 0*rhs(t, y)
            from hedge.tools import count_dofs, has_data_in_numpy_arrays
            self.dof_count = count_dofs(self.residual)

            self.use_jit = self.allow_jit and has_data_in_numpy_arrays(
                    y, allow_objarray_levels=1)

        if self.use_jit:
            from hedge.tools import numpy_linear_comb

            for a, b, c in self.coeffs:
                this_rhs = rhs(t + c*dt, y)

                sub_timer = self.timer.start_sub_timer()
                self.residual = numpy_linear_comb([(a, self.residual), (dt, this_rhs)])
                del this_rhs
                y = numpy_linear_comb([(1, y), (b, self.residual)])
                sub_timer.stop().submit()
        else:
            for a, b, c in self.coeffs:
                this_rhs = rhs(t + c*dt, y)

                sub_timer = self.timer.start_sub_timer()
                self.residual = a*self.residual + dt*this_rhs
                del this_rhs
                y = y + b * self.residual
                sub_timer.stop().submit()

        # 5 is the number of flops above, *NOT* the number of stages,
        # which is already captured in len(self.coeffs)
        self.flop_counter.add(len(self.coeffs)*self.dof_count*5)

        return y
