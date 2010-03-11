"""Runge-Kutta ODE timestepper."""

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
from hedge.timestep.base import TimeStepper




# {{{ Carpenter/Kennedy low-storage fourth-order Runge-Kutta ------------------
class LSRK4TimeStepper(TimeStepper):
    """A low storage fourth-order Runge-Kutta method

    See JSH, TW: Nodal Discontinuous Galerkin Methods p.64
    or 
    Carpenter, M.H., and Kennedy, C.A., Fourth-order-2N-storage 
    Runge-Kutta schemes, NASA Langley Tech Report TM 109112, 1994
    """

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
            #1,
            ]

    dt_fudge_factor = 1

    def __init__(self, dtype=numpy.float64, rcon=None):
        from pytools.log import IntervalTimer, EventCounter
        timer_factory = IntervalTimer
        if rcon is not None:
            timer_factory = rcon.make_timer

        self.timer = timer_factory(
                "t_rk4", "Time spent doing algebra in RK4")
        self.flop_counter = EventCounter(
                "n_flops_rk4", "Floating point operations performed in RK4")

        from pytools import match_precision
        self.dtype = numpy.dtype(dtype)
        self.rcon = rcon
        self.scalar_dtype = match_precision(
                numpy.dtype(numpy.float64), self.dtype)
        self.coeffs = numpy.array([self._RK4A, self._RK4B, self._RK4C], 
                dtype=self.scalar_dtype).T

    def get_stability_relevant_init_args(self):
        return ()

    def add_instrumentation(self, logmgr):
        logmgr.add_quantity(self.timer)
        logmgr.add_quantity(self.flop_counter)

    def __call__(self, y, t, dt, rhs):
        try:
            self.residual
        except AttributeError:
            self.residual = 0*rhs(t, y)
            from hedge.tools import count_dofs
            self.dof_count = count_dofs(self.residual)

            from hedge.vector_primitives import make_linear_combiner
            self.linear_combiner = make_linear_combiner(
                    self.dtype, self.scalar_dtype, self.residual,
                    arg_count=2, rcon=self.rcon)

        lc = self.linear_combiner

        for a, b, c in self.coeffs:
            this_rhs = rhs(t + c*dt, y)

            sub_timer = self.timer.start_sub_timer()
            self.residual = lc((a, self.residual), (dt, this_rhs))
            del this_rhs
            y = lc((1, y), (b, self.residual))
            sub_timer.stop().submit()

        # 5 is the number of flops above, *NOT* the number of stages,
        # which is already captured in len(self.coeffs)
        self.flop_counter.add(len(self.coeffs)*self.dof_count*5)

        return y




class RK4TimeStepper(LSRK4TimeStepper):
    def __init__(self, *args, **kwargs):
        from warnings import warn
        warn("RK4TimeStepper is a deprecated name of LSRK4TimeStepper",
                DeprecationWarning, stacklevel=2)

        LSRK4TimeStepper.__init__(self, *args, **kwargs)
# }}}




# {{{ Embedded Runge-Kutta schemes base class ---------------------------------
class EmbeddedRungeKuttaTimeStepperBase(TimeStepper):
    def __init__(self, use_high_order=True, dtype=numpy.float64, rcon=None,
            atol=0, rtol=0):
        from pytools.log import IntervalTimer, EventCounter
        timer_factory = IntervalTimer
        if rcon is not None:
            timer_factory = rcon.make_timer

        self.timer = timer_factory(
                "t_rk", "Time spent doing algebra in Runge-Kutta")
        self.flop_counter = EventCounter(
                "n_flops_rk", "Floating point operations performed in Runge-Kutta")

        self.use_high_order = use_high_order

        self.dtype = numpy.dtype(dtype)
        self.rcon = rcon

        self.adaptive = bool(atol or rtol)
        self.atol = atol
        self.rtol = rtol

        from pytools import match_precision
        self.scalar_dtype = match_precision(
                numpy.dtype(numpy.float64), self.dtype)

    def get_stability_relevant_init_args(self):
        return (self.use_high_order,)

    def add_instrumentation(self, logmgr):
        logmgr.add_quantity(self.timer)
        logmgr.add_quantity(self.flop_counter)

    def __call__(self, y, t, dt, rhs):
        from hedge.tools import count_dofs

        # {{{ preparation, linear combiners
        try:
            self.last_rhs
        except AttributeError:
            self.last_rhs = rhs(t, y)
            self.dof_count = count_dofs(self.last_rhs)

            from hedge.vector_primitives import (
                    make_linear_combiner,
                    make_inner_product)

            self.linear_combiners = dict(
                    (arg_count, make_linear_combiner(
                        self.dtype, self.scalar_dtype, self.last_rhs,
                        arg_count=arg_count, rcon=self.rcon))
                    for arg_count in (
                        set(1+len(coeffs) for t_frac, coeffs 
                            in self.butcher_tableau)
                        | set([2, 1+len(self.low_order_coeffs),
                            1+len(self.high_order_coeffs)]))
                    if arg_count)

            self.ip = make_inner_product(self.last_rhs, rcon=self.rcon)

        lcs = self.linear_combiners
        ip = self.ip

        def norm(a):
            return numpy.sqrt(ip(a, a))

        # }}}

        flop_count = [0]

        while True:
            rhss = []

            # {{{ stage loop

            for i, (c, coeffs) in enumerate(self.butcher_tableau):
                if len(coeffs) == 0:
                    assert c == 0
                    this_rhs = self.last_rhs
                else:
                    sub_timer = self.timer.start_sub_timer()
                    args = [(1, y)] + [
                            (dt*coeff, rhss[j]) for j, coeff in enumerate(coeffs)
                            if coeff]
                    flop_count[0] += len(args)*2
                    sub_y = lcs[len(args)](*args)
                    sub_timer.stop().submit()

                    this_rhs = rhs(t + c*dt, sub_y)

                rhss.append(this_rhs)


            # }}}

            def finish_solution(coeffs):
                args = [(1, y)] + [
                        (dt*coeff, rhss[i]) for i, coeff in enumerate(coeffs)
                        if coeff]
                flop_count[0] += len(args)*2
                return lcs[len(args)](*args)

            if not self.adaptive:
                if self.use_high_order:
                    y = finish_solution(self.high_order_coeffs)
                else:
                    y = finish_solution(self.low_order_coeffs)

                self.last_rhs = this_rhs
                self.flop_counter.add(self.dof_count*flop_count)
                return y
            else:
                # {{{ step size adaptation
                high_order_end_y = finish_solution(self.high_order_coeffs)
                low_order_end_y = finish_solution(self.low_order_coeffs)

                normalization = self.atol + self.rtol*max(
                            norm(low_order_end_y), norm(y))

                error = lcs[2](*[
                    (1/normalization, high_order_end_y),
                    (-1/normalization, low_order_end_y)
                    ])
                flop_count[0] += 1

                rel_err = numpy.sqrt(self.ip(error, error)/count_dofs(error))
                if rel_err == 0:
                   rel_err = 1e-14

                if rel_err > 1 or numpy.isnan(rel_err):
                    # reject step

                    last_dt = dt
                    if not numpy.isnan(rel_err):
                        dt = max(
                                0.9 * dt * rel_err**(-1/self.low_order),
                                0.1 * dt)
                    else:
                        dt = 0.1*dt

                    if t + dt == t:
                        raise RuntimeError("stepsize underflow")
                    # ... and go back to top of loop

                else:
                    # accept step

                    next_dt = min(
                            0.9 * dt * rel_err**(-1/self.high_order),
                            5*dt)

                    # finish up
                    self.last_rhs = this_rhs
                    self.flop_counter.add(self.dof_count*flop_count[0])

                    return high_order_end_y, t+dt, dt, next_dt

                # }}}

# }}}




# {{{ Bogacki-Shampine second/third-order Runge-Kutta -------------------------
class ODE23TimeStepper(EmbeddedRungeKuttaTimeStepperBase):
    """Bogacki-Shampine second/third-order Runge-Kutta.

    (same as Matlab's ode23)

    Bogacki, Przemyslaw; Shampine, Lawrence F. (1989), "A 3(2) pair of
    Runge-Kutta formulas", Applied Mathematics Letters 2 (4): 321-325,
    doi:10.1016/0893-9659(89)90079-7
    """

    dt_fudge_factor = 1

    butcher_tableau = [
            (0, []),
            (1/2, [1/2]),
            (3/4, [0, 3/4]),
            (1, [2/9, 1/3, 4/9])
            ]

    low_order = 2
    low_order_coeffs = [7/24, 1/4, 1/3, 1/8]
    high_order = 3
    high_order_coeffs = [2/9, 1/3, 4/9, 0]

# }}}




# {{{ Dormand-Prince fourth/fifth-order Runge-Kutta ---------------------------
class ODE45TimeStepper(EmbeddedRungeKuttaTimeStepperBase):
    """Dormand-Prince fourth/fifth-order Runge-Kutta.

    (same as Matlab's ode45)

    Dormand, J. R.; Prince, P. J. (1980), "A family of embedded Runge-Kutta
    formulae", Journal of Computational and Applied Mathematics 6 (1): 19-26,
    doi:10.1016/0771-050X(80)90013-3.
    """

    dt_fudge_factor = 1

    butcher_tableau = [
            (0, []),
            (1/5, [1/5]),
            (3/10, [3/40, 9/40]),
            (4/5, [44/45, -56/15, 32/9]),
            (8/9, [19372/6561, -25360/2187, 64448/6561, -212/729]),
            (1, [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656]),
            (1, [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84])
            ]

    low_order = 4
    low_order_coeffs = [5179/57600, 0, 7571/16695, 393/640, -92097/339200, 
            187/2100, 1/40]
    high_order = 5
    high_order_coeffs = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]

# }}}




# vim: foldmethod=marker
