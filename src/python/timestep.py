"""ODE solvers: timestepping support, such as Runge-Kutta, Adams-Bashforth, etc."""

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



_ABCoefficients = [
        # from R. Verfuerth,
        # "Skript Numerische Behandlung von Differentialgleichungen"
        None,
        [1],
        [3/2, -1/2],
        [23/12, -16/12, 5/12],
        [55/24, -59/24, 37/24, -9/24],
        [1901/720, -2774/720, 2616/720, -1274/720, 251/720]
        ]




class TimeStepper(object):
    pass




class RK4TimeStepper(TimeStepper):
    def __init__(self):
        from pytools.log import IntervalTimer
        self.timer = IntervalTimer(
                "t_rk4", "Time spent doing algebra in RK4")

    def add_instrumentation(self, logmgr):
        logmgr.add_quantity(self.timer)

    def __call__(self, y, t, dt, rhs):
        try:
            self.residual
        except AttributeError:
            self.residual = 0*rhs(t, y)

        for a, b, c in zip(_RK4A, _RK4B, _RK4C):
            this_rhs = rhs(t + c*dt, y)

            self.timer.start()
            self.residual = a*self.residual + dt*this_rhs
            y = y + b * self.residual
            self.timer.stop()

        return y




class AdamsBashforthTimeStepper(TimeStepper):
    def __init__(self, order, startup_stepper=RK4TimeStepper()):
        try:
            self.coefficients = _ABCoefficients[order]
        except IndexError:
            self.coefficients = None

        if self.coefficients is None:
            raise ValueError, "unsupported order in Adams-Bashforth"

        self.f_history = []
        self.startup_stepper = startup_stepper

    def __call__(self, y, t, dt, rhs):
        if len(self.f_history) == 0:
            # insert IC
            self.f_history.append(rhs(t, y))

        if len(self.f_history) < len(self.coefficients):
            ynew = self.startup_stepper(y, t, dt, rhs)
            if len(self.f_history) == len(self.coefficients) - 1:
                # here's some memory we won't need any more
                del self.startup_stepper

        else:
            from operator import add

            assert len(self.coefficients) == len(self.f_history)
            ynew = y + dt * reduce(add,
                    (coeff * f 
                        for coeff, f in 
                        zip(self.coefficients, self.f_history)))

            self.f_history.pop()

        self.f_history.insert(0, rhs(t+dt, ynew))
        return ynew

