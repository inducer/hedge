# -*- coding: utf8 -*-
from __future__ import division

__copyright__ = "Copyright (C) 2009 Andreas Stock, Andreas Kloeckner"

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
from math import sqrt, log, sin, cos
from cmath import exp
import numpy
from cmath import pi
from numpy import matrix
from hedge.timestep.multirate_ab.methods import methods
from matplotlib.pyplot import *




class bisection:
    def __init__(self, method, order, step_ratio, ode, err):
        self.method      = method
        self.order       = order
        self.step_ratio  = step_ratio
        self.ode         = ode()
        # Abort criteria:
        self.prec = 1e-3
        # maximal error between exact and numerical solution ------------------
        self.max_error = err

    def is_stable(self, dt):
        # initialize stepper --------------------------------------------------
        from hedge.timestep.multirate_ab import \
                TwoRateAdamsBashforthTimeStepper
        stepper = TwoRateAdamsBashforthTimeStepper(self.method, dt, self.step_ratio, self.order)

        # set intial conditions -----------------------------------------------
        t = 0
        y = self.ode.initial_values
        #print "running dt:", dt
        # run integration -----------------------------------------------------
        y_zero = y
        log_y = []
        log_soln = []
        log_t = []
        i = 0
        while t <= 20 or i <= 20:
        #while i <= 20:
            #err = abs(
            #        sqrt(y[0]**2 + y[1]**2)
            #        - sqrt(self.ode.soln_0(t)**2 + self.ode.soln_1(t)**2)
            #        )
            err0 = abs(y[0] - self.ode.soln_0(t))
            err1 = abs(y[1] - self.ode.soln_1(t))
            log_y.append(sqrt(y[0]**2 + y[1]**2))
            log_soln.append(sqrt(self.ode.soln_0(t)**2 + self.ode.soln_1(t)**2))
            log_t.append(t)
            i += 1
            #if err > self.max_error:
            #if (y[0] > (y_zero[0] + self.max_error)) or (y[1] > (y_zero[1] + self.max_error)):
            if (err0 > 1) or (err1 > 1):
            #if (abs(y[0]) > 2) or (abs(y[1]) > 2):
                if False:
                    print "steps:", i
                    log_t = numpy.array(log_t)
                    log_y = numpy.array(log_y)
                    log_soln = numpy.array(log_soln)
                    figure()
                    xlabel("t")
                    ylabel("y")
                    grid()
                    line1 = plot(log_t,log_y)
                    line2 = plot(log_t,log_soln)
                    show()
                    raw_input()

                return False
            y = stepper(y, t, (self.ode.f2f_rhs,
                self.ode.s2f_rhs,
                self.ode.f2s_rhs, 
                self.ode.s2s_rhs))
            t += dt

        # make a nice graph ---------------------------------------------------
        if False:
            print "steps:", i
            log_t = numpy.array(log_t)
            log_y = numpy.array(log_y)
            log_soln = numpy.array(log_soln)
            figure()
            xlabel("t")
            ylabel("y")
            grid()
            line1 = plot(log_t,log_y)
            line2 = plot(log_t,log_soln)
            show()
            raw_input()

        return True


    def refine(self, stable_dt, unstable_dt):
        assert self.is_stable(stable_dt)
        assert not self.is_stable(unstable_dt)
        while abs(stable_dt-unstable_dt) > self.prec:
            mid_dt = (stable_dt+unstable_dt)/2
            if self.is_stable(mid_dt):
                stable_dt = mid_dt
            else:
                unstable_dt = mid_dt
        else:
            return stable_dt

    def __call__(self):
        dt = 0.5

        # bisection method ----------------------------------------------------
        if self.is_stable(dt):
            dt *= 2
            while self.is_stable(dt):
                dt *= 2

                if dt > 2**8:
                    return dt
            return self.refine(dt/2, dt)
        else:
            while not self.is_stable(dt):
                dt /= 2

                if dt < self.prec:
                    return dt
            return self.refine(dt, dt*2)
