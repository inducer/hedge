# -*- coding: utf8 -*-
"""Automatic size finding for stability regions."""

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
if False:
    methods_man = ['f_f_1a', 'f_f_1b',
            's_f_1', 's_f_1_nr',
            's_f_2a', 's_f_2a_nr',
            's_f_2b', 's_f_2b_nr',
            's_f_3a', 's_f_3a_nr',
            's_f_3b', 's_f_3b_nr',
            's_f_4', 's_f_4_nr']

else:
    methods_man = ['f_f_1a']
import pickle

class calc_stab_reg():
    def __init__(self, method):
    # Two-Rate-AB settings ----------------------------------------------------
        self.order = 2
        self.step_ratio = 10
        self.method = method

    # Case 1: Negative real EW's ----------------------------------------------
    def case_1(self):
        stab_dt=[]
        #ev_ratio = [1e-3, 5e-2, 1e-2, 1e-1, 0.5, 1, 2, 10, 100, 500]
        ev_ratio = [1e-4, 1e-3, 5e-2, 1e-2, 1e-1, 0.5, 1]
        for a in ev_ratio:
            #print "ev ratio:", a
            lambda_1 = -1
            lambda_2 = -a
            d_matrix = matrix([[lambda_1,0],[0,lambda_2]])

            get_stab_reg = calculate_stability_region(self.method,
                            self.order,
                            self.step_ratio,
                            lambda_1,
                            lambda_2,
                            d_matrix)

            stab_dt.append(get_stab_reg())

        # Save stuff to file --------------------------------------------------
        dt_small = numpy.array(stab_dt)/self.step_ratio
        print dt_small
        print stab_dt
        print ev_ratio
        filename = "case_1_res_%s_%s_%s.dat" % (self.method,self.order,self.step_ratio)
        print "finished:", filename
        results = {"stab_dt" : numpy.array(stab_dt), "ev_ratio": numpy.array(ev_ratio)}
        pickle.dump(results, open(filename,"w"))

        # make a nice graph ---------------------------------------------------
        #from matplotlib.pyplot import *
        #figure()
        #title_name = "stabile dt: %s" %self.method
        #title(title_name)
        #xlabel("ev ratio")
        #ylabel("stable dt")
        #grid()
        #line1 = plot(numpy.array(ev_ratio),numpy.array(stab_dt))
        #fname = "out/stable_dt_%s.pdf"  %self.method
        #savefig("out/",fname)
        #show()

def make_serial_stab_reg():
    # initiate case 1 ---------------------------------------------------------
    calc_stab_reg("f_f_1a").case_1()





class calculate_stability_region:
    def __init__(self, method, order, step_ratio, lambda_1, lambda_2,
            a_matrix):
        self.method      = method
        self.order       = order
        self.step_ratio  = step_ratio
        self.lambda_1    = lambda_1
        self.lambda_2    = lambda_2
        self.a           = a_matrix
        # Abort criteria:
        self.prec = 1e-3
        # maximal error between exact and numerical solution ------------------
        self.max_error = 1

    # define rhs's ------------------------------------------------------------
    # A = V D V⁻¹
    # with D = [[λ₁ , 0 ]
    #           [0  , λ₂]]
    def f2f_rhs(self, t, y_f, y_s):
            return self.a[0,0] * y_f()

    def s2f_rhs(self, t, y_f, y_s):
            return self.a[0,1] * y_s()

    def f2s_rhs(self, t, y_f, y_s):
            return self.a[1,0] * y_f()

    def s2s_rhs(self, t, y_f, y_s):
            return self.a[1,1] * y_s()


    # exact solution ------------------------------------------------------
    # w = [exp(λ₁*t , exp(λ₂*t)]
    # soln_y = V w
    # in case 2 only the realpart contributes to the solution:
    def exact_soln(self, t):
        exp_lambda_1 =  exp(self.lambda_1*t)
        exp_lambda_2 =  exp(self.lambda_2*t)
        w = matrix([[exp_lambda_1],[exp_lambda_2]])
        return (self.a * w).real

    def is_stable(self, dt):
        # initialize stepper --------------------------------------------------
        from hedge.timestep.multirate_ab import \
                TwoRateAdamsBashforthTimeStepper
        stepper = TwoRateAdamsBashforthTimeStepper(self.method, dt, self.step_ratio, self.order)

        # set intial conditions -----------------------------------------------
        t = 0
        y = self.exact_soln(t)
        #print "running dt:", dt
        # run integration -----------------------------------------------------
        log_y = []
        log_soln = []
        log_t = []
        i = 0
        while t <= 20 or i <= 20:
            soln = self.exact_soln(t)
            err = abs(
                    sqrt(y[0]**2 + y[1]**2)
                    - sqrt(soln[0]**2 + soln[1]**2)
                    )
            log_y.append(sqrt(y[0]**2 + y[1]**2))
            log_soln.append(sqrt(soln[0]**2 + soln[1]**2))
            log_t.append(t)
            i += 1
            if err > self.max_error:
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
            y = stepper(y, t, (self.f2f_rhs,
                               self.s2f_rhs,
                               self.f2s_rhs,
                               self.s2s_rhs))
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




if __name__ == "__main__":
    make_serial_stab_reg()

