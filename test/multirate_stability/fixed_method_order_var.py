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
import os
if True:
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
    def __init__(self, method, order, step_ratio, ode, outfile, err):
    # Two-Rate-AB settings ----------------------------------------------------
        self.order = order
        self.step_ratio = step_ratio
        self.method = method
        self.outfile = outfile
        self.err = err
        self.ode = ode

    # Case 1: Negative real EW's ----------------------------------------------
    def case_1(self):
        import fpformat as fpf
        get_stab_reg = calculate_stability_region(self.method,
                self.order,
                self.step_ratio,
                self.ode,
                self.err)
        stab_dt_res = get_stab_reg()
        self.outfile.write("& %s & %s " % (fpf.fix(stab_dt_res/self.step_ratio,2),
            fpf.fix(stab_dt_res,2)))


def make_serial_stab_reg():
    import fpformat as fpf
    from ode_systems import NonStiffUncoupled, \
            WeakCoupledInit, \
            StrongCoupled, \
            ExtForceStiff, \
            ExtForceNonStiff
    ode = WeakCoupledInit
    single_ab_dt_dict = {"2":0.53 , "3": 0.77, "4": 0.64, "5":0.51, "6":0.47, "7":0.45}
    method_dict = {'f_f_1a':'FFw' , 'f_f_1b':'FFs',
            's_f_1':'SF1r', 's_f_1_nr':'SF1',
            's_f_2a':'SF2wr', 's_f_2a_nr':'SF2w',
            's_f_2b':'SF2sr', 's_f_2b_nr':'SF2s',
            's_f_3a':'SF3wr', 's_f_3a_nr':'SF3w',
            's_f_3b':'SF3sr', 's_f_3b_nr':'SF3s',
            's_f_4':'SF4r', 's_f_4_nr':'SF4'}
    order_list = [2, 3, 4, 5, 6]
    step_ratio = 10
    err = 1
    method = "f_f_1a"
    # outputfile setup: ---------------------------------------------
    out_method = str(method)
    ode_str = str(ode).strip("ode_systems.")
    outfilename = "stab-mrab-out/%s_order_var_r_%s_%s.tex" % (ode_str,step_ratio,out_method)
    outfile = open(outfilename, "w")

    # Init table:
    outfile.write("\\begin{tabular}{l|c|cc}" + "\n")
    outfile.write("N & AB & two-rate& AB"+"\\""\\" + "\n")
    outfile.write("\\hline" + "\n")
    outfile.write(" & $\Delta t$ & $\Delta t$ fast & $\Delta t$ slow" + "\\""\\" + "\n")

    # initiate case 1 ---------------------------------------------------------
    for order in order_list:
        outfile.write("\\hline" + "\n")
        outfile.write("%s & %s" %(order,single_ab_dt_dict.get(str(order))))
        calc_stab_reg(method, order, step_ratio, ode, outfile, err).case_1()
        outfile.write("\\""\\" + "\n")

    outfile.write("\\hline" + "\n")
    outfile.write("\\end{tabular}" + "\n")




class calculate_stability_region:
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
        log_y = []
        log_soln = []
        log_t = []
        i = 0
        while t <= 20 or i <= 20:
            err = abs(
                    sqrt(y[0]**2 + y[1]**2)
                    - sqrt(self.ode.soln_0(t)**2 + self.ode.soln_1(t)**2)
                    )
            log_y.append(sqrt(y[0]**2 + y[1]**2))
            log_soln.append(sqrt(self.ode.soln_0(t)**2 + self.ode.soln_1(t)**2))
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




if __name__ == "__main__":
    make_serial_stab_reg()

