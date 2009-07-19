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


""" Calculating stability regions for Two-Rate-Adams-Bashforth method
    Method:
    2 cases we are interested in:
    1. Compelx conjungated EV
    2. Real EV

"""

from math import sqrt, log, sin, cos
from cmath import exp
import numpy
from cmath import pi
from numpy import matrix
from hedge.timestep.multirate_ab.methods import methods
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
    def __init__(self):
        # Two-Rate-AB settings ----------------------------------------------------
        self.order = 2
        self.step_ratio = 2

        # parameterization dense --------------------------------------------------
        self.p_dense = 20

    # Case 1: Negative real EW's ----------------------------------------------
    def case_1(self):
    #def case_1(self,a_start,a_end,
     #       a_dense):
        points_1=[]
        for a in numpy.arange(-1, 0, 1/self.p_dense):
            lambda_1 = 1
            lambda_2 = a
            d_matrix = matrix([[lambda_1,0],[0,lambda_2]])
            for b in numpy.arange(2*pi/self.p_dense,2*pi, 2*pi/self.p_dense):
                print b
                for c in numpy.arange(2*pi/self.p_dense+b,2*pi+b, 2*pi/self.p_dense):
                    v_matrix = matrix([
                        [sin(b)*exp(complex(0,-c/2)), sin(b)*exp(complex(0,c/2))],
                        [cos(b)*exp(complex(0,c/2)), cos(b)*exp(complex(0,-c/2))]
                        ])
                    # ---------------------------------------------------------
                    # A = V D V⁻¹
                    # with D = [[λ₁ , 0 ]
                    #           [0  , λ₂]
                    a_matrix = (v_matrix*d_matrix*v_matrix.I).real
                    get_stab_reg = calculate_stability_region('f_f_1a',
                            self.order,
                            self.step_ratio,
                            lambda_1,
                            lambda_2,
                            a_matrix, v_matrix)

                    points_1.append(numpy.array([get_stab_reg(),a,b,c,]))

        filename = "case_1_res_a_start_%s_a_end_%s" %(a_start, a_end)
        case_1_res = {"case_1_res" : numpy.array(points_1.append)}
        pickle.dump(case_1_res, open(filename,"w"))



    # Case 2: Complex Conjungated EW's ----------------------------------------
    def case_2(self):
        points_2 = []
        for a in numpy.arange(pi*0.5, pi, 1/self.p_dense):
            lambda_1 = complex(cos(a),sin(a))
            lambda_2 = complex(cos(a),-sin(a))
            d_matrix = matrix([[lambda_1,0],[0,lambda_2]])
            for b in numpy.arange(2*pi/self.p_dense,2*pi, 2*pi/self.p_dense):
                #print b
                for c in numpy.arange(2*pi/self.p_dense+b,2*pi+b, 2*pi/self.p_dense):
                    v_matrix = matrix([
                        [sin(b)*exp(complex(0,-c/2)), sin(b)*exp(complex(0,c/2))],
                        [cos(b)*exp(complex(0,c/2)), cos(b)*exp(complex(0,-c/2))]
                        ])
                    # ---------------------------------------------------------
                    # A = V D V⁻¹
                    # with D = [[λ₁ , 0 ]
                    #           [0  , λ₂]
                    a_matrix = (v_matrix*d_matrix*v_matrix.I).real
                    get_stab_reg = calculate_stability_region('f_f_1a',
                            self.order,
                            self.step_ratio,
                            lambda_1,
                            lambda_2,
                            a_matrix, v_matrix)

                    points_2.append(numpy.array([get_stab_reg(),a,b,c,]))

        filename = "case_2_res_a_start_%s_a_end_%s" %(a_start, a_end)
        case_2_res = {"case_2_res" : numpy.array(points_1.append)}
        pickle.dump(case_1_res, open(filename,"w"))


def make_mpi_stab_reg(rank)
    a = calc_stab_reg()
    a.case_1()
    #a.case_2()
    #n_per_rank = 2
    #p_dense = 20
    #a_step = 2*pi/p_dense
    #a.case_1(0*a_step,a_step,2*pi/p_dense)


#@memoize
class calculate_stability_region:
    def __init__(self, method, order, step_ratio, lambda_1, lambda_2,
            a_matrix, v_matrix):
        self.method      = method
        self.order       = order
        self.step_ratio  = step_ratio
        self.lambda_1    = lambda_1
        self.lambda_2    = lambda_2
        self.a           = a_matrix
        self.v           = v_matrix

        # Abort criteria:
        self.prec = 1e-3
        # maximal error between exact and numerical solution ------------------
        self.max_error = 1e-5

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
        return (self.v * w).real

    def is_stable(self, dt):
        # initialize stepper --------------------------------------------------
        from hedge.timestep.multirate_ab import \
                TwoRateAdamsBashforthTimeStepper
        stepper = TwoRateAdamsBashforthTimeStepper(self.method, dt, self.step_ratio, self.order)

        # set intial conditions -----------------------------------------------
        t = 0
        y = self.exact_soln(t)

        # run integration -----------------------------------------------------
        for i in range(20):
            soln = self.exact_soln(t)
            err = abs(
                    sqrt(y[0]**2 + y[1]**2)
                    - sqrt(soln[0]**2 + soln[1]**2)
                    )
            if err > self.max_error:
                return False
            y = stepper(y, t, (self.f2f_rhs,
                               self.s2f_rhs,
                               self.f2s_rhs,
                               self.s2s_rhs))
            t += dt
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
    a = calc_stab_reg()
    a.case_1()
    #a.case_2()


