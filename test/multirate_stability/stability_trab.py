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

import boostmpi as mpi
import boostmpi.autoinit
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
    def __init__(self, p_dense, a_start, a_end, int_len):
        # Two-Rate-AB settings ----------------------------------------------------
        self.order = 2
        self.step_ratio = 2

        # parameterization dense --------------------------------------------------
        self.p_dense = p_dense
        self.a_start = a_start
        self.a_end   = a_end
        self.int_len = int_len

        # ---------------------------------------------------------
        # A = V D V⁻¹
        # with D = [[λ₁ , 0 ]
        #           [0  , λ₂]

    # Case 1: Negative real EW's ----------------------------------------------
    def case_1_mod(self):
        mod_dense = 20
        data = numpy.empty((1, mod_dense, mod_dense), dtype=numpy.float64, order="F")
        lambda_1 = 1
        lambda_2 = 0.001
        d_matrix = matrix([[lambda_1,0],[0,lambda_2]])
        i = -1
        for b in numpy.arange(0, 2*pi, 2*pi/mod_dense):
            i += 1
            j = i
            print "-----------"
            # run c from b on through 2*pi intervall
            for c in numpy.arange(2*pi/mod_dense+b,
                    2*pi+b+2*pi/mod_dense,
                    2*pi/mod_dense):
                j += 1
                print i, numpy.fmod(j,mod_dense)
                v_matrix = matrix([
                    [cos(b), cos(c)],
                    [sin(b), sin(c)]
                    ])

                a_matrix = (v_matrix*d_matrix*v_matrix.I).real

                get_stab_reg = calculate_stability_region('f_f_1a',
                        self.order,
                        self.step_ratio,
                        lambda_1,
                        lambda_2,
                        a_matrix, v_matrix)

                data[0][i][numpy.fmod(j,mod_dense)] = get_stab_reg()

        filename = "case_1_res_a_%s.dat" %(lambda_2)
        print "finished:", filename
        case_1_res = {"case_1_res" : data}
        pickle.dump(case_1_res, open(filename,"w"))


    # Case 1: Negative real EW's ----------------------------------------------
    def case_1(self):
        points_1=[]
        for a in numpy.linspace(self.a_start,
                self.a_end-self.int_len/self.p_dense,
                self.p_dense/mpi.size):
            lambda_1 = 1
            lambda_2 = a
            d_matrix = matrix([[lambda_1,0],[0,lambda_2]])
            for b in numpy.arange(0, 2*pi, 2*pi/self.p_dense):
                # run c from b on through 2*pi intervall
                for c in numpy.arange(2*pi/self.p_dense+b,
                        2*pi+b+2*pi/self.p_dense,
                        2*pi/self.p_dense):
                    v_matrix = matrix([
                        [cos(b), cos(c)],
                        [sin(b), sin(c)]
                        ])

                    a_matrix = (v_matrix*d_matrix*v_matrix.I).real

                    get_stab_reg = calculate_stability_region('f_f_1a',
                            self.order,
                            self.step_ratio,
                            lambda_1,
                            lambda_2,
                            a_matrix, v_matrix)

                    points_1.append(numpy.array([get_stab_reg(),a,b,c,]))

        filename = "case_1_res_a_start_%s_a_end_%s.dat" %(self.a_start, self.a_end)
        print "finished:", filename
        case_1_res = {"case_1_res" : numpy.array(points_1)}
        pickle.dump(case_1_res, open(filename,"w"))



    # Case 2: Complex Conjungated EW's ----------------------------------------
    def case_2(self):
        points_2 = []
        for a in numpy.linspace(self.a_start,
                self.a_end-self.int_len/self.p_dense,
                self.p_dense/mpi.size):
            lambda_1 = complex(cos(a),sin(a))
            lambda_2 = complex(cos(a),-sin(a))
            d_matrix = matrix([[lambda_1,0],[0,lambda_2]])
            for b in numpy.arange(0, 2*pi, 2*pi/self.p_dense):
                if numpy.fmod(b,pi*0.5) == 0:
                    b += 1e-05
                for c in numpy.arange(1, 0, -1/self.p_dense):
                    v_matrix = matrix([
                        [sin(b)*exp(complex(0,-c/2)), sin(b)*exp(complex(0,c/2))],
                        [cos(b)*exp(complex(0,c/2)), cos(b)*exp(complex(0,-c/2))]
                        ])

                    a_matrix = (v_matrix*d_matrix*v_matrix.I)
                    assert a_matrix.imag.max > 1e-10
                    a_matrix = a_matrix.real

                    get_stab_reg = calculate_stability_region('f_f_1a',
                            self.order,
                            self.step_ratio,
                            lambda_1,
                            lambda_2,
                            a_matrix, v_matrix)

                    points_2.append(numpy.array([get_stab_reg(),a,b,c,]))

        filename = "case_2_res_a_start_%s_a_end_%s.dat" %(self.a_start, self.a_end)
        print "finished:", filename
        case_2_res = {"case_2_res" : numpy.array(points_2)}
        pickle.dump(case_2_res, open(filename,"w"))


def make_serial_stab_reg():
    # initiate case 1 ---------------------------------------------------------
    int_a = -1
    int_b = 0
    int_len = abs(int_a-int_b)
    for i in range(mpi.size):
        a_start = int_a + i/mpi.size * abs(int_b-int_a)
        a_end = int_a + (i+1)/mpi.size * abs(int_b-int_a)
        calc_stab_reg(1, a_start, a_end, int_len).case_1_mod()


def make_mpi_stab_reg(p_dense):
    assert p_dense >= mpi.size
    # number of ranks should fit to p_dense in an integer relation
    work_list = []
    # initiate case 1 ---------------------------------------------------------
    int_a = -1
    int_b = 0
    int_len = abs(int_a-int_b)
    for i in range(mpi.size):
        a_start = int_a + i/mpi.size * abs(int_b-int_a)
        a_end = int_a + (i+1)/mpi.size * abs(int_b-int_a)
        work_list.append(calc_stab_reg(p_dense, a_start, a_end, int_len).case_1)
    # initiate case 2 ---------------------------------------------------------
    int_a = pi * 0.5
    int_b = pi
    # Due to the implementation pi will never be calculated. But this is not necessary
    # since it describes the case that we are on the real axis. This case is allready 
    # covered by case 1.
    int_len = abs(int_a-int_b)
    for i in range(mpi.size):
        a_start = int_a + i/mpi.size * abs(int_b-int_a)
        a_end = int_a + (i+1)/mpi.size * abs(int_b-int_a)
        work_list.append(calc_stab_reg(p_dense, a_start, a_end, int_len).case_2)

    if mpi.rank == 0:
        print "Worklist length:", len(work_list)
        print "Number of processes:", mpi.size
    # make shure that the p_dense and the number of prozesses have an interger
    # relation.
    #assert int(p_dense/mpi.size) == p_dense/mpi.size

    # dispose to different processes ------------------------------------------
    divisor = int(len(work_list)/mpi.size)

    # first pop all all not wished list entries without executing them:
    for i in range(mpi.rank):
       for j in range(divisor):
           work_list.pop()

    # pop all entries which shall be executed:
    for i in range(divisor):
        work_list.pop()()

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
    #a = calc_stab_reg(20, -0.9, -0.8)
    #a.case_1()
    #a.case_2()
    #make_mpi_stab_reg(10)
    make_serial_stab_reg()

