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
        from bisection import bisection
        get_stab_dt = bisection(self.method,
                self.order,
                self.step_ratio,
                self.ode,
                self.err)
        stab_dt = get_stab_dt()
        self.outfile.write("& %s & %s " % (fpf.fix(stab_dt/self.step_ratio,2),
            fpf.fix(stab_dt,2)))




def make_serial_stab_reg():
    import fpformat as fpf
    from ode_systems import NonStiffUncoupled, \
            StiffUncoupled, \
            WeakCoupledInit, \
            StrongCoupled, \
            ExtForceStiff, \
            ExtForceNonStiff
    ode = ExtForceStiff
    order_list = [2, 3, 4, 5]
    method_dict = {'f_f_1a':'FFw' , 'f_f_1b':'FFs',
            's_f_1':'SF1r', 's_f_1_nr':'SF1',
            's_f_2a':'SF2wr', 's_f_2a_nr':'SF2w',
            's_f_2b':'SF2sr', 's_f_2b_nr':'SF2s',
            's_f_3a':'SF3wr', 's_f_3a_nr':'SF3w',
            's_f_3b':'SF3sr', 's_f_3b_nr':'SF3s',
            's_f_4':'SF4r', 's_f_4_nr':'SF4'}
    step_ratio = 10
    err = 1
    # outputfile setup: ---------------------------------------------
    ode_str = str(ode).strip("ode_systems.")
    outfile = "stab-mrab-out/%s_all_methods_order_var_r_%s.tex" % (ode_str,
            step_ratio)
    outfile = open(outfile, "w")

    # Init table:
    outfile.write("\\begin{tabular}{l")
    for i in order_list:
        outfile.write("|cc")
    outfile.write("}" + "\n")
    outfile.write("N")
    for i in order_list:
        outfile.write("& %s  &  " %(i))
    outfile.write("\\""\\" + "\n")
    outfile.write("\\hline" + "\n")
    outfile.write("method")
    for i in order_list:
        outfile.write("& $\Delta t_{fast}$  & $\Delta t_{slow}$")
    outfile.write("\\""\\" + "\n")
    outfile.write("\\hline" + "\n")

    # initiate case 1 ---------------------------------------------------------
    for method in methods_man:
        #outfile.write("\\hline" + "\n")
        outfile.write("%s " %(method_dict.get(str(method))))
        for order in order_list:
            calc_stab_reg(method, order, step_ratio, ode, outfile, err).case_1()
        outfile.write("\\""\\" + "\n")

    outfile.write("\\hline" + "\n")
    outfile.write("\\end{tabular}" + "\n")




# -----------------------------------------------------------------------------
if __name__ == "__main__":
    make_serial_stab_reg()

