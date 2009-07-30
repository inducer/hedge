from __future__ import division
from math import sqrt, log, sin, cos
from cmath import exp
import numpy
from cmath import pi
from numpy import matrix
from matplotlib.pyplot import *
from matplotlib.lines import *


from ode_systems import NonStiffUncoupled, \
            StiffUncoupled, \
            WeakCoupled, \
            StrongCoupled, \
            ExtForceStiff, \
            ExtForceNonStiff, \
            StiffCoupled2, \
            Full, Inh, CC, Comp, StiffComp2, StiffOscil, WeakCoupledInit

#ode = WeakCoupledInit()
ode_list = [NonStiffUncoupled, StiffUncoupled, WeakCoupled,\
        StrongCoupled, ExtForceStiff, ExtForceNonStiff, \
        StiffCoupled2, StiffComp2, StiffOscil, WeakCoupledInit]
ode_list = [StiffUncoupled]

for ode_arg in ode_list:
    ode = ode_arg()


    # set intial conditions ----------------------------------------------
    t = numpy.linspace(0,20,100)

    y0 = []
    y1 = []
    for i in range(100):
        y0.append(ode.soln_0(t[i]))
        y1.append(ode.soln_1(t[i]))

    y0 = numpy.array(y0)
    y1 = numpy.array(y1)

    figure(1)
    xlabel("t")
    ylabel("y")
    grid()
    line1 = plot(t,y0,'r', linewidth=4) # Fields: strong damped
    line2 = plot(t,y1,'b--', linewidth=4) # Particles: weak damped
    legend((line1,line2),('$y_{fast}$','$y_{slow}$'))
    ode_str = str(ode_arg).strip("ode_systems.")
    fname = "stab-mrab-out/%s_plot.png" % (ode_str)
    savefig(fname, dpi = 300, format='png')
    close(1)
    #show()

