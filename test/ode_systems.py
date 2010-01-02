# -*- coding: utf8 -*-
# Hedge - the Hybrid'n'Easy DG Environment
# Copyright (C) 2009 Andreas Stock
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import division




import numpy
import numpy.linalg as la
from hedge.backends.jit import Discretization as JITDiscretization
from math import sqrt, log, sin, cos, exp


class LinearODESystemsBase():
    """Check that the multirate timestepper has the advertised accuracy

    Solve linear ODE-system:

                            ∂w/∂t = A w,

    with w = [u,v] = [u,∂u/∂t]. The system gets solved for differen matrix A.
    """
    def __init__(self):
        self.t_start = 0
        self.t_end = 1
        self.initial_values = numpy.array([self.soln_0(self.t_start),
            self.soln_1(self.t_start)])





class Basic(LinearODESystemsBase):
    """
    ODE-system - basic
    ∂u/∂t = v
    ∂v/∂t = -u/t²
    A = [[0, 1]
        [-1/t², 0]].
    """
    def __init__(self):
        self.t_start = 1
        self.t_end = 2
        self.initial_values = numpy.array([1, 3])

    def f2f_rhs(self, t, u, v):
        return 0

    def s2f_rhs(self, t, u, v):
        return v()

    def f2s_rhs(self, t, u, v):
        return -u()/t**2

    def s2s_rhs(self, t, u, v):
        return 0

    def soln_0(self, t):
        inner = sqrt(3)/2*log(t)
        return sqrt(t)*(
                5*sqrt(3)/3*sin(inner)
                + cos(inner)
                )





class Full(LinearODESystemsBase):
    """
    ODE-system - full
    From:
        Gewöhnliche Differentialgleichungen
        Theorie und Praxis - vertieft und visualisiert mit Maple
        Wilhelm Forst and Dieter Hoffmann
        2005, Springer Berlin Heidelberg
        p. 145
    A = [[cos(2*t),-(sin(2*t)-1)]
        [(sin(2*t)+1),-cos(2*t)]].
    """
    def __init__(self):
        LinearODESystemsBase.__init__(self)

    def f2f_rhs(self, t, u, v):
        return cos(2*t)*u()

    def s2f_rhs(self, t, u, v):
        return (sin(2*t)-1)*v()

    def f2s_rhs(self, t, u, v):
        return (sin(2*t)+1)*u()

    def s2s_rhs(self, t, u, v):
        return -cos(2*t)*v()

    def soln_0(self, t):
        return exp(t)*cos(t)

    def soln_1(self, t):
        return exp(t)*sin(t)





class Real(LinearODESystemsBase):
    """
    ODE-system - real
    A = [[-1,3]
        [2,-2]],
    with the real eigenvalues λ₁=1 and λ₂=-4 which are quite far away from each
    other representing a recognizable difference between the speed of the
    two systems.
    """
    def __init__(self):
        LinearODESystemsBase.__init__(self)

    def f2f_rhs(self, t, u, v):
        return -u()

    def s2f_rhs(self, t, u, v):
        return 3*v()

    def f2s_rhs(self, t, u, v):
        return 2*u()

    def s2s_rhs(self, t, u, v):
        return -2*v()

    def soln_0(self, t):
        return exp(-4*t)*(exp(5*t)+1)

    def soln_1(self, t):
        return 1/3*exp(-4*t)*(2*exp(5*t)-3)




class Comp(LinearODESystemsBase):
    """
    ODE-system - complex
    A = [[0,1]
        [-1,0]],
    with pure complex eigenvalues λ₁=i and λ₂=-i. This is a
    skew-symmetric matrix which also is used in the Maxwell
    operator for the curl operator.
    """
    def __init__(self):
        LinearODESystemsBase.__init__(self)

    def f2f_rhs(self, t, u, v):
        return 0

    def s2f_rhs(self, t, u, v):
        return v()

    def f2s_rhs(self, t, u, v):
        return -u()

    def s2s_rhs(self, t, u, v):
        return 0

    def soln_0(self, t):
        return sin(t)*sin(2*t)+cos(t)*(cos(2*t)+1)

    def soln_1(self, t):
        return sin(t)*(cos(2*t)-1)-cos(t)*sin(2*t)





class CC(LinearODESystemsBase):
    """
    ODE-system - complex-conjungated
    A = [[1,1]
        [-1,1]]
    with the complex conjungated eigenvalues λ₁=1-i and λ₂=1+i.
    """
    def __init__(self):
        LinearODESystemsBase.__init__(self)

    def f2f_rhs(self, t, u, v):
        return u()

    def s2f_rhs(self, t, u, v):
        return v()

    def f2s_rhs(self, t, u, v):
        return -u()

    def s2s_rhs(self, t, u, v):
        return v()

    def soln_0(self, t):
        return exp(t)*sin(t)

    def soln_1(self, t):
        return exp(t)*cos(t)





class Tria(LinearODESystemsBase):
    """
    ODE-system - tria
    ∂²u/∂t² + ∂u/∂t + u = 0
    gets to:
    ∂u/∂t = v
    ∂v/∂t = -v -u.
    """
    def __init__(self):
        self.t_start = 0
        self.t_end = 2
        self.initial_values = numpy.array([1, 3])

    def f2f_rhs(self, t, u, v):
        return 0

    def s2f_rhs(self, t, u, v):
        return v()

    def f2s_rhs(self, t, u, v):
        return -u()

    def s2s_rhs(self, t, u, v):
        return -v()

    def soln_0(self, t):
        inner = sqrt(3)/2*t
        return exp(-t/2)*(
                7*sqrt(3)/3*sin(inner)
                + cos(inner)
                )





class Inh(LinearODESystemsBase):
    """
    ODE-system - inhom
    from: L. Papula, Math for Engineers Part 2, p.592, Vieweg Verlag
    solve inhomogeneous system:
            ∂w/∂t = A w + f(t)
    A = [[-2,3]
        [-3,-2]]
    f= [[exp(t)]
        [   0  ]]
    """
    def __init__(self):
        LinearODESystemsBase.__init__(self)

    def f2f_rhs(self, t, u, v):
        return -2*u() + 2*exp(2*t)

    def s2f_rhs(self, t, u, v):
        return 3*v()

    def f2s_rhs(self, t, u, v):
        return -3*u()

    def s2s_rhs(self, t, u, v):
        return -2*v()

    def soln_0(self, t):
        return exp(-2*t) * (6/25*sin(3*t)
                +42/25*cos(3*t))+8/25*exp(2*t)

    def soln_1(self, t):
        return exp(-2*t) * (-42/25*sin(3*t)
                +6/25*cos(3*t))-6/25*exp(2*t)




class Inh2(LinearODESystemsBase):
    """
    ODE-system - inhom_2
    from: L. Papula, Math for Engineers Part 2, p.592, Vieweg Verlag
    solve inhomogene system:
            ∂w/∂t = A w + f(t)
    A = [[-1,3]
        [2,-2]]
    f= [[t]
        [ exp(-t)]]
    initial conditions:
    w(t=0) = [[0]
              [0]]
    """
    def __init__(self):
        LinearODESystemsBase.__init__(self)

    def f2f_rhs(self, t, u, v):
        return -1*u() + t

    def s2f_rhs(self, t, u, v):
        return 3*v()

    def f2s_rhs(self, t, u, v):
        return 2*u()

    def s2s_rhs(self, t, u, v):
        return -2*v() + exp(-t)

    def soln_0(self, t):
        return 9/40*exp(-4*t)+9/10*exp(t)-0.5*t-5/8-0.5*exp(-t)

    def soln_1(self, t):
        return -9/40*exp(-4*t)+3/5*exp(t)-0.5*t-3/8





class ExtForceStiff(LinearODESystemsBase):
    """
    ∂w/∂t = A w + f(t)
    A = [[-1,0.001]
        [0.001,-0.001]]
    f= [[0]
        [ exp(-0.001) + exp(-t)]]
    """
    def __init__(self):
        self.lambda_1 = -1.000001001
        self.lambda_2 = -0.000998999
        self.t_start = 0
        self.t_end = 2
        self.initial_values = numpy.array([1, 0])
        self.c1 = (999-2001*(self.lambda_2 + 1))/(self.lambda_1-self.lambda_2)
        self.c2 = 2001 - (999-2001*(self.lambda_2 + 1))/(self.lambda_1-self.lambda_2)

    def f2f_rhs(self, t, u, v):
        return -1*u()

    def s2f_rhs(self, t, u, v):
        return 0.001*v()

    def f2s_rhs(self, t, u, v):
        return 0.001*u()

    def s2s_rhs(self, t, u, v):
        return -0.001*v() + exp(-t) + exp(-0.001*t)

    def soln_0(self, t):
        return self.c1 * exp(self.lambda_1*t)\
                + self.c2 * exp(self.lambda_2*t)\
                - 1000 * (exp(-t) + exp(-0.001*t))

    def soln_1(self, t):
        return 1000 * (self.c1 * exp(self.lambda_1*t) * (self.lambda_1 + 1)\
                + self.c2 * exp(self.lambda_2*t) * (self.lambda_2 + 1)\
                - 999 * exp(-0.001*t))




class StiffUncoupled(LinearODESystemsBase):
    """
    ∂w/∂t = A w
    A = [[-1,0]
        [0,-0.001]]
    """
    def __init__(self):
        self.lambda_1 = -1
        self.lambda_2 = -0.001
        LinearODESystemsBase.__init__(self)

    def f2f_rhs(self, t, u, v):
        return -1*u()

    def s2f_rhs(self, t, u, v):
        return 0

    def f2s_rhs(self, t, u, v):
        return 0

    def s2s_rhs(self, t, u, v):
        return -0.001*v()

    def soln_0(self, t):
        return exp(self.lambda_1*t)

    def soln_1(self, t):
        return exp(self.lambda_2*t)





class NonStiffUncoupled(LinearODESystemsBase):
    """
    ∂w/∂t = A w
    A = [[-1,0]
        [0,-1]]
    """
    def __init__(self):
        self.lambda_1 = -1
        self.lambda_2 = -1
        LinearODESystemsBase.__init__(self)

    def f2f_rhs(self, t, u, v):
        return -1*u()

    def s2f_rhs(self, t, u, v):
        return 0

    def f2s_rhs(self, t, u, v):
        return 0

    def s2s_rhs(self, t, u, v):
        return -1*v()

    def soln_0(self, t):
        return exp(self.lambda_1*t)

    def soln_1(self, t):
        return exp(self.lambda_2*t)





class WeakCoupled(LinearODESystemsBase):
    """
    A = [[-1,0.001]
        [0.001,-0.001]]
    """
    def __init__(self):
        self.lambda_1 = -(sqrt(998005)+1001)/2000  #-1.000001001
        self.lambda_2 = (sqrt(998005)-1001)/2000   #-0.000998999
        LinearODESystemsBase.__init__(self)
        self.t_end = 2

    def f2f_rhs(self, t, u, v):
        return -1*u()

    def s2f_rhs(self, t, u, v):
        return 0.001*v()

    def f2s_rhs(self, t, u, v):
        return 0.001*u()

    def s2s_rhs(self, t, u, v):
        return -0.001*v()

    def soln_0(self, t):
        return exp(self.lambda_1*t) + exp(self.lambda_2*t)

    def soln_1(self, t):
        return 0.5 * (
                exp(self.lambda_2*t) * (sqrt(998005)+999)
                + exp(self.lambda_1*t) * (999-sqrt(998005))
                )




class WeakCoupledInit(LinearODESystemsBase):
    """
    A = [[-1,0.001]
        [0.001,-0.001]]
    """
    def __init__(self):
        self.lambda_1 = -(sqrt(998005)+1001)/2000  #-1.000001001
        self.lambda_2 = (sqrt(998005)-1001)/2000   #-0.000998999
        self.a11 = -1
        self.a12 = 0.001
        self.t_start = 0
        self.t_end = 1
        self.c2 = (self.a12+self.a11-self.lambda_1)/(self.lambda_2 - self.lambda_1)
        self.c1 = 1 - self.c2
        self.initial_values = numpy.array([self.soln_0(self.t_start),
            self.soln_1(self.t_start)])
        #print self.initial_values
        #raw_input("init values")
        #self.initial_values = numpy.array([1,1])

    def f2f_rhs(self, t, u, v):
        return self.a11*u()

    def s2f_rhs(self, t, u, v):
        return self.a12*v()

    def f2s_rhs(self, t, u, v):
        return 0.001*u()

    def s2s_rhs(self, t, u, v):
        return -0.001*v()

    def soln_0(self, t):
        return self.c1 * exp(self.lambda_1*t)\
                + self.c2 * exp(self.lambda_2*t)

    def soln_1(self, t):
        return 1/self.a12 * (
                exp(self.lambda_1*t) * self.c1 * (self.lambda_1 - self.a11)
                + exp(self.lambda_2*t) * self.c2 * (self.lambda_2 - self.a11)
                )





class StrongCoupled(LinearODESystemsBase):
    """
    A = [[-1,-1]
        [0.001,-0.001]]
    """
    def __init__(self):
        self.lambda_1 = -(sqrt(994001)+1001)/2000
        self.lambda_2 = (sqrt(994001)-1001)/2000
        LinearODESystemsBase.__init__(self)
        self.t_end = 2

    def f2f_rhs(self, t, u, v):
        return -1*u()

    def s2f_rhs(self, t, u, v):
        return -1*v()

    def f2s_rhs(self, t, u, v):
        return 0.001*u()

    def s2s_rhs(self, t, u, v):
        return -0.001*v()

    def soln_0(self, t):
        return exp(self.lambda_1*t) + exp(self.lambda_2*t)

    def soln_1(self, t):
        return -0.0005 * (
                exp(self.lambda_2*t) * (sqrt(994001)+999)
                + exp(self.lambda_1*t) * (999-sqrt(994001))
                )





class ExtForceNonStiff(LinearODESystemsBase):
    """
    ∂w/∂t = A w + f(t)
    A = [[-10,1]
        [1,-1]]
    f= [[0]
        [ exp(-10) + exp(-t)]]
    """
    def __init__(self):
        self.lambda_1 = -11/2 + sqrt(121/4-9)
        self.lambda_2 = -11/2 - sqrt(121/4-9)
        self.t_start = 0
        self.t_end = 2
        self.initial_values = numpy.array([1, 0])
        self.c1 = (9-3*(self.lambda_2 + 10))/(self.lambda_1-self.lambda_2)
        self.c2 = 3 - (9-3*(self.lambda_2 + 10))/(self.lambda_1-self.lambda_2)

    def f2f_rhs(self, t, u, v):
        return -10*u()

    def s2f_rhs(self, t, u, v):
        return 1*v()

    def f2s_rhs(self, t, u, v):
        return 1*u()

    def s2s_rhs(self, t, u, v):
        return -1*v() + exp(-t) + exp(-10*t)

    def soln_0(self, t):
        return self.c1 * exp(self.lambda_1*t) + self.c2 * exp(self.lambda_2*t)\
                - exp(-10*t) - exp(-t)

    def soln_1(self, t):
        return self.c1 * (self.lambda_1 + 10) * exp(self.lambda_1*t)\
                + self.c2 *(self.lambda_2 + 10) * exp(self.lambda_2*t)\
                - 9 * exp(-t)






class StiffCoupled2(LinearODESystemsBase):
    """
    A = [[-1,0.999]
        [0,-0.001]]
    """
    def __init__(self):
        self.lambda_1 = -1
        self.lambda_2 = -1/1000
        LinearODESystemsBase.__init__(self)
        self.t_end = 2

    def f2f_rhs(self, t, u, v):
        return -1*u()

    def s2f_rhs(self, t, u, v):
        return 0.999*v()

    def f2s_rhs(self, t, u, v):
        return 0*u()

    def s2s_rhs(self, t, u, v):
        return -0.001*v()

    def soln_0(self, t):
        return exp(self.lambda_1*t) + exp(self.lambda_2*t)

    def soln_1(self, t):
        return  exp(self.lambda_2*t)





class StiffComp(LinearODESystemsBase):
    """
    A = [[0,1]
        [-100,0]]
    """
    def __init__(self):
        self.lambda_1 = -10
        self.lambda_2 = -10
        LinearODESystemsBase.__init__(self)
        self.t_end = 2

    def f2f_rhs(self, t, u, v):
        return 0*u()

    def s2f_rhs(self, t, u, v):
        return 1*v()

    def f2s_rhs(self, t, u, v):
        return -100*u()

    def s2s_rhs(self, t, u, v):
        return 0*v()

    def soln_0(self, t):
        return 2*cos(10*t)

    def soln_1(self, t):
        return  -20*sin(10*t)





class StiffComp2(LinearODESystemsBase):
    def __init__(self):
        self.alpha = -1001/2000
        self.omega = sqrt(3999001999)/2000
        self.a11 = -1
        self.a12 = -1000
        self.t_start = 0
        self.t_end = 1
        self.c1 = (self.alpha-self.a12-self.a11)/(-self.omega)
        self.c2 = 1
        self.initial_values = numpy.array([self.soln_0(self.t_start),
            self.soln_1(self.t_start)])
        #print self.initial_values
        #raw_input("init values")
        #self.initial_values = numpy.array([1,1])

    def f2f_rhs(self, t, u, v):
        return self.a11*u()

    def s2f_rhs(self, t, u, v):
        return self.a12*v()

    def f2s_rhs(self, t, u, v):
        return 1*u()

    def s2s_rhs(self, t, u, v):
        return -0.001*v()

    def soln_0(self, t):
        return exp(self.alpha*t)\
                * (self.c1 * sin(t*self.omega)
                        + self.c2 * cos(t*self.omega))

    def soln_1(self, t):
        return -1/self.a12 * (
                (exp(self.alpha*t)*self.omega*sin(t*self.omega)
                    + (self.a11-self.alpha)*exp(self.alpha*t)*cos(t*self.omega))*self.c2
                + ((self.a11-self.alpha)*exp(self.alpha*t)*sin(t*self.omega)
                    - exp(self.alpha*t)*self.omega*cos(t*self.omega))*self.c1
                )





class StiffOscil(LinearODESystemsBase):
    """
    A = [[10*sin(t),0]
        [0,sin(t)]]
    """
    def __init__(self):
        self.lambda_1 = -1
        self.lambda_2 = -1/1000
        LinearODESystemsBase.__init__(self)
        self.t_end = 2

    def f2f_rhs(self, t, u, v):
        return cos(t*10) + 0*u()

    def s2f_rhs(self, t, u, v):
        return 0*v()

    def f2s_rhs(self, t, u, v):
        return 0*u()

    def s2s_rhs(self, t, u, v):
        return cos(t)+0*v()

    def soln_0(self, t):
        return 1/10*sin(t*10)

    def soln_1(self, t):
        return  sin(t)
