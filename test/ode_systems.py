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

    def initial_values(self):
        return self.initial_values

    def t_start(self):
        return self.t_start

    def t_end(self):
        return self.t_end





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
    solve inhomogene system:
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
