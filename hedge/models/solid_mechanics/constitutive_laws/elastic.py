# -*- coding: utf8 -*-
"""Implementation of Neohookean constitutive law"""

from __future__ import division

__copyright__ = "Copyright (C) 2013 Felipe Hernandez"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import hedge.models.solid_mechanics.mat3 as mat3
from hedge.models.solid_mechanics.materials import Material

class Elastic(Material):
    # interface
    def celerity(self, Fn, ndm):
        """
        Calculate critical wave speed
        """
        return ((self.l + 2*self.mu)/self.rho) ** (1/2)

    def stress(self, Fn, ndm):
        """
        Calculate the stress
        """
        F = [1,0,0, \
             0,1,0, \
             0,0,1]

        assert ndm == 2 or ndm == 3, "dimensions should be 2 or 3"
        if ndm == 2:
            F = mat3.copyMat2ToMat3(Fn)
        else:
            F = mat3.copyMat3(Fn)

        F = mat3.add(F, mat3.scaleMat(-1))
        trace = mat3.trace(F)

        mu = self.mu
        l = self.l

        P = [0,0,0, \
             0,0,0, \
             0,0,0]

        P[0] = l*trace+2*mu*F[0]
        P[4] = l*trace+2*mu*F[4]
        P[8] = l*trace+2*mu*F[8]

        P[1] = P[3] = mu*(F[1] + F[3])
        P[2] = P[6] = mu*(F[2] + F[6])
        P[5] = P[7] = mu*(F[5] + F[7])

        return P

    def tangent_moduli(self, Fn, ndf, ndm):
        """
        Computes elastic tangent moduli C = dW/dF
        """

        tangent = [0,]*81
        Miiii = self.l + 2*self.mu
        Miijj = self.l
        Mijij = self.mu

        iiii_indices = [0,40,80]
        iijj_indices = [4,8,36,44,72,76]
        ijij_indices = [10,12,20,24,28,30,50,52,56, \
                        60,68,70]

        for x in iiii_indices:
            tangent[x] = Miiii
        for x in iijj_indices:
            tangent[x] = Miijj
        for x in ijij_indices:
            tangent[x] = Mijij

        return tangent

    # meta-methods
    def __init__(self, rho, E, nu, **kwds):
        self.rho = rho
        self.E = E
        self.nu = nu
        self.l = nu * E / (1+nu) / (1-2*nu)
        self.mu = E / 2 / (1+nu)
        return
