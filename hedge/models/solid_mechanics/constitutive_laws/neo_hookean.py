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

import hedge.models.solid_mechanics.constitutive_laws.mat3 as mat3
from hedge.models.solid_mechanics.constitutive_laws.material import Material

def _perform_log(x):
    y = x-1
    return y - y**2/2
    from hedge.optemplate.primitives import CFunction
    return CFunction("log")(x)

class NeoHookean(Material):

    # interface
    def celerity(self, Fn, ndm):
        """
        Calculate critical wave speed
        """
        F = [1,0,0, \
             0,1,0, \
             0,0,1]

        assert ndm == 2 or ndm == 3, "dimensions should be 2 or 3"
        if ndm == 2:
            F = mat3.copyMat2ToMat3(Fn)
        else:
            F = mat3.copyMat3(Fn)

        C = mat3.mults(F, F)
        detC = mat3.det(C)
        Cinv = mat3.inv(C, detA=detC)
        p = self.l * 0.5 * _perform_log(detC)
        coef = self.mu - p
        rhot = self.rho / (detC**(1/2))

        return ((self.l + 2*coef)/rhot) ** (1/2)

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

        C = mat3.mults(F, F)
        detC = mat3.det(C)
        #if detC < 1e-10:
        #    raise NegativeJacobianError()
        Cinv = mat3.inv(C, detA=detC)
        defVol = 0.5 * _perform_log(detC)
        p = self.l* defVol
        trace = mat3.trace(C)
        coef = p - self.mu
        S = mat3.scalarMult(coef, Cinv)
        S = mat3.add(S, mat3.scaleMat(self.mu))
        P = mat3.mulss(F, S)
        return tuple(P)

    def tangent_moduli(self, Fn, ndf, ndm):
        """
        Computes elastic tangent moduli C = dW/dF
        """
        # -- Don't worry about repeated computation --

        F = [1,0,0, \
             0,1,0, \
             0,0,1]

        assert ndm == 2 or ndm == 3, "dimensions should be 2 or 3"
        if ndm == 2:
            F = mat3.copyMat2ToMat3(Fn)
        else:
            F = mat3.copyMat3(Fn)

        C = mat3.mults(F, F)
        detC = mat3.det(C)
        #if detC < 1e-10:
        #    raise NegativeJacobianError()
        Cinv = mat3.inv(C, detA=detC)
        defVol = 0.5 * _perform_log(detC)
        p = self.l* defVol
        trace = mat3.trace(C)
        coef = p - self.mu
        S = mat3.scalarMult(coef, Cinv)
        S = mat3.add(S, mat3.scaleMat(self.mu))

        coef = self.mu - p

        # have to repeat awful summing scheme!!
        M = [0,]*81;
        ijkl = 0
        ij = 0
        for i in range(3):
            for j in range(3):
                kl = 0
                for k in range(3):
                    for l in range(3):
                        M[ijkl] = self.l*Cinv[ij]*Cinv[kl] \
                                + coef*(Cinv[3*i+k]*Cinv[3*j+l] \
                                +       Cinv[3*i+l]*Cinv[3*j+k])
                        ijkl = ijkl + 1
                        kl = kl + 1
            ij = ij + 1

        ijkl = 0

        tangent = [0,]*81
        # will be very slow, but can improve with binding
        for i in range(ndf):
            for j in range(ndm):
                for k in range(ndf):
                    for l in range(ndm):
                        for n in range(3):
                            for m in range(3):
                                tangent[ijkl] = tangent[ijkl] + \
                                        F[3*i+m]*M[9*(3*j+m)+(3*l+n)]*F[3*k+n]

                        if i == k:
                            tangent[ijkl] = tangent[ijkl] + S[3*j+l];

                        ijkl = ijkl + 1

        return tuple(tangent)

    # meta-methods
    def __init__(self, rho, E, nu, **kwds):
        self.rho = rho
        self.E = E
        self.nu = nu
        self.l = nu * E / (1+nu) / (1-2*nu)
        self.mu = E / 2 / (1+nu)
        return
