# -*- coding: utf8 -*-
"""Lattice-Boltzmann operator."""

from __future__ import division

__copyright__ = "Copyright (C) 2011 Andreas Kloeckner"

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

import numpy as np
import numpy.linalg as la
from hedge.models import HyperbolicOperator
from pytools.obj_array import make_obj_array





class LBMMethodBase(object):
    def __len__(self):
        return len(self.direction_vectors)

    def find_opposites(self):
        self.opposites = np.zeros(len(self))

        for alpha in xrange(len(self)):
            if self.opposites[alpha]:
                continue

            found = False
            for alpha_2 in xrange(alpha, len(self)):
                if la.norm(
                        self.direction_vectors[alpha] 
                        + self.direction_vectors[alpha_2]) < 1e-12:
                    self.opposites[alpha] = alpha_2
                    self.opposites[alpha_2] = alpha
                    found = True

            if not found:
                raise RuntimeError(
                        "direction %s had no opposite" 
                        % self.direction_vectors[alpha])




class D2Q9LBMMethod(LBMMethodBase):
    def __init__(self):
        self.dimensions = 2

        alphas = np.arange(0, 9)
        thetas = (alphas-1)*np.pi/2
        thetas[5:9] += np.pi/4

        direction_vectors = np.vstack([
            np.cos(thetas), np.sin(thetas)]).T

        direction_vectors[0] *= 0
        direction_vectors[5:9] *= np.sqrt(2)

        direction_vectors[np.abs(direction_vectors) < 1e-12] = 0

        self.direction_vectors = direction_vectors

        self.weights = np.array([4/9] + [1/9]*4 + [1/36]*4)

        self.speed_of_sound = 1/np.sqrt(3)
        self.find_opposites()

    def f_equilibrium(self, rho, alpha, u):
        e_alpha = self.direction_vectors[alpha]
        c_s = self.speed_of_sound
        return self.weights[alpha]*rho*(
                1
                + np.dot(e_alpha, u)/c_s**2
                + 1/2*np.dot(e_alpha, u)**2/c_s**4
                - 1/2*np.dot(u, u)/c_s**2)




class LatticeBoltzmannOperator(HyperbolicOperator):
    def __init__(self, method, lbm_delta_t, nu, flux_type="upwind"):
        self.method = method
        self.lbm_delta_t = lbm_delta_t
        self.nu = nu

        self.flux_type = flux_type

    @property
    def tau(self):
        return (self.nu
                /
                (self.lbm_delta_t*self.method.speed_of_sound**2))

    def get_advection_flux(self, velocity):
        from hedge.flux import make_normal, FluxScalarPlaceholder
        from pymbolic.primitives import IfPositive

        u = FluxScalarPlaceholder(0)
        normal = make_normal(self.method.dimensions)

        if self.flux_type == "central":
            return u.avg*np.dot(normal, velocity)
        elif self.flux_type == "lf":
            return u.avg*np.dot(normal, velocity) \
                    + 0.5*la.norm(v)*(u.int - u.ext)
        elif self.flux_type == "upwind":
            return (np.dot(normal, velocity)*
                    IfPositive(np.dot(normal, velocity),
                        u.int, # outflow
                        u.ext, # inflow
                        ))
        else:
            raise ValueError, "invalid flux type"

    def get_advection_op(self, q, velocity):
        from hedge.optemplate import (
                BoundaryPair,
                get_flux_operator,
                make_stiffness_t,
                InverseMassOperator)

        stiff_t = make_stiffness_t(self.method.dimensions)

        flux_op = get_flux_operator(self.get_advection_flux(velocity))
        return InverseMassOperator()(
                np.dot(velocity, stiff_t*q) - flux_op(q))

    def f_bar(self):
        from hedge.optemplate import make_sym_vector
        return make_sym_vector("f_bar", len(self.method))

    def rho(self, f_bar):
        return sum(f_bar)

    def rho_u(self, f_bar):
        return sum(
                dv_i * field_i
                for dv_i, field_i in
                zip(self.method.direction_vectors, f_bar))

    def stream_rhs(self, f_bar):
        return make_obj_array([
            self.get_advection_op(f_bar_alpha, e_alpha)
            for e_alpha, f_bar_alpha in
            zip(self.method.direction_vectors, f_bar)])

    def collision_update(self, f_bar):
        from hedge.optemplate.primitives import make_common_subexpression as cse
        rho = cse(self.rho(f_bar), "rho")
        rho_u = self.rho_u(f_bar)
        u = cse(rho_u/rho, "u")

        f_eq_func = self.method.f_equilibrium
        f_eq = make_obj_array([
            f_eq_func(rho, alpha, u) for alpha in range(len(self.method))])

        return f_bar - 1/(self.tau+1/2)*(f_bar - f_eq)

    def bind_rhs(self, discr):
        compiled_op_template = discr.compile(
                self.stream_rhs(self.f_bar()))

        #from hedge.mesh import check_bc_coverage, TAG_ALL
        #check_bc_coverage(discr.mesh, [TAG_ALL])

        def rhs(t, f_bar):
            return compiled_op_template(f_bar=f_bar)

        return rhs

    def bind(self, discr, what):
        f_bar_sym = self.f_bar()

        from hedge.optemplate.mappers.type_inference import (
                type_info, NodalRepresentation)

        type_hints = dict(
                (f_bar_i, type_info.VolumeVector(NodalRepresentation()))
                for f_bar_i in f_bar_sym)

        compiled_op_template = discr.compile(what(f_bar_sym), type_hints=type_hints)

        def rhs(f_bar):
            return compiled_op_template(f_bar=f_bar)

        return rhs

    def max_eigenvalue(self, t=None, fields=None, discr=None):
        return max(
                la.norm(v) for v in self.method.direction_vectors)
