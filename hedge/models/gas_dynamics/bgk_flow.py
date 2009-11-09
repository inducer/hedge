# -*- coding: utf8 -*-
"""Operator for BGK-approximate Boltzmann moments flow."""

from __future__ import division

__copyright__ = "Copyright (C) 2009 Andreas Kloeckner"

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



import numpy
from pytools import memoize_method
from hedge.models import HyperbolicOperator
from pymbolic.mapper.stringifier import PREC_NONE
from hedge.optemplate import StringifyMapper





class MaximaStringifyMapper(StringifyMapper):
    def map_subscript(self, expr, prec):
        from pymbolic.primitives import Variable
        assert isinstance(expr.aggregate, Variable)
        assert isinstance(expr.index, int)

        return "%s%d" % (expr.aggregate.name, expr.index)

    def map_operator_binding(self, expr, prec):
        from hedge.optemplate import DifferentiationOperator
        AXES = ["x", "y", "z"]
        if isinstance(expr.op, DifferentiationOperator):
            return "diff(%s, %s)" % (
                    self.rec(expr.field, PREC_NONE),
                    AXES[expr.op.xyz_axis])
        else:
            return StringifyMapper.map_operator_binding(expr, prec)

class ShortMaximaStringifyMapper(MaximaStringifyMapper):
    def map_numpy_array(self, expr, prec):
        return "[%s]" % (
                ", ".join(
                    self.rec(child, PREC_NONE)
                    for child in expr))

class LongMaximaStringifyMapper(MaximaStringifyMapper):
    def map_numpy_array(self, expr, prec):
        return "[\n%s\n]" % (
                ",\n".join(
                    self.rec(child, PREC_NONE)
                    for child in expr))




class BGKFlowOperator(HyperbolicOperator):
    """A discretization of (moments) of the Boltzmann equation,
    following [1].

    [1] J. Tölke, M. Krafczyk, M. Schulz, and E. Rank,
    "Discretization of the Boltzmann equation in velocity space
    using a Galerkin approach,” Computer Physics Communications,
    2000, pp. 91-99.

    Vector order is as in [1], except that the directional
    number of directional moments (and hence the index of
    what is a4 in [1]) changes.
    """

    def __init__(self, bc, dimensions, temperature=293.15, tau=1e-3):
        """
        :param temperature: is the temperature in Kelvin.
        """

        self.bc = bc

        self.dimensions = dimensions
        self.temperature = temperature
        r = 8.314472 # J / (KG*mol) (??)
        self.sqrt_rt = numpy.sqrt(r*temperature)
        self.tau = tau

        # FIXME
        self.sqrt_rt = 1
        self.tau = tau

    def flux(self):
        from hedge.flux import FluxVectorPlaceholder, make_normal

        d = self.dimensions
        w = FluxVectorPlaceholder(2*d+2)
        normal = make_normal(d)

        r = self.sqrt_rt

        from hedge.tools import join_fields, make_obj_array
        sqrt = numpy.sqrt

        if False:
            # Tim's pure-upwind flow

            nx = normal[0]
            ny = normal[1]

            v2 = make_obj_array([
                1/sqrt(6), -nx/sqrt(2), -ny/sqrt(2), nx*ny*sqrt(2/3),
                nx**2/sqrt(3), ny**2/sqrt(3)])

            v4 = make_obj_array([
                0, -ny/sqrt(2), nx/sqrt(2), -(nx**2-ny**2)/sqrt(2), 
                nx*ny, -nx*ny])

            return 0.5*self.sqrt_rt*numpy.dot(
                    sqrt(3)*numpy.outer(v2, v2) + numpy.outer(v4, v4),
                    w.ext - w.int)

        return join_fields(
                -((2*sqrt(3)-2*sqrt(3)*normal[0]**2)*w[5].int*r
                  +2*sqrt(3)*normal[0]**2*w[4].int*r
                  +2*sqrt(2)*sqrt(3)*normal[0]*normal[1]*w[3].int*r
                  -3*sqrt(2)*normal[1]*w[2].int*r-3*sqrt(2)*normal[0]*w[1].int*r
                  +sqrt(2)*sqrt(3)*w[0].int*r+(2*sqrt(3)*normal[0]**2-2*sqrt(3))*w[5].ext*r
                  -2*sqrt(3)*normal[0]**2*w[4].ext*r
                  -2*sqrt(2)*sqrt(3)*normal[0]*normal[1]*w[3].ext*r
                  +3*sqrt(2)*normal[1]*w[2].ext*r+3*sqrt(2)*normal[0]*w[1].ext*r
                  -sqrt(2)*sqrt(3)*w[0].ext*r)
                  /(6*sqrt(2)),
                 -(-2*normal[0]*w[4].int*r-sqrt(2)*normal[1]*w[3].int*r
                                           +(sqrt(2)*sqrt(3)-sqrt(2))
                                            *normal[0]*normal[1]*w[2].int*r
                                           +((sqrt(2)*sqrt(3)-sqrt(2))*normal[0]**2+sqrt(2))
                                            *w[1].int*r-sqrt(2)*normal[0]*w[0].int*r
                                           +2*normal[0]*w[4].ext*r
                                           +sqrt(2)*normal[1]*w[3].ext*r
                                           +(sqrt(2)-sqrt(2)*sqrt(3))
                                            *normal[0]*normal[1]*w[2].ext*r
                                           +((sqrt(2)-sqrt(2)*sqrt(3))*normal[0]**2-sqrt(2))
                                            *w[1].ext*r+sqrt(2)*normal[0]*w[0].ext*r)
                  /(2*sqrt(2)),
                 -(-2*normal[1]*w[5].int*r-sqrt(2)*normal[0]*w[3].int*r
                                           +((sqrt(2)-sqrt(2)*sqrt(3))*normal[0]**2
                                            +sqrt(2)*sqrt(3))
                                            *w[2].int*r
                                           +(sqrt(2)*sqrt(3)-sqrt(2))
                                            *normal[0]*normal[1]*w[1].int*r
                                           -sqrt(2)*normal[1]*w[0].int*r
                                           +2*normal[1]*w[5].ext*r
                                           +sqrt(2)*normal[0]*w[3].ext*r
                                           +((sqrt(2)*sqrt(3)-sqrt(2))*normal[0]**2
                                            -sqrt(2)*sqrt(3))
                                            *w[2].ext*r
                                           +(sqrt(2)-sqrt(2)*sqrt(3))
                                            *normal[0]*normal[1]*w[1].ext*r
                                           +sqrt(2)*normal[1]*w[0].ext*r)
                  /(2*sqrt(2)),
                 -(((12-4*sqrt(3))*normal[0]**3+(4*sqrt(3)-6)*normal[0])*normal[1]*w[5].int*r
                  +((4*sqrt(3)-12)*normal[0]**3+6*normal[0])*normal[1]*w[4].int*r
                  +((12*sqrt(2)-4*sqrt(2)*sqrt(3))*normal[0]**4
                   +(4*sqrt(2)*sqrt(3)-12*sqrt(2))*normal[0]**2+3*sqrt(2))
                   *w[3].int*r-3*sqrt(2)*normal[0]*w[2].int*r
                  -3*sqrt(2)*normal[1]*w[1].int*r
                  +2*sqrt(2)*sqrt(3)*normal[0]*normal[1]*w[0].int*r
                  +((4*sqrt(3)-12)*normal[0]**3+(6-4*sqrt(3))*normal[0])*normal[1]*w[5].ext*r
                  +((12-4*sqrt(3))*normal[0]**3-6*normal[0])*normal[1]*w[4].ext*r
                  +((4*sqrt(2)*sqrt(3)-12*sqrt(2))*normal[0]**4
                   +(12*sqrt(2)-4*sqrt(2)*sqrt(3))*normal[0]**2-3*sqrt(2))
                   *w[3].ext*r+3*sqrt(2)*normal[0]*w[2].ext*r
                  +3*sqrt(2)*normal[1]*w[1].ext*r
                  -2*sqrt(2)*sqrt(3)*normal[0]*normal[1]*w[0].ext*r)
                  /(6*sqrt(2)),
                 -(((6-2*sqrt(3))*normal[0]**4+(2*sqrt(3)-6)*normal[0]**2)*w[5].int*r
                  +((2*sqrt(3)-6)*normal[0]**4+6*normal[0]**2)*w[4].int*r
                  +((2*sqrt(2)*sqrt(3)-6*sqrt(2))*normal[0]**3+3*sqrt(2)*normal[0])
                   *normal[1]*w[3].int*r-3*sqrt(2)*normal[0]*w[1].int*r
                  +sqrt(2)*sqrt(3)*normal[0]**2*w[0].int*r
                  +((2*sqrt(3)-6)*normal[0]**4+(6-2*sqrt(3))*normal[0]**2)*w[5].ext*r
                  +((6-2*sqrt(3))*normal[0]**4-6*normal[0]**2)*w[4].ext*r
                  +((6*sqrt(2)-2*sqrt(2)*sqrt(3))*normal[0]**3-3*sqrt(2)*normal[0])
                   *normal[1]*w[3].ext*r+3*sqrt(2)*normal[0]*w[1].ext*r
                  -sqrt(2)*sqrt(3)*normal[0]**2*w[0].ext*r)
                  /6,
                 (((6-2*sqrt(3))*normal[0]**4+(4*sqrt(3)-6)*normal[0]**2-2*sqrt(3))*w[5].int*r
                  +((2*sqrt(3)-6)*normal[0]**4+(6-2*sqrt(3))*normal[0]**2)*w[4].int*r
                  +((2*sqrt(2)*sqrt(3)-6*sqrt(2))*normal[0]**3
                   +(3*sqrt(2)-2*sqrt(2)*sqrt(3))*normal[0])
                   *normal[1]*w[3].int*r+3*sqrt(2)*normal[1]*w[2].int*r
                  +(sqrt(2)*sqrt(3)*normal[0]**2-sqrt(2)*sqrt(3))*w[0].int*r
                  +((2*sqrt(3)-6)*normal[0]**4+(6-4*sqrt(3))*normal[0]**2+2*sqrt(3))*w[5].ext*r
                  +((6-2*sqrt(3))*normal[0]**4+(2*sqrt(3)-6)*normal[0]**2)*w[4].ext*r
                  +((6*sqrt(2)-2*sqrt(2)*sqrt(3))*normal[0]**3
                   +(2*sqrt(2)*sqrt(3)-3*sqrt(2))*normal[0])
                   *normal[1]*w[3].ext*r-3*sqrt(2)*normal[1]*w[2].ext*r
                  +(sqrt(2)*sqrt(3)-sqrt(2)*sqrt(3)*normal[0]**2)*w[0].ext*r)
                  /6
                )

    @memoize_method
    def field_placeholders(self):
        from hedge.optemplate import make_vector_field

        d = self.dimensions

        w = make_vector_field("w", 2*d+2)

        from pytools import Record
        class BGKField(Record):
            pass

        return BGKField(
            # names follow [1], with vectors named after their
            # first entry
            a1=w[0],
            a2=w[1:1+d],
            a4=w[1+d],
            a5=w[2+d:],
            w=w)

    def local_op(self, sym_constants=False):
        d = self.dimensions

        if sym_constants:
            from pymbolic.primitives import Variable
            sqrt_rt = Variable("sqrt_rt")
            sqrt_2 = Variable("sqrt(2)")
            tau = Variable("tau")
        else:
            sqrt_rt = self.sqrt_rt
            sqrt_2 = numpy.sqrt(2)
            tau = self.tau

        from hedge.optemplate import make_nabla
        nabla = make_nabla(d)
        f = self.field_placeholders()

        from hedge.tools import join_fields
        return join_fields(
                sqrt_rt*numpy.dot(nabla, f.a2),

                # FIXME this likely not the right nD generalization
                sqrt_rt*(
                    nabla*f.a1+nabla[::-1]*f.a4
                    + sqrt_2*nabla*f.a5),

                # FIXME this likely not the right nD generalization
                sqrt_rt*numpy.dot(
                    nabla[::-1], f.a2)
                + 1/tau*(f.a4-numpy.product(f.a2)/f.a1),

                sqrt_2*sqrt_rt*(nabla*f.a2)
                + 1/tau*(f.a5-f.a2**2/(sqrt_2*f.a1)))

    def op_template(self):
        from hedge.optemplate import \
                BoundaryPair, \
                get_flux_operator, \
                InverseMassOperator, \
                BoundarizeOperator, \
                make_vector_field

        f = self.field_placeholders()
        d = self.dimensions

        bc_w = make_vector_field("bc_w", 2*d+2)

        flux_op = get_flux_operator(self.flux())

        from hedge.mesh import TAG_ALL
        return - self.local_op() + \
                InverseMassOperator() * (
                    flux_op*f.w
                    + flux_op * BoundaryPair(f.w, bc_w, TAG_ALL))

    def bind(self, discr):
        compiled_op_template = discr.compile(self.op_template())

        from hedge.mesh import TAG_ALL

        def rhs(t, w):
            return compiled_op_template(
                    w=w,
                    bc_w=self.bc.boundary_interpolant(t, discr, TAG_ALL)
                    )

        return rhs

    def write_maxima_expressions(self):
        smapper = ShortMaximaStringifyMapper()
        lmapper = LongMaximaStringifyMapper()

        op = self.local_op(sym_constants=True)
        from hedge.optemplate import OperatorBinder
        op = OperatorBinder()(op)

        return ("d:%d;\n\n"
                "vars: %s;\n\n"
                "depends(vars, coords);\n\n"
                "bgk_neg_rhss: %s;"
                ) % (
                        self.dimensions,
                        smapper(self.field_placeholders().w,
                            PREC_NONE),
                        lmapper(op, PREC_NONE))

    def max_eigenvalue(self, t, fields=None, discr=None):
        return numpy.sqrt(3)*self.sqrt_rt

    def from_primitive(self, x_vec, rho, u=None, sigma=None):
        def get_var(var):
            if var is None:
                return numpy.zeros_like(x_vec[0])
            else:
                return var

        def get_index(var, idx):
            if var is None:
                return numpy.zeros_like(x_vec[0])
            else:
                return var[idx]

        from hedge.tools import make_obj_array
        d = self.dimensions
        rho = get_var(rho)
        u = make_obj_array([get_index(u, i) for i in range(d)])
        # order 11,22,12
        sigma = make_obj_array([
            get_index(sigma, i) for i in range(d**2+d//2)])

        from hedge.tools import join_fields
        return join_fields(
                rho,
                u[0]*rho/self.sqrt_rt,
                u[1]*rho/self.sqrt_rt,
                (u[0]*u[1]*rho-sigma[2])/self.sqrt_rt**2,
                (u[0]**2*rho-sigma[0])/numpy.sqrt(2),
                (u[1]**2*rho-sigma[1])/numpy.sqrt(2),
                )

    def to_primitive(self, x_vec, a):
        def get_index(var, idx):
            if var is None:
                return numpy.zeros_like(x_vec[0])
            else:
                return var[idx]

        from hedge.tools import make_obj_array
        d = self.dimensions

        # initial 0 to make indices same as [1]
        a = make_obj_array([0]+[
            get_index(a, i) for i in range(2*d+2)])

        from hedge.tools import join_fields
        return join_fields(
                # rho
                a[1],

                # u
                a[2]*self.sqrt_rt/a[1],
                a[3]*self.sqrt_rt/a[1],

                -self.sqrt_rt**2*(
                    numpy.sqrt(2)*a[5]-a[2]**2/a[1]),
                -self.sqrt_rt**2*(
                    numpy.sqrt(2)*a[6]-a[3]**2/a[1]),
                -self.sqrt_rt**2*(a[4]-a[2]*a[3]/a[1]),
                )
