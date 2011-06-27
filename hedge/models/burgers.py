# -*- coding: utf8 -*-
"""Burgers operator."""

from __future__ import division

__copyright__ = "Copyright (C) 2009 Andreas Kloeckner"

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




from hedge.models import HyperbolicOperator
import numpy
from hedge.second_order import CentralSecondDerivative




class BurgersOperator(HyperbolicOperator):
    def __init__(self, dimensions, viscosity=None, 
            viscosity_scheme=CentralSecondDerivative()):
        # yes, you read that right--no BCs, 1D only.
        # (well--you can run the operator on a 2D grid. If you must.)
        self.dimensions = dimensions
        self.viscosity = viscosity
        self.viscosity_scheme = viscosity_scheme

    def characteristic_velocity_optemplate(self, field):
        from hedge.optemplate.operators import (
                ElementwiseMaxOperator)
        return ElementwiseMaxOperator()(field**2)**0.5

    def bind_characteristic_velocity(self, discr):
        from hedge.optemplate import Field
        compiled = discr.compile(
                self.characteristic_velocity_optemplate(
                    Field("u")))

        def do(u):
            return compiled(u=u)

        return do

    def op_template(self, with_sensor):
        from hedge.optemplate import (
                Field,
                make_stiffness_t,
                make_nabla,
                InverseMassOperator,
                ElementwiseMaxOperator,
                get_flux_operator)

        from hedge.optemplate.operators import (
                QuadratureGridUpsampler,
                QuadratureInteriorFacesGridUpsampler)

        to_quad = QuadratureGridUpsampler("quad")
        to_if_quad = QuadratureInteriorFacesGridUpsampler("quad")

        u = Field("u")
        u0 = Field("u0")

        # boundary conditions -------------------------------------------------
        minv_st = make_stiffness_t(self.dimensions)
        nabla = make_nabla(self.dimensions)
        m_inv = InverseMassOperator()

        def flux(u):
            return u**2/2
            #return u0*u

        emax_u = self.characteristic_velocity_optemplate(u)
        from hedge.flux.tools import make_lax_friedrichs_flux
        from pytools.obj_array import make_obj_array
        num_flux = make_lax_friedrichs_flux(
                #u0,
                to_if_quad(emax_u),
                make_obj_array([to_if_quad(u)]), 
                [make_obj_array([flux(to_if_quad(u))])], 
                [], strong=False)[0]

        from hedge.second_order import SecondDerivativeTarget

        if self.viscosity is not None or with_sensor:
            viscosity_coeff = 0
            if with_sensor:
                viscosity_coeff += Field("sensor")

            if isinstance(self.viscosity, float):
                viscosity_coeff += self.viscosity
            elif self.viscosity is None:
                pass
            else:
                raise TypeError("unsupported type of viscosity coefficient")

            # strong_form here allows IPDG to reuse the value of grad u.
            grad_tgt = SecondDerivativeTarget(
                    self.dimensions, strong_form=True,
                    operand=u)

            self.viscosity_scheme.grad(grad_tgt, bc_getter=None,
                    dirichlet_tags=[], neumann_tags=[])

            div_tgt = SecondDerivativeTarget(
                    self.dimensions, strong_form=False,
                    operand=viscosity_coeff*grad_tgt.minv_all)

            self.viscosity_scheme.div(div_tgt,
                    bc_getter=None,
                    dirichlet_tags=[], neumann_tags=[])

            viscosity_bit = div_tgt.minv_all
        else:
            viscosity_bit = 0

        return m_inv((minv_st[0](flux(to_quad(u)))) - num_flux) \
                + viscosity_bit

    def bind(self, discr, u0=1, sensor=None):
        compiled_op_template = discr.compile(
                self.op_template(with_sensor=sensor is not None))

        from hedge.mesh import check_bc_coverage
        check_bc_coverage(discr.mesh, [])

        def rhs(t, u):
            kwargs = {}
            if sensor is not None:
                kwargs["sensor"] = sensor(u)
            return compiled_op_template(u=u, u0=u0, **kwargs)

        return rhs

    def max_eigenvalue(self, t=None, fields=None, discr=None):
        return discr.nodewise_max(fields)
