# -*- coding: utf8 -*-
"""Burgers operator."""

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
import numpy.linalg as la

from hedge.models import HyperbolicOperator
import hedge.data
from hedge.tools.second_order import CentralSecondDerivative




class BurgersOperator(HyperbolicOperator):
    def __init__(self, dimensions, viscosity=None, 
            viscosity_scheme=CentralSecondDerivative()):
        # yes, you read that right--no BCs, 1D only.
        # (well--you can run the operator on a 2D grid. If you must.)
        self.dimensions = dimensions
        self.viscosity = viscosity
        self.viscosity_scheme = viscosity_scheme

    def op_template(self, with_sensor):
        from hedge.optemplate import \
                Field, \
                BoundaryPair, \
                get_flux_operator, \
                make_minv_stiffness_t, \
                make_nabla, \
                InverseMassOperator, \
                ElementwiseMaxOperator, \
                ScalarParameter

        u = Field("u")

        # boundary conditions -------------------------------------------------
        minv_st = make_minv_stiffness_t(self.dimensions)
        nabla = make_nabla(self.dimensions)
        m_inv = InverseMassOperator()

        def flux(u):
            return u**2/2

        emax_u = ElementwiseMaxOperator()(u**2)**0.5
        from hedge.flux.tools import make_lax_friedrichs_flux
        from pytools.obj_array import make_obj_array
        lf_flux = make_lax_friedrichs_flux(
                emax_u, 
                make_obj_array([u]), 
                [make_obj_array([flux(u)])], 
                [], strong=False)

        from hedge.tools.second_order import SecondDerivativeTarget

        if self.viscosity is not None or with_sensor:
            # strong_form here allows LDG to reuse the value of grad u.
            grad_tgt = SecondDerivativeTarget(
                    self.dimensions, strong_form=True,
                    operand=u)

            self.viscosity_scheme.grad(grad_tgt, bc_getter=None,
                    dirichlet_tags=[], neumann_tags=[])

            div_tgt = SecondDerivativeTarget(
                    self.dimensions, strong_form=False,
                    operand=grad_tgt.minv_all)

            self.viscosity_scheme.div(div_tgt,
                    bc_getter=None,
                    dirichlet_tags=[], neumann_tags=[])

            viscosity_coeff = 0
            if with_sensor:
                viscosity_coeff += Field("sensor")

            if isinstance(self.viscosity, float):
                viscosity_coeff += self.viscosity
            if self.viscosity is None:
                pass
            else:
                raise TypeError("unsupported type of viscosity coefficient")

            viscosity_bit = Field("sensor")*div_tgt.minv_all
        else:
            viscosity_bit = 0

        return (minv_st[0](flux(u))) - m_inv(lf_flux[0]) \
                + viscosity_bit

    def bind(self, discr, sensor=None):
        compiled_op_template = discr.compile(
                self.op_template(with_sensor=sensor is not None))

        from hedge.mesh import check_bc_coverage
        check_bc_coverage(discr.mesh, [])

        def rhs(t, u):
            kwargs = {}
            if sensor is not None:
                kwargs["sensor"] = sensor(u)
            return compiled_op_template(u=u, **kwargs)

        return rhs

    def max_eigenvalue(self, t=None, fields=None, discr=None):
        return discr.nodewise_max(fields)
