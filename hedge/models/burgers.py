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




class BurgersOperator(HyperbolicOperator):
    def __init__(self, dimensions):
        # yes, you read that right--no BCs, 1D only.
        self.dimensions = dimensions

    def op_template(self):
        from hedge.optemplate import \
                Field, \
                BoundaryPair, \
                get_flux_operator, \
                make_minv_stiffness_t, \
                make_nabla, \
                InverseMassOperator, \
                ElementwiseMaxOperator

        u = Field("u")

        # boundary conditions -------------------------------------------------
        minv_st = make_minv_stiffness_t(self.dimensions)
        nabla = make_nabla(self.dimensions)
        m_inv = InverseMassOperator()

        def flux(u):
            return u**2/2

        emax_u = ElementwiseMaxOperator()(u**2)**0.5
        from hedge.tools import make_lax_friedrichs_flux
        from pytools.obj_array import make_obj_array
        lf_flux = make_lax_friedrichs_flux(
                emax_u, 
                make_obj_array([u]), 
                [make_obj_array([flux(u)])], 
                [], strong=True)

        return -(nabla[0](flux(u))) + m_inv(lf_flux[0])

    def max_eigenvalue(self, t=None, fields=None, discr=None):
        return discr.nodewise_max(fields)

    def bind(self, discr):
        compiled_op_template = discr.compile(self.op_template())

        from hedge.mesh import check_bc_coverage
        check_bc_coverage(discr.mesh, [])

        def rhs(t, u):
            return compiled_op_template(u=u)

        return rhs
