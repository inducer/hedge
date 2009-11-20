# -*- coding: utf8 -*-
"""Canned operators for multivariable calculus."""

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
from hedge.models import Operator
from hedge.mesh import TAG_NONE




class GradientOperator(Operator):
    def __init__(self, dimensions):
        self.dimensions = dimensions

    def flux(self):
        from hedge.flux import make_normal, FluxScalarPlaceholder
        u = FluxScalarPlaceholder()

        normal = make_normal(self.dimensions)
        return u.int*normal - u.avg*normal

    def op_template(self):
        from hedge.mesh import TAG_ALL
        from hedge.optemplate import Field, BoundaryPair, \
                make_nabla, InverseMassOperator, get_flux_operator

        u = Field("u")
        bc = Field("bc")

        nabla = make_nabla(self.dimensions)
        flux_op = get_flux_operator(self.flux())

        return nabla*u - InverseMassOperator()*(
                flux_op * u +
                flux_op * BoundaryPair(u, bc, TAG_ALL)
                )

    def bind(self, discr):
        compiled_op_template = discr.compile(self.op_template())

        def op(u):
            from hedge.mesh import TAG_ALL

            return compiled_op_template(u=u,
                    bc=discr.boundarize_volume_field(u, TAG_ALL))

        return op




class DivergenceOperator(Operator):
    def __init__(self, dimensions, subset=None):
        self.dimensions = dimensions

        if subset is None:
            self.subset = dimensions * [True,]
        else:
            # chop off any extra dimensions
            self.subset = subset[:dimensions]

        from hedge.tools import count_subset
        self.arg_count = count_subset(self.subset)

    def flux(self):
        from hedge.flux import make_normal, FluxVectorPlaceholder

        v = FluxVectorPlaceholder(self.arg_count)

        normal = make_normal(self.dimensions)

        flux = 0
        idx = 0

        for i, i_enabled in enumerate(self.subset):
            if i_enabled and i < self.dimensions:
                flux += (v.int-v.avg)[idx]*normal[i]
                idx += 1

        return flux

    def op_template(self):
        from hedge.mesh import TAG_ALL
        from hedge.optemplate import make_vector_field, BoundaryPair, \
                get_flux_operator, make_nabla, InverseMassOperator

        nabla = make_nabla(self.dimensions)
        m_inv = InverseMassOperator()

        v = make_vector_field("v", self.arg_count)
        bc = make_vector_field("bc", self.arg_count)

        local_op_result = 0
        idx = 0
        for i, i_enabled in enumerate(self.subset):
            if i_enabled and i < self.dimensions:
                local_op_result += nabla[i]*v[idx]
                idx += 1

        flux_op = get_flux_operator(self.flux())

        return local_op_result - m_inv*(
                flux_op * v +
                flux_op * BoundaryPair(v, bc, TAG_ALL))

    def bind(self, discr):
        compiled_op_template = discr.compile(self.op_template())

        def op(v):
            from hedge.mesh import TAG_ALL
            return compiled_op_template(v=v,
                    bc=discr.boundarize_volume_field(v, TAG_ALL))

        return op




# second derivative targets ---------------------------------------------------
class SecondDerivativeTarget(object):
    def __init__(self, dimensions, strong_form,
            operand, unflux_operand=None,
            lower_order_operand=None):
        self.dimensions = dimensions
        self.operand = operand
        self.unflux_operand = unflux_operand
        self.lower_order_operand = lower_order_operand

        self.strong_form = strong_form
        if strong_form:
            self.strong_neg = -1
        else:
            self.strong_neg = 1

        self.local_derivatives = 0
        self.inner_fluxes = 0
        self.boundary_fluxes = 0

    def add_local_derivatives(self, expr):
        self.local_derivatives = self.local_derivatives \
                + (-self.strong_neg)*expr

    def _local_nabla(self):
        if self.strong_form:
            from hedge.optemplate import make_stiffness
            return make_stiffness(self.dimensions)
        else:
            from hedge.optemplate import make_stiffness_t
            return make_stiffness_t(self.dimensions)

    def add_grad(self, operand=None):
        nabla = self._local_nabla()

        if operand is None:
            operand = self.operand

        from pytools.obj_array import make_obj_array
        self.add_local_derivatives(make_obj_array(
            [nabla[i](self.operand) for i in range(self.dimensions)]))

    def add_div(self, operand=None):
        nabla = self._local_nabla()

        if operand is None:
            operand = self.operand

        if len(operand) != self.dimensions:
            raise ValueError("operand of divergence must have %d dimensions"
                    % self.dimensions)

        from pytools.obj_array import make_obj_array
        self.add_local_derivatives(
                sum(nabla[i](self.operand[i]) for i in range(self.dimensions)))

    def add_inner_fluxes(self, flux, expr):
        from hedge.optemplate import get_flux_operator
        self.inner_fluxes = self.inner_fluxes \
                + get_flux_operator(self.strong_neg*flux)(expr)

    def add_boundary_flux(self, flux, volume_expr, bdry_expr, tag):
        from hedge.optemplate import BoundaryPair, get_flux_operator
        self.boundary_fluxes = self.boundary_fluxes + \
                get_flux_operator(self.strong_neg*flux)(BoundaryPair(
                        volume_expr, bdry_expr, tag))

    @property
    def fluxes(self):
        return self.inner_fluxes + self.boundary_fluxes

    @property
    def all(self):
        return self.local_derivatives + self.fluxes




# second derivative schemes ---------------------------------------------------
class SecondDerivativeBase(object):
    pass




class LDGSecondDerivative(SecondDerivativeBase):
    def beta(self, tgt):
        return numpy.array([0.5]*tgt.dimensions, dtype=numpy.float64)

    def first_grad(self, tgt,
            dirichlet_tags_and_bcs=[],
            neumann_tags_and_bcs=[]):

        from numpy import dot
        from hedge.flux import FluxScalarPlaceholder, make_normal
        normal = make_normal(tgt.dimensions)

        u = FluxScalarPlaceholder()

        flux = normal*(u.avg - (u.int-u.ext)*dot(normal, self.beta(tgt)))

        if tgt.strong_form:
            flux = u.int*normal - flux

        tgt.add_grad()
        tgt.add_inner_fluxes(flux, tgt.operand)

        for tag, bc in dirichlet_tags_and_bcs:
            # FIXME: Adjust for strong form
            tgt.add_boundary_flux(normal * u.ext, tgt.operand, bc, tag)

        for tag, bc in neumann_tags_and_bcs:
            # FIXME: Adjust for strong form
            tgt.add_boundary_flux(normal * u.int, tgt.operand, 0, tag)

    def second_div(self, tgt, 
            dirichlet_tags_and_bcs=[],
            neumann_tags_and_bcs=[]):

        from numpy import dot
        from hedge.optemplate import make_common_subexpression as cse
        from hedge.flux import FluxVectorPlaceholder, make_normal, PenaltyTerm
        normal = make_normal(tgt.dimensions)

        vec = FluxVectorPlaceholder(1+tgt.dimensions)
        u = vec[0]
        v = vec[1:]

        stab_term = cse(10 * PenaltyTerm() * (u.int - u.ext), "stab")
        flux = dot(v.avg + cse(dot(v.int - v.ext, normal), "jump_v")*self.beta(tgt),
            normal) - stab_term

        if tgt.strong_form:
            flux = dot(v.int, normal) - flux

        from pytools.obj_array import join_fields
        op_w = join_fields(tgt.lower_order_operand, tgt.operand)

        tgt.add_div()
        tgt.add_inner_fluxes(flux, op_w)

        for tag, bc in dirichlet_tags_and_bcs:
            tgt.add_boundary_flux(dot(v.int, normal) - stab_term,
                    op_w, 0, tag)

        from hedge.optemplate import make_normal as make_op_normal

        from pytools.obj_array import make_obj_array
        loc_bc_vec = make_obj_array([0]*(tgt.dimensions+1))

        for tag, bc in neumann_tags_and_bcs:
            # FIXME add post-treatment
            # FIXME vector BC may not be like this in CNS
            neu_bc_w = join_fields(0, make_op_normal(tag, tgt.dimensions)*bc)
            tgt.add_boundary_flux(dot(normal, v.ext), loc_bc_vec, neu_bc_w, tag)
