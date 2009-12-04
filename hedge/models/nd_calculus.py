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

        return nabla*u - InverseMassOperator()(
                flux_op(u) +
                flux_op(BoundaryPair(u, bc, TAG_ALL)))

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

        return local_op_result - m_inv(
                flux_op(v) +
                flux_op(BoundaryPair(v, bc, TAG_ALL)))

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
            operand, lower_order_operand=None,
            process_vector=None):
        self.dimensions = dimensions
        self.operand = operand
        self.lower_order_operand = lower_order_operand

        self.strong_form = strong_form
        if strong_form:
            self.strong_neg = -1
        else:
            self.strong_neg = 1

        self.local_derivatives = 0
        self.inner_fluxes = 0
        self.boundary_fluxes = 0

        if process_vector is None:
            self.process_vector = lambda x: x
        else:
            self.process_vector = process_vector

    def add_local_derivatives(self, expr):
        self.local_derivatives = self.local_derivatives \
                + (-self.strong_neg)*expr

    def vec_times(self, vec, operand):
        from pytools.obj_array import is_obj_array

        if is_obj_array(operand):
            if len(operand) != self.dimensions:
                raise ValueError("operand of vec_times must have %d dimensions"
                        % self.dimensions)

            return numpy.dot(vec, operand)
        else:
            return vec*operand

    def normal_times_flux(self, flux):
        from hedge.flux import make_normal
        return self.vec_times(make_normal(self.dimensions), flux)

    def apply_diff(self, nabla, operand):
        from pytools.obj_array import make_obj_array, is_obj_array
        if is_obj_array(operand):
            if len(operand) != self.dimensions:
                raise ValueError("operand of apply_diff must have %d dimensions"
                        % self.dimensions)

            from pytools.obj_array import make_obj_array
            return sum(nabla[i](operand[i]) for i in range(self.dimensions))
        else:
            return make_obj_array(
                [nabla[i](operand) for i in range(self.dimensions)])

    def _local_nabla(self):
        if self.strong_form:
            from hedge.optemplate import make_stiffness
            return make_stiffness(self.dimensions)
        else:
            from hedge.optemplate import make_stiffness_t
            return make_stiffness_t(self.dimensions)

    def add_derivative(self, operand=None):
        if operand is None:
            operand = self.operand

        from pytools.obj_array import make_obj_array, is_obj_array
        self.add_local_derivatives(
                self.apply_diff(self._local_nabla(), operand))

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
    def __init__(self, beta_value=0.5, stab_coefficient=10):
        self.beta_value = beta_value
        self.stab_coefficient = stab_coefficient

    def beta(self, tgt):
        return numpy.array([self.beta_value]*tgt.dimensions, dtype=numpy.float64)

    def first_derivative(self, tgt,
            dirichlet_tags_and_bcs=[],
            neumann_tags_and_bcs=[]):

        from numpy import dot
        from hedge.optemplate import make_common_subexpression as cse
        from hedge.flux import FluxScalarPlaceholder, make_normal
        normal = make_normal(tgt.dimensions)

        n_times = tgt.normal_times_flux
        v_times = tgt.vec_times

        if tgt.strong_form:
            def adjust_flux(f):
                return n_times(u.int) - f
        else:
            def adjust_flux(f):
                return f

        u = FluxScalarPlaceholder()

        flux = n_times(
                cse(u.avg, "u_avg") 
                - v_times(self.beta(tgt), n_times(u.int-u.ext)))

        tgt.add_derivative()
        tgt.add_inner_fluxes(adjust_flux(flux), tgt.operand)

        for tag, bc in dirichlet_tags_and_bcs:
            tgt.add_boundary_flux(
                    adjust_flux(n_times(u.ext)),
                    tgt.operand, bc, tag)

        for tag, bc in neumann_tags_and_bcs:
            tgt.add_boundary_flux(
                    adjust_flux(n_times(u.int)), 
                    tgt.operand, 0, tag)

    def second_derivative(self, tgt, 
            dirichlet_tags_and_bcs=[],
            neumann_tags_and_bcs=[]):

        from numpy import dot
        from hedge.optemplate import make_common_subexpression as cse
        from hedge.flux import FluxVectorPlaceholder, make_normal, PenaltyTerm
        normal = make_normal(tgt.dimensions)

        n_times = tgt.normal_times_flux
        v_times = tgt.vec_times

        if tgt.strong_form:
            def adjust_flux(f):
                return tgt.normal_times_flux(v.int) - f
        else:
            def adjust_flux(f):
                return f

        from pytools.obj_array import gen_len, gen_slice, make_obj_array

        vec = FluxVectorPlaceholder(1+tgt.dimensions)
        low_order_len = gen_len(tgt.lower_order_operand)
        u = gen_slice(vec, slice(low_order_len))
        v = gen_slice(vec, slice(low_order_len, None))

        stab_term = cse(
                self.stab_coefficient * PenaltyTerm() * (u.int - u.ext), "stab")
        flux = n_times(v.avg 
                + v_times(self.beta(tgt), cse(n_times(v.int - v.ext), "jump_v"))
                - n_times(stab_term))

        processed_v = cse(tgt.process_vector(tgt.operand))
        from pytools.obj_array import join_fields
        op_w = join_fields(tgt.lower_order_operand, processed_v)

        tgt.add_derivative(processed_v)
        tgt.add_inner_fluxes(adjust_flux(flux), op_w)

        for tag, bc in dirichlet_tags_and_bcs:
            dir_bc_w = join_fields(bc, [0]*tgt.dimensions)
            tgt.add_boundary_flux(
                    adjust_flux(n_times(v.int- n_times(stab_term))),
                    op_w, dir_bc_w, tag)

        from hedge.optemplate import make_normal as make_op_normal
        loc_bc_vec = make_obj_array([0]*(tgt.dimensions+1))

        for tag, bc in neumann_tags_and_bcs:
            # FIXME vector BC may not be like this in CNS
            neu_bc_w = join_fields(0, 
                    tgt.process_vector(
                        make_op_normal(tag, tgt.dimensions)*bc, 
                        tag=tag))

            tgt.add_boundary_flux(
                    adjust_flux(n_times(v.ext)),
                    loc_bc_vec, neu_bc_w, tag)




class StabilizedCentralSecondDerivative(LDGSecondDerivative):
    def __init__(self):
        LDGSecondDerivative.__init__(self, 0)




class CentralSecondDerivative(LDGSecondDerivative):
    def __init__(self):
        LDGSecondDerivative.__init__(self, 0, 0)




class IPDGSecondDerivative(SecondDerivativeBase):
    def __init__(self, stab_coefficient=10):
        self.stab_coefficient = stab_coefficient

    def first_derivative(self, tgt,
            dirichlet_tags_and_bcs=[],
            neumann_tags_and_bcs=[]):

        from numpy import dot
        from hedge.optemplate import make_common_subexpression as cse
        from hedge.flux import FluxScalarPlaceholder, make_normal
        normal = make_normal(tgt.dimensions)

        n_times = tgt.normal_times_flux

        if tgt.strong_form:
            def adjust_flux(f):
                return n_times(u.int) - f
        else:
            def adjust_flux(f):
                return f

        u = FluxScalarPlaceholder()

        flux = n_times(cse(u.avg, "u_avg"))

        tgt.add_derivative()
        tgt.add_inner_fluxes(adjust_flux(flux), tgt.operand)

        for tag, bc in dirichlet_tags_and_bcs:
            tgt.add_boundary_flux(
                    adjust_flux(n_times(u.ext)),
                    tgt.operand, bc, tag)

        for tag, bc in neumann_tags_and_bcs:
            tgt.add_boundary_flux(
                    adjust_flux(n_times(u.int)), 
                    tgt.operand, 0, tag)

    def second_derivative(self, tgt, 
            dirichlet_tags_and_bcs=[],
            neumann_tags_and_bcs=[]):

        from numpy import dot
        from hedge.optemplate import make_common_subexpression as cse
        from hedge.flux import FluxVectorPlaceholder, make_normal, PenaltyTerm
        normal = make_normal(tgt.dimensions)

        n_times = tgt.normal_times_flux

        if tgt.strong_form:
            def adjust_flux(f):
                return tgt.normal_times_flux(v.int) - f
        else:
            def adjust_flux(f):
                return f

        from pytools.obj_array import gen_len, gen_slice, make_obj_array

        vec = FluxVectorPlaceholder(1+tgt.dimensions)
        low_order_len = gen_len(tgt.lower_order_operand)
        u = gen_slice(vec, slice(low_order_len))
        v = gen_slice(vec, slice(low_order_len, None))

        stab_term = cse(
                self.stab_coefficient * PenaltyTerm() * (u.int - u.ext), "stab")
        flux = n_times(v.avg - n_times(stab_term))

        from hedge.optemplate import make_nabla
        processed_unflux_v = cse(tgt.process_vector(
            tgt.apply_diff(
                make_nabla(tgt.dimensions), 
                tgt.lower_order_operand)), "diff_v")

        from pytools.obj_array import join_fields
        op_w = join_fields(tgt.lower_order_operand, processed_unflux_v)

        processed_v = cse(tgt.process_vector(tgt.operand))
        tgt.add_derivative(processed_v)
        tgt.add_inner_fluxes(adjust_flux(flux), op_w)

        for tag, bc in dirichlet_tags_and_bcs:
            dir_bc_w = join_fields(bc, [0]*tgt.dimensions)
            tgt.add_boundary_flux(
                    adjust_flux(n_times(v.int- n_times(stab_term))),
                    op_w, dir_bc_w, tag)

        from hedge.optemplate import make_normal as make_op_normal
        loc_bc_vec = make_obj_array([0]*(tgt.dimensions+1))

        for tag, bc in neumann_tags_and_bcs:
            # FIXME vector BC may not be like this in CNS
            neu_bc_w = join_fields(0, 
                    tgt.process_vector(
                        make_op_normal(tag, tgt.dimensions)*bc, 
                        tag=tag))

            tgt.add_boundary_flux(
                    adjust_flux(n_times(v.ext)),
                    loc_bc_vec, neu_bc_w, tag)
