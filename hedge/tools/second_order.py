# -*- coding: utf8 -*-
"""Schemes for second-order derivatives."""

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
import hedge.optemplate




# stabilization term generator ------------------------------------------------
class StabilizationTermGenerator(hedge.optemplate.IdentityMapper):
    def __init__(self, flux_args):
        hedge.optemplate.IdentityMapper.__init__(self)
        self.flux_args = flux_args
        self.flux_arg_lookup = dict(
                (flux_arg, i) for i, flux_arg in enumerate(flux_args))

    def get_flux_arg_idx(self, expr):
        try:
            return self.flux_arg_lookup[expr]
        except KeyError:
            flux_arg_idx = len(self.flux_args)
            self.flux_arg_lookup[expr] = flux_arg_idx
            self.flux_args.append(expr)
            return flux_arg_idx

    def map_operator_binding(self, expr):
        if isinstance(expr.op, hedge.optemplate.DiffOperatorBase):
            flux_arg_idx = self.get_flux_arg_idx(expr.field)

            from hedge.optemplate import \
                    WeakFormDiffOperatorBase, \
                    StrongFormDiffOperatorBase
            if isinstance(expr.op, WeakFormDiffOperatorBase):
                factor = -1
            elif isinstance(expr.op, StrongFormDiffOperatorBase):
                factor = 1
            else:
                raise RuntimeError("unknown type of differentiation "
                        "operator encountered by stab term generator")

            from hedge.flux import Normal, FluxScalarPlaceholder
            sph = FluxScalarPlaceholder(flux_arg_idx)
            return (factor
                    * Normal(expr.op.xyz_axis)
                    * (sph.int - sph.ext))

        elif isinstance(expr.op, hedge.optemplate.FluxOperatorBase):
            return 0
        elif isinstance(expr.op, hedge.optemplate.InverseMassOperator):
            return self.rec(expr.field)
        else:
            raise ValueError("stabilization term generator doesn't know "
                    "what to do with '%s'" % expr)

    def map_variable(self, expr):
        from hedge.flux import FieldComponent
        return FieldComponent(self.get_flux_arg_idx(expr), is_interior=True)

    def map_subscript(self, expr):
        from pymbolic.primitives import Variable
        assert isinstance(expr.aggregate, Variable)
        from hedge.flux import FieldComponent
        return FieldComponent(self.get_flux_arg_idx(expr), is_interior=True)



class NeumannBCGenerator(hedge.optemplate.IdentityMapper):
    def __init__(self, tag, bc):
        hedge.optemplate.IdentityMapper.__init__(self)
        self.tag = tag
        self.bc = bc

    def map_operator_binding(self, expr):
        if isinstance(expr.op, hedge.optemplate.DiffOperatorBase):
            from hedge.optemplate import \
                    WeakFormDiffOperatorBase, \
                    StrongFormDiffOperatorBase
            if isinstance(expr.op, WeakFormDiffOperatorBase):
                factor = -1
            elif isinstance(expr.op, StrongFormDiffOperatorBase):
                factor = 1
            else:
                raise RuntimeError("unknown type of differentiation "
                        "operator encountered by stab term generator")

            from hedge.optemplate import BoundaryNormalComponent
            return (self.bc * factor * 
                    BoundaryNormalComponent(self.tag, expr.op.xyz_axis))

        elif isinstance(expr.op, hedge.optemplate.FluxOperatorBase):
            return 0
        elif isinstance(expr.op, hedge.optemplate.InverseMassOperator):
            return self.rec(expr.field)
        else:
            raise ValueError("neumann normal direction generator doesn't know "
                    "what to do with '%s'" % expr)




class IPDGDerivativeGenerator(hedge.optemplate.IdentityMapper):
    def map_operator_binding(self, expr):
        if isinstance(expr.op, hedge.optemplate.DiffOperatorBase):
            from hedge.optemplate import \
                    WeakFormDiffOperatorBase, \
                    StrongFormDiffOperatorBase
            if isinstance(expr.op, WeakFormDiffOperatorBase):
                factor = -1
            elif isinstance(expr.op, StrongFormDiffOperatorBase):
                factor = 1
            else:
                raise RuntimeError("unknown type of differentiation "
                        "operator encountered by stab term generator")

            from hedge.optemplate import DifferentiationOperator
            return factor*DifferentiationOperator(expr.op.xyz_axis)(expr.field)

        elif isinstance(expr.op, hedge.optemplate.FluxOperatorBase):
            return 0
        elif isinstance(expr.op, hedge.optemplate.InverseMassOperator):
            return self.rec(expr.field)
        else:
            raise ValueError("IPDG derivative generator doesn't know "
                    "what to do with '%s'" % expr)




# second derivative target ----------------------------------------------------
class SecondDerivativeTarget(object):
    def __init__(self, dimensions, strong_form,
            operand, process_vector=None):
        self.dimensions = dimensions
        self.operand = operand

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

    @property
    def minv_all(self):
        from hedge.optemplate import make_common_subexpression as cse
        from hedge.optemplate import InverseMassOperator
        return (cse(InverseMassOperator()(self.local_derivatives), "grad_loc") 
                + cse(InverseMassOperator()(self.fluxes), "grad_flux"))




# second derivative schemes ---------------------------------------------------
class SecondDerivativeBase(object):
    def grad(self, tgt, bc_getter, dirichlet_tags, neumann_tags):
        """
        :param bc_getter: a function (tag, volume_expr) -> boundary expr.
          *volume_expr* will be None to query the Neumann condition.
        """
        from numpy import dot
        from hedge.optemplate import make_common_subexpression as cse

        n_times = tgt.normal_times_flux

        if tgt.strong_form:
            def adjust_flux(f):
                return n_times(u.int) - f
        else:
            def adjust_flux(f):
                return f

        from hedge.flux import FluxScalarPlaceholder
        u = FluxScalarPlaceholder()

        tgt.add_derivative()
        tgt.add_inner_fluxes(
                adjust_flux(self.grad_interior_flux(tgt, u)), 
                tgt.operand)

        for tag in dirichlet_tags:
            tgt.add_boundary_flux(
                    adjust_flux(n_times(u.ext)),
                    tgt.operand, bc_getter(tag, tgt.operand), tag)

        for tag in neumann_tags:
            tgt.add_boundary_flux(
                    adjust_flux(n_times(u.int)), 
                    tgt.operand, 0, tag)

    def add_div_bcs(self, tgt, bc_getter, dirichlet_tags, neumann_tags,
            stab_term_generator, stab_term, adjust_flux, flux_v, flux_arg_int,
            grad_flux_arg_count):
        from pytools.obj_array import make_obj_array, join_fields
        n_times = tgt.normal_times_flux

        for tag in dirichlet_tags:
            dir_bc_w = join_fields(
                    [0]*grad_flux_arg_count,
                    [bc_getter(tag, vol_expr) for vol_expr in 
                        stab_term_generator.flux_args[grad_flux_arg_count:]])
            tgt.add_boundary_flux(
                    adjust_flux(n_times(flux_v.int-stab_term)),
                    flux_arg_int, dir_bc_w, tag)

        from hedge.optemplate import make_normal as make_op_normal
        loc_bc_vec = make_obj_array([0]*len(stab_term_generator.flux_args))

        for tag in neumann_tags:
            neu_bc_w = join_fields(
                    NeumannBCGenerator(tag, bc_getter(tag, None))(tgt.operand),
                    [0]*len(flux_arg_int))

            tgt.add_boundary_flux(
                    adjust_flux(n_times(flux_v.ext)),
                    loc_bc_vec, neu_bc_w, tag)




class LDGSecondDerivative(SecondDerivativeBase):
    def __init__(self, beta_value=0.5, stab_coefficient=1):
        self.beta_value = beta_value
        self.stab_coefficient = stab_coefficient

    def beta(self, tgt):
        return numpy.array([self.beta_value]*tgt.dimensions, dtype=numpy.float64)

    def grad_interior_flux(self, tgt, u):
        from hedge.optemplate import make_common_subexpression as cse
        n_times = tgt.normal_times_flux
        v_times = tgt.vec_times

        return n_times(
                cse(u.avg, "u_avg") 
                - v_times(self.beta(tgt), n_times(u.int-u.ext)))

    def div(self, tgt, bc_getter, dirichlet_tags, neumann_tags):
        """
        :param bc_getter: a function (tag, volume_expr) -> boundary expr.
          *volume_expr* will be None to query the Neumann condition.
        """

        from numpy import dot
        from hedge.optemplate import make_common_subexpression as cse
        from hedge.flux import FluxVectorPlaceholder, make_normal, PenaltyTerm
        normal = make_normal(tgt.dimensions)

        n_times = tgt.normal_times_flux
        v_times = tgt.vec_times

        if tgt.strong_form:
            def adjust_flux(f):
                return n_times(flux_v.int) - f
        else:
            def adjust_flux(f):
                return f

        flux_v = FluxVectorPlaceholder(tgt.dimensions)

        stab_term_generator = StabilizationTermGenerator(
                list(tgt.operand))
        stab_term = (self.stab_coefficient * PenaltyTerm() 
                * stab_term_generator(tgt.operand))
        flux = n_times(flux_v.avg 
                + v_times(self.beta(tgt), cse(n_times(flux_v.int - flux_v.ext), "jump_v"))
                - stab_term)

        from pytools.obj_array import make_obj_array, join_fields
        flux_arg_int = cse(make_obj_array(stab_term_generator.flux_args))

        tgt.add_derivative(cse(tgt.operand))
        tgt.add_inner_fluxes(adjust_flux(flux), flux_arg_int)

        self.add_div_bcs(tgt, bc_getter, dirichlet_tags, neumann_tags,
                stab_term_generator, stab_term, adjust_flux, flux_v, flux_arg_int,
                tgt.dimensions)




class StabilizedCentralSecondDerivative(LDGSecondDerivative):
    def __init__(self, stab_coefficient=1):
        LDGSecondDerivative.__init__(self, 0, stab_coefficient=stab_coefficient)




class CentralSecondDerivative(LDGSecondDerivative):
    def __init__(self):
        LDGSecondDerivative.__init__(self, 0, 0)




class IPDGSecondDerivative(SecondDerivativeBase):
    def __init__(self, stab_coefficient=1):
        self.stab_coefficient = stab_coefficient

    def grad_interior_flux(self, tgt, u):
        from hedge.optemplate import make_common_subexpression as cse
        n_times = tgt.normal_times_flux
        return n_times(cse(u.avg, "u_avg"))

    def div(self, tgt, bc_getter, dirichlet_tags, neumann_tags):
        """
        :param bc_getter: a function (tag, volume_expr) -> boundary expr.
          *volume_expr* will be None to query the Neumann condition.
        """

        from numpy import dot
        from hedge.optemplate import make_common_subexpression as cse
        from hedge.flux import FluxVectorPlaceholder, make_normal, PenaltyTerm
        normal = make_normal(tgt.dimensions)

        n_times = tgt.normal_times_flux
        v_times = tgt.vec_times

        if tgt.strong_form:
            def adjust_flux(f):
                return n_times(flux_v.int) - f
        else:
            def adjust_flux(f):
                return f

        dim = tgt.dimensions

        flux_w = FluxVectorPlaceholder(2*tgt.dimensions)
        flux_v = flux_w[:dim]
        pure_diff_v = flux_w[dim:]
        flux_args = (
                list(tgt.operand) 
                + list(IPDGDerivativeGenerator()(tgt.operand)))

        stab_term_generator = StabilizationTermGenerator(flux_args)
        stab_term = (self.stab_coefficient * PenaltyTerm() 
                * stab_term_generator(tgt.operand))
        flux = n_times(pure_diff_v.avg - stab_term)

        from pytools.obj_array import make_obj_array, join_fields
        flux_arg_int = cse(make_obj_array(stab_term_generator.flux_args))

        tgt.add_derivative(cse(tgt.operand))
        tgt.add_inner_fluxes(adjust_flux(flux), flux_arg_int)

        self.add_div_bcs(tgt, bc_getter, dirichlet_tags, neumann_tags,
                stab_term_generator, stab_term, adjust_flux, flux_v, flux_arg_int,
                2*tgt.dimensions)

