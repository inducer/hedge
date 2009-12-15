"""Operator template mappers."""

from __future__ import division

__copyright__ = "Copyright (C) 2008 Andreas Kloeckner"

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





import pymbolic.primitives
import pymbolic.mapper.stringifier
import pymbolic.mapper.evaluator
import pymbolic.mapper.dependency
import pymbolic.mapper.substitutor
import pymbolic.mapper.constant_folder
import pymbolic.mapper.flop_counter
from pymbolic.mapper import CSECachingMapperMixin




# mappers ---------------------------------------------------------------------
class LocalOpReducerMixin(object):
    """Reduces calls to mapper methods for all local differentiation
    operators to a single mapper method, and likewise for mass
    operators.
    """
    def map_diff(self, expr, *args, **kwargs):
        return self.map_diff_base(expr, *args, **kwargs)

    def map_minv_st(self, expr, *args, **kwargs):
        return self.map_diff_base(expr, *args, **kwargs)

    def map_stiffness(self, expr, *args, **kwargs):
        return self.map_diff_base(expr, *args, **kwargs)

    def map_stiffness_t(self, expr, *args, **kwargs):
        return self.map_diff_base(expr, *args, **kwargs)

    def map_mass(self, expr, *args, **kwargs):
        return self.map_mass_base(expr, *args, **kwargs)

    def map_inverse_mass(self, expr, *args, **kwargs):
        return self.map_mass_base(expr, *args, **kwargs)




class FluxOpReducerMixin(object):
    """Reduces calls to mapper methods for all flux
    operators to a smaller number of mapper methods.
    """
    def map_flux(self, expr, *args, **kwargs):
        return self.map_flux_base(expr, *args, **kwargs)

    def map_lift(self, expr, *args, **kwargs):
        return self.map_flux_base(expr, *args, **kwargs)




class OperatorReducerMixin(LocalOpReducerMixin, FluxOpReducerMixin):
    """Reduces calls to *any* operator mapping function to just one."""
    def map_diff_base(self, expr, *args, **kwargs):
        return self.map_operator(expr, *args, **kwargs)

    map_mass_base = map_diff_base
    map_flux_base = map_diff_base
    map_elementwise_max = map_diff_base
    map_boundarize = map_diff_base
    map_flux_exchange = map_diff_base




class CombineMapperMixin(object):
    def map_operator_binding(self, expr):
        return self.combine([self.rec(expr.op), self.rec(expr.field)])

    def map_boundary_pair(self, expr):
        return self.combine([self.rec(expr.field), self.rec(expr.bfield)])




class CombineMapper(CombineMapperMixin, pymbolic.mapper.CombineMapper):
    pass




class IdentityMapperMixin(LocalOpReducerMixin, FluxOpReducerMixin):
    def map_operator_binding(self, expr, *args, **kwargs):
        assert not isinstance(self, BoundOpMapperMixin), \
                "IdentityMapper instances cannot be combined with " \
                "the BoundOpMapperMixin"

        return expr.__class__(
                self.rec(expr.op, *args, **kwargs),
                self.rec(expr.field, *args, **kwargs))

    def map_boundary_pair(self, expr, *args, **kwargs):
        assert not isinstance(self, BoundOpMapperMixin), \
                "IdentityMapper instances cannot be combined with " \
                "the BoundOpMapperMixin"

        return expr.__class__(
                self.rec(expr.field, *args, **kwargs),
                self.rec(expr.bfield, *args, **kwargs),
                expr.tag)

    def map_mass_base(self, expr, *args, **kwargs):
        assert not isinstance(self, BoundOpMapperMixin), \
                "IdentityMapper instances cannot be combined with " \
                "the BoundOpMapperMixin"

        # it's a leaf--no changing children
        return expr

    def map_scalar_parameter(self, expr, *args, **kwargs):
        # it's a leaf--no changing children
        return expr

    map_diff_base = map_mass_base
    map_flux_base = map_mass_base
    map_elementwise_max = map_mass_base
    map_boundarize = map_mass_base
    map_flux_exchange = map_mass_base

    map_normal_component = map_mass_base




class DependencyMapper(
        CombineMapperMixin,
        pymbolic.mapper.dependency.DependencyMapper,
        OperatorReducerMixin):
    def __init__(self,
            include_operator_bindings=True,
            composite_leaves=None,
            **kwargs):
        if composite_leaves == False:
            include_operator_bindings = False
        if composite_leaves == True:
            include_operator_bindings = True

        pymbolic.mapper.dependency.DependencyMapper.__init__(self,
                composite_leaves=composite_leaves, **kwargs)

        self.include_operator_bindings = include_operator_bindings

    def map_operator_binding(self, expr):
        if self.include_operator_bindings:
            return set([expr])
        else:
            return CombineMapperMixin.map_operator_binding(self, expr)

    def map_operator(self, expr):
        return set()

    def map_scalar_parameter(self, expr):
        return set([expr])

    def map_normal_component(self, expr):
        return set()



class FlopCounter(
        CombineMapperMixin,
        pymbolic.mapper.flop_counter.FlopCounter):
    def map_operator_binding(self, expr):
        return self.rec(expr.field)

    def map_scalar_parameter(self, expr):
        return 0




class IdentityMapper(
        IdentityMapperMixin,
        pymbolic.mapper.IdentityMapper):
    pass





class SubstitutionMapper(pymbolic.mapper.substitutor.SubstitutionMapper,
        IdentityMapperMixin):
    pass




class StringifyMapper(pymbolic.mapper.stringifier.StringifyMapper):
    def __init__(self, constant_mapper=str, flux_stringify_mapper=None):
        pymbolic.mapper.stringifier.StringifyMapper.__init__(
                self, constant_mapper=constant_mapper)

        if flux_stringify_mapper is None:
            from hedge.flux import FluxStringifyMapper
            flux_stringify_mapper = FluxStringifyMapper()

        self.flux_stringify_mapper = flux_stringify_mapper

    def map_boundary_pair(self, expr, enclosing_prec):
        from pymbolic.mapper.stringifier import PREC_NONE
        return "BPair(%s, %s, %s)" % (
                self.rec(expr.field, PREC_NONE),
                self.rec(expr.bfield, PREC_NONE),
                repr(expr.tag))

    def map_diff(self, expr, enclosing_prec):
        return "Diff%d" % expr.xyz_axis

    def map_minv_st(self, expr, enclosing_prec):
        return "MInvST%d" % expr.xyz_axis

    def map_stiffness(self, expr, enclosing_prec):
        return "Stiff%d" % expr.xyz_axis

    def map_stiffness_t(self, expr, enclosing_prec):
        return "StiffT%d" % expr.xyz_axis

    def map_mass(self, expr, enclosing_prec):
        return "M"

    def map_inverse_mass(self, expr, enclosing_prec):
        return "InvM"

    def map_flux(self, expr, enclosing_prec):
        from pymbolic.mapper.stringifier import PREC_NONE
        return "Flux(%s)" % self.flux_stringify_mapper(expr.flux, PREC_NONE)

    def map_lift(self, expr, enclosing_prec):
        from pymbolic.mapper.stringifier import PREC_NONE
        return "Lift(%s)" % self.flux_stringify_mapper(expr.flux, PREC_NONE)

    def map_whole_domain_flux(self, expr, enclosing_prec):
        # used from hedge.backends.cuda.optemplate
        if expr.is_lift:
            opname = "WLift"
        else:
            opname = "WFlux"

        from pymbolic.mapper.stringifier import PREC_NONE
        return "%s(%s)" % (opname,
                self.rec(expr.rebuild_optemplate(), PREC_NONE))

    def map_elementwise_max(self, expr, enclosing_prec):
        return "ElWMax"

    def map_boundarize(self, expr, enclosing_prec):
        return "Boundarize<tag=%s>" % expr.tag

    def map_flux_exchange(self, expr, enclosing_prec):
        return "FExch<idx=%s,rank=%d>" % (expr.index, expr.rank)

    def map_normal_component(self, expr, enclosing_prec):
        return "Normal<tag=%s>[%d]" % (expr.tag, expr.axis)

    def map_operator_binding(self, expr, enclosing_prec):
        from pymbolic.mapper.stringifier import PREC_NONE
        return "<%s>(%s)" % (
                self.rec(expr.op, PREC_NONE),
                self.rec(expr.field, PREC_NONE))

    def map_scalar_parameter(self, expr, enclosing_prec):
        return "ScalarPar[%s]" % expr.name




class PrettyStringifyMapper(
        pymbolic.mapper.stringifier.CSESplittingStringifyMapperMixin,
        StringifyMapper):
    def __init__(self):
        pymbolic.mapper.stringifier.CSESplittingStringifyMapperMixin.__init__(self)
        StringifyMapper.__init__(self)

        self.flux_to_number = {}
        self.flux_string_list = []

        self.bc_to_number = {}
        self.bc_string_list = []

        from hedge.flux import PrettyFluxStringifyMapper
        self.flux_stringify_mapper = PrettyFluxStringifyMapper()

    def get_flux_number(self, flux):
        try:
            return self.flux_to_number[flux]
        except KeyError:
            from pymbolic.mapper.stringifier import PREC_NONE
            str_flux = self.flux_stringify_mapper(flux, PREC_NONE)

            flux_number = len(self.flux_to_number)
            self.flux_string_list.append(str_flux)
            self.flux_to_number[flux] = flux_number
            return flux_number

    def map_boundary_pair(self, expr, enclosing_prec):
        try:
            bc_number = self.bc_to_number[expr]
        except KeyError:
            from pymbolic.mapper.stringifier import PREC_NONE
            str_bc = StringifyMapper.map_boundary_pair(self, expr, PREC_NONE)

            bc_number = len(self.bc_to_number)
            self.bc_string_list.append(str_bc)
            self.bc_to_number[expr] = bc_number

        return "BC%d@%s" % (bc_number, expr.tag)

    def map_operator_binding(self, expr, enclosing_prec):
        from hedge.optemplate import BoundarizeOperator
        if isinstance(expr.op, BoundarizeOperator):
            from pymbolic.mapper.stringifier import PREC_CALL, PREC_SUM
            return self.parenthesize_if_needed(
                    "%s@%s" % (
                        self.rec(expr.field, PREC_CALL),
                        expr.op.tag),
                    enclosing_prec, PREC_SUM)
        else:
            return StringifyMapper.map_operator_binding(
                    self, expr, enclosing_prec)

    def get_bc_strings(self):
        return ["BC%d : %s" % (i, bc_str)
                for i, bc_str in enumerate(self.bc_string_list)]

    def get_flux_strings(self):
        return ["Flux%d : %s" % (i, flux_str)
                for i, flux_str in enumerate(self.flux_string_list)]

    def map_flux(self, expr, enclosing_prec):
        return "Flux%d" % self.get_flux_number(expr.flux)

    def map_lift(self, expr, enclosing_prec):
        return "Lift-Flux%d" % self.get_flux_number(expr.flux)




def pretty_print_optemplate(optemplate):
    stringify_mapper = PrettyStringifyMapper()
    from pymbolic.mapper.stringifier import PREC_NONE
    result = stringify_mapper(optemplate, PREC_NONE)

    splitter = "="*75 + "\n"

    bc_strs = stringify_mapper.get_bc_strings()
    if bc_strs:
        result = "\n".join(bc_strs)+"\n"+splitter+result

    cse_strs = stringify_mapper.get_cse_strings()
    if cse_strs:
        result = "\n".join(cse_strs)+"\n"+splitter+result

    flux_strs = stringify_mapper.get_flux_strings()
    if flux_strs:
        result = "\n".join(flux_strs)+"\n"+splitter+result

    flux_cses = stringify_mapper.flux_stringify_mapper.get_cse_strings()
    if flux_cses:
        result = "\n".join("flux "+fs for fs in flux_cses)+"\n\n"+result

    return result




class NoCSEStringifyMapper(StringifyMapper):
    def map_common_subexpression(self, expr, enclosing_prec):
        return self.rec(expr.child, enclosing_prec)




class BoundOpMapperMixin(object):
    def map_operator_binding(self, expr, *args, **kwargs):
        return expr.op.get_mapper_method(self)(expr.op, expr.field, *args, **kwargs)



class OperatorBinder(CSECachingMapperMixin, IdentityMapper):
    map_common_subexpression_uncached = \
            IdentityMapper.map_common_subexpression

    def map_product(self, expr):
        if len(expr.children) == 0:
            return expr

        from pymbolic.primitives import flattened_product, Product
        from hedge.optemplate import Operator, OperatorBinding

        first = expr.children[0]
        if isinstance(first, Operator):
            prod = flattened_product(expr.children[1:])
            if isinstance(prod, Product) and len(prod.children) > 1:
                from warnings import warn
                warn("Binding an %s to more than one "
                        "operand in a product is ambiguous - "
                        "use the parenthesized form instead."
                        % first)
            return OperatorBinding(first, self.rec(prod))
        else:
            return first * self.rec(flattened_product(expr.children[1:]))




# simplifications / optimizations ---------------------------------------------
class CommutativeConstantFoldingMapper(
        pymbolic.mapper.constant_folder.CommutativeConstantFoldingMapper,
        IdentityMapperMixin):

    def __init__(self):
        pymbolic.mapper.constant_folder.CommutativeConstantFoldingMapper.__init__(self)
        self.dep_mapper = DependencyMapper()

    def is_constant(self, expr):
        return not bool(self.dep_mapper(expr))




class EmptyFluxKiller(CSECachingMapperMixin, IdentityMapper):
    def __init__(self, mesh):
        IdentityMapper.__init__(self)
        self.mesh = mesh

    map_common_subexpression_uncached = \
            IdentityMapper.map_common_subexpression

    def map_operator_binding(self, expr):
        from hedge.optemplate import (
                FluxOperatorBase, LiftingFluxOperator, BoundaryPair)

        if (isinstance(expr.op, (
            FluxOperatorBase,
            LiftingFluxOperator))
            and
            isinstance(expr.field, BoundaryPair)
            and
            len(self.mesh.tag_to_boundary.get(expr.field.tag, [])) == 0):
            return 0
        else:
            return IdentityMapper.map_operator_binding(self, expr)




class _InnerDerivativeJoiner(pymbolic.mapper.RecursiveMapper):
    def map_operator_binding(self, expr, derivatives):
        from hedge.optemplate import DifferentiationOperator

        if isinstance(expr.op, DifferentiationOperator):
            derivatives.setdefault(expr.op, []).append(expr.field)
            return 0
        else:
            return DerivativeJoiner()(expr)

    def map_common_subexpression(self, expr, derivatives):
        # no use preserving these if we're moving derivatives around
        return self.rec(expr.child, derivatives)

    def map_sum(self, expr, derivatives):
        from pymbolic.primitives import flattened_sum
        return flattened_sum(tuple(
            self.rec(child, derivatives) for child in expr.children))

    def map_product(self, expr, derivatives):
        from hedge.optemplate import ScalarParameter

        def is_scalar(expr):
            return isinstance(expr, (int, float, complex, ScalarParameter))

        from pytools import partition
        scalars, nonscalars = partition(is_scalar, expr.children)

        if len(nonscalars) != 1:
            return DerivativeJoiner()(expr)
        else:
            from pymbolic import flattened_product
            factor = flattened_product(scalars)
            nonscalar, = nonscalars

            sub_derivatives = {}
            nonscalar = self.rec(nonscalar, sub_derivatives)
            def do_map(expr):
                if is_scalar(expr):
                    return expr
                else:
                    return self.rec(expr, derivatives)

            for operator, operands in sub_derivatives.iteritems():
                for operand in operands:
                    derivatives.setdefault(operator, []).append(
                            factor*operand)

            return factor*nonscalar

    def map_constant(self, expr, *args):
        return DerivativeJoiner()(expr)

    def map_scalar_parameter(self, expr, *args):
        return DerivativeJoiner()(expr)

    # these two are necessary because they're forwarding targets
    def map_algebraic_leaf(self, expr, *args):
        return DerivativeJoiner()(expr)

    def map_quotient(self, expr, *args):
        return DerivativeJoiner()(expr)




class DerivativeJoiner(CSECachingMapperMixin, IdentityMapper):
    """Joins derivatives:

    .. math::

        \frac{\partial A}{\partial x} + \frac{\partial B}{\partial x} 
        \rightarrow
        \frac{\partial (A+B)}{\partial x}.
    """
    map_common_subexpression_uncached = \
            IdentityMapper.map_common_subexpression

    def map_sum(self, expr):
        idj = _InnerDerivativeJoiner()

        derivatives = {}
        new_children = [idj(child, derivatives)
                for child in expr.children]

        for operator, operands in derivatives.iteritems():
            new_children.insert(0, operator(
                sum(self.rec(operand) for operand in operands)))

        from pymbolic.primitives import flattened_sum
        return flattened_sum(new_children)




class _InnerInverseMassContractor(pymbolic.mapper.RecursiveMapper):
    def __init__(self, outer_mass_contractor):
        self.outer_mass_contractor = outer_mass_contractor

    def map_constant(self, expr):
        from hedge.tools import is_zero
        from hedge.optemplate import InverseMassOperator, OperatorBinding

        if is_zero(expr):
            return 0
        else:
            return OperatorBinding(
                    InverseMassOperator(),
                    self.outer_mass_contractor(expr))

    def map_algebraic_leaf(self, expr):
        from hedge.optemplate import InverseMassOperator, OperatorBinding

        return OperatorBinding(
                InverseMassOperator(),
                self.outer_mass_contractor(expr))

    def map_operator_binding(self, binding):
        from hedge.optemplate import (
                MassOperator, StiffnessOperator, StiffnessTOperator,
                FluxOperator, OperatorBinding, DifferentiationOperator,
                MInvSTOperator, LiftingFluxOperator, InverseMassOperator)

        if isinstance(binding.op, MassOperator):
            return binding.field
        elif isinstance(binding.op, StiffnessOperator):
            return OperatorBinding(
                    DifferentiationOperator(binding.op.xyz_axis),
                    self.outer_mass_contractor(binding.field))
        elif isinstance(binding.op, StiffnessTOperator):
            return OperatorBinding(
                    MInvSTOperator(binding.op.xyz_axis),
                    self.outer_mass_contractor(binding.field))
        elif isinstance(binding.op, FluxOperator):
            return OperatorBinding(
                    LiftingFluxOperator(binding.op.flux),
                    self.outer_mass_contractor(binding.field))
        else:
            return OperatorBinding(
                InverseMassOperator(),
                self.outer_mass_contractor(binding))

    def map_sum(self, expr):
        return expr.__class__(tuple(self.rec(child) for child in expr.children))

    def map_product(self, expr):
        from hedge.optemplate import (
                InverseMassOperator, OperatorBinding, ScalarParameter)

        def is_scalar(expr):
            return isinstance(expr, (int, float, complex, ScalarParameter))

        from pytools import len_iterable
        nonscalar_count = len_iterable(ch
                for ch in expr.children
                if not is_scalar(ch))

        if nonscalar_count > 1:
            # too complicated, don't touch it
            return OperatorBinding(
                    InverseMassOperator(),
                    self.outer_mass_contractor(expr))
        else:
            def do_map(expr):
                if is_scalar(expr):
                    return expr
                else:
                    return self.rec(expr)
            return expr.__class__(tuple(
                do_map(child) for child in expr.children))





class InverseMassContractor(CSECachingMapperMixin, IdentityMapper):
    # assumes all operators to be bound
    map_common_subexpression_uncached = \
            IdentityMapper.map_common_subexpression

    def map_boundary_pair(self, bp):
        from hedge.optemplate import BoundaryPair
        return BoundaryPair(self.rec(bp.field), self.rec(bp.bfield), bp.tag)

    def map_operator_binding(self, binding):
        # we only care about bindings of inverse mass operators
        from hedge.optemplate import InverseMassOperator

        if isinstance(binding.op, InverseMassOperator):
            return _InnerInverseMassContractor(self)(binding.field)
        else:
            return binding.__class__(binding.op,
                    self.rec(binding.field))




# BC-to-flux rewriting --------------------------------------------------------
class BCToFluxRewriter(CSECachingMapperMixin, IdentityMapper):
    """Operates on :class:`FluxOperator` instances bound to :class:`BoundaryPair`. If the
    boundary pair's *bfield* is an expression of what's available in the
    *field*, we can avoid fetching the data for the explicit boundary
    condition and just substitute the *bfield* expression into the flux. This
    mapper does exactly that.
    """

    map_common_subexpression_uncached = \
            IdentityMapper.map_common_subexpression

    def map_operator_binding(self, expr):
        from hedge.optemplate import FluxOperator, BoundaryPair
        if not (isinstance(expr.op, FluxOperator)
                and isinstance(expr.field, BoundaryPair)):
            return IdentityMapper.map_operator_binding(self, expr)

        bpair = expr.field
        vol_field = bpair.field
        bdry_field = bpair.bfield
        flux = expr.op.flux

        bdry_dependencies = DependencyMapper(
                    include_calls="descend_args",
                    include_operator_bindings=True)(bdry_field)

        vol_dependencies = DependencyMapper(
                include_operator_bindings=True)(vol_field)

        vol_bdry_intersection = bdry_dependencies & vol_dependencies
        if vol_bdry_intersection:
            raise RuntimeError("Variables are being used as both "
                    "boundary and volume quantities: %s"
                    % ", ".join(str(v) for v in vol_bdry_intersection))

        # Step 1: Find maximal flux-evaluable subexpression of boundary field
        # in given BoundaryPair.

        class MaxBoundaryFluxEvaluableExpressionFinder(
                IdentityMapper, OperatorReducerMixin):
            def __init__(self, vol_expr_list):
                self.vol_expr_list = vol_expr_list
                self.vol_expr_to_idx = dict((vol_expr, idx)
                        for idx, vol_expr in enumerate(vol_expr_list))

                self.bdry_expr_list = []
                self.bdry_expr_to_idx = {}

            def register_boundary_expr(self, expr):
                try:
                    return self.bdry_expr_to_idx[expr]
                except KeyError:
                    idx = len(self.bdry_expr_to_idx)
                    self.bdry_expr_to_idx[expr] = idx
                    self.bdry_expr_list.append(expr)
                    return idx

            def register_volume_expr(self, expr):
                try:
                    return self.vol_expr_to_idx[expr]
                except KeyError:
                    idx = len(self.vol_expr_to_idx)
                    self.vol_expr_to_idx[expr] = idx
                    self.vol_expr_list.append(expr)
                    return idx

            def map_normal(self, expr):
                raise RuntimeError("Your operator template contains a flux normal. "
                        "You may find this confusing, but you can't do that. "
                        "It turns out that you need to use "
                        "hedge.optemplate.make_normal() for normals in boundary "
                        "terms of operator templates.")

            def map_normal_component(self, expr):
                if expr.tag != bpair.tag:
                    raise RuntimeError("BoundaryNormalComponent and BoundaryPair "
                            "do not agree about boundary tag: %s vs %s"
                            % (expr.tag, bpair.tag))

                from hedge.flux import Normal
                return Normal(expr.axis)

            def map_variable(self, expr):
                return FieldComponent(
                        self.register_boundary_expr(expr),
                        is_interior=False)

            map_subscript = map_variable

            def map_operator_binding(self, expr):
                from hedge.optemplate import (BoundarizeOperator,
                        FluxExchangeOperator)

                if isinstance(expr.op, BoundarizeOperator):
                    if expr.op.tag != bpair.tag:
                        raise RuntimeError("BoundarizeOperator and BoundaryPair "
                                "do not agree about boundary tag: %s vs %s"
                                % (expr.op.tag, bpair.tag))

                    return FieldComponent(
                            self.register_volume_expr(expr.field),
                            is_interior=True)
                elif isinstance(expr.op, FluxExchangeOperator):
                    from hedge.mesh import TAG_RANK_BOUNDARY
                    op_tag = TAG_RANK_BOUNDARY(expr.op.rank)
                    if bpair.tag != op_tag:
                        raise RuntimeError("BoundarizeOperator and FluxExchangeOperator "
                                "do not agree about boundary tag: %s vs %s"
                                % (op_tag, bpair.tag))
                    return FieldComponent(
                            self.register_boundary_expr(expr),
                            is_interior=False)
                else:
                    raise RuntimeError("Found '%s' in a boundary term. "
                            "To the best of my knowledge, no hedge operator applies "
                            "directly to boundary data, so this is likely in error."
                            % expr.op)

        from hedge.tools import is_obj_array
        if not is_obj_array(vol_field):
            vol_field = [vol_field]

        mbfeef = MaxBoundaryFluxEvaluableExpressionFinder(list(vol_field))
        new_bdry_field = mbfeef(bdry_field)

        # Step II: Substitute the new_bdry_field into the flux.
        from hedge.flux import FluxSubstitutionMapper, FieldComponent

        def sub_bdry_into_flux(expr):
            if isinstance(expr, FieldComponent) and not expr.is_interior:
                if expr.index == 0 and not is_obj_array(bdry_field):
                    return new_bdry_field
                else:
                    return new_bdry_field[expr.index]
            else:
                return None

        new_flux = FluxSubstitutionMapper(sub_bdry_into_flux)(flux)

        from hedge.tools import is_zero, make_obj_array
        if is_zero(new_flux):
            return 0
        else:
            from hedge.optemplate import OperatorBinding
            return OperatorBinding(
                    FluxOperator(new_flux), BoundaryPair(
                        make_obj_array([self.rec(e) for e in mbfeef.vol_expr_list]),
                        make_obj_array([self.rec(e) for e in mbfeef.bdry_expr_list]),
                        bpair.tag))




# error checker ---------------------------------------------------------------
class ErrorChecker(CSECachingMapperMixin, IdentityMapper):
    map_common_subexpression_uncached = \
            IdentityMapper.map_common_subexpression

    def __init__(self, mesh):
        self.mesh = mesh

    def map_operator_binding(self, expr):
        from hedge.optemplate import DiffOperatorBase

        if isinstance(expr.op, DiffOperatorBase):
            if (self.mesh is not None 
                    and expr.op.xyz_axis >= self.mesh.dimensions):
                raise ValueError("optemplate tries to differentiate along a "
                        "non-existent axis (e.g. Z in 2D)")

        # FIXME: Also check fluxes
        return IdentityMapper.map_operator_binding(self, expr)

    def map_normal(self, expr):
        if self.mesh is not None and expr.axis >= self.mesh.dimensions:
            raise ValueError("optemplate tries to differentiate along a "
                    "non-existent axis (e.g. Z in 2D)")

        return expr




# collecting ------------------------------------------------------------------
class CollectorMixin(LocalOpReducerMixin, FluxOpReducerMixin):
    def combine(self, values):
        from pytools import flatten
        return set(flatten(values))

    def map_constant(self, bpair):
        return set()

    map_mass_base = map_constant
    map_diff_base = map_constant
    map_flux_base = map_constant
    map_variable = map_constant
    map_elementwise_max = map_constant
    map_boundarize = map_constant
    map_flux_exchange = map_constant
    map_normal_component = map_constant
    map_scalar_parameter = map_constant




class FluxCollector(CSECachingMapperMixin, CollectorMixin, CombineMapper):
    map_common_subexpression_uncached = \
            CombineMapper.map_common_subexpression

    def map_operator_binding(self, expr):
        from hedge.optemplate import FluxOperatorBase

        if isinstance(expr.op, FluxOperatorBase):
            result = set([expr])
        else:
            result = set()

        return result | CombineMapper.map_operator_binding(self, expr)




class BoundaryTagCollector(CollectorMixin, CombineMapper):
    def map_boundary_pair(self, bpair):
        return set([bpair.tag])




class BoundOperatorCollector(CSECachingMapperMixin, CollectorMixin, CombineMapper):
    def __init__(self, op_class):
        self.op_class = op_class

    map_common_subexpression_uncached = \
            CombineMapper.map_common_subexpression

    def map_operator_binding(self, expr):
        if isinstance(expr.op, self.op_class):
            result = set([expr])
        else:
            result = set()

        return result | CombineMapper.map_operator_binding(self, expr)



# evaluation ------------------------------------------------------------------
class Evaluator(pymbolic.mapper.evaluator.EvaluationMapper):
    def map_boundary_pair(self, bp):
        from hedge.optemplate.primitives import BoundaryPair
        return BoundaryPair(self.rec(bp.field), self.rec(bp.bfield), bp.tag)
