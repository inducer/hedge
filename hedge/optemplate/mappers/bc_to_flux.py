"""Operator template mapper: BC-to-flux rewriting."""

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




from pytools import memoize_method
from pymbolic.mapper import CSECachingMapperMixin
from hedge.optemplate.mappers import (
        IdentityMapper, DependencyMapper, CombineMapper,
        OperatorReducerMixin)




class ExpensiveBoundaryOperatorDetector(CombineMapper):
    def combine(self, values):
        for val in values:
            if val:
                return True

        return False

    def map_operator_binding(self, expr):
        from hedge.optemplate import (BoundarizeOperator,
                FluxExchangeOperator,
                QuadratureGridUpsampler,
                QuadratureBoundaryGridUpsampler)

        if isinstance(expr.op, BoundarizeOperator):
            return False

        elif isinstance(expr.op, FluxExchangeOperator):
            # FIXME: Duplication of these is an even bigger problem!
            return True

        elif isinstance(expr.op, (
            QuadratureGridUpsampler,
            QuadratureBoundaryGridUpsampler)):
            return True

        else:
            raise RuntimeError("Found '%s' in a boundary term. "
                    "To the best of my knowledge, no hedge operator applies "
                    "directly to boundary data, so this is likely in error."
                    % expr.op)

    def map_common_subexpression(self, expr):
        # If there are any expensive operators below here, this
        # CSE will catch them, so we can easily flux-CSE down to
        # here.

        return False

    def map_normal_component(self, expr):
        return False

    map_variable = map_normal_component
    map_constant = map_normal_component

    @memoize_method
    def __call__(self, expr):
        return CombineMapper.__call__(self, expr)





class BCToFluxRewriter(CSECachingMapperMixin, IdentityMapper):
    """Operates on :class:`FluxOperator` instances bound to :class:`BoundaryPair`. If the
    boundary pair's *bfield* is an expression of what's available in the
    *field*, we can avoid fetching the data for the explicit boundary
    condition and just substitute the *bfield* expression into the flux. This
    mapper does exactly that.
    """

    map_common_subexpression_uncached = \
            IdentityMapper.map_common_subexpression

    def __init__(self):
        self.expensive_bdry_op_detector = \
                ExpensiveBoundaryOperatorDetector()

    def map_operator_binding(self, expr):
        from hedge.optemplate.operators import FluxOperatorBase
        from hedge.optemplate.primitives import BoundaryPair
        from hedge.flux import FluxSubstitutionMapper, FieldComponent

        if not (isinstance(expr.op, FluxOperatorBase)
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

            def __init__(self, vol_expr_list, expensive_bdry_op_detector):
                self.vol_expr_list = vol_expr_list
                self.vol_expr_to_idx = dict((vol_expr, idx)
                        for idx, vol_expr in enumerate(vol_expr_list))

                self.bdry_expr_list = []
                self.bdry_expr_to_idx = {}

                self.expensive_bdry_op_detector = expensive_bdry_op_detector

            # {{{ expression registration
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

            # }}}

            # {{{ map_xxx routines

            @memoize_method
            def map_common_subexpression(self, expr):
                # Here we need to decide whether this CSE should be turned into
                # a flux CSE or not. This is a good idea if the transformed
                # expression only contains "bare" volume or boundary
                # expressions.  However, as soon as an operator is applied
                # somewhere in the subexpression, the CSE should not be touched
                # in order to avoid redundant evaluation of that operator.
                #
                # Observe that at the time of this writing (Feb 2010), the only
                # operators that may occur in boundary expressions are
                # quadrature-related.

                has_expensive_operators = \
                        self.expensive_bdry_op_detector(expr.child)

                if has_expensive_operators:
                    return FieldComponent(
                            self.register_boundary_expr(expr),
                            is_interior=False)
                else:
                    return IdentityMapper.map_common_subexpression(self, expr)

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
                        FluxExchangeOperator,
                        QuadratureGridUpsampler,
                        QuadratureBoundaryGridUpsampler)

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

                elif isinstance(expr.op, QuadratureBoundaryGridUpsampler):
                    if bpair.tag != expr.op.boundary_tag:
                        raise RuntimeError("BoundarizeOperator "
                                "and QuadratureBoundaryGridUpsampler "
                                "do not agree about boundary tag: %s vs %s"
                                % (expr.op.boundary_tag, bpair.tag))
                    return FieldComponent(
                            self.register_boundary_expr(expr),
                            is_interior=False)

                elif isinstance(expr.op, QuadratureGridUpsampler):
                    # We're invoked before operator specialization, so we may
                    # see these instead of QuadratureBoundaryGridUpsampler.
                    return FieldComponent(
                            self.register_boundary_expr(expr),
                            is_interior=False)

                else:
                    raise RuntimeError("Found '%s' in a boundary term. "
                            "To the best of my knowledge, no hedge operator applies "
                            "directly to boundary data, so this is likely in error."
                            % expr.op)

            def map_flux_exchange(self, expr):
                return FieldComponent(
                        self.register_boundary_expr(expr),
                        is_interior=False)
            # }}}

        from hedge.tools import is_obj_array
        if not is_obj_array(vol_field):
            vol_field = [vol_field]

        mbfeef = MaxBoundaryFluxEvaluableExpressionFinder(list(vol_field),
                self.expensive_bdry_op_detector)
        #from hedge.optemplate.tools import pretty_print_optemplate
        #print pretty_print_optemplate(bdry_field)
        #raw_input("YO")
        new_bdry_field = mbfeef(bdry_field)

        # Step II: Substitute the new_bdry_field into the flux.
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
            return type(expr.op)(new_flux, *expr.op.__getinitargs__()[1:])(
                    BoundaryPair(
                        make_obj_array([self.rec(e) for e in mbfeef.vol_expr_list]),
                        make_obj_array([self.rec(e) for e in mbfeef.bdry_expr_list]),
                        bpair.tag))

