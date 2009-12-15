"""Interface with Nvidia CUDA."""

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



from pytools import Record, memoize_method
import pymbolic.primitives
import pymbolic.mapper
import hedge.optemplate
from pymbolic.mapper import CSECachingMapperMixin



def get_flux_dependencies(flux, field, bdry="all"):
    from hedge.flux import FluxDependencyMapper, FieldComponent
    in_fields = list(FluxDependencyMapper(
        include_calls=False)(flux))

    # check that all in_fields are FieldComponent instances
    assert not [in_field
        for in_field in in_fields
        if not isinstance(in_field, FieldComponent)]

    def maybe_index(fld, index):
        from hedge.tools import is_obj_array
        if is_obj_array(fld):
            return fld[inf.index]
        else:
            return fld

    from hedge.tools import is_zero
    from hedge.optemplate import BoundaryPair
    if isinstance(field, BoundaryPair):
        for inf in in_fields:
            if inf.is_interior:
                if bdry in ["all", "int"]:
                    value = maybe_index(field.field, inf.index)

                    if not is_zero(value):
                        yield value
            else:
                if bdry in ["all", "ext"]:
                    value = maybe_index(field.bfield, inf.index)

                    if not is_zero(value):
                        yield value
    else:
        for inf in in_fields:
            value = maybe_index(field, inf.index)
            if not is_zero(value):
                yield value




class WholeDomainFluxOperator(pymbolic.primitives.Leaf):
    class FluxInfo(Record):
        __slots__ = []

        def __repr__(self):
            # override because we want flux_expr in infix
            return "%s(%s)" % (
                    self.__class__.__name__,
                    ", ".join("%s=%s" % (fld, getattr(self, fld))
                        for fld in self.__class__.fields
                        if hasattr(self, fld)))

    class InteriorInfo(FluxInfo):
        # attributes: flux_expr, field_expr,

        @property
        @memoize_method
        def dependencies(self):
            return set(get_flux_dependencies(
                self.flux_expr, self.field_expr))

    class BoundaryInfo(FluxInfo):
        # attributes: flux_expr, bpair

        @property
        @memoize_method
        def int_dependencies(self):
            return set(get_flux_dependencies(
                    self.flux_expr, self.bpair, bdry="int"))

        @property
        @memoize_method
        def ext_dependencies(self):
            return set(get_flux_dependencies(
                    self.flux_expr, self.bpair, bdry="ext"))


    def __init__(self, is_lift, interiors, boundaries):
        self.is_lift = is_lift

        self.interiors = tuple(interiors)
        self.boundaries = tuple(boundaries)

        from pytools import set_sum
        interior_deps = set_sum(iflux.dependencies
                for iflux in interiors)
        boundary_int_deps = set_sum(bflux.int_dependencies
                for bflux in boundaries)
        boundary_ext_deps = set_sum(bflux.ext_dependencies
                for bflux in boundaries)

        self.interior_deps = list(interior_deps)
        self.boundary_int_deps = list(boundary_int_deps)
        self.boundary_ext_deps = list(boundary_ext_deps)
        self.boundary_deps = list(boundary_int_deps | boundary_ext_deps)

        self.dep_to_tag = {}
        for bflux in boundaries:
            for dep in get_flux_dependencies(
                    bflux.flux_expr, bflux.bpair, bdry="ext"):
                self.dep_to_tag[dep] = bflux.bpair.tag

    @memoize_method
    def rebuild_optemplate(self):
        from hedge.optemplate import OperatorBinding, \
                FluxOperator, LiftingFluxOperator

        if self.is_lift:
            f_op = LiftingFluxOperator
        else:
            f_op = FluxOperator

        summands = []
        for i in self.interiors:
            summands.append(OperatorBinding(
                    f_op(i.flux_expr), i.field_expr))
        for b in self.boundaries:
            summands.append(OperatorBinding(
                    f_op(b.flux_expr), b.bpair))

        from pymbolic.primitives import flattened_sum
        return flattened_sum(summands)

    # infrastructure interaction
    def get_hash(self):
        return hash((self.__class__, self.rebuild_optemplate()))

    def is_equal(self, other):
        return (other.__class__ == WholeDomainFluxOperator
                and self.rebuild_optemplate() == other.rebuild_optemplate())

    def __getinitargs__(self):
        return self.is_lift, self.interiors, self.boundaries

    def stringifier(self):
        return hedge.optemplate.StringifyMapper

    def get_mapper_method(self, mapper):
        return mapper.map_whole_domain_flux




class BoundaryCombiner(CSECachingMapperMixin, hedge.optemplate.IdentityMapper):
    """Combines inner fluxes and boundary fluxes into a
    single, whole-domain operator of type
    L{hedge.backends.cuda.execute.WholeDomainFluxOperator}.
    """
    def __init__(self, mesh):
        self.mesh = mesh

    flux_op_types = (hedge.optemplate.FluxOperator,
            hedge.optemplate.LiftingFluxOperator)

    map_common_subexpression_uncached = \
            hedge.optemplate.IdentityMapper.map_common_subexpression

    def gather_one_wdflux(self, expressions):
        from hedge.optemplate import OperatorBinding, \
                LiftingFluxOperator, BoundaryPair

        interiors = []
        boundaries = []
        is_lift = None

        rest = []

        for ch in expressions:
            if (isinstance(ch, OperatorBinding)
                    and isinstance(ch.op, self.flux_op_types)):
                my_is_lift = isinstance(ch.op, LiftingFluxOperator)

                if is_lift is None:
                    is_lift = my_is_lift
                else:
                    if is_lift != my_is_lift:
                        rest.append(ch)
                        continue

                if isinstance(ch.field, BoundaryPair):
                    bpair = self.rec(ch.field)
                    if self.mesh.tag_to_boundary.get(bpair.tag, []):
                        boundaries.append(WholeDomainFluxOperator.BoundaryInfo(
                            flux_expr=ch.op.flux,
                            bpair=bpair))
                else:
                    interiors.append(WholeDomainFluxOperator.InteriorInfo(
                            flux_expr=ch.op.flux,
                            field_expr=self.rec(ch.field)))
            else:
                rest.append(ch)

        if interiors or boundaries:
            wdf = WholeDomainFluxOperator(
                    is_lift=is_lift,
                    interiors=interiors,
                    boundaries=boundaries)
        else:
            wdf = None
        return wdf, rest

    def map_operator_binding(self, expr):
        if isinstance(expr.op, self.flux_op_types):
            wdf, rest = self.gather_one_wdflux([expr])
            assert not rest
            return wdf
        else:
            return hedge.optemplate.IdentityMapper \
                    .map_operator_binding(self, expr)

    def map_sum(self, expr):
        from pymbolic.primitives import flattened_sum

        result = 0
        expressions = expr.children
        while True:
            wdf, expressions = self.gather_one_wdflux(expressions)
            if wdf is not None:
                result += wdf
            else:
                return result + flattened_sum(self.rec(r_i) for r_i in expressions)




# collectors ------------------------------------------------------------------
class FluxCollector(CSECachingMapperMixin,
        hedge.optemplate.CollectorMixin, hedge.optemplate.CombineMapper):
    map_common_subexpression_uncached = \
            hedge.optemplate.CombineMapper.map_common_subexpression

    def map_whole_domain_flux(self, wdflux):
        result = set([wdflux])

        for intr in wdflux.interiors:
            result |= self.rec(intr.field_expr)
        for bdry in wdflux.boundaries:
            result |= self.rec(bdry.bpair)

        return result




class BoundOperatorCollector(hedge.optemplate.BoundOperatorCollector):
    def map_whole_domain_flux(self, expr):
        result = set()

        for ii in expr.interiors:
            result.update(self.rec(ii.field_expr))

        for bi in expr.boundaries:
            result.update(self.rec(bi.bpair.field))
            result.update(self.rec(bi.bpair.bfield))

        return result
