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



import pymbolic.primitives
import pymbolic.mapper
import hedge.optemplate




class StringifyMapper(hedge.optemplate.StringifyMapper):
    def map_whole_domain_flux(self, expr, enclosing_prec):
        if expr.is_lift:
            tag = "WLift"
        else:
            tag = "WFlux"

        from pymbolic.mapper.stringifier import PREC_NONE
        return "%s(%s)" % (tag, self.rec(expr.flux_optemplate, PREC_NONE))
        return "%s(is_lift=%s, int=%s, tag->bdry=%s, bdry->flux=%s)" % (tag, 
                expr.is_lift,
                expr.interiors,
                expr.tag_to_bdry_id,
                ["%s:%s" % item for item in expr.bdry_id_to_fluxes.iteritems()])




def get_flux_dependencies(flux, field):
    """For a multi-dependency scalar flux passed in,
    return a list of tuples C{(in_field_idx, int, ext)}, where C{int}
    and C{ext} are the (expressions of the) flux coefficients for the
    dependency with number C{in_field_idx}.
    """

    def in_fields_cmp(a, b):
        return cmp(a.index, b.index) \
                or cmp(a.is_local, b.is_local)

    from hedge.flux import FluxDependencyMapper, FieldComponent
    in_fields = list(FluxDependencyMapper(composite_leaves=True)(flux))

    # check that all in_fields are FieldComponent instances
    for in_field in in_fields:
        if not isinstance(in_field, FieldComponent):
            raise ValueError, "flux depends on invalid term `%s'" % str(in_field)
        
    in_fields.sort(in_fields_cmp)

    from hedge.tools import is_zero
    from hedge.optemplate import BoundaryPair
    if isinstance(field, BoundaryPair):
        for inf in in_fields:
            if inf.is_local:
                if not is_zero(field.field[inf.index]):
                    yield field.field[inf.index]
            else:
                if not is_zero(field.bfield[inf.index]):
                    yield field.bfield[inf.index]
    else:
        for inf in in_fields:
            if not is_zero(field[inf.index]):
                yield field[inf.index]




class WholeDomainFluxOperator(pymbolic.primitives.Leaf):
    def __init__(self, is_lift, interiors, boundaries, 
            flux_optemplate=None):
        self.is_lift = is_lift

        from pytools import Record
        self.interiors = interiors
        self.boundaries = boundaries

        def set_sum(set_iterable):
            from operator import or_
            return reduce(or_, set_iterable, set())

        interior_deps = set_sum(
                set(get_flux_dependencies(iflux.flux_expr, iflux.field_expr))
                for iflux in interiors)
        boundary_deps = set_sum(
                set(get_flux_dependencies(bflux.flux_expr, bflux.bpair))
                for bflux in boundaries)

        self.interior_deps = list(interior_deps)
        self.boundary_deps = list(boundary_deps)
        self.all_deps = list(interior_deps|boundary_deps)

        self.tag_to_bdry_id = {}
        self.bdry_id_to_fluxes = {}

        for bflux in boundaries:
            bdry_id = self.tag_to_bdry_id.setdefault(
                    bflux.bpair.tag, len(self.tag_to_bdry_id))
            self.bdry_id_to_fluxes.setdefault(bdry_id, []).append(bflux)
                
        self.flux_optemplate = flux_optemplate

    def get_hash(self):
        return hash((self.__class__, self.flux_optemplate))

    def is_equal(self, other):
        return (other.__class__ == WholeDomainFluxOperator
                and self.flux_optemplate == other.flux_optemplate)

    def __getinitargs__(self):
        return self.is_lift, self.interiors, self.boundaries
        
    @staticmethod
    def short_name(field):
        from pymbolic.primitives import Subscript
        if isinstance(field, Subscript):
            return "%s%d" % (field.aggregate, field.index)
        else:
            return str(field)

    def stringifier(self):
        return StringifyMapper

    def get_mapper_method(self, mapper): 
        return mapper.map_whole_domain_flux




class BoundaryCombiner(hedge.optemplate.OpTemplateIdentityMapper):
    """Combines inner fluxes and boundary fluxes on disjoint parts of the
    boundary into a single, whole-domain operator of type
    L{hedge.backends.cuda.execute.WholeDomainFluxOperator}.
    """
    def __init__(self, mesh):
        self.mesh = mesh

    def handle_unsupported_expression(self, expr):
        return expr

    def map_sum(self, expr):
        from hedge.optemplate import OperatorBinding, \
                FluxOperator, LiftingFluxOperator, \
                Field, BoundaryPair

        flux_op_types = (FluxOperator, LiftingFluxOperator)

        def is_valid_arg(arg):
            from pymbolic.primitives import Subscript, Variable
            from hedge.tools import is_obj_array
            if isinstance(arg, BoundaryPair):
                return is_valid_arg(arg.field) and is_valid_arg(arg.bfield)
            elif isinstance(arg, Subscript):
                return isinstance(arg.aggregate, Variable) and isinstance(arg.index, int)
            elif is_obj_array(arg):
                for entry in arg:
                    if not is_valid_arg(entry):
                        return False
                return True
            else:
                return isinstance(arg, Variable) or (
                        isinstance(arg, (int, float)) and arg == 0)

        def gather_one_wdflux(expressions):
            interiors = []
            boundaries = []
            is_lift = None

            rest = []
            flux_optemplate_summands = []

            from pytools import Record
            class BoundaryInfo(Record): pass
            class InteriorInfo(Record): pass

            for ch in expressions:
                if (isinstance(ch, OperatorBinding) 
                        and isinstance(ch.op, flux_op_types)
                        and is_valid_arg(ch.field)):
                    my_is_lift = isinstance(ch.op, LiftingFluxOperator),

                    if is_lift is None:
                        is_lift = my_is_lift
                    else:
                        if is_lift != my_is_lift:
                            rest.append(ch)
                            continue

                    flux_optemplate_summands.append(ch)

                    if isinstance(ch.field, BoundaryPair):
                        bpair = ch.field
                        if self.mesh.tag_to_boundary.get(bpair.tag, []):
                            boundaries.append(BoundaryInfo(
                                flux_expr=ch.op.flux,
                                bpair=bpair,
                                ))
                    else:
                        interiors.append(InteriorInfo(
                                flux_expr=ch.op.flux,
                                field_expr=ch.field,
                                short_name=WholeDomainFluxOperator.short_name(ch.field)))
                else:
                    rest.append(ch)

            if interiors or boundaries:
                from pymbolic.primitives import flattened_sum
                wdf = WholeDomainFluxOperator(
                        is_lift=is_lift,
                        interiors=interiors,
                        boundaries=boundaries,
                        flux_optemplate=flattened_sum(flux_optemplate_summands))
            else:
                wdf = None
            return wdf, rest

        result = 0
        
        from pymbolic.primitives import flattened_sum
        expressions = expr.children
        while True:
            wdf, expressions = gather_one_wdflux(expressions)
            if wdf is not None:
                result += wdf
            else:
                return result + flattened_sum(self.rec(r_i) for r_i in expressions)




class BoundaryTagCollector(pymbolic.mapper.CombineMapper):
    def combine(self, values):
        from pytools import flatten
        return set(flatten(values))

    def map_algebraic_leaf(self, expr):
        return set()

    def handle_unsupported_expression(self, expr):
        return set()

    def map_constant(self, bpair):
        return set()
    
    def map_boundary_pair(self, bpair):
        return set([bpair.tag])

    def map_operator_binding(self, expr):
        return self.rec(expr.field)




class FluxCollector(hedge.optemplate.LocalOpReducerMixin,
        pymbolic.mapper.CombineMapper):
    def combine(self, values):
        from pytools import flatten
        return set(flatten(values))

    def map_algebraic_leaf(self, expr):
        return set()

    def map_constant(self, bpair):
        return set()
    
    def map_operator_binding(self, expr):
        return self.rec(expr.op) | self.rec(expr.field)

    def map_whole_domain_flux(self, wdflux):
        return set([wdflux])

    def map_common_subexpression(self, expr):
        return self.rec(expr.child)

    def map_mass_base(self, expr):
        return set()
    
    def map_diff_base(self, expr):
        return set()


