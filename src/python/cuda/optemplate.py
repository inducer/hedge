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
import hedge.optemplate




class StringifyMapper(hedge.optemplate.StringifyMapper):
    def map_whole_domain_flux(self, expr, enclosing_prec):
        if expr.is_lift:
            tag = "WLift"
        else:
            tag = "WFlux"

        return "%s(int=%s, bdry=%s)" % (tag, 
                expr.interiors,
                expr.boundaries)




class WholeDomainFluxOperator(pymbolic.primitives.Leaf):
    def __init__(self, discr, is_lift, interiors, boundaries, 
            flux_optemplate=None):
        self.discr = discr
        self.is_lift = is_lift

        self.fluxes = []

        from pytools import Record
        self.interiors = interiors

        interior_deps = set(
                iflux.field_expr for iflux in interiors)
        boundary_deps = (
                set(bflux.field_expr for bflux in boundaries)
                |
                set(bflux.bfield_expr for bflux in boundaries)
                )
        self.interior_deps = list(interior_deps)
        self.boundary_deps = list(boundary_deps)
        self.all_deps = list(interior_deps|boundary_deps)

        tag_to_bdry_id = {}
        self.bdry_id_to_fluxes = {}

        for bflux in boundaries:
            bdry_id = tag_to_bdry_id.setdefault(bflux.tag, len(tag_to_bdry_id))
            self.bdry_id_to_fluxes.setdefault(bdry_id, []).append(bflux)
                
        self.flux_optemplate = flux_optemplate

        self.elface_to_bdry_id = {}
        for btag, bdry_id in tag_to_bdry_id.iteritems():
            for elface in discr.mesh.tag_to_boundary.get(btag, []):
                if elface in self.elface_to_bdry_id:
                    raise ValueError, "face contained in two boundaries of WholeDomainFlux"
                self.elface_to_bdry_id[elface] = bdry_id

    @staticmethod
    def short_name(field):
        from pymbolic.primitives import Subscript
        if isinstance(field, Subscript):
            return "%s%d" % (field.aggregate, field.index)
        else:
            return str(field)

    def boundary_elface_to_bdry_id(self, elface):
        try:
            return self.elface_to_bdry_id[elface]
        except KeyError:
            return len(self.fluxes)

    def stringifier(self):
        return StringifyMapper

    def get_mapper_method(self, mapper): 
        return mapper.map_whole_domain_flux




class BoundaryCombiner(hedge.optemplate.IdentityMapper):
    """Combines inner fluxes and boundary fluxes on disjoint parts of the
    boundary into a single, whole-domain operator.
    """
    def __init__(self, discr):
        self.discr = discr

    def handle_unsupported_expression(self, expr):
        return expr

    def map_sum(self, expr):
        from hedge.optemplate import OperatorBinding, \
                FluxCoefficientOperator, LiftingFluxCoefficientOperator, \
                Field, BoundaryPair

        flux_op_types = (FluxCoefficientOperator, LiftingFluxCoefficientOperator)

        def is_valid_arg(arg):
            from pymbolic.primitives import Subscript, Variable
            if isinstance(arg, BoundaryPair):
                return is_valid_arg(arg.field) and is_valid_arg(arg.bfield)
            elif isinstance(arg, Subscript):
                return isinstance(arg.aggregate, Variable) and isinstance(arg.index, int)
            else:
                return isinstance(arg, Variable)

        def gather_one_wdflux(expressions):
            interiors = []
            boundaries = []
            is_lift = None

            rest = []
            flux_optemplate_summands = []

            from pytools import Record
            for ch in expressions:
                if (isinstance(ch, OperatorBinding) 
                        and isinstance(ch.op, flux_op_types)
                        and is_valid_arg(ch.field)):
                    my_is_lift = isinstance(ch.op, LiftingFluxCoefficientOperator),

                    if is_lift is None:
                        is_lift = my_is_lift
                    else:
                        if is_lift != my_is_lift:
                            rest.append(ch)
                            continue

                    flux_optemplate_summands.append(ch)

                    if isinstance(ch.field, BoundaryPair):
                        bp = ch.field
                        if ch.op.discr.mesh.tag_to_boundary.get(bp.tag, []):
                            boundaries.append(Record(
                                tag=bp.tag,
                                int_coeff=ch.op.int_coeff,
                                ext_coeff=ch.op.ext_coeff,
                                field_expr=bp.field,
                                bfield_expr=bp.bfield,
                                field_short_name=\
                                        WholeDomainFluxOperator.short_name(bp.field),
                                bfield_short_name=
                                        WholeDomainFluxOperator.short_name(bp.bfield),
                                ))
                    else:
                        interiors.append(Record(
                                int_coeff=ch.op.int_coeff,
                                ext_coeff=ch.op.ext_coeff,
                                field_expr=ch.field,
                                short_name=WholeDomainFluxOperator.short_name(ch.field)))
                else:
                    rest.append(ch)

            if interiors or boundaries:
                from pymbolic.primitives import flattened_sum
                wdf = WholeDomainFluxOperator(
                        self.discr,
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
