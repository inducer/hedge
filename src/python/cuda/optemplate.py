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



import hedge.optemplate




class StringifyMapper(hedge.optemplate.StringifyMapper):
    def map_whole_domain_flux(self, expr, enclosing_prec):
        if expr.is_lift:
            tag = "WLift"
        else:
            tag = "WFlux"

        return "%s(int=%s, ext=%s, %s)" % (tag, 
                expr.int_coeff,
                expr.ext_coeff,
                expr.boundaries)




class WholeDomainFluxOperator(hedge.optemplate.Operator):
    def __init__(self, discr, is_lift, int_coeff, ext_coeff, boundaries, 
            flux_optemplate=None):
        """@arg boundaries: A list of C{(tag, int_coeff, ext_coeff, bfield)} tuples.
        """
        flux_to_number = {}
        def register_flux(int_coeff, ext_coeff):
            try:
                return flux_to_number[int_coeff, ext_coeff]
            except KeyError:
                number = len(self.fluxes)
                flux_to_number[int_coeff, ext_coeff] = number
                self.fluxes.append((int_coeff, ext_coeff))
                return number

        hedge.optemplate.Operator.__init__(self, discr)
        self.fluxes = []
        self.is_lift = is_lift
        self.interior_flux_number = \
                register_flux(int_coeff, ext_coeff)
        from pytools import Record
        self.boundaries = [
                Record(
                    tag=tag,
                    flux_number=register_flux(int_coeff, ext_coeff),
                    bfield_expr=bfield_expr)
                for tag, int_coeff, ext_coeff, bfield_expr in boundaries]
                
        self.flux_optemplate = flux_optemplate

        self.elface_to_boundary = {}
        for b in self.boundaries:
            for elface in discr.mesh.tag_to_boundary.get(b.tag, []):
                if elface in self.elface_to_boundary:
                    raise ValueError, "face contained in two boundaries of WholeDomainFlux"
                self.elface_to_boundary[elface] = b

    def elface_to_flux_number(self, elface):
        try:
            return self.elface_to_boundary[elface].flux_number
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
        arg_to_flux = {}

        from hedge.optemplate import OperatorBinding, \
                FluxCoefficientOperator, LiftingFluxCoefficientOperator, \
                Field, BoundaryPair
        flux_op_types = (FluxCoefficientOperator, LiftingFluxCoefficientOperator)
        valid_arg_types = (Field, BoundaryPair)

        result = []
        for ch in expr.children:
            if (isinstance(ch, OperatorBinding) 
                    and isinstance(ch.op, flux_op_types)
                    and isinstance(ch.field, valid_arg_types)):
                arg_to_flux.setdefault(ch.field, []).append(ch.op)
            else:
                result.append(self.rec(ch))

        for inner_var in arg_to_flux:
            if isinstance(inner_var, BoundaryPair):
                # not an inner-flux variable
                continue

            inner_flux_ops = arg_to_flux[inner_var]
            try:
                inner_flux_op = inner_flux_ops.pop()
            except IndexError:
                # empty already--that's ok
                continue

            flux_optemplate_summands = [OperatorBinding(inner_flux_op, inner_var)]

            boundaries = []
            for bp in arg_to_flux:
                if isinstance(bp, BoundaryPair) and bp.field == inner_var:
                    bflux_ops = arg_to_flux[bp]
                    for i in range(len(bflux_ops)):
                        if isinstance(bflux_ops[i], type(inner_flux_op)):
                            bflux_op = bflux_ops.pop(i)
                            boundaries.append((
                                bp.tag,
                                bflux_op.int_coeff,
                                bflux_op.ext_coeff,
                                bp.bfield))
                            flux_optemplate_summands.append(
                                    OperatorBinding(bflux_op, bp))

            from pymbolic import flattened_sum
            wflux = WholeDomainFluxOperator(
                    self.discr,
                    is_lift=isinstance(inner_flux_op, LiftingFluxCoefficientOperator),
                    int_coeff=inner_flux_op.int_coeff,
                    ext_coeff=inner_flux_op.ext_coeff,
                    boundaries=boundaries,
                    flux_optemplate=flattened_sum(flux_optemplate_summands))

            from hedge.optemplate import OperatorBinding
            result.append(OperatorBinding(wflux, inner_var))

        # make sure we got everything
        for var, flux_ops in arg_to_flux.iteritems():
            assert not flux_ops

        from pymbolic.primitives import flattened_sum
        return flattened_sum(result)
