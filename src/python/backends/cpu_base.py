"""Base functionality for both JIT and dynamic CPU backends."""

from __future__ import division

__copyright__ = "Copyright (C) 2007 Andreas Kloeckner"

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




import pymbolic.mapper
import hedge.optemplate




class ExecutionMapper(hedge.optemplate.Evaluator,
        hedge.optemplate.BoundOpMapperMixin, 
        hedge.optemplate.LocalOpReducerMixin):
    def __init__(self, context, discr):
        pymbolic.mapper.evaluator.EvaluationMapper.__init__(self, context)
        self.discr = discr
        self.diff_rst_cache = {}

        from hedge._internal import perform_double_sided_flux, lift_flux
        from hedge.tools import time_count_flop_if_instrumented
        if discr.instrumented:
            self.map_diff_base = \
                    time_count_flop_if_instrumented(
                            self.map_diff_base,
                            self.discr.diff_timer,
                            self.discr.diff_counter,
                            self.discr.diff_flop_counter,
                            self.diff_flops(self.discr))
            self.map_mass_base = \
                    time_count_flop_if_instrumented(
                            self.map_mass_base,
                            self.discr.mass_timer,
                            self.discr.mass_counter,
                            self.discr.mass_flop_counter,
                            self.mass_flops(self.discr))

            gather_flops = 0
            for eg in self.discr.element_groups:
                ldis = eg.local_discretization
                gather_flops += (
                        ldis.face_node_count()
                        * ldis.face_count()
                        * len(eg.members)
                        * (1 # facejac-mul
                            + 2 * # int+ext
                            3 # const-mul, normal-mul, add
                            )
                        )

            self.perform_double_sided_flux = \
                    time_count_flop_if_instrumented(
                            perform_double_sided_flux,
                            self.discr.gather_timer,
                            self.discr.gather_counter,
                            self.discr.gather_flop_counter,
                            gather_flops)

            from hedge._internal import lift_flux
            self.lift_flux = \
                    time_count_flop_if_instrumented(
                            lift_flux,
                            self.discr.lift_timer,
                            self.discr.lift_counter,
                            self.discr.lift_flop_counter,
                            self.lift_flops(self.discr))
        else:
            self.perform_double_sided_flux = perform_double_sided_flux
            self.lift_flux = lift_flux

    # implementation stuff ----------------------------------------------------
    def diff_rst(self, op, rst_axis, field):
        result = self.discr.volume_zeros()

        from hedge.tools import make_vector_target
        target = make_vector_target(field, result)

        target.begin(len(self.discr), len(self.discr))

        from hedge._internal import perform_elwise_operator
        for eg in self.discr.element_groups:
            perform_elwise_operator(eg.ranges, eg.ranges, 
                    op.matrices(eg)[rst_axis], target)

        target.finalize()

        return result

    def diff_xyz(self, op, expr, field, result):
        try:
            rst_derivatives = self.diff_rst_cache[op.__class__, expr]
        except KeyError:
            rst_derivatives = self.diff_rst_cache[op.__class__, expr] = \
                    [self.diff_rst(op, i, field) 
                            for i in range(self.discr.dimensions)]

        from hedge.tools import make_vector_target
        from hedge._internal import perform_elwise_scale

        for rst_axis in range(self.discr.dimensions):
            target = make_vector_target(rst_derivatives[rst_axis], result)

            target.begin(len(self.discr), len(self.discr))
            for eg in self.discr.element_groups:
                perform_elwise_scale(eg.ranges,
                        op.coefficients(eg)[op.xyz_axis][rst_axis],
                        target)
            target.finalize()
        return result

    def scalar_inner_flux(self, int_coeff, ext_coeff, field, lift, out=None):
        if out is None:
            out = self.discr.volume_zeros()

        if isinstance(field, (int, float, complex)) and field == 0:
            return 0

        from hedge._internal import ChainedFlux

        for fg in self.discr.face_groups:
            fluxes_on_faces = numpy.zeros(
                    (fg.face_count*fg.face_length()*fg.element_count(),),
                    dtype=field.dtype)
            
            self.perform_double_sided_flux(fg, 
                    ChainedFlux(int_coeff), ChainedFlux(ext_coeff),
                    field, fluxes_on_faces)

            if lift:
                self.lift_flux(fg, fg.ldis_loc.lifting_matrix(),
                        fg.local_el_inverse_jacobians, fluxes_on_faces, out)
            else:
                self.lift_flux(fg, fg.ldis_loc.multi_face_mass_matrix(),
                        None, fluxes_on_faces, out)

        return out


    def scalar_bdry_flux(self, int_coeff, ext_coeff, field, bfield, tag, lift, out=None):
        if out is None:
            out = self.discr.volume_zeros()

        bdry = self.discr.get_boundary(tag)
        if not bdry.nodes:
            return 0

        from hedge._internal import \
                perform_single_sided_flux, ChainedFlux, ZeroVector, \
                lift_flux
        if isinstance(field, (int, float, complex)) and field == 0:
            field = ZeroVector()
            dtype = bfield.dtype
        else:
            dtype = field.dtype

        if isinstance(bfield, (int, float, complex)) and bfield == 0:
            bfield = ZeroVector()

        if bdry.nodes:
            for fg in bdry.face_groups:
                fluxes_on_faces = numpy.zeros(
                        (fg.face_count*fg.face_length()*fg.element_count(),),
                        dtype=dtype)

                perform_single_sided_flux(
                        fg, ChainedFlux(int_coeff), ChainedFlux(ext_coeff),
                        field, bfield, fluxes_on_faces)

                if lift:
                    lift_flux(fg, fg.ldis_loc.lifting_matrix(),
                            fg.local_el_inverse_jacobians, 
                            fluxes_on_faces, out)
                else:
                    lift_flux(fg, fg.ldis_loc.multi_face_mass_matrix(),
                            None, 
                            fluxes_on_faces, out)

        return out




    # entry points ------------------------------------------------------------
    def map_diff_base(self, op, field_expr, out=None):
        field = self.rec(field_expr)

        if out is None:
            out = self.discr.volume_zeros()
        self.diff_xyz(op, field_expr, field, out)
        return out

    def map_mass_base(self, op, field_expr, out=None):
        field = self.rec(field_expr)

        if isinstance(field, (float, int)) and field == 0:
            return 0

        from hedge.tools import make_vector_target
        if out is None:
            out = self.discr.volume_zeros()

        from hedge._internal import perform_elwise_scaled_operator
        target = make_vector_target(field, out)

        target.begin(len(self.discr), len(self.discr))
        for eg in self.discr.element_groups:
            perform_elwise_scaled_operator(eg.ranges, eg.ranges,
                   op.coefficients(eg), op.matrix(eg), 
                   target)
        target.finalize()

        return out

    def map_flux_coefficient(self, op, field_expr, out=None, lift=False):
        from hedge.optemplate import BoundaryPair

        if isinstance(field_expr, BoundaryPair):
            bp = field_expr
            return self.scalar_bdry_flux(
                    op.int_coeff, op.ext_coeff,
                    self.rec(bp.field), self.rec(bp.bfield), 
                    bp.tag, lift, out)
        else:
            field = self.rec(field_expr)
            return self.scalar_inner_flux(
                    op.int_coeff, op.ext_coeff,
                    field, lift, out)

    def map_lift_coefficient(self, op, field_expr, out=None):
        return self.map_flux_coefficient(op, field_expr, out, lift=True)

    def map_sum(self, expr, out=None):
        if out is None:
            out = self.discr.volume_zeros()
        for child in expr.children:
            result = self.rec(child, out)
            if result is not out:
                out += result
        return out

    def map_product(self, expr, out=None):
        return hedge.optemplate.Evaluator.map_product(self, expr)

    def map_variable(self, expr, out=None):
        return hedge.optemplate.Evaluator.map_variable(self, expr)

    def map_sum(self, expr, out=None):
        if out is None:
            out = self.discr.volume_zeros()
        for child in expr.children:
            result = self.rec(child, out)
            if result is not out:
                out += result
        return out

    def map_product(self, expr, out=None):
        return hedge.optemplate.Evaluator.map_product(self, expr)

    def map_variable(self, expr, out=None):
        return hedge.optemplate.Evaluator.map_variable(self, expr)

