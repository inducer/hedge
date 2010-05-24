"""Just-in-time compiling backend."""

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




import hedge.discretization
import hedge.optemplate
from hedge.backends.exec_common import ExecutionMapperBase
import numpy




# {{{ exec mapper -------------------------------------------------------------
class ExecutionMapper(ExecutionMapperBase):
    # {{{ code execution functions --------------------------------------------
    def exec_assign(self, insn):
        return [(name, self.rec(expr))
                for name, expr in zip(insn.names, insn.exprs)], []

    def exec_vector_expr_assign(self, insn):
        if self.discr.instrumented:
            def stats_callback(n, vec_expr):
                self.discr.vector_math_flop_counter.add(n*insn.flop_count())
                return self.discr.vector_math_timer
        else:
            stats_callback = None

        if insn.flop_count() == 0:
            return [(name, self(expr))
                for name, expr in zip(insn.names, insn.exprs)], []
        else:
            compiled = insn.compiled(self.executor)
            return zip(compiled.result_names(),
                    compiled(self, stats_callback)), []

    def exec_flux_batch_assign(self, insn):
        from pymbolic.primitives import is_zero

        class ZeroSpec:
            pass
        class BoundaryZeros(ZeroSpec):
            pass
        class VolumeZeros(ZeroSpec):
            pass

        def eval_arg(arg_spec):
            arg_expr, is_int = arg_spec
            arg = self.rec(arg_expr)
            if is_zero(arg):
                if insn.is_boundary and not is_int:
                    return BoundaryZeros()
                else:
                    return VolumeZeros()
            else:
                return arg

        args = [eval_arg(arg_expr)
                for arg_expr in insn.flux_var_info.arg_specs]

        from pytools import common_dtype
        max_dtype = common_dtype(
                [a.dtype for a in args if not isinstance(a, ZeroSpec)],
                self.discr.default_scalar_type)

        def cast_arg(arg):
            if isinstance(arg, BoundaryZeros):
                return self.discr.boundary_zeros(
                        insn.repr_op.boundary_tag, dtype=max_dtype)
            elif isinstance(arg, VolumeZeros):
                return self.discr.volume_zeros(
                        dtype=max_dtype)
            elif isinstance(arg, numpy.ndarray):
                return numpy.asarray(arg, dtype=max_dtype)
            else:
                return arg

        args = [cast_arg(arg) for arg in args]

        if insn.quadrature_tag is None:
            if insn.is_boundary:
                face_groups = self.discr.get_boundary(insn.repr_op.boundary_tag)\
                        .face_groups
            else:
                face_groups = self.discr.face_groups
        else:
            if insn.is_boundary:
                face_groups = self.discr.get_boundary(insn.repr_op.boundary_tag)\
                        .get_quadrature_info(insn.quadrature_tag).face_groups
            else:
                face_groups = self.discr.get_quadrature_info(insn.quadrature_tag) \
                        .face_groups

        result = []

        for fg in face_groups:
            # grab module
            module = insn.get_module(self.discr, max_dtype)
            func = module.gather_flux

            # set up argument structure
            arg_struct = module.ArgStruct()
            for arg_name, arg in zip(insn.flux_var_info.arg_names, args):
                setattr(arg_struct, arg_name, arg)
            for arg_num, scalar_arg_expr in enumerate(insn.flux_var_info.scalar_parameters):
                setattr(arg_struct,
                        "_scalar_arg_%d" % arg_num,
                        self.rec(scalar_arg_expr))

            fof_shape = (fg.face_count*fg.face_length()*fg.element_count(),)
            all_fluxes_on_faces = [
                    numpy.zeros(fof_shape, dtype=max_dtype)
                    for f in insn.expressions]
            for i, fof in enumerate(all_fluxes_on_faces):
                setattr(arg_struct, "flux%d_on_faces" % i, fof)

            # make sure everything ended up in Boost.Python attributes
            # (i.e. empty __dict__)
            assert not arg_struct.__dict__, arg_struct.__dict__.keys()

            # perform gather
            func(fg, arg_struct)

            # do lift, produce output
            for name, flux_bdg, fluxes_on_faces in zip(insn.names, insn.expressions,
                    all_fluxes_on_faces):

                if insn.quadrature_tag is None:
                    if flux_bdg.op.is_lift:
                        mat = fg.ldis_loc.lifting_matrix()
                        scaling = fg.local_el_inverse_jacobians
                    else:
                        mat = fg.ldis_loc.multi_face_mass_matrix()
                        scaling = None
                else:
                    assert not flux_bdg.op.is_lift
                    mat = fg.ldis_loc_quad_info.multi_face_mass_matrix()
                    scaling = None

                out = self.discr.volume_zeros(dtype=fluxes_on_faces.dtype)
                self.executor.lift_flux(fg, mat, scaling, fluxes_on_faces, out)

                if self.discr.instrumented:
                    from hedge.tools import lift_flops

                    # correct for quadrature, too.
                    self.discr.lift_flop_counter.add(lift_flops(fg))

                result.append((name, out))

        if not face_groups:
            # No face groups? Still assign context variables.
            for name, flux_bdg in zip(insn.names, insn.expressions):
                result.append((name, self.discr.volume_zeros()))

        return result, []

    def exec_diff_batch_assign(self, insn):
        xyz_diff = self.executor.diff(insn.operators, self.rec(insn.field))

        return [(name, diff) for name, diff in zip(insn.names, xyz_diff)], []

    exec_quad_diff_batch_assign = exec_diff_batch_assign

    # }}}

    # {{{ expression mappings -------------------------------------------------
    def map_if_positive(self, expr):
        crit = self.rec(expr.criterion)
        bool_crit = crit > 0
        then = self.rec(expr.then)
        else_ = self.rec(expr.else_)

        true_indices = numpy.nonzero(bool_crit)
        false_indices = numpy.nonzero(~bool_crit)

        result = numpy.empty_like(crit)

        if isinstance(then, numpy.ndarray):
            then = then[true_indices]
        if isinstance(else_, numpy.ndarray):
            else_ = else_[false_indices]

        result[true_indices] = then
        result[false_indices] = else_
        return result

    def map_ref_diff_base(self, op, field_expr):
        raise NotImplementedError(
                "differentiation should be happening in batched form")

    def map_elementwise_linear(self, op, field_expr):
        field = self.rec(field_expr)

        from hedge.tools import is_zero
        if is_zero(field):
            return 0

        out = self.discr.volume_zeros()
        self.executor.do_elementwise_linear(op, field, out)
        return out

    def map_ref_quad_mass(self, op, field_expr):
        field = self.rec(field_expr)

        from hedge.tools import is_zero
        if is_zero(field):
            return 0

        qtag = op.quadrature_tag

        from hedge._internal import perform_elwise_operator

        out = self.discr.volume_zeros()
        for eg in self.discr.element_groups:
            eg_quad_info = eg.quadrature_info[qtag]

            perform_elwise_operator(eg_quad_info.ranges, eg.ranges,
                    eg_quad_info.ldis_quad_info.mass_matrix(),
                    field, out)

        return out

    def map_quad_grid_upsampler(self, op, field_expr):
        field = self.rec(field_expr)

        from hedge.tools import is_zero
        if is_zero(field):
            return 0

        qtag = op.quadrature_tag

        from hedge._internal import perform_elwise_operator
        quad_info = self.discr.get_quadrature_info(qtag)

        out = numpy.zeros(quad_info.node_count, field.dtype)
        for eg in self.discr.element_groups:
            eg_quad_info = eg.quadrature_info[qtag]

            perform_elwise_operator(eg.ranges, eg_quad_info.ranges,
                eg_quad_info.ldis_quad_info.volume_up_interpolation_matrix(),
                field, out)

        return out

    def map_quad_int_faces_grid_upsampler(self, op, field_expr):
        field = self.rec(field_expr)

        from hedge.tools import is_zero
        if is_zero(field):
            return 0

        qtag = op.quadrature_tag

        from hedge._internal import perform_elwise_operator
        quad_info = self.discr.get_quadrature_info(qtag)

        out = numpy.zeros(quad_info.int_faces_node_count, field.dtype)
        for eg in self.discr.element_groups:
            eg_quad_info = eg.quadrature_info[qtag]

            perform_elwise_operator(eg.ranges, eg_quad_info.el_faces_ranges,
                eg_quad_info.ldis_quad_info.volume_to_face_up_interpolation_matrix(),
                field, out)

        return out

    def map_quad_bdry_grid_upsampler(self, op, field_expr):
        field = self.rec(field_expr)

        from hedge.tools import is_zero
        if is_zero(field):
            return 0

        bdry = self.discr.get_boundary(op.boundary_tag)
        bdry_q_info = bdry.get_quadrature_info(op.quadrature_tag)

        out = numpy.zeros(bdry_q_info.node_count, field.dtype)

        from hedge._internal import perform_elwise_operator
        for fg, from_ranges, to_ranges, ldis_quad_info in zip(
                bdry.face_groups,
                bdry.fg_ranges,
                bdry_q_info.fg_ranges,
                bdry_q_info.fg_ldis_quad_infos):
            perform_elwise_operator(from_ranges, to_ranges,
                ldis_quad_info.face_up_interpolation_matrix(),
                field, out)

        return out

    def map_elementwise_max(self, op, field_expr):
        from hedge._internal import perform_elwise_max
        field = self.rec(field_expr)

        out = self.discr.volume_zeros(dtype=field.dtype)
        for eg in self.discr.element_groups:
            perform_elwise_max(eg.ranges, field, out)

        return out

    def map_call(self, expr):
        from pymbolic.primitives import Variable
        assert isinstance(expr.function, Variable)
        func_name = expr.function.name

        try:
            func = self.discr.exec_functions[func_name]
        except KeyError:
            func = getattr(numpy, expr.function.name)

        return func(*[self.rec(p) for p in expr.parameters])

    # }}}

# }}}

# {{{ executor ----------------------------------------------------------------
class Executor(object):
    def __init__(self, discr, optemplate, post_bind_mapper, type_hints):
        self.discr = discr
        self.code = self.compile_optemplate(discr, optemplate, 
                post_bind_mapper, type_hints)
        self.elwise_linear_cache = {}

        if "dump_op_code" in discr.debug:
            from hedge.tools import open_unique_debug_file
            open_unique_debug_file("op-code", ".txt").write(
                    str(self.code))

        def bench_diff(f):
            test_field = discr.volume_zeros()
            from hedge.optemplate import ReferenceDifferentiationOperator
            from time import time

            start = time()
            f([ReferenceDifferentiationOperator(i)
                for i in range(discr.dimensions)], test_field)
            return time() - start

        def bench_lift(f):
            if len(discr.face_groups) == 0:
                return 0

            fg = discr.face_groups[0]
            out = discr.volume_zeros()
            from time import time

            xyz_needed = range(discr.dimensions)

            fof_shape = (fg.face_count*fg.face_length()*fg.element_count(),)
            fof = numpy.zeros(fof_shape, dtype=self.discr.default_scalar_type)

            start = time()
            f(fg, fg.ldis_loc.lifting_matrix(), fg.local_el_inverse_jacobians, fof, out)
            return time() - start

        def pick_faster_func(benchmark, choices, attempts=3):
            from pytools import argmin2
            return argmin2(
                    (f, min(benchmark(f) for i in range(attempts)))
                    for f in choices)

        from hedge.backends.jit.diff import JitDifferentiator
        self.diff = pick_faster_func(bench_diff,
                [self.diff_builtin, JitDifferentiator(discr)])
        from hedge.backends.jit.lift import JitLifter
        self.lift_flux = pick_faster_func(bench_lift,
                [self.lift_flux, JitLifter(discr)])

    def compile_optemplate(self, discr, optemplate, post_bind_mapper,
            type_hints):
        from hedge.optemplate import process_optemplate

        stage = [0]

        def dump_optemplate(name, optemplate):
            if "dump_optemplate_stages" in discr.debug:
                from hedge.tools import open_unique_debug_file
                from hedge.optemplate import pretty_print_optemplate
                open_unique_debug_file("%02d-%s" % (stage[0], name), ".txt").write(
                        pretty_print_optemplate(optemplate))
                stage[0] += 1

        optemplate = process_optemplate(optemplate,
                post_bind_mapper=post_bind_mapper,
                dumper=dump_optemplate,
                mesh=discr.mesh,
                type_hints=type_hints)

        from hedge.backends.jit.compiler import OperatorCompiler
        return OperatorCompiler(discr)(optemplate, type_hints)

    def instrument(self):
        discr = self.discr
        assert discr.instrumented

        from pytools.log import time_and_count_function
        from hedge.tools import time_count_flop

        from hedge.tools import \
                diff_rst_flops, diff_rescale_one_flops, mass_flops

        if discr.quad_min_degrees:
            from warnings import warn
            warn("flop counts for quadrature may be wrong")

        self.diff_rst = \
                time_count_flop(
                        self.diff_rst,
                        discr.diff_timer,
                        discr.diff_counter,
                        discr.diff_flop_counter,
                        diff_rst_flops(discr))

        self.do_elementwise_linear = \
                time_count_flop(
                        self.do_elementwise_linear,
                        discr.el_local_timer,
                        discr.el_local_counter,
                        discr.el_local_flop_counter,
                        mass_flops(discr))

        self.lift_flux = \
                time_and_count_function(
                        self.lift_flux,
                        discr.lift_timer,
                        discr.lift_counter)

    def lift_flux(self, fgroup, matrix, scaling, field, out):
        from hedge._internal import lift_flux
        from pytools import to_uncomplex_dtype
        lift_flux(fgroup,
                matrix.astype(to_uncomplex_dtype(field.dtype)),
                scaling, field, out)

    def diff_rst(self, op, field):
        result = self.discr.volume_zeros(dtype=field.dtype)

        from hedge._internal import perform_elwise_operator
        for eg in self.discr.element_groups:
            perform_elwise_operator(op.preimage_ranges(eg), eg.ranges,
                    op.matrices(eg)[op.rst_axis].astype(field.dtype),
                    field, result)

        return result

    def diff_builtin(self, operators, field):
        """For the batch of reference differentiation operators in
        *operators*, return the local corresponding derivatives of
        *field*.
        """

        return [self.diff_rst(op, field) for op in operators]

    def do_elementwise_linear(self, op, field, out):
        for eg in self.discr.element_groups:
            try:
                matrix, coeffs = self.elwise_linear_cache[eg, op, field.dtype]
            except KeyError:
                matrix = numpy.asarray(op.matrix(eg), dtype=field.dtype)
                coeffs = op.coefficients(eg)
                self.elwise_linear_cache[eg, op, field.dtype] = matrix, coeffs

            from hedge._internal import (
                    perform_elwise_scaled_operator,
                    perform_elwise_operator)

            if coeffs is None:
                perform_elwise_operator(eg.ranges, eg.ranges,
                        matrix, field, out)
            else:
                perform_elwise_scaled_operator(eg.ranges, eg.ranges,
                        coeffs, matrix, field, out)

    def __call__(self, **context):
        return self.code.execute(
                self.discr.exec_mapper_class(context, self))

# }}}

# {{{ discretization ----------------------------------------------------------
class Discretization(hedge.discretization.Discretization):
    exec_mapper_class = ExecutionMapper
    executor_class = Executor

    @classmethod
    def all_debug_flags(cls):
        return hedge.discretization.Discretization.all_debug_flags() | set([
            "jit_dont_optimize_large_exprs",
            ])

    @classmethod
    def noninteractive_debug_flags(cls):
        return hedge.discretization.Discretization.noninteractive_debug_flags() | set([
            "jit_dont_optimize_large_exprs",
            ])

    def __init__(self, *args, **kwargs):
        toolchain = kwargs.pop("toolchain", None)

        # tolerate (and ignore) the CUDA backend's tune_for argument
        _ = kwargs.pop("tune_for", None)

        hedge.discretization.Discretization.__init__(self, *args, **kwargs)

        if toolchain is None:
            from codepy.toolchain import guess_toolchain
            toolchain = guess_toolchain()
            toolchain = toolchain.with_optimization_level(3)

        from codepy.libraries import add_hedge
        add_hedge(toolchain)

        self.toolchain = toolchain

# }}}




# vim: foldmethod=marker
