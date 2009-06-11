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




import hedge.backends.cpu_base
import hedge.discretization
import hedge.optemplate
from hedge.backends.exec_common import \
        CPUExecutorBase, \
        CPUExecutionMapperBase
import numpy




# exec mapper -----------------------------------------------------------------
class ExecutionMapper(CPUExecutionMapperBase):
    # code execution functions ------------------------------------------------
    def exec_assign(self, insn):
        if self.discr.instrumented:
            sub_timer = self.discr.vector_math_timer.start_sub_timer()
            result = self(insn.expr)
            sub_timer.stop().submit()

            from hedge.tools import count_dofs
            self.discr.vector_math_flop_counter.add(count_dofs(result)*insn.flop_count)
        else:
            result = self(insn.expr)

        return [(insn.name, result)], []

    def exec_vector_expr_assign(self, insn):
        if self.discr.instrumented:
            def stats_callback(n, vec_expr):
                self.discr.vector_math_flop_counter.add(n*vec_expr.flop_count)
                return self.discr.vector_math_timer
        else:
            stats_callback = None

        return [(insn.name, insn.compiled(self, stats_callback))], []

    def exec_flux_batch_assign(self, insn):
        from hedge.backends.jit.compiler import BoundaryFluxKind
        is_bdry = isinstance(insn.kind, BoundaryFluxKind)

        from pymbolic.primitives import is_zero

        def eval_arg(arg_spec):
            arg_expr, is_int = arg_spec
            arg = self.rec(arg_expr)
            if is_zero(arg):
                if is_bdry and not is_int:
                    return self.discr.boundary_zeros(insn.kind.tag)
                else:
                    return self.discr.volume_zeros()
            else:
                return arg

        args = [eval_arg(arg_expr) for arg_expr in insn.arg_specs]

        if is_bdry:
            bdry = self.discr.get_boundary(insn.kind.tag)
            face_groups = bdry.face_groups
        else:
            face_groups = self.discr.face_groups

        result = []

        for fg in face_groups:
            fof_shape = (fg.face_count*fg.face_length()*fg.element_count(),)
            all_fluxes_on_faces = [
                    numpy.zeros(fof_shape, dtype=self.discr.default_scalar_type)
                    for f in insn.fluxes]
            insn.compiled_func(fg, *(all_fluxes_on_faces+args))
            
            for name, flux, fluxes_on_faces in zip(insn.names, insn.fluxes, 
                    all_fluxes_on_faces):
                from hedge.optemplate import LiftingFluxOperator

                out = self.discr.volume_zeros()
                if isinstance(flux.op, LiftingFluxOperator):
                    self.executor.lift_flux(fg, fg.ldis_loc.lifting_matrix(),
                            fg.local_el_inverse_jacobians, fluxes_on_faces, out)
                else:
                    self.executor.lift_flux(fg, fg.ldis_loc.multi_face_mass_matrix(),
                            None, fluxes_on_faces, out)

                result.append((name, out))

        if not face_groups:
            # No face groups? Still assign context variables.
            for name, flux in zip(insn.names, insn.fluxes):
                result.append((name, self.discr.volume_zeros()))

        return result, []

    def exec_diff_batch_assign(self, insn):
        xyz_diff = self.executor.diff(insn.op_class, self.rec(insn.field),
                xyz_needed=[op.xyz_axis for op in insn.operators])

        return [(name, diff)
                for name, op, diff in zip(
                    insn.names, insn.operators, xyz_diff)], []

    def exec_mass_assign(self, insn):
        field = self.rec(insn.field)

        if isinstance(field, (float, int)) and field == 0:
            return 0

        out = self.discr.volume_zeros(dtype=field.dtype)
        self.executor.do_mass(insn.op_class, field, out)

        return [(insn.name, out)], []




class Executor(CPUExecutorBase):
    def __init__(self, discr, optemplate, post_bind_mapper):
        self.discr = discr
        self.code = self.compile_optemplate(discr, optemplate, post_bind_mapper)

        if "print_op_code" in discr.debug:
	    from hedge.tools import get_rank
	    if get_rank(discr) == 0:
	        print self.code
	        raw_input()

	if "print_op_code" in discr.debug:
	    from hedge.tools import get_rank
	    if get_rank(discr) == 0:
		print self.code
		raw_input()

        def bench_diff(f):
            test_field = discr.volume_zeros()
            from hedge.optemplate import DifferentiationOperator
            from time import time

            xyz_needed = range(discr.dimensions)

            start = time()
            f(DifferentiationOperator, test_field, xyz_needed)
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

    def compile_optemplate(self, discr, optemplate, post_bind_mapper):
        from hedge.optemplate import \
                OperatorBinder, \
                InverseMassContractor, \
                BCToFluxRewriter, \
                EmptyFluxKiller

        from hedge.optemplate import CommutativeConstantFoldingMapper

        prepared_optemplate = (
                InverseMassContractor()(
                    EmptyFluxKiller(discr)(
                        CommutativeConstantFoldingMapper()(
                            post_bind_mapper(
                                BCToFluxRewriter()(
                                    OperatorBinder()(
                                        optemplate)))))))
        from hedge.backends.jit.compiler import OperatorCompiler
        return OperatorCompiler(discr)(prepared_optemplate)

    def diff_builtin(self, op_class, field, xyz_needed):
        rst_derivatives = [
                self.diff_rst(op_class, i, field) 
                for i in range(self.discr.dimensions)]

        return [self.diff_rst_to_xyz(op_class(i), rst_derivatives)
                for i in xyz_needed]

    def __call__(self, **context):
        return self.code.execute(
                self.discr.exec_mapper_class(context, self))






# discretization --------------------------------------------------------------
class Discretization(hedge.discretization.Discretization):
    exec_mapper_class = ExecutionMapper
    executor_class = Executor

    def __init__(self, *args, **kwargs):
        hedge.discretization.Discretization.__init__(self, *args, **kwargs)

        toolchain = kwargs.pop("toolchain", None)

        if toolchain is None:
            from codepy.jit import guess_toolchain
            toolchain = guess_toolchain()

        toolchain = toolchain.with_max_optimization()
        
        from codepy.libraries import add_hedge
        add_hedge(toolchain)

        self._toolchain = toolchain

    def nodewise_max(self,a):
        return numpy.max(a)
