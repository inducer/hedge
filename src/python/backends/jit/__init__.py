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
from hedge.backends.cpu_base import ExecutorBase, ExecutionMapperBase
from pymbolic.mapper.c_code import CCodeMapper
import numpy




# exec mapper -----------------------------------------------------------------
class ExecutionMapper(ExecutionMapperBase):
    # code execution functions ------------------------------------------------
    def exec_discard(self, insn):
        del self.context[insn.name]

    def exec_assign(self, insn):
        self.context[insn.name] = self(insn.expr)

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

                self.context[name] = out

    def exec_diff_batch_assign(self, insn):
        xyz_diff = self.executor.diff(insn.op_class, self.rec(insn.field),
                xyz_needed=[op.xyz_axis for op in insn.operators])

        for name, op, diff in zip(insn.names, insn.operators, xyz_diff):
            self.context[name] = diff

    # mapper functions --------------------------------------------------------
    def map_mass_base(self, op, field_expr):
        field = self.rec(field_expr)

        if isinstance(field, (float, int)) and field == 0:
            return 0

        out = self.discr.volume_zeros(dtype=field.dtype)
        self.executor.do_mass(op, field, out)
        return out




class Executor(ExecutorBase):
    def __init__(self, discr, op_data):
        ExecutorBase.__init__(self, discr, op_data)

        def bench_diff(f):
            test_field = discr.volume_zeros()
            from hedge.optemplate import DifferentiationOperator
            from time import time

            xyz_needed = range(discr.dimensions)

            start = time()
            f(DifferentiationOperator, test_field, xyz_needed)
            return time() - start

        def bench_lift(f):
            fg = discr.face_groups[0]
            out = discr.volume_zeros()
            from hedge.optemplate import DifferentiationOperator
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

    def diff_builtin(self, op_class, field, xyz_needed):
        rst_derivatives = [
                self.diff_rst(op_class, i, field) 
                for i in range(self.discr.dimensions)]

        return [self.diff_rst_to_xyz(op_class(i), rst_derivatives)
                for i in xyz_needed]






# discretization --------------------------------------------------------------
class Discretization(hedge.discretization.Discretization):
    exec_mapper_class = ExecutionMapper

    def __init__(self, *args, **kwargs):
        hedge.discretization.Discretization.__init__(self, *args, **kwargs)

        plat = kwargs.pop("platform", None)

        if plat is None:
            from codepy.jit import guess_platform
            plat = guess_platform()

        plat = plat.with_max_optimization()
        
        from codepy.libraries import add_hedge
        add_hedge(plat)

        self._platform = plat

    def compile(self, optemplate, post_bind_mapper=lambda x: x):
        from hedge.optemplate import \
                OperatorBinder, \
                InverseMassContractor, \
                BCToFluxRewriter, \
                EmptyFluxKiller

        from hedge.optemplate import CommutativeConstantFoldingMapper

        prepared_optemplate = (
                InverseMassContractor()(
                    EmptyFluxKiller(self)(
                        CommutativeConstantFoldingMapper()(
                            post_bind_mapper(
                                BCToFluxRewriter()(
                                    OperatorBinder()(
                                        optemplate)))))))

        from hedge.backends.jit.compiler import OperatorCompiler
        ex = Executor(self, 
                OperatorCompiler(self)(prepared_optemplate))
        ex.instrument()
        return ex
