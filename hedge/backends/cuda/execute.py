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



import numpy
import numpy.linalg as la
from pytools import memoize_method
import hedge.optemplate
from hedge.compiler import OperatorCompilerBase, \
        Assign, FluxBatchAssign
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import pymbolic.mapper.stringifier
from hedge.backends.exec_common import ExecutionMapperBase




# debug stuff -----------------------------------------------------------------
def get_vec_structure(vec, point_size, segment_size, block_size,
        other_char=lambda snippet: "."):
    """Prints a structured view of a vector--one character per `point_size` floats,
    `segment_size` characters partitioned off by spaces, `block_size` segments
    per line.

    The caracter printed is either an 'N' if any NaNs are encountered, a zero
    if the entire snippet is zero, or otherwise whatever `other_char` returns,
    defaulting to a period.
    """

    result = ""
    for block in range(len(vec) // block_size):
        struc = ""
        for segment in range(block_size//segment_size):
            for point in range(segment_size//point_size):
                offset = block*block_size + segment*segment_size + point*point_size
                snippet = vec[offset:offset+point_size]

                if numpy.isnan(snippet).any():
                    struc += "N"
                elif (snippet == 0).any():
                    struc += "0"
                else:
                    struc += other_char(snippet)

            struc += " "
        result += struc + "\n"
    return result




def print_error_structure(discr, computed, reference, diff,
        eventful_only=False, detail=True):
    norm_ref = la.norm(reference)
    struc_lines = []

    if norm_ref == 0:
        norm_ref = 1

    from hedge.tools import relative_error
    numpy.set_printoptions(precision=2, linewidth=130, suppress=True)
    for block in discr.blocks:
        add_lines = []
        struc_line  = "%7d " % (block.number * discr.flux_plan.dofs_per_block())
        i_el = 0
        eventful = False
        for mb in block.microblocks:
            for el in mb:
                s = discr.find_el_range(el.id)
                relerr = relative_error(la.norm(diff[s]), norm_ref)
                if relerr > 1e-4:
                    eventful = True
                    struc_line += "*"
                    if detail:
                        print "block %d, el %d, global el #%d, rel.l2err=%g" % (
                                block.number, i_el, el.id, relerr)
                        print computed[s]
                        print reference[s]
                        print diff[s]
                        print diff[s]/norm_ref
                        print la.norm(diff[s]), norm_ref
                        raw_input()
                elif numpy.isnan(diff[s]).any():
                    eventful = True
                    struc_line += "N"
                    add_lines.append(str(diff[s]))

                    if detail:
                        print "block %d, el %d, global el #%d, rel.l2err=%g" % (
                                block.number, i_el, el.id, relerr)
                        print computed[s]
                        print reference[s]
                        print diff[s]
                        raw_input()
                else:
                    if numpy.max(numpy.abs(reference[s])) == 0:
                        struc_line += "0"
                    else:
                        if False:
                            print "block %d, el %d, global el #%d, rel.l2err=%g" % (
                                    block.number, i_el, el.id, relerr)
                            print computed[s]
                            print reference[s]
                            print diff[s]
                            raw_input()
                        struc_line += "."
                i_el += 1
            struc_line += " "
        if (not eventful_only) or eventful:
            struc_lines.append(struc_line)
            if detail:
                struc_lines.extend(add_lines)
    print
    print "\n".join(struc_lines)




# exec mapper -----------------------------------------------------------------
class ExecutionMapper(ExecutionMapperBase):
    def exec_assign(self, insn):
        return [(insn.name, self(insn.expr))], []

    def exec_vector_expr_assign(self, insn):
        if self.executor.discr.instrumented:
            def stats_callback(n, vec_expr, t_func):
                self.executor.discr.vector_math_timer.add_timer_callable(t_func)
                self.executor.discr.vector_math_flop_counter.add(
                        n*insn.flop_count())
                self.executor.discr.gmem_bytes_vector_math.add(
                        self.executor.discr.given.float_size() * n *
                        (len(vec_expr.vector_deps)+len(insn.exprs)))
        else:
            stats_callback = None

        if insn.flop_count() == 0:
            return [(name, self(expr))
                for name, expr in zip(insn.names, insn.exprs)], []
        else:
            compiled = insn.compiled(self.executor)
            return zip(compiled.result_names(),
                    compiled(self, stats_callback)), []

    def exec_diff_batch_assign(self, insn):
        field = self.rec(insn.field)

        discr = self.executor.discr
        if discr.instrumented:
            discr.diff_counter.add(discr.dimensions)
            discr.diff_flop_counter.add(discr.dimensions*(
                self.executor.diff_rst_flops + self.executor.diff_rescale_one_flops))

        xyz_diff = self.executor.diff_kernel(insn.op_class, field)

        if set(["cuda_diff", "cuda_compare"]) <= discr.debug:
            field = self.rec(insn.field)
            f = discr.volume_from_gpu(field)
            assert not numpy.isnan(f).any(), "Initial field contained NaNs."
            cpu_xyz_diff = [discr.volume_from_gpu(xd) for xd in xyz_diff]
            dx = cpu_xyz_diff[0]

            test_discr = discr.test_discr
            real_dx = test_discr.nabla[0].apply(f.astype(numpy.float64))

            diff = dx - real_dx

            for i, xd in enumerate(cpu_xyz_diff):
                if numpy.isnan(xd).any():
                    self.print_error_structure(xd, xd, xd-xd,
                            eventful_only=False, detail=False)
                    assert False, "Resulting field %d contained NaNs." % i

            from hedge.tools import relative_error
            rel_err_norm = relative_error(la.norm(diff), la.norm(real_dx))
            print "diff", rel_err_norm
            if not (rel_err_norm < 5e-5):
                self.print_error_structure(dx, real_dx, diff,
                        eventful_only=False, detail=False)

            assert rel_err_norm < 5e-5

        return [(name, xyz_diff[op.xyz_axis])
                for name, op in zip(insn.names, insn.operators)], []


    def exec_flux_batch_assign(self, insn):
        discr = self.executor.discr

        kernel = insn.kernel(self.executor)
        all_fofs = kernel(self.rec, discr.fluxlocal_plan)
        elgroup, = discr.element_groups

        result = [
            (name, self.executor.fluxlocal_kernel(
                fluxes_on_faces,
                *self.executor.flux_local_data(
                    self.executor.fluxlocal_kernel, elgroup, wdflux.is_lift)))
            for name, wdflux, fluxes_on_faces in zip(
                insn.names, insn.fluxes, all_fofs)]

        if discr.instrumented:
            given = discr.given

            flux_count = len(insn.fluxes)
            dep_count = len(kernel.interior_deps)

            discr.gather_counter.add(
                    flux_count*dep_count)
            discr.gather_flop_counter.add(
                    flux_count
                    * given.dofs_per_face()
                    * given.faces_per_el()
                    * len(discr.mesh.elements)
                    * (1 # facejac-mul
                        + 2 * # int+ext
                        3*dep_count # const-mul, normal-mul, add
                        )
                    )

            discr.lift_counter.add(flux_count)
            discr.lift_flop_counter.add(flux_count*self.executor.lift_flops)

        # debug ---------------------------------------------------------------
        if discr.debug & set(["cuda_lift", "cuda_flux"]):
            fplan = discr.flux_plan

            for fluxes_on_faces in all_fofs:
                useful_size = (len(discr.blocks)
                        * given.aligned_face_dofs_per_microblock()
                        * fplan.microblocks_per_block())
                fof = fluxes_on_faces.get()

                fof = fof[:useful_size]

                have_used_nans = False
                for i_b, block in enumerate(discr.blocks):
                    offset = i_b*(given.aligned_face_dofs_per_microblock()
                            *fplan.microblocks_per_block())
                    size = (len(block.el_number_map)
                            *given.dofs_per_face()
                            *given.faces_per_el())
                    if numpy.isnan(la.norm(fof[offset:offset+size])).any():
                        have_used_nans = True

                if have_used_nans:
                    struc = (given.dofs_per_face(),
                            given.dofs_per_face()*given.faces_per_el(),
                            given.aligned_face_dofs_per_microblock(),
                            )

                    print self.get_vec_structure(fof, *struc)
                    raise RuntimeError("Detected used NaNs in flux gather output.")

                assert not have_used_nans

        if "cuda_lift" in discr.debug:
            cuda.Context.synchronize()
            print "NANCHECK"

            for name in insn.names:
                flux = self.context[name]
                copied_flux = discr.convert_volume(flux, kind="numpy")
                contains_nans = numpy.isnan(copied_flux).any()
                if contains_nans:
                    print "examining", name
                    print_error_structure(discr,
                            copied_flux, copied_flux, copied_flux-copied_flux,
                            eventful_only=True)
                assert not contains_nans, "Resulting flux contains NaNs."

        return result, []

    def exec_mass_assign(self, insn):
        elgroup, = self.executor.discr.element_groups
        kernel = self.executor.discr.element_local_kernel()
        return [(insn.name, kernel(
                self.rec(insn.field),
                *self.executor.mass_data(kernel, elgroup, insn.op_class)))], []

    def map_elementwise_max(self, op, field_expr):

        field = self.rec(field_expr)
        field_out = gpuarray.zeros_like(field)

        func_rec = self.executor.get_elwise_max_kernel(field.dtype)

        func_rec.func.prepared_call((func_rec.grid_dim, 1),
            field.gpudata, field_out.gpudata, func_rec.mb_count)

        return field_out

    def map_if_positive(self, expr):
        crit = self.rec(expr.criterion)
        then_ = self.rec(expr.then_)
        else_ = self.rec(expr.else_)

        return gpuarray.if_positive(crit, then_, else_)




# compiler stuff --------------------------------------------------------------
class VectorExprAssign(Assign):
    __slots__ = ["compiled"]

    def get_executor_method(self, executor):
        return executor.exec_vector_expr_assign

    comment = "compiled"

    @memoize_method
    def compiled(self, executor):
        discr = executor.discr

        from hedge.backends.vector_expr import \
                VectorExpressionInfo, simple_result_dtype_getter
        from hedge.backends.cuda.vector_expr import CompiledVectorExpression
        return CompiledVectorExpression(
                [VectorExpressionInfo(
                    name=name,
                    expr=expr,
                    do_not_return=dnr)
                    for name, expr, dnr in zip(
                        self.names, self.exprs, self.do_not_return)],
                result_dtype_getter=simple_result_dtype_getter,
                allocator=discr.pool.allocate)

class CUDAFluxBatchAssign(FluxBatchAssign):
    @memoize_method
    def get_dependencies(self):
        deps = set()
        for wdflux in self.fluxes:
            deps |= set(wdflux.interior_deps)
            deps |= set(wdflux.boundary_deps)

        dep_mapper = self.dep_mapper_factory()

        from pytools import flatten
        return set(flatten(dep_mapper(dep) for dep in deps))

    @memoize_method
    def kernel(self, executor):
        return executor.discr.flux_plan.make_kernel(
                executor.discr, executor, self.fluxes)



class OperatorCompiler(OperatorCompilerBase):
    from hedge.backends.cuda.optemplate import \
            BoundOperatorCollector \
            as bound_op_collector_class

    def get_contained_fluxes(self, expr):
        from hedge.backends.cuda.optemplate import FluxCollector
        return [self.FluxRecord(
            flux_expr=wdflux,
            dependencies=set(wdflux.interior_deps) | set(wdflux.boundary_deps),
            kind="whole-domain")
            for wdflux in FluxCollector()(expr)]

    def internal_map_flux(self, wdflux):
        from hedge.backends.cuda.optemplate import WholeDomainFluxOperator
        return WholeDomainFluxOperator(
            wdflux.is_lift,
            [wdflux.InteriorInfo(
                flux_expr=ii.flux_expr,
                field_expr=self.rec(ii.field_expr))
                for ii in wdflux.interiors],
            [wdflux.BoundaryInfo(
                flux_expr=bi.flux_expr,
                bpair=self.rec(bi.bpair))
                for bi in wdflux.boundaries])

    def map_whole_domain_flux(self, wdflux):
        return self.map_planned_flux(wdflux)

    def make_flux_batch_assign(self, names, fluxes, kind):
        return CUDAFluxBatchAssign(names=names, fluxes=fluxes, kind=kind,
                dep_mapper_factory=self.dep_mapper_factory)

    def finalize_multi_assign(self, names, exprs, do_not_return, priority):
        return VectorExprAssign(names=names, exprs=exprs,
                do_not_return=do_not_return,
                dep_mapper_factory=self.dep_mapper_factory,
                priority=priority)




class Executor(object):
    exec_mapper_class = ExecutionMapper

    def __init__(self, discr, optemplate, post_bind_mapper):
        self.discr = discr

        from hedge.tools import diff_rst_flops, diff_rescale_one_flops, \
                mass_flops, lift_flops
        self.diff_rst_flops = diff_rst_flops(discr)
        self.diff_rescale_one_flops = diff_rescale_one_flops(discr)
        self.mass_flops = mass_flops(discr)
        self.lift_flops = sum(lift_flops(fg) for fg in discr.face_groups)

        optemplate_stage1 = self.prepare_optemplate_stage1(
                optemplate, post_bind_mapper)

        # build a boundary tag bitmap
        from hedge.optemplate import BoundaryTagCollector
        self.boundary_tag_to_number = {}
        for btag in BoundaryTagCollector()(optemplate_stage1):
            self.boundary_tag_to_number.setdefault(btag,
                    len(self.boundary_tag_to_number))

        e2bb = self.elface_to_bdry_bitmap = {}

        for btag, bdry_number in self.boundary_tag_to_number.iteritems():
            bdry_bit = 1 << bdry_number
            for elface in discr.mesh.tag_to_boundary.get(btag, []):
                e2bb[elface] = (e2bb.get(elface, 0) | bdry_bit)

        # compile the optemplate
        from struct import calcsize
        self.code = OperatorCompiler(
                max_vectors_in_batch_expr=220 // calcsize("P")
                )(
                self.prepare_optemplate_stage2(discr.mesh, optemplate_stage1,
                    discr.debug))

        # build the local kernels
        self.diff_kernel = self.discr.diff_plan.make_kernel(discr)
        self.fluxlocal_kernel = self.discr.fluxlocal_plan.make_kernel(discr,
                with_index_check=False)

        if "dump_op_code" in discr.debug:
            from hedge.tools import open_unique_debug_file
            open_unique_debug_file("op-code", ".txt").write(
                    str(self.code))

    @staticmethod
    def prepare_optemplate_stage2(mesh, optemplate, debug_flags=set()):
        stage = [0]

        def dump_optemplate(name, optemplate):
            if "dump_optemplate_stages" in debug_flags:
                from hedge.tools import open_unique_debug_file
                from hedge.optemplate import pretty_print_optemplate
                open_unique_debug_file("%02d-%s" % (stage[0], name), ".txt").write(
                        pretty_print_optemplate(optemplate))
                stage[0] += 1

        from hedge.backends.cuda.optemplate import BoundaryCombiner
        from hedge.optemplate import process_optemplate

        optemplate = process_optemplate(optemplate, 
                post_bind_mapper=None, # we've already applied it in stage 1
                dumper=dump_optemplate,
                mesh=mesh)

        optemplate = BoundaryCombiner(mesh)(optemplate)
        dump_optemplate("final", optemplate)

        return optemplate

    @staticmethod
    def prepare_optemplate_stage1(optemplate, post_bind_mapper=lambda x: x):
        from hedge.optemplate import OperatorBinder
        return post_bind_mapper(OperatorBinder()(optemplate))

    @classmethod
    def prepare_optemplate(cls, mesh, optemplate, post_bind_mapper=lambda x: x):
        return cls.prepare_optemplate_stage2(mesh,
                cls.prepare_optemplate_stage1(optemplate, post_bind_mapper))

    @classmethod
    def get_first_flux_batch(cls, mesh, optemplate):
        compiler = OperatorCompiler()
        compiler(cls.prepare_optemplate(mesh, optemplate))

        if compiler.flux_batches:
            return compiler.flux_batches[0]
        else:
            return None

    def instrument(self):
        pass

    # actual execution --------------------------------------------------------
    def __call__(self, **vars):
        def pre_assign_check(tgt, value):
            if numpy.isnan(self.discr.convert_volume(value, kind="numpy")).any():
                print "%s was assigned a nan-containing value." % tgt

        return self.code.execute(
                self.discr.exec_mapper_class(vars, self),
                #pre_assign_check=pre_assign_check
                )

    # data caches for execution -----------------------------------------------
    @memoize_method
    def flux_local_data(self, kernel, elgroup, is_lift):
        if is_lift:
            mat = elgroup.local_discretization.lifting_matrix()
            prep_scaling = kernel.prepare_scaling(elgroup, elgroup.inverse_jacobians)
        else:
            mat = elgroup.local_discretization.multi_face_mass_matrix()
            prep_scaling = None

        prep_mat = kernel.prepare_matrix(mat)

        return prep_mat, prep_scaling

    @memoize_method
    def mass_data(self, kernel, elgroup, op_class):
        return (kernel.prepare_matrix(op_class.matrix(elgroup)),
                kernel.prepare_scaling(elgroup, op_class.coefficients(elgroup)))

    @memoize_method
    def get_elwise_max_kernel(self, dtype):
        discr = self.discr

        dofs_per_el = discr.given.dofs_per_el()
        floats_per_mb = discr.given.microblock.aligned_floats

        # set block_size to floats_per_mb (best result of benchmarking 
        # orders 3-6)
        mbs_per_block = 1 
        block_size = floats_per_mb * mbs_per_block

        if block_size > discr.given.devdata.max_threads:
            # rounding up took us beyond the max block size,
            # round down instead
            block_size = block_size - floats_per_mb

        if block_size > discr.given.devdata.max_threads or block_size == 0:
            raise RuntimeError("Computation of thread block size for "
                    "elementwise max gave invalid result %d." % block_size)

        mb_count = len(discr.blocks) * discr.given.microblocks_per_block

        grid_dim = (mb_count + mbs_per_block - 1) // mbs_per_block
        if dtype == numpy.float64:
            max_func = "fmax"
        elif dtype == numpy.float32:
            max_func = "fmaxf"
        else:
            raise TypeError("Could not find a maximum function"
                " due to unsupported field.dtype.")

        from pycuda.tools import dtype_to_ctype
        from pycuda.compiler import SourceModule
        mod = SourceModule("""
        #include <pycuda-helpers.hpp>

        #define DOFS_PER_EL %(dofs_per_el)d
        #define MAX_FUNC %(max_func)s
        #define BLOCK_SIZE %(block_size)d
        #define FLOATS_PER_MB %(floats_per_mb)d
        #define MBS_PER_BLOCK %(mbs_per_block)d

        #define MY_INFINITY (1./0.)

        #define NODE_IN_MB_IDX  threadIdx.x
        #define MB_IN_BLOCK_IDX threadIdx.y
        #define BLOCK_IDX blockIdx.x
        #define MB_NUMBER (BLOCK_IDX * MBS_PER_BLOCK + MB_IN_BLOCK_IDX)

        typedef %(value_type)s value_type;

        __global__ void elwise_max(value_type *field, value_type *field_out,
          unsigned mb_count)
        {
          if (MB_NUMBER >= mb_count)
            return;

          int idx =  MB_NUMBER * FLOATS_PER_MB + NODE_IN_MB_IDX;
          int element_base_idx = FLOATS_PER_MB * MB_IN_BLOCK_IDX +
              (NODE_IN_MB_IDX / DOFS_PER_EL) * DOFS_PER_EL;

          __shared__ value_type whole_block[BLOCK_SIZE];
          int idx_in_block = FLOATS_PER_MB * MB_IN_BLOCK_IDX + NODE_IN_MB_IDX;
          whole_block[idx_in_block] = field[idx];

          __syncthreads();

          value_type own_value = -MY_INFINITY;

          for (int i=0; i<DOFS_PER_EL; i++)
              own_value = MAX_FUNC(own_value, whole_block[element_base_idx+i]);
          field_out[idx] = own_value;
        }
        """% {
        "dofs_per_el": dofs_per_el,
        "max_func": max_func,
        "block_size": block_size,
        "floats_per_mb": floats_per_mb,
        "mbs_per_block": mbs_per_block,
        "value_type": dtype_to_ctype(dtype),
        })

        func = mod.get_function("elwise_max")
        func.prepare("PPI", block=(floats_per_mb, block_size//floats_per_mb, 1))

        from pytools import Record
        class ElementwiseMaxKernelInfo(Record):
            pass

        return ElementwiseMaxKernelInfo(
                func=func,
                grid_dim=grid_dim,
                mb_count=mb_count)
