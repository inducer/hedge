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
from pytools import memoize_method, memoize
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
from hedge.backends.cuda.tools import FakeGPUArray
import hedge.backends.cuda.plan 




# plan ------------------------------------------------------------------------
class ExecutionPlan(hedge.backends.cuda.plan.SegmentedMatrixLocalOpExecutionPlan):
    def __init__(self, given, parallelism, segment_size, max_unroll,
            use_prefetch_branch, debug_name,
           aligned_preimage_dofs_per_microblock, preimage_dofs_per_el):
        hedge.backends.cuda.plan.SegmentedMatrixLocalOpExecutionPlan.__init__(
                self, given, parallelism, segment_size, max_unroll)

        self.use_prefetch_branch = use_prefetch_branch

        self.debug_name = debug_name
        self.aligned_preimage_dofs_per_microblock = \
                aligned_preimage_dofs_per_microblock
        self.preimage_dofs_per_el = preimage_dofs_per_el

    def columns(self):
        return self.preimage_dofs_per_el

    def gpu_matrix_columns(self):
        columns = self.preimage_dofs_per_el

        # avoid smem fetch bank conflicts by ensuring odd col count
        if columns % 2 == 0:
            columns += 1

        return columns

    @memoize_method
    def gpu_matrix_block_floats(self):
        return self.given.devdata.align_dtype(
                self.gpu_matrix_columns()*self.segment_size, 
                self.given.float_size())

    def registers(self):
        return 12 + self.parallelism.inline

    def fetch_buffer_segments(self):
        return 1

    def __str__(self):
        return "%s prefetch_branch=%s" % (
                hedge.backends.cuda.plan.SegmentedMatrixLocalOpExecutionPlan.__str__(self),
                self.use_prefetch_branch)

    @staticmethod
    def feature_columns():
        return ("type text",
                "parallel integer", 
                "inline integer", 
                "serial integer", 
                "segment_size integer", 
                "max_unroll integer",
                "mb_elements integer",
                "lmem integer",
                "smem integer",
                "registers integer",
                "threads integer",
                )

    def features(self, lmem, smem, registers):
        return ("smem_matrix",
                self.parallelism.parallel,
                self.parallelism.inline,
                self.parallelism.serial,
                self.segment_size,
                self.max_unroll,
                self.given.microblock.elements,
                lmem,
                smem,
                registers,
                self.threads(),
                )

    def make_kernel(self, discr):
        return Kernel(discr, self)



# kernel ----------------------------------------------------------------------
class Kernel:
    def __init__(self, discr, plan):
        self.discr = discr
        self.plan = plan

        from hedge.backends.cuda.tools import int_ceiling
        self.grid = (plan.segments_per_microblock(), 
                int_ceiling(
                    self.plan.given.total_dofs()
                    / plan.dofs_per_macroblock())
                )

    def benchmark(self):
        discr = self.discr
        given = self.plan.given
        elgroup, = discr.element_groups

        try:
            kernel, in_vector_texref, scaling_texref = \
                    self.get_kernel(with_scaling=True, for_benchmark=True)
        except cuda.CompileError:
            return None

        def vol_empty():
            from hedge.backends.cuda.tools import int_ceiling
            dofs = int_ceiling(
                    given.total_dofs(), self.plan.dofs_per_macroblock())

            return gpuarray.empty((dofs,), dtype=given.float_type,
                    allocator=discr.pool.allocate)

        out_vector = vol_empty()
        in_vector = gpuarray.empty(
                given.matmul_preimage_shape(self.plan), 
                dtype=given.float_type,
                allocator=discr.pool.allocate)
        in_vector.bind_to_texref_ext(in_vector_texref, allow_double_hack=True)

        fake_matrix = self.prepare_matrix(
                numpy.ones(
                    (given.dofs_per_el(), self.plan.preimage_dofs_per_el),
                    dtype=given.float_type))

        from hedge.backends.cuda.kernelbase import fake_elwise_scaling
        fake_scaling = fake_elwise_scaling(self.plan.given)
        fake_scaling.bind_to_texref_ext(scaling_texref, allow_double_hack=True)

        if "cuda_fastbench" in discr.debug:
            count = 1
        else:
            count = 20

        start = cuda.Event()
        start.record()
        cuda.Context.synchronize()
        for i in range(count):
            try:
                kernel.prepared_call(
                        self.grid,
                        out_vector.gpudata, 
                        fake_matrix,
                        0)
            except cuda.LaunchError:
                return None

        stop = cuda.Event()
        stop.record()
        stop.synchronize()

        return (1e-3/count * stop.time_since(start),
                kernel.lmem, kernel.smem, kernel.registers)

    def __call__(self, in_vector, prepped_mat, prepped_scaling):
        discr = self.discr
        elgroup, = discr.element_groups

        kernel, in_vector_texref, scaling_texref = \
                self.get_kernel(prepped_scaling is not None)

        out_vector = discr.volume_empty() 
        in_vector.bind_to_texref_ext(in_vector_texref, allow_double_hack=True)
        if prepped_scaling is not None:
            prepped_scaling.bind_to_texref_ext(scaling_texref,
                    allow_double_hack=True)

        if set([self.plan.debug_name, "cuda_debugbuf"]) <= discr.debug:
            debugbuf = gpuarray.zeros((1024,), dtype=self.plan.given.float_type)
        else:
            debugbuf = FakeGPUArray()

        if discr.instrumented:
            discr.el_local_timer.add_timer_callable(
                    kernel.prepared_timed_call(
                        self.grid,
                        out_vector.gpudata, 
                        prepped_mat,
                        debugbuf.gpudata))

            given = self.plan.given

            from pytools import product
            discr.gmem_bytes_el_local.add(
                    given.float_size()
                    * (
                        # matrix fetch
                        self.plan.gpu_matrix_block_floats() * product(self.grid)
                        # field fetch
                        + self.plan.preimage_dofs_per_el
                        * given.dofs_per_el() * given.microblock.elements
                        * self.grid[1] * self.plan.parallelism.total()
                        # field store
                        + len(discr.nodes)
                        ))
        else:
            kernel.prepared_call(
                    self.grid,
                    out_vector.gpudata, 
                    prepped_mat,
                    debugbuf.gpudata)

        if set([self.plan.debug_name, "cuda_debugbuf"]) <= discr.debug:
            copied_debugbuf = debugbuf.get()[:144*7].reshape((144,7))
            print "DEBUG"
            numpy.set_printoptions(linewidth=100)
            copied_debugbuf.shape = (144,7)
            numpy.set_printoptions(threshold=3000)

            print copied_debugbuf
            raw_input()

        return out_vector

    @memoize_method
    def get_kernel(self, with_scaling, for_benchmark=False):
        from codepy.cgen import \
                Pointer, POD, Value, ArrayOf, Const, \
                Module, FunctionDeclaration, FunctionBody, Block, \
                Comment, Line, Define, Include, \
                Static, \
                Constant, Initializer, If, For, Statement, Assign, \
                ArrayInitializer

        from codepy.cgen import dtype_to_ctype
        from codepy.cgen.cuda import CudaShared, CudaConstant, CudaGlobal

        discr = self.discr
        d = discr.dimensions
        dims = range(d)
        given = self.plan.given

        float_type = given.float_type

        f_decl = CudaGlobal(FunctionDeclaration(Value("void", "apply_el_local_mat_smem_mat"), 
            [
                Pointer(POD(float_type, "out_vector")),
                Pointer(POD(numpy.uint8, "gmem_matrix")),
                Pointer(POD(float_type, "debugbuf")),
                ]
            ))

        cmod = Module([
                Include("pycuda-helpers.hpp"),
                Line(),
                Value("texture<fp_tex_%s, 1, cudaReadModeElementType>"
                    % dtype_to_ctype(float_type), 
                    "in_vector_tex"),
                ])
        if with_scaling:
            cmod.append(
                Value("texture<fp_tex_%s, 1, cudaReadModeElementType>"
                    % dtype_to_ctype(float_type), 
                    "scaling_tex"),
                )

        par = self.plan.parallelism

        cmod.extend([
                Line(),
                Define("DIMENSIONS", discr.dimensions),
                Define("DOFS_PER_EL", given.dofs_per_el()),
                Define("PREIMAGE_DOFS_PER_EL", self.plan.preimage_dofs_per_el),
                Line(),
                Define("SEGMENT_DOF", "threadIdx.x"),
                Define("PAR_MB_NR", "threadIdx.y"),
                Line(),
                Define("MB_SEGMENT", "blockIdx.x"),
                Define("MACROBLOCK_NR", "blockIdx.y"),
                Line(),
                Define("DOFS_PER_SEGMENT", self.plan.segment_size),
                Define("SEGMENTS_PER_MB", self.plan.segments_per_microblock()),
                Define("ALIGNED_DOFS_PER_MB", given.microblock.aligned_floats),
                Define("ALIGNED_PREIMAGE_DOFS_PER_MB", 
                    self.plan.aligned_preimage_dofs_per_microblock),
                Define("MB_EL_COUNT", given.microblock.elements),
                Line(),
                Define("PAR_MB_COUNT", par.parallel),
                Define("INLINE_MB_COUNT", par.inline),
                Define("SEQ_MB_COUNT", par.serial),
                Line(),
                Define("THREAD_NUM", "(SEGMENT_DOF+PAR_MB_NR*DOFS_PER_SEGMENT)"),
                Define("COALESCING_THREAD_COUNT", "(PAR_MB_COUNT*DOFS_PER_SEGMENT)"),
                Line(),
                Define("MB_DOF_BASE", "(MB_SEGMENT*DOFS_PER_SEGMENT)"),
                Define("MB_DOF", "(MB_DOF_BASE+SEGMENT_DOF)"),
                Define("GLOBAL_MB_NR_BASE", 
                    "(MACROBLOCK_NR*PAR_MB_COUNT*INLINE_MB_COUNT*SEQ_MB_COUNT)"),
                Define("GLOBAL_MB_NR", 
                    "(GLOBAL_MB_NR_BASE"
                    "+ (seq_mb_number*PAR_MB_COUNT + PAR_MB_NR)*INLINE_MB_COUNT)"),
                Define("GLOBAL_MB_DOF_BASE", "(GLOBAL_MB_NR*ALIGNED_DOFS_PER_MB)"),
                Define("GLOBAL_MB_PREIMG_DOF_BASE", "(GLOBAL_MB_NR*ALIGNED_PREIMAGE_DOFS_PER_MB)"),
                Line(),
                Define("MATRIX_COLUMNS", self.plan.gpu_matrix_columns()),
                Define("MATRIX_SEGMENT_FLOATS", self.plan.gpu_matrix_block_floats()),
                Define("MATRIX_SEGMENT_BYTES", 
                    "(MATRIX_SEGMENT_FLOATS*%d)" % given.float_size()),

                Line(),
                CudaShared(ArrayOf(POD(float_type, "smem_matrix"), 
                    "MATRIX_SEGMENT_FLOATS")),
                CudaShared(
                    ArrayOf(
                        ArrayOf(
                            ArrayOf(
                                POD(float_type, "dof_buffer"), 
                                "PAR_MB_COUNT"),
                            "INLINE_MB_COUNT"),
                        "DOFS_PER_SEGMENT"),
                    ),
                CudaShared(POD(numpy.uint16, "segment_start_el")),
                CudaShared(POD(numpy.uint16, "segment_stop_el")),
                CudaShared(POD(numpy.uint16, "segment_el_count")),
                Line(),
                ArrayInitializer(
                        CudaConstant(
                            ArrayOf(
                                POD(numpy.uint32, "segment_start_el_lookup"),
                            "SEGMENTS_PER_MB")),
                        [(chk*self.plan.segment_size)//given.dofs_per_el()
                            for chk in range(self.plan.segments_per_microblock())]
                        ),
                ArrayInitializer(
                        CudaConstant(
                            ArrayOf(
                                POD(numpy.uint32, "segment_stop_el_lookup"),
                            "SEGMENTS_PER_MB")),
                        [min(given.microblock.elements, 
                            (chk*self.plan.segment_size+self.plan.segment_size-1)
                                //given.dofs_per_el()+1)
                            for chk in range(self.plan.segments_per_microblock())]
                        ),
                ])

        S = Statement
        f_body = Block()
            
        f_body.extend_log_block("calculate this dof's element", [
            Initializer(POD(numpy.uint8, "mb_el"),
                "MB_DOF/DOFS_PER_EL") ])

        if self.plan.use_prefetch_branch:
            f_body.extend_log_block("calculate segment responsibility data", [
                If("THREAD_NUM==0",
                    Block([
                        Assign("segment_start_el", "segment_start_el_lookup[MB_SEGMENT]"),
                        Assign("segment_stop_el", "segment_stop_el_lookup[MB_SEGMENT]"),
                        Assign("segment_el_count", "segment_stop_el-segment_start_el"),
                        ])
                    ),
                S("__syncthreads()")
                ])

        from hedge.backends.cuda.tools import get_load_code
        f_body.extend(
            get_load_code(
                dest="smem_matrix",
                base=("gmem_matrix + MB_SEGMENT*MATRIX_SEGMENT_BYTES"),
                bytes="MATRIX_SEGMENT_BYTES",
                descr="load matrix segment")
            +[S("__syncthreads()")]
            )

        # ---------------------------------------------------------------------
        def get_batched_fetch_mat_mul_code(el_fetch_count):
            result = []
            dofs = range(self.plan.preimage_dofs_per_el)

            for load_segment_start in range(0, self.plan.preimage_dofs_per_el,
                    self.plan.segment_size):
                result.extend(
                        [S("__syncthreads()")]
                        +[Assign(
                            "dof_buffer[PAR_MB_NR][%d][SEGMENT_DOF]" % inl,
                            "fp_tex1Dfetch(in_vector_tex, "
                            "GLOBAL_MB_PREIMG_DOF_BASE"
                            " + %d*ALIGNED_PREIMAGE_DOFS_PER_MB"
                            " + (segment_start_el)*PREIMAGE_DOFS_PER_EL + %d + SEGMENT_DOF)"
                            % (inl, load_segment_start)
                            )
                        for inl in range(par.inline)]
                        +[S("__syncthreads()"),
                        Line(),
                        ])

                for dof in dofs[load_segment_start:load_segment_start+self.plan.segment_size]:
                    for inl in range(par.inline):
                        result.append(
                                S("result%d += "
                                    "smem_matrix[SEGMENT_DOF*MATRIX_COLUMNS + %d]"
                                    "*"
                                    "dof_buffer[PAR_MB_NR][%d][%d]"
                                    % (inl, dof, inl, dof-load_segment_start))
                                )
                result.append(Line())
            return result

        from hedge.backends.cuda.tools import unroll
        def get_direct_tex_mat_mul_code():
            return (
                    [POD(float_type, "fof%d" % inl) for inl in range(par.inline)]
                    + [POD(float_type, "lm"), Line()]
                    + unroll(
                        lambda j: [
                        Assign("fof%d" % inl,
                            "fp_tex1Dfetch(in_vector_tex, "
                            "GLOBAL_MB_PREIMG_DOF_BASE"
                            " + %(inl)d * ALIGNED_PREIMAGE_DOFS_PER_MB"
                            " + mb_el*PREIMAGE_DOFS_PER_EL+%(j)s)"
                            % {"j":j, "inl":inl, "row": "SEGMENT_DOF"},)
                        for inl in range(par.inline)
                        ]+[
                        Assign("lm",
                            "smem_matrix["
                            "%(row)s*MATRIX_COLUMNS + %(j)s]"
                            % {"j":j, "row": "SEGMENT_DOF"},
                            )
                        ]+[
                        S("result%(inl)d += fof%(inl)d*lm" % {"inl":inl})
                        for inl in range(par.inline)
                        ],
                        total_number=self.plan.preimage_dofs_per_el,
                        max_unroll=self.plan.max_unroll) 
                    + [ Line(), ])

        def get_mat_mul_code(el_fetch_count):
            if el_fetch_count == 1:
                return get_batched_fetch_mat_mul_code(el_fetch_count)
            else:
                return get_direct_tex_mat_mul_code()

        def mat_mul_outer_loop(fetch_count):
            if with_scaling:
                inv_jac_multiplier = ("fp_tex1Dfetch(scaling_tex,"
                        "(GLOBAL_MB_NR + %(inl)d)*MB_EL_COUNT + mb_el)")
            else:
                inv_jac_multiplier = "1"

            return For("unsigned short seq_mb_number = 0",
                "seq_mb_number < SEQ_MB_COUNT",
                "++seq_mb_number",
                Block([
                    Initializer(POD(float_type, "result%d" % inl), 0)
                    for inl in range(par.inline)
                    ]+[ Line() ]
                    +get_mat_mul_code(fetch_count)
                    +[
                    If("MB_DOF < DOFS_PER_EL*MB_EL_COUNT",
                        Block([
                            Assign(
                                "out_vector[GLOBAL_MB_DOF_BASE"
                                " + %d*ALIGNED_DOFS_PER_MB"
                                " + MB_DOF]" % inl,
                                "result%d * %s" % (inl, (inv_jac_multiplier % {"inl":inl}))
                                )
                            for inl in range(par.inline)
                            ])
                        )
                    ])
                )

        if self.plan.use_prefetch_branch:
            from codepy.cgen import make_multiple_ifs
            f_body.append(make_multiple_ifs([
                    ("segment_el_count == %d" % fetch_count,
                        mat_mul_outer_loop(fetch_count))
                    for fetch_count in 
                    range(1, self.plan.max_elements_touched_by_segment()+1)]
                    ))
        else:
            f_body.append(mat_mul_outer_loop(0))

        # finish off ----------------------------------------------------------
        cmod.append(FunctionBody(f_decl, f_body))

        if not for_benchmark and "cuda_dumpkernels" in discr.debug:
            open("%s.cu" % self.plan.debug_name, "w").write(str(cmod))

        mod = SourceModule(cmod, 
                keep="cuda_keep_kernels" in discr.debug, 
                #options=["--maxrregcount=12"]
                )

        if self.plan.debug_name in discr.debug:
            print "%s: lmem=%d smem=%d regs=%d" % (
                    self.plan.debug_name, mod.lmem, mod.smem, mod.registers)

        in_vector_texref = mod.get_texref("in_vector_tex")
        texrefs = [in_vector_texref]

        if with_scaling:
            scaling_texref = mod.get_texref("scaling_tex")
            texrefs.append(scaling_texref)
        else:
            scaling_texref = None

        func = mod.get_function("apply_el_local_mat_smem_mat")
        func.prepare(
                "PPP", 
                block=(self.plan.segment_size, self.plan.parallelism.parallel, 1),
                texrefs=texrefs)

        return func, in_vector_texref, scaling_texref

    # data blocks -------------------------------------------------------------
    def prepare_matrix(self, matrix):
        discr = self.discr
        given = self.plan.given

        assert matrix.shape == (given.dofs_per_el(), self.plan.preimage_dofs_per_el)

        columns = self.plan.gpu_matrix_columns()
        block_floats = self.plan.gpu_matrix_block_floats()

        vstacked_matrix = numpy.vstack(
                given.microblock.elements*(matrix,)
                )

        if vstacked_matrix.shape[1] < columns:
            vstacked_matrix = numpy.hstack((
                vstacked_matrix,
                numpy.zeros((
                    vstacked_matrix.shape[0],
                    columns-vstacked_matrix.shape[1]
                    ))
                ))
                
        segments = [
                buffer(numpy.asarray(
                    vstacked_matrix[
                        segment_start:segment_start+self.plan.segment_size],
                    dtype=given.float_type,
                    order="C"))
                for segment_start in range(
                    0, given.microblock.elements*given.dofs_per_el(), 
                    self.plan.segment_size)
                ]
        
        from hedge.backends.cuda.tools import pad_and_join

        return cuda.to_device(
                pad_and_join(segments, block_floats*given.float_size()))

    def prepare_scaling(self, elgroup, scaling):
        ij = scaling[self.discr.elgroup_microblock_indices(elgroup)]
        return gpuarray.to_gpu(
                ij.astype(self.plan.given.float_type))

