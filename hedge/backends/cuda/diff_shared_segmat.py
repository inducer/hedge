# -*- coding: utf-8 -*-
"""Interface with Nvidia CUDA."""

from __future__ import division

__copyright__ = "Copyright (C) 2008 Andreas Kloeckner"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""




import numpy
from pytools import memoize_method
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import hedge.backends.cuda.plan




# plan ------------------------------------------------------------------------
class ExecutionPlan(hedge.backends.cuda.plan.SegmentedMatrixLocalOpExecutionPlan):
    def columns(self):
        return self.given.dofs_per_el() * self.given.ldis.dimensions # r,s,t

    def registers(self):
        return 16 + 4 * (self.parallelism.inline-1)

    def fetch_buffer_segments(self):
        return 0

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

        given = self.plan.given
        self.grid = (plan.segments_per_microblock(),
                    int_ceiling(given.total_dofs()/plan.dofs_per_macroblock()))

    def benchmark(self):
        discr = self.discr
        given = self.plan.given
        elgroup, = discr.element_groups

        from hedge.optemplate \
                import ReferenceDifferentiationOperator as op_class
        try:
            func, field_texref = self.get_kernel(op_class, elgroup, for_benchmark=True)
        except cuda.CompileError:
            return None

        def vol_empty():
            from hedge.backends.cuda.tools import int_ceiling
            dofs = int_ceiling(
                    given.total_dofs(), self.plan.dofs_per_macroblock())
            assert dofs == self.grid[1] * self.plan.dofs_per_macroblock()

            import pycuda.gpuarray as gpuarray
            return gpuarray.empty((dofs,), dtype=given.float_type,
                    allocator=discr.pool.allocate)

        field = vol_empty()
        field.fill(0)

        field.bind_to_texref_ext(field_texref, allow_double_hack=True)
        rst_diff = [vol_empty() for axis in range(discr.dimensions)]
        rst_diff_gpudata = [subarray.gpudata for subarray in rst_diff]

        if "cuda_fastbench" in self.discr.debug:
            count = 1
        else:
            count = 20

        gpu_diffmats = self.gpu_diffmats(op_class, elgroup)

        start = cuda.Event()
        start.record()
        cuda.Context.synchronize()
        for i in range(count):
            try:
                func.prepared_call(self.grid, gpu_diffmats.device_memory,
                        *rst_diff_gpudata)
            except cuda.LaunchError:
                return None

        stop = cuda.Event()
        stop.record()
        stop.synchronize()

        return (1e-3/count * stop.time_since(start),
                func.local_size_bytes, func.shared_size_bytes, func.num_regs)

    def __call__(self, op_class, field):
        discr = self.discr
        given = self.plan.given

        d = discr.dimensions
        elgroup, = discr.element_groups

        func, field_texref = self.get_kernel(op_class, elgroup)

        assert field.dtype == given.float_type

        field.bind_to_texref_ext(field_texref, allow_double_hack=True)

        rst_diff = [discr.volume_empty() for axis in range(d)]
        rst_diff_gpudata = [subarray.gpudata for subarray in rst_diff]

        gpu_diffmats = self.gpu_diffmats(op_class, elgroup)

        if discr.instrumented:
            discr.diff_op_timer.add_timer_callable(func.prepared_timed_call(
                    self.grid, gpu_diffmats.device_memory,
                    *rst_diff_gpudata))

            from pytools import product
            discr.gmem_bytes_diff.add(
                    given.float_size()
                    * (
                        # matrix fetch
                        gpu_diffmats.block_floats * product(self.grid)
                        # field fetch
                        + given.dofs_per_el()
                        * given.dofs_per_el() * given.microblock.elements
                        * self.grid[1] * self.plan.parallelism.total()
                        # field store
                        + len(discr.nodes)
                        ))
        else:
            func.prepared_call(self.grid, gpu_diffmats.device_memory,
                    *rst_diff_gpudata)

        if False:
            copied_debugbuf = debugbuf.get()
            print "DEBUG"
            #print numpy.reshape(copied_debugbuf, (len(copied_debugbuf)//16, 16))
            print copied_debugbuf[:100].reshape((10,10))
            raw_input()

        return rst_diff

    @memoize_method
    def get_kernel(self, diff_op_cls, elgroup, for_benchmark=False):
        from cgen import \
                Pointer, POD, Value, ArrayOf, \
                Module, FunctionDeclaration, FunctionBody, Block, \
                Line, Define, Include, \
                Initializer, If, For, Statement, Assign

        from cgen import dtype_to_ctype
        from cgen.cuda import CudaShared, CudaGlobal

        discr = self.discr
        d = discr.dimensions
        dims = range(d)
        given = self.plan.given

        par = self.plan.parallelism

        diffmat_data = self.gpu_diffmats(diff_op_cls, elgroup)
        elgroup, = discr.element_groups

        float_type = given.float_type

        f_decl = CudaGlobal(FunctionDeclaration(Value("void", "apply_diff_mat"),
            [Pointer(POD(numpy.uint8, "gmem_diff_rst_mat")),
                #Pointer(POD(float_type, "debugbuf")),
                ] + [Pointer(POD(float_type, "drst%d_global" % i)) for i in dims]
            ))

        rst_channels = given.devdata.make_valid_tex_channel_count(d)
        cmod = Module([
                Include("pycuda-helpers.hpp"),
                Line(),
                Value("texture<fp_tex_%s, 1, cudaReadModeElementType>"
                    % dtype_to_ctype(float_type),
                    "field_tex"),
                Line(),
                Define("DIMENSIONS", discr.dimensions),
                Define("DOFS_PER_EL", given.dofs_per_el()),
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
                Define("ELS_PER_MB", given.microblock.elements),
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
                Line(),
                Define("DIFFMAT_SEGMENT_FLOATS", diffmat_data.block_floats),
                Define("DIFFMAT_SEGMENT_BYTES", "(DIFFMAT_SEGMENT_FLOATS*%d)"
                     % given.float_size()),
                Define("DIFFMAT_COLUMNS", diffmat_data.matrix_columns),
                Line(),
                CudaShared(ArrayOf(POD(float_type, "smem_diff_rst_mat"),
                    "DIFFMAT_COLUMNS*DOFS_PER_SEGMENT")),
                Line(),
                ])

        S = Statement
        f_body = Block()

        f_body.extend_log_block("calculate responsibility data", [
            Initializer(POD(numpy.uint16, "mb_el"),
                "MB_DOF/DOFS_PER_EL"),
            ])

        from hedge.backends.cuda.tools import get_load_code
        f_body.extend(
            get_load_code(
                dest="smem_diff_rst_mat",
                base="gmem_diff_rst_mat + MB_SEGMENT*DIFFMAT_SEGMENT_BYTES",
                bytes="DIFFMAT_SEGMENT_BYTES",
                descr="load diff mat segment")
            +[S("__syncthreads()"), Line()])

        # ---------------------------------------------------------------------
        def get_scalar_diff_code():
            code = []
            for inl in range(par.inline):
                for axis in dims:
                    code.append(
                        Initializer(POD(float_type, "d%drst%d" % (inl, axis)), 0))

            code.append(Line())

            def get_mat_entry(row, col, axis):
                return ("smem_diff_rst_mat["
                        "%(row)s*DIFFMAT_COLUMNS + %(axis)s*DOFS_PER_EL"
                        " + %(col)s"
                        "]" % {"row":row, "col":col, "axis":axis}
                        )

            tex_channels = ["x", "y", "z", "w"]
            from hedge.backends.cuda.tools import unroll
            code.extend(
                    [POD(float_type, "field_value%d" % inl)
                        for inl in range(par.inline)]
                    +[Line()]
                    +unroll(lambda j: [
                        Assign("field_value%d" % inl,
                            "fp_tex1Dfetch(field_tex, GLOBAL_MB_DOF_BASE + %d*ALIGNED_DOFS_PER_MB "
                            "+ mb_el*DOFS_PER_EL + %s)" % (inl, j)
                            )
                        for inl in range(par.inline)]
                        +[Line()]
                        +[S("d%drst%d += %s * field_value%d"
                            % (inl, axis, get_mat_entry("SEGMENT_DOF", j, axis), inl))
                        for axis in dims
                        for inl in range(par.inline)]
                        +[Line()],
                        given.dofs_per_el(), self.plan.max_unroll)
                    )

            store_code = Block()
            for inl in range(par.inline):
                for rst_axis in dims:
                    store_code.append(Assign(
                        "drst%d_global[GLOBAL_MB_DOF_BASE"
                        " + %d*ALIGNED_DOFS_PER_MB + MB_DOF]" % (rst_axis, inl),
                        "d%drst%d" % (inl, rst_axis),
                        ))

            code.append(If("MB_DOF < DOFS_PER_EL*ELS_PER_MB", store_code))

            return code

        f_body.extend([
            For("unsigned short seq_mb_number = 0",
                "seq_mb_number < SEQ_MB_COUNT",
                "++seq_mb_number",
                Block(get_scalar_diff_code()))
            ])

        # finish off ----------------------------------------------------------
        cmod.append(FunctionBody(f_decl, f_body))

        if not for_benchmark and "cuda_dump_kernels" in discr.debug:
            from hedge.tools import open_unique_debug_file
            open_unique_debug_file("diff", ".cu").write(str(cmod))

        mod = SourceModule(cmod,
                keep="cuda_keep_kernels" in discr.debug,
                #options=["--maxrregcount=10"]
                )

        field_texref = mod.get_texref("field_tex")

        func = mod.get_function("apply_diff_mat")
        func.prepare(
                discr.dimensions*[float_type] + ["P"],
                block=(self.plan.segment_size, par.parallel, 1),
                texrefs=[field_texref])

        if "cuda_diff" in discr.debug:
            print "diff: lmem=%d smem=%d regs=%d" % (
                    func.local_size_bytes,
                    func.shared_size_bytes,
                    func.num_regs)

        return func, field_texref

    # data blocks -------------------------------------------------------------
    @memoize_method
    def gpu_diffmats(self, diff_op_cls, elgroup):
        discr = self.discr
        given = self.plan.given

        columns = given.dofs_per_el()*discr.dimensions
        additional_columns = 0
        # avoid smem fetch bank conflicts by ensuring odd col count
        if columns % 2 == 0:
            columns += 1
            additional_columns += 1

        block_floats = given.devdata.align_dtype(
                columns*self.plan.segment_size, given.float_size())

        vstacked_matrices = [
                numpy.vstack(given.microblock.elements*(m,))
                for m in diff_op_cls.matrices(elgroup)
                ]

        segments = []

        from pytools import single_valued
        for segment_start in range(0, given.microblock.elements*given.dofs_per_el(), self.plan.segment_size):
            matrices = [
                m[segment_start:segment_start+self.plan.segment_size]
                for m in vstacked_matrices]

            matrices.append(
                numpy.zeros((single_valued(m.shape[0] for m in matrices),
                    additional_columns))
                )

            diffmats = numpy.asarray(
                    numpy.hstack(matrices),
                    dtype=given.float_type,
                    order="C")
            segments.append(buffer(diffmats))

        from hedge.backends.cuda.tools import pad_and_join

        from pytools import Record
        class GPUDifferentiationMatrices(Record):
            pass

        return GPUDifferentiationMatrices(
                device_memory=cuda.to_device(
                    pad_and_join(segments, block_floats*given.float_size())),
                block_floats=block_floats,
                matrix_columns=columns)
