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
import hedge.cuda.plan
from hedge.cuda.kernelbase import DiffKernelBase




# plan ------------------------------------------------------------------------
class ChunkedDiffExecutionPlan(hedge.cuda.plan.ChunkedMatrixLocalOpExecutionPlan):
    def columns(self):
        return self.given.dofs_per_el() * self.given.ldis.dimensions # r,s,t

    def registers(self):
        return 16 + 4 * (self.parallelism.inline-1)

    def fetch_buffer_chunks(self):
        return 0

    @staticmethod
    def feature_columns():
        return ("type text",
                "parallel integer", 
                "inline integer", 
                "serial integer", 
                "chunk_size integer", 
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
                self.chunk_size,
                self.max_unroll,
                self.given.microblock.elements,
                lmem,
                smem,
                registers,
                self.threads(),
                )

    def make_kernel(self, discr):
        return DiffKernel(discr, self)





def make_plan(discr, given):
    def generate_plans():
        chunk_sizes = range(given.microblock.align_size, 
                given.microblock.elements*given.dofs_per_el()+1, 
                given.microblock.align_size)

        from hedge.cuda.plan import Parallelism

        for pe in range(1,32+1):
            for inline in range(1, 4+1):
                for seq in range(1, 4):
                    localop_par = Parallelism(pe, inline, seq)
                    for chunk_size in chunk_sizes:
                        yield ChunkedDiffExecutionPlan(
                                given, localop_par, chunk_size, 
                                max_unroll=given.dofs_per_el())

        from hedge.cuda.diff_alt import SMemFieldDiffExecutionPlan

        for pe in range(1,32+1):
            for inline in range(1, 4+1):
                localop_par = Parallelism(pe, inline, 1)
                yield SMemFieldDiffExecutionPlan(given, localop_par, 
                        max_unroll=given.dofs_per_el())

    def target_func(plan):
        return plan.make_kernel(discr).benchmark()

    from hedge.cuda.plan import optimize_plan
    return optimize_plan(generate_plans, target_func, maximize=False,
            debug="cuda_diff_plan" in discr.debug,
            log_filename="diff-%d" % given.order())




# kernel ----------------------------------------------------------------------
class DiffKernel(DiffKernelBase):
    def __init__(self, discr, plan):
        self.discr = discr
        self.plan = plan

        from hedge.cuda.tools import int_ceiling

        given = self.plan.given
        self.grid = (plan.chunks_per_microblock(), 
                    int_ceiling(given.total_dofs()/plan.dofs_per_macroblock()))

    def benchmark(self):
        discr = self.discr
        given = self.plan.given
        elgroup, = discr.element_groups

        from hedge.optemplate import DifferentiationOperator as op_class
        try:
            func, field_texref = self.get_kernel(op_class, elgroup, for_benchmark=True)
        except cuda.CompileError:
            return None

        def vol_empty():
            from hedge.cuda.tools import int_ceiling
            dofs = int_ceiling(
                    given.total_dofs(), self.plan.dofs_per_macroblock())

            import pycuda.gpuarray as gpuarray
            return gpuarray.empty((dofs,), dtype=given.float_type,
                    allocator=discr.pool.allocate)

        field = vol_empty()
        field.fill(0)

        field.bind_to_texref(field_texref)
        xyz_diff = [vol_empty() for axis in range(discr.dimensions)]
        xyz_diff_gpudata = [subarray.gpudata for subarray in xyz_diff] 

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
                        *xyz_diff_gpudata)
            except cuda.LaunchError:
                return None

        stop = cuda.Event()
        stop.record()
        stop.synchronize()

        return (1e-3/count * stop.time_since(start),
                func.lmem, func.smem, func.registers)

    def __call__(self, op_class, field):
        discr = self.discr
        given = self.plan.given

        d = discr.dimensions
        elgroup, = discr.element_groups

        func, field_texref = self.get_kernel(op_class, elgroup)

        assert field.dtype == given.float_type

        field.bind_to_texref(field_texref)

        xyz_diff = [discr.volume_empty() for axis in range(d)]
        xyz_diff_gpudata = [subarray.gpudata for subarray in xyz_diff] 

        gpu_diffmats = self.gpu_diffmats(op_class, elgroup)

        if discr.instrumented:
            discr.diff_op_timer.add_timer_callable(func.prepared_timed_call(
                    self.grid, gpu_diffmats.device_memory, 
                    *xyz_diff_gpudata))
        else:
            func.prepared_call(self.grid, gpu_diffmats.device_memory, 
                    *xyz_diff_gpudata)

        if False:
            copied_debugbuf = debugbuf.get()
            print "DEBUG"
            #print numpy.reshape(copied_debugbuf, (len(copied_debugbuf)//16, 16))
            print copied_debugbuf[:100].reshape((10,10))
            raw_input()

        return xyz_diff

    @memoize_method
    def get_kernel(self, diff_op_cls, elgroup, for_benchmark=False):
        from hedge.cuda.cgen import \
                Pointer, POD, Value, ArrayOf, Const, \
                Module, FunctionDeclaration, FunctionBody, Block, \
                Comment, Line, \
                CudaShared, CudaGlobal, Static, \
                Define, \
                Constant, Initializer, If, For, Statement, Assign
                
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
                ] + [Pointer(POD(float_type, "dxyz%d" % i)) for i in dims]
            ))

        rst_channels = given.devdata.make_valid_tex_channel_count(d)
        cmod = Module([
                Value("texture<float%d, 2, cudaReadModeElementType>"
                    % rst_channels, 
                    "rst_to_xyz_tex"),
                Value("texture<float, 1, cudaReadModeElementType>", 
                    "field_tex"),
                Line(),
                Define("DIMENSIONS", discr.dimensions),
                Define("DOFS_PER_EL", given.dofs_per_el()),
                Line(),
                Define("CHUNK_DOF", "threadIdx.x"),
                Define("PAR_MB_NR", "threadIdx.y"),
                Line(),
                Define("MB_CHUNK", "blockIdx.x"),
                Define("MACROBLOCK_NR", "blockIdx.y"),
                Line(),
                Define("DOFS_PER_CHUNK", self.plan.chunk_size),
                Define("CHUNKS_PER_MB", self.plan.chunks_per_microblock()),
                Define("ALIGNED_DOFS_PER_MB", given.microblock.aligned_floats),
                Define("ELS_PER_MB", given.microblock.elements),
                Line(),
                Define("PAR_MB_COUNT", par.parallel),
                Define("INLINE_MB_COUNT", par.inline),
                Define("SEQ_MB_COUNT", par.serial),
                Line(),
                Define("THREAD_NUM", "(CHUNK_DOF+PAR_MB_NR*DOFS_PER_CHUNK)"),
                Define("COALESCING_THREAD_COUNT", "(PAR_MB_COUNT*DOFS_PER_CHUNK)"),
                Line(),
                Define("MB_DOF_BASE", "(MB_CHUNK*DOFS_PER_CHUNK)"),
                Define("MB_DOF", "(MB_DOF_BASE+CHUNK_DOF)"),
                Define("GLOBAL_MB_NR_BASE", 
                    "(MACROBLOCK_NR*PAR_MB_COUNT*INLINE_MB_COUNT*SEQ_MB_COUNT)"),
                Define("GLOBAL_MB_NR", 
                    "(GLOBAL_MB_NR_BASE"
                    "+ (seq_mb_number*PAR_MB_COUNT + PAR_MB_NR)*INLINE_MB_COUNT)"),
                Define("GLOBAL_MB_DOF_BASE", "(GLOBAL_MB_NR*ALIGNED_DOFS_PER_MB)"),
                Line(),
                Define("DIFFMAT_CHUNK_FLOATS", diffmat_data.block_floats),
                Define("DIFFMAT_CHUNK_BYTES", "(DIFFMAT_CHUNK_FLOATS*%d)"
                     % given.float_size()),
                Define("DIFFMAT_COLUMNS", diffmat_data.matrix_columns),
                Line(),
                CudaShared(ArrayOf(POD(float_type, "smem_diff_rst_mat"), 
                    "DIFFMAT_COLUMNS*DOFS_PER_CHUNK")),
                Line(),
                ])

        S = Statement
        f_body = Block()
            
        f_body.extend_log_block("calculate responsibility data", [
            Initializer(POD(numpy.uint16, "mb_el"),
                "MB_DOF/DOFS_PER_EL"),
            ])

        from hedge.cuda.tools import get_load_code
        f_body.extend(
            get_load_code(
                dest="smem_diff_rst_mat",
                base="gmem_diff_rst_mat + MB_CHUNK*DIFFMAT_CHUNK_BYTES",
                bytes="DIFFMAT_CHUNK_BYTES",
                descr="load diff mat chunk")
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
            from hedge.cuda.tools import unroll
            code.extend(
                    [POD(float_type, "field_value%d" % inl)
                        for inl in range(par.inline)]
                    +[Line()]
                    +unroll(lambda j: [
                        Assign("field_value%d" % inl, 
                            "tex1Dfetch(field_tex, GLOBAL_MB_DOF_BASE + %d*ALIGNED_DOFS_PER_MB "
                            "+ mb_el*DOFS_PER_EL + %s)" % (inl, j)
                            )
                        for inl in range(par.inline)]
                        +[Line()]
                        +[S("d%drst%d += %s * field_value%d" 
                            % (inl, axis, get_mat_entry("CHUNK_DOF", j, axis), inl))
                        for axis in dims
                        for inl in range(par.inline)]
                        +[Line()],
                        given.dofs_per_el(), self.plan.max_unroll)
                    )

            store_code = Block()
            for inl in range(par.inline):
                for glob_axis in dims:
                    store_code.append(Block([
                        Initializer(Value("float%d" % rst_channels, "rst_to_xyz"),
                            "tex2D(rst_to_xyz_tex, %d, "
                            "(GLOBAL_MB_NR+%d)*ELS_PER_MB + mb_el)" 
                            % (glob_axis, inl)
                            ),
                        Assign(
                            "dxyz%d[GLOBAL_MB_DOF_BASE + %d*ALIGNED_DOFS_PER_MB + MB_DOF]" 
                            % (glob_axis, inl),
                            " + ".join(
                                "rst_to_xyz.%s"
                                "*"
                                "d%drst%d" % (tex_channels[loc_axis], inl, loc_axis)
                                for loc_axis in dims
                                )
                            )
                        ]))

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

        if not for_benchmark and "cuda_dumpkernels" in discr.debug:
            open("diff.cu", "w").write(str(cmod))

        mod = cuda.SourceModule(cmod, keep=True, 
                #options=["--maxrregcount=10"]
                )

        if for_benchmark:
            rst_to_xyz_array = self.fake_localop_rst_to_xyz()
        else:
            rst_to_xyz_array = self.localop_rst_to_xyz(diff_op_cls, elgroup)

        rst_to_xyz_texref = mod.get_texref("rst_to_xyz_tex")
        cuda.bind_array_to_texref(rst_to_xyz_array, rst_to_xyz_texref)

        field_texref = mod.get_texref("field_tex")

        func = mod.get_function("apply_diff_mat")
        func.prepare(
                discr.dimensions*[float_type] + ["P"],
                block=(self.plan.chunk_size, par.parallel, 1),
                texrefs=[field_texref, rst_to_xyz_texref])

        print "diff: lmem=%d smem=%d regs=%d" % (func.lmem, func.smem, func.registers)

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
                columns*self.plan.chunk_size, given.float_size())

        vstacked_matrices = [
                numpy.vstack(given.microblock.elements*(m,))
                for m in diff_op_cls.matrices(elgroup)
                ]

        chunks = []

        from pytools import single_valued
        for chunk_start in range(0, given.microblock.elements*given.dofs_per_el(), self.plan.chunk_size):
            matrices = [
                m[chunk_start:chunk_start+self.plan.chunk_size] 
                for m in vstacked_matrices]

            matrices.append(
                numpy.zeros((single_valued(m.shape[0] for m in matrices), 
                    additional_columns))
                )

            diffmats = numpy.asarray(
                    numpy.hstack(matrices),
                    dtype=given.float_type,
                    order="C")
            chunks.append(buffer(diffmats))
        
        from hedge.cuda.tools import pad_and_join

        from pytools import Record
        class GPUDifferentiationMatrices(Record): pass

        return GPUDifferentiationMatrices(
                device_memory=cuda.to_device(
                    pad_and_join(chunks, block_floats*given.float_size())),
                block_floats=block_floats,
                matrix_columns=columns)
