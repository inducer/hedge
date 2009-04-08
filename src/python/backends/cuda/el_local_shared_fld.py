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
class ExecutionPlan(hedge.backends.cuda.plan.SMemFieldLocalOpExecutionPlan):
    def __init__(self, given, parallelism, max_unroll, debug_name,
           aligned_preimage_dofs_per_microblock, preimage_dofs_per_el):
        hedge.backends.cuda.plan.SMemFieldLocalOpExecutionPlan.__init__(
                self, given, parallelism, max_unroll)

        self.debug_name = debug_name
        self.aligned_preimage_dofs_per_microblock = \
                aligned_preimage_dofs_per_microblock
        self.preimage_dofs_per_el = preimage_dofs_per_el

    def registers(self):
        return 16

    @memoize_method
    def shared_mem_use(self):
        given = self.given
        
        return (64 # parameters, block header, small extra stuff
               + given.float_size() * (
                   self.parallelism.parallel 
                   * self.parallelism.inline
                   * self.aligned_preimage_dofs_per_microblock))

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
        return ("smem_field",
                self.parallelism.parallel,
                self.parallelism.inline,
                self.parallelism.serial,
                None,
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
        self.grid = (int_ceiling(
            self.plan.given.total_dofs()
            / self.plan.dofs_per_macroblock()),
            1)

    def benchmark(self):
        discr = self.discr
        given = self.plan.given
        elgroup, = discr.element_groups

        try:
            kernel, mat_texref, scaling_texref = \
                    self.get_kernel(with_scaling=True, for_benchmark=True)
        except cuda.CompileError:
            return None

        fake_matrix = self.prepare_matrix(
                numpy.ones(
                    (given.dofs_per_el(), self.plan.preimage_dofs_per_el),
                    dtype=given.float_type))
        mat_texref.set_array(fake_matrix)

        from hedge.backends.cuda.kernelbase import fake_elwise_scaling
        fake_scaling = fake_elwise_scaling(self.plan.given)
        fake_scaling.bind_to_texref(scaling_texref)

        def vol_empty():
            from hedge.backends.cuda.tools import int_ceiling
            dofs = int_ceiling(given.total_dofs(), self.plan.dofs_per_macroblock())

            return gpuarray.empty((dofs,), dtype=given.float_type,
                    allocator=discr.pool.allocate)

        out_vector = vol_empty()
        in_vector = gpuarray.empty(
                given.matmul_preimage_shape(self.plan), 
                dtype=given.float_type,
                allocator=discr.pool.allocate)

        if set([self.plan.debug_name, "cuda_debugbuf"]) <= discr.debug:
            debugbuf = gpuarray.zeros((1024,), dtype=numpy.float32)
        else:
            debugbuf = FakeGPUArray()

        if "cuda_fastbench" in discr.debug:
            count = 1
        else:
            count = 20

        start = cuda.Event()
        start.record()
        cuda.Context.synchronize()
        for i in range(count):
            try:
                kernel.prepared_call(self.grid,
                        out_vector.gpudata, 
                        in_vector.gpudata,
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

        kernel, mat_texref, scaling_texref = \
                self.get_kernel(with_scaling=prepped_scaling is not None)

        mat_texref.set_array(prepped_mat)
        if prepped_scaling is not None:
            prepped_scaling.bind_to_texref(scaling_texref)

        out_vector = discr.volume_empty() 

        if set([self.plan.debug_name, "cuda_debugbuf"]) <= discr.debug:
            debugbuf = gpuarray.zeros((1024,), dtype=numpy.float32)
        else:
            debugbuf = FakeGPUArray()

        if discr.instrumented:
            discr.el_local_timer.add_timer_callable(
                    kernel.prepared_timed_call(self.grid, 
                        out_vector.gpudata, 
                        in_vector.gpudata,
                        debugbuf.gpudata))

            given = self.discr.given

            block_gmem_floats = (
                        # matrix fetch
                        given.microblock.aligned_floats
                        * self.plan.preimage_dofs_per_el
                        * self.plan.parallelism.serial
                        * self.plan.parallelism.parallel
                        # field fetch
                        + self.plan.preimage_dofs_per_el
                        * given.microblock.elements
                        * self.plan.parallelism.total()
                        )
            gmem_bytes = given.float_size() * ( 
                    self.grid[0] * block_gmem_floats 
                    # field store
                    + len(discr.nodes))

            discr.gmem_bytes_el_local.add(gmem_bytes)
        else:
            kernel.prepared_call(self.grid,
                    out_vector.gpudata, 
                    in_vector.gpudata,
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
                Comment, Line, \
                Static, \
                Define, \
                Constant, Initializer, If, For, Statement, Assign, \
                ArrayInitializer

        from codepy.cgen.cuda import CudaShared, CudaConstant, CudaGlobal
        discr = self.discr
        d = discr.dimensions
        dims = range(d)
        given = self.plan.given

        float_type = given.float_type

        f_decl = CudaGlobal(FunctionDeclaration(Value("void", "apply_el_local_mat_smem_field"), 
            [
                Pointer(POD(float_type, "out_vector")),
                Pointer(POD(float_type, "in_vector")),
                Pointer(POD(float_type, "debugbuf")),
                ]
            ))

        cmod = Module([
                Value("texture<float, 2, cudaReadModeElementType>", 
                    "mat_tex"),
                ])
        if with_scaling:
            cmod.append(
                Value("texture<float, 1, cudaReadModeElementType>",
                    "scaling_tex"),
                )

        par = self.plan.parallelism

        cmod.extend([
                Line(),
                Define("DIMENSIONS", discr.dimensions),
                Define("DOFS_PER_EL", given.dofs_per_el()),
                Define("MB_EL_COUNT", given.microblock.elements),
                Define("PREIMAGE_DOFS_PER_EL", self.plan.preimage_dofs_per_el),
                Line(),
                Define("DOFS_PER_MB", "(DOFS_PER_EL*MB_EL_COUNT)"),
                Define("ALIGNED_DOFS_PER_MB", given.microblock.aligned_floats),
                Define("ALIGNED_PREIMAGE_DOFS_PER_MB", 
                    self.plan.aligned_preimage_dofs_per_microblock),
                Line(),
                Define("CHUNK_SIZE", given.devdata.smem_granularity),
                Define("CHUNK_DOF", "threadIdx.x"),
                Define("PAR_MB_NR", "threadIdx.y"),
                Define("CHUNK_NR", "threadIdx.z"),
                Define("MB_DOF", "(CHUNK_NR*CHUNK_SIZE+CHUNK_DOF)"),
                Define("EL_DOF", "(MB_DOF - mb_el*DOFS_PER_EL)"),
                Line(),
                Define("MACROBLOCK_NR", "blockIdx.x"),
                Line(),
                Define("PAR_MB_COUNT", par.parallel),
                Define("INLINE_MB_COUNT", par.inline),
                Define("SEQ_MB_COUNT", par.serial),
                Line(),
                Define("GLOBAL_MB_NR_BASE", 
                    "(MACROBLOCK_NR*PAR_MB_COUNT*INLINE_MB_COUNT*SEQ_MB_COUNT)"),
                Define("GLOBAL_MB_NR", 
                    "(GLOBAL_MB_NR_BASE"
                    "+ (seq_mb_number*PAR_MB_COUNT + PAR_MB_NR)*INLINE_MB_COUNT)"),
                Define("GLOBAL_MB_DOF_BASE", "(GLOBAL_MB_NR*ALIGNED_DOFS_PER_MB)"),
                Define("GLOBAL_MB_PREIMAGE_BASE", "(GLOBAL_MB_NR*ALIGNED_PREIMAGE_DOFS_PER_MB)"),
                Line(),
                CudaShared(
                    ArrayOf(
                        ArrayOf(
                            ArrayOf(
                                POD(float_type, "smem_in_vector"), 
                                "PAR_MB_COUNT"),
                            "INLINE_MB_COUNT"),
                        "ALIGNED_PREIMAGE_DOFS_PER_MB")),
                Line(),
                ])

        S = Statement
        f_body = Block([
            Initializer(Const(POD(numpy.uint16, "mb_el")),
                "MB_DOF / DOFS_PER_EL"),
            Line(),
            ])

        def get_load_code():
            mb_dofs = given.microblock.aligned_floats
            mb_preimg_dofs = self.plan.aligned_preimage_dofs_per_microblock
            preimg_dofs_over_dofs = (mb_preimg_dofs+mb_dofs-1) // mb_dofs

            load_code = []
            store_code = []

            var_num = 0
            for load_block in range(preimg_dofs_over_dofs):
                for inl in range(par.inline):
                    # load and store are split for better pipelining
                    # compiler can't figure that out because of branch

                    var = "tmp%d" % var_num
                    var_num += 1
                    load_code.append(POD(float_type, var))

                    block_addr = "%d * ALIGNED_DOFS_PER_MB + MB_DOF" % load_block
                    load_instr = Assign(var, 
                        "in_vector[GLOBAL_MB_PREIMAGE_BASE"
                        " + %d*ALIGNED_PREIMAGE_DOFS_PER_MB"
                        " + %s]" % (inl, block_addr))
                    store_instr = Assign(
                            "smem_in_vector[PAR_MB_NR][%d][%s]" % (inl, block_addr),
                            var
                            )
                    if (load_block+1)*mb_dofs >= mb_preimg_dofs:
                        cond = "%s < ALIGNED_PREIMAGE_DOFS_PER_MB" % block_addr
                        load_instr = If(cond, load_instr)
                        store_instr = If(cond, store_instr)

                    load_code.append(load_instr)
                    store_code.append(store_instr)
            return load_code + [Line()] + store_code

        def get_matmul_code():
            from hedge.backends.cuda.tools import unroll

            if with_scaling:
                inv_jac_multiplier = ("tex1Dfetch(scaling_tex,"
                        "(GLOBAL_MB_NR + %(inl)d)*MB_EL_COUNT + mb_el)")
            else:
                inv_jac_multiplier = "1"

            return Block([
                Comment("everybody needs to be done with the old data"),
                S("__syncthreads()"),
                Line(),
                ]+get_load_code()+[
                Line(),
                Comment("all the new data must be loaded"),
                S("__syncthreads()"),
                Line(),
                ]+[
                Initializer(POD(float_type, "result%d" % inl), 0)
                for inl in range(par.inline)
                ]+[
                Line(),
                POD(float_type, "mat_entry"),
                Line(),
                If("MB_DOF < DOFS_PER_MB", Block(unroll(
                    lambda j:
                    [Assign("mat_entry", "tex2D(mat_tex, EL_DOF, %s)" % j)]
                    +[
                    S("result%d += mat_entry "
                    "* smem_in_vector[PAR_MB_NR][%d][mb_el*PREIMAGE_DOFS_PER_EL + %s]" 
                    % (inl, inl, j))
                    for inl in range(par.inline)
                    ],
                    total_number=self.plan.preimage_dofs_per_el,
                    max_unroll=self.plan.max_unroll)
                    +[ Line(), ]
                    +[ Assign(
                        "out_vector[GLOBAL_MB_DOF_BASE + %d*ALIGNED_DOFS_PER_MB + MB_DOF]" % inl,
                        "result%d*%s" % (inl, (inv_jac_multiplier % {"inl": inl})))
                    for inl in range(par.inline)
                    ]))
                ])

        f_body.append(For("unsigned short seq_mb_number = 0",
            "seq_mb_number < SEQ_MB_COUNT",
            "++seq_mb_number", get_matmul_code()))

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

        mat_texref = mod.get_texref("mat_tex")
        texrefs = [mat_texref]

        if with_scaling:
            scaling_texref = mod.get_texref("scaling_tex")
            texrefs.append(scaling_texref)
        else:
            scaling_texref = None

        func = mod.get_function("apply_el_local_mat_smem_field")
        func.prepare(
                "PPP", 
                block=(
                    given.devdata.smem_granularity, 
                    self.plan.parallelism.parallel, 
                    given.microblock.aligned_floats//given.devdata.smem_granularity),
                texrefs=texrefs)

        return func, mat_texref, scaling_texref

    # data blocks -------------------------------------------------------------
    def prepare_matrix(self, matrix):
        given = self.plan.given

        assert matrix.shape == (given.dofs_per_el(), self.plan.preimage_dofs_per_el)

        return cuda.matrix_to_array(matrix.astype(given.float_type), "F")

    def prepare_scaling(self, elgroup, scaling):
        ij = scaling[self.discr.elgroup_microblock_indices(elgroup)]
        return gpuarray.to_gpu(
                ij.astype(self.plan.given.float_type))

