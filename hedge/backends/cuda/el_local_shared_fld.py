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
from pytools import memoize_method
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
from hedge.backends.cuda.tools import FakeGPUArray
import hedge.backends.cuda.plan

from hedge.backends.cuda.plan import ExecutionPlan as \
        ExecutionPlanBase


# plan ------------------------------------------------------------------------
class ExecutionPlan(ExecutionPlanBase):
    def __init__(self, given, parallelism, debug_name,
           aligned_preimage_dofs_per_microblock, preimage_dofs_per_el,
           aligned_image_dofs_per_microblock, image_dofs_per_el):
        ExecutionPlanBase.__init__(self, given.devdata)

        self.given = given
        self.parallelism = parallelism

        self.debug_name = debug_name

        self.aligned_preimage_dofs_per_microblock = \
                aligned_preimage_dofs_per_microblock
        self.preimage_dofs_per_el = preimage_dofs_per_el
        self.aligned_image_dofs_per_microblock = \
                aligned_image_dofs_per_microblock
        self.image_dofs_per_el = image_dofs_per_el

    def image_dofs_per_macroblock(self):
        return (self.parallelism.total()
                * self.aligned_image_dofs_per_microblock)

    def preimage_dofs_per_macroblock(self):
        return (self.parallelism.total()
                * self.aligned_preimage_dofs_per_microblock)

    def threads(self):
        return (self.parallelism.parallel 
                * self.aligned_image_dofs_per_microblock)

    def __str__(self):
            return "smem_field %s par=%s" % (
                ExecutionPlanBase.__str__(self),
                self.parallelism)

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
            len(discr.blocks)
            * self.plan.given.microblocks_per_block
            * plan.aligned_image_dofs_per_microblock
            / self.plan.image_dofs_per_macroblock()),
            1)

    def benchmark(self):
        discr = self.discr
        given = self.plan.given
        elgroup, = discr.element_groups

        try:
            kernel, mat_texref = \
                    self.get_kernel(with_scaling=True, for_benchmark=True)
        except cuda.CompileError:
            return None

        fake_matrix = self.prepare_matrix(
                numpy.ones(
                    (given.dofs_per_el(), self.plan.preimage_dofs_per_el),
                    dtype=given.float_type))
        mat_texref.set_array(fake_matrix)

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
            debugbuf = gpuarray.zeros((1024,), dtype=given.float_type)
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
                estimated_mb_count = given.block_count*given.microblocks_per_block
                kernel.prepared_call(self.grid,
                        out_vector.gpudata,
                        in_vector.gpudata,
                        0,
                        estimated_mb_count,
                        )
            except cuda.LaunchError:
                return None
        stop = cuda.Event()
        stop.record()
        stop.synchronize()

        return (1e-3/count * stop.time_since(start),
                kernel.local_size_bytes, kernel.shared_size_bytes, kernel.num_regs)

    def __call__(self, in_vector, prepped_mat, out_vector=None):
        discr = self.discr
        elgroup, = discr.element_groups
        given = self.discr.given

        kernel, mat_texref = self.get_kernel()

        mat_texref.set_array(prepped_mat)

        if out_vector is None:
            out_vector = discr.volume_empty()

        if set([self.plan.debug_name, "cuda_debugbuf"]) <= discr.debug:
            debugbuf = gpuarray.zeros((1024,), dtype=self.plan.given.float_type)
        else:
            debugbuf = FakeGPUArray()

        if discr.instrumented:
            discr.el_local_timer.add_timer_callable(
                    kernel.prepared_timed_call(self.grid,
                        out_vector.gpudata,
                        in_vector.gpudata,
                        debugbuf.gpudata,
                        len(discr.blocks)*given.microblocks_per_block,
                        ))

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
                    debugbuf.gpudata,
                    len(discr.blocks)*given.microblocks_per_block,
                    )

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
    def get_kernel(self, for_benchmark=False):
        from codepy.cgen import \
                Pointer, POD, Value, ArrayOf, Const, \
                Module, FunctionDeclaration, FunctionBody, Block, \
                Comment, Line, Include, \
                Define, \
                Initializer, If, For, Statement, Assign

        from codepy.cgen import dtype_to_ctype
        from codepy.cgen.cuda import CudaShared, CudaGlobal

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
                POD(numpy.uint32, "microblock_count"),
                ]
            ))

        cmod = Module([
                Include("pycuda-helpers.hpp"),
                Line(),
                Value("texture<fp_tex_%s, 2, cudaReadModeElementType>"
                    % dtype_to_ctype(float_type),
                    "mat_tex"),
                ])

        plan = self.plan
        par = plan.parallelism

        assert (given.microblock.aligned_floats // given.dofs_per_el()
                == given.microblock.elements)

        cmod.extend([
                Line(),
                Define("DIMENSIONS", discr.dimensions),
                Define("IMAGE_DOFS_PER_EL", plan.image_dofs_per_el),
                Define("PREIMAGE_DOFS_PER_EL", plan.preimage_dofs_per_el),
                Define("ALIGNED_IMAGE_DOFS_PER_MB", plan.aligned_image_dofs_per_microblock),
                Define("ALIGNED_PREIMAGE_DOFS_PER_MB",
                    plan.aligned_preimage_dofs_per_microblock),
                Line(),
                Define("IMAGE_MB_EL_COUNT", 
                    "(ALIGNED_IMAGE_DOFS_PER_MB/IMAGE_DOFS_PER_EL)"),
                Define("PREIMAGE_MB_EL_COUNT", 
                    "(ALIGNED_PREIMAGE_DOFS_PER_MB/PREIMAGE_DOFS_PER_EL)"),
                Line(),
                Define("IMAGE_DOFS_PER_MB", "(IMAGE_DOFS_PER_EL*IMAGE_MB_EL_COUNT)"),
                Line(),
                Define("CHUNK_SIZE", given.devdata.smem_granularity),
                Define("CHUNK_DOF", "threadIdx.x"),
                Define("PAR_MB_NR", "threadIdx.y"),
                Define("CHUNK_NR", "threadIdx.z"),
                Define("IMAGE_MB_DOF", "(CHUNK_NR*CHUNK_SIZE+CHUNK_DOF)"),
                Define("IMAGE_EL_DOF", "(IMAGE_MB_DOF - mb_el*IMAGE_DOFS_PER_EL)"),
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
                Define("GLOBAL_MB_IMAGE_DOF_BASE", "(GLOBAL_MB_NR*ALIGNED_IMAGE_DOFS_PER_MB)"),
                Define("GLOBAL_MB_PREIMAGE_DOF_BASE", "(GLOBAL_MB_NR*ALIGNED_PREIMAGE_DOFS_PER_MB)"),
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
                "IMAGE_MB_DOF / IMAGE_DOFS_PER_EL"),
            Line(),
            ])

        def get_load_code():
            mb_img_dofs = plan.aligned_image_dofs_per_microblock
            mb_preimg_dofs = plan.aligned_preimage_dofs_per_microblock
            preimg_dofs_over_dofs = (mb_preimg_dofs+mb_img_dofs-1) // mb_img_dofs

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

                    block_addr = "%d * ALIGNED_IMAGE_DOFS_PER_MB + IMAGE_MB_DOF" % load_block
                    load_instr = Assign(var,
                        "in_vector[GLOBAL_MB_PREIMAGE_DOF_BASE"
                        " + %d*ALIGNED_PREIMAGE_DOFS_PER_MB"
                        " + %s]" % (inl, block_addr))
                    store_instr = Assign(
                            "smem_in_vector[PAR_MB_NR][%d][%s]" % (inl, block_addr),
                            var
                            )
                    if (load_block+1)*mb_img_dofs >= mb_preimg_dofs:
                        cond = "%s < ALIGNED_PREIMAGE_DOFS_PER_MB" % block_addr
                        load_instr = If(cond, load_instr)
                        store_instr = If(cond, store_instr)

                    load_code.append(load_instr)
                    store_code.append(store_instr)
            return Block(load_code + [Line()] + store_code)

        def get_matmul_code():
            from hedge.backends.cuda.tools import unroll

            index_check_condition = "GLOBAL_MB_NR < microblock_count"

            def if_(conditions, then):
                final_cond = " && ".join(cond for cond in conditions if cond)
                if final_cond:
                    return If(final_cond, then)
                else:
                    return then

            result = Block([
                Comment("everybody needs to be done with the old data"),
                S("__syncthreads()"), Line(),
                ]+[If(index_check_condition, get_load_code())]+[
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
                ])

            result.append(if_(["IMAGE_MB_DOF < IMAGE_DOFS_PER_MB", index_check_condition],
                Block(unroll(lambda j:
                    [Assign("mat_entry", "fp_tex2D(mat_tex, IMAGE_EL_DOF, %s)" % j)]
                    +[
                    S("result%d += mat_entry "
                    "* smem_in_vector[PAR_MB_NR][%d][mb_el*PREIMAGE_DOFS_PER_EL + %s]"
                    % (inl, inl, j))
                    for inl in range(par.inline)
                    ],
                    total_number=plan.preimage_dofs_per_el)
                    +[Line()]
                    +[Assign(
                        "out_vector[GLOBAL_MB_IMAGE_DOF_BASE + "
                        "%d*ALIGNED_IMAGE_DOFS_PER_MB + IMAGE_MB_DOF]" % inl,
                        "result%d" % inl)
                    for inl in range(par.inline)]
                    )))

            return result

        f_body.append(For("unsigned short seq_mb_number = 0",
            "seq_mb_number < SEQ_MB_COUNT",
            "++seq_mb_number", get_matmul_code()))

        # finish off ----------------------------------------------------------
        cmod.append(FunctionBody(f_decl, f_body))

        if not for_benchmark and "cuda_dump_kernels" in discr.debug:
            from hedge.tools import open_unique_debug_file
            open_unique_debug_file(plan.debug_name, ".cu").write(str(cmod))

        mod = SourceModule(cmod,
                keep="cuda_keep_kernels" in discr.debug,
                #options=["--maxrregcount=12"]
                )

        func = mod.get_function("apply_el_local_mat_smem_field")

        if plan.debug_name in discr.debug:
            print "%s: lmem=%d smem=%d regs=%d" % (
                    plan.debug_name,
                    func.local_size_bytes,
                    func.shared_size_bytes,
                    func.num_regs)

        mat_texref = mod.get_texref("mat_tex")
        texrefs = [mat_texref]

        func.prepare(
                "PPPI",
                block=(
                    given.devdata.smem_granularity,
                    plan.parallelism.parallel,
                    plan.aligned_image_dofs_per_microblock
                        //given.devdata.smem_granularity),
                texrefs=texrefs)

        return func, mat_texref

    # data blocks -------------------------------------------------------------
    def prepare_matrix(self, matrix):
        plan = self.plan
        given = plan.given

        assert matrix.shape == (
                plan.image_dofs_per_el, plan.preimage_dofs_per_el)

        return cuda.matrix_to_array(matrix.astype(given.float_type), "F",
                allow_double_hack=True)
