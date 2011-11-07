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
import hedge.backends.cuda.plan
from pytools import memoize_method
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import hedge.backends.cuda.plan
from pycuda.compiler import SourceModule

from hedge.backends.cuda.plan import ExecutionPlan as \
        ExecutionPlanBase




# plan ------------------------------------------------------------------------
class ExecutionPlan(ExecutionPlanBase):
    def __init__(self, given, parallelism,
           aligned_preimage_dofs_per_microblock, preimage_dofs_per_el,
           aligned_image_dofs_per_microblock, image_dofs_per_el):
        ExecutionPlanBase.__init__(self, given.devdata)
        self.given = given
        self.parallelism = parallelism

        self.aligned_preimage_dofs_per_microblock = \
                aligned_preimage_dofs_per_microblock
        self.preimage_dofs_per_el = preimage_dofs_per_el
        self.aligned_image_dofs_per_microblock = \
                aligned_image_dofs_per_microblock
        self.image_dofs_per_el = image_dofs_per_el

    def dofs_per_macroblock(self):
        return self.parallelism.total() * self.given.microblock.aligned_floats

    def preimage_dofs_per_macroblock(self):
        return (self.parallelism.total()
                * self.aligned_preimage_dofs_per_microblock)

    def threads(self):
        return self.parallelism.parallel * self.given.microblock.aligned_floats

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
            self.plan.given.total_dofs()
            / self.plan.dofs_per_macroblock()), 1)

    def benchmark(self):
        if set(["cuda_diff", "cuda_debugbuf"]) <= self.discr.debug:
            return 0

        discr = self.discr
        given = self.plan.given
        elgroup, = discr.element_groups

        from hedge.optemplate \
                import ReferenceDifferentiationOperator as op_class
        try:
            block, func = self.get_kernel(op_class, elgroup, for_benchmark=True)
        except cuda.CompileError:
            return None

        def vol_empty():
            from hedge.backends.cuda.tools import int_ceiling
            dofs = int_ceiling(
                    given.total_dofs(), self.plan.dofs_per_macroblock())

            import pycuda.gpuarray as gpuarray
            return gpuarray.empty((dofs,), dtype=given.float_type,
                    allocator=discr.pool.allocate)

        field = vol_empty()
        field.fill(0)

        rst_diff = [vol_empty() for axis in range(discr.dimensions)]
        rst_diff_gpudata = [subarray.gpudata for subarray in rst_diff]

        if "cuda_fastbench" in discr.debug:
            count = 1
        else:
            count = 20

        start = cuda.Event()
        start.record()
        cuda.Context.synchronize()
        for i in range(count):
            try:
                func.prepared_call(self.grid, block,
                        0, # debugbuf
                        field.gpudata,
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

        block, func = self.get_kernel(op_class, elgroup)

        assert field.dtype == given.float_type

        use_debugbuf = set(["cuda_diff", "cuda_debugbuf"]) <= discr.debug
        if use_debugbuf:
            debugbuf = gpuarray.zeros((512,), dtype=given.float_type)
        else:
            from hedge.backends.cuda.tools import FakeGPUArray
            debugbuf = FakeGPUArray()

        rst_diff = [discr.volume_empty() for axis in range(d)]
        rst_diff_gpudata = [subarray.gpudata for subarray in rst_diff]

        if discr.instrumented:
            discr.diff_op_timer.add_timer_callable(
                    func.prepared_timed_call(self.grid, block,
                        debugbuf.gpudata, field.gpudata, *rst_diff_gpudata))

            block_gmem_floats = (
                    # matrix fetch
                    given.microblock.aligned_floats
                    * discr.dimensions
                    * given.dofs_per_el()
                    * self.plan.parallelism.serial
                    * self.plan.parallelism.parallel
                    # field fetch
                    + given.microblock.aligned_floats
                    * self.plan.parallelism.total()
                    )

            gmem_bytes = given.float_size() * (
                    self.grid[0] * block_gmem_floats
                    # field store
                    + len(discr.nodes))

            discr.gmem_bytes_diff.add(gmem_bytes)
        else:
            func.prepared_call(self.grid, block,
                    debugbuf.gpudata, field.gpudata, *rst_diff_gpudata)

        if use_debugbuf:
            copied_debugbuf = debugbuf.get()
            print "DEBUG"
            print field.shape
            #print numpy.reshape(copied_debugbuf, (len(copied_debugbuf)//16, 16))
            print copied_debugbuf
            raw_input()

        return rst_diff

    @memoize_method
    def get_kernel(self, diff_op, elgroup, for_benchmark=False):
        from cgen import \
                Pointer, POD, Value, ArrayOf, Const, \
                Module, FunctionDeclaration, FunctionBody, Block, \
                Comment, Line, Define, Include, \
                Initializer, If, For, Statement, Assign

        from pycuda.tools import dtype_to_ctype
        from cgen.cuda import CudaShared, CudaGlobal

        discr = self.discr
        d = discr.dimensions
        dims = range(d)
        plan = self.plan
        given = plan.given

        elgroup, = discr.element_groups
        float_type = given.float_type

        f_decl = CudaGlobal(FunctionDeclaration(Value("void", "apply_diff_mat_smem"),
            [Pointer(POD(float_type, "debugbuf")), Pointer(POD(float_type, "field")), ]
            + [Pointer(POD(float_type, "drst%d_global" % i)) for i in dims]
            ))

        par = plan.parallelism

        cmod = Module([
                Include("pycuda-helpers.hpp"),
                ])

        if float_type == numpy.float64:
            cmod.append(Value("texture<fp_tex_double, 1, cudaReadModeElementType>",
                    "diff_rst_mat_tex"))
        elif float_type == numpy.float32:
            rst_channels = given.devdata.make_valid_tex_channel_count(d)
            cmod.append(Value("texture<float%d, 1, cudaReadModeElementType>"
                    % rst_channels, "diff_rst_mat_tex"))
        else:
            raise ValueError("unsupported float type: %s" % float_type)

        # only preimage size variation is supported here
        assert plan.image_dofs_per_el == given.dofs_per_el()
        assert plan.aligned_image_dofs_per_microblock == given.microblock.aligned_floats

        # FIXME: aligned_image_dofs_per_microblock must be divisible
        # by this, therefore hardcoding for now.
        chunk_size = 16

        cmod.extend([
                Line(),
                Define("DIMENSIONS", discr.dimensions),

                Define("IMAGE_DOFS_PER_EL", plan.image_dofs_per_el),
                Define("PREIMAGE_DOFS_PER_EL", plan.preimage_dofs_per_el),
                Define("ALIGNED_IMAGE_DOFS_PER_MB", plan.aligned_image_dofs_per_microblock),
                Define("ALIGNED_PREIMAGE_DOFS_PER_MB", plan.aligned_preimage_dofs_per_microblock),
                Define("ELS_PER_MB", given.microblock.elements),
                Define("IMAGE_DOFS_PER_MB", "(IMAGE_DOFS_PER_EL*ELS_PER_MB)"),
                Line(),
                Define("CHUNK_SIZE", chunk_size),
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
                                POD(float_type, "smem_field"),
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

        # ---------------------------------------------------------------------
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
                        "field[GLOBAL_MB_PREIMAGE_DOF_BASE"
                        " + %d*ALIGNED_PREIMAGE_DOFS_PER_MB"
                        " + %s]" % (inl, block_addr))
                    store_instr = Assign(
                            "smem_field[PAR_MB_NR][%d][%s]" % (inl, block_addr),
                            var
                            )
                    if (load_block+1)*mb_img_dofs >= mb_preimg_dofs:
                        cond = "%s < ALIGNED_PREIMAGE_DOFS_PER_MB" % block_addr
                        load_instr = If(cond, load_instr)
                        store_instr = If(cond, store_instr)

                    load_code.append(load_instr)
                    store_code.append(store_instr)
            return Block(load_code + [Line()] + store_code)

        def get_scalar_diff_code():
            code = []
            for inl in range(par.inline):
                for axis in dims:
                    code.append(
                        Initializer(POD(float_type, "d%drst%d" % (inl, axis)), 0))

            code.append(Line())

            tex_channels = ["x", "y", "z", "w"]

            store_code = Block()
            for inl in range(par.inline):
                for rst_axis in dims:
                    store_code.append(Assign(
                        "drst%d_global[GLOBAL_MB_IMAGE_DOF_BASE + "
                        "%d*ALIGNED_IMAGE_DOFS_PER_MB + IMAGE_MB_DOF]"
                        % (rst_axis, inl),
                        "d%drst%d" % (inl, rst_axis)
                        ))

            from hedge.backends.cuda.tools import unroll
            code.extend([
                Comment("everybody needs to be done with the old data"),
                S("__syncthreads()"),
                Line(),
                get_load_code(),
                Line(),
                Comment("all the new data must be loaded"),
                S("__syncthreads()"),
                Line(),
                ])

            if float_type == numpy.float32:
                code.append(Value("float%d" % rst_channels, "dmat_entries"))

            code.extend([
                POD(float_type, "field_value%d" % inl)
                for inl in range(par.inline)
                ]+[Line()])

            def unroll_body(j):
                result = [
                    Assign("field_value%d" % inl,
                        "smem_field[PAR_MB_NR][%d][mb_el*PREIMAGE_DOFS_PER_EL+%s]" % (inl, j))
                    for inl in range(par.inline)
                    ]

                if float_type == numpy.float32:
                    result.append(Assign("dmat_entries",
                        "tex1Dfetch(diff_rst_mat_tex, IMAGE_EL_DOF + %s*IMAGE_DOFS_PER_EL)" % j))
                    result.extend(
                        S("d%drst%d += dmat_entries.%s * field_value%d"
                            % (inl, axis, tex_channels[axis], inl))
                        for inl in range(par.inline)
                        for axis in dims)
                elif float_type == numpy.float64:
                    result.extend(
                        S("d%(inl)drst%(axis)d += "
                            "fp_tex1Dfetch(diff_rst_mat_tex, %(axis)d "
                            "+ DIMENSIONS*(IMAGE_EL_DOF + %(j)d*IMAGE_DOFS_PER_EL))"
                            "* field_value%(inl)d" % {
                            "inl": inl,
                            "axis": axis,
                            "j": j
                            })
                        for inl in range(par.inline)
                        for axis in dims)
                else:
                    assert False

                return result

            code.append(If("IMAGE_MB_DOF < IMAGE_DOFS_PER_MB", Block(unroll(unroll_body,
                    total_number=plan.preimage_dofs_per_el)
                    +[store_code])))

            return code

        f_body.extend([
            For("unsigned short seq_mb_number = 0",
                "seq_mb_number < SEQ_MB_COUNT",
                "++seq_mb_number",
                Block(get_scalar_diff_code())
                )
            ])

        # finish off ----------------------------------------------------------
        cmod.append(FunctionBody(f_decl, f_body))

        if not for_benchmark and "cuda_dump_kernels" in discr.debug:
            from hedge.tools import open_unique_debug_file
            open_unique_debug_file("diff", ".cu").write(str(cmod))

        mod = SourceModule(cmod,
                keep="cuda_keep_kernels" in discr.debug,
                #options=["--maxrregcount=16"]
                )

        func = mod.get_function("apply_diff_mat_smem")

        if "cuda_diff" in discr.debug:
            print "diff: lmem=%d smem=%d regs=%d" % (
                    func.local_size_bytes,
                    func.shared_size_bytes,
                    func.registers)

        diff_rst_mat_texref = mod.get_texref("diff_rst_mat_tex")
        gpu_diffmats = self.gpu_diffmats(diff_op, elgroup)

        if given.float_type == numpy.float32:
            gpu_diffmats.bind_to_texref_ext(diff_rst_mat_texref, rst_channels)
        elif given.float_type == numpy.float64:
            gpu_diffmats.bind_to_texref_ext(diff_rst_mat_texref,
                    allow_double_hack=True)
        else:
            assert False

        assert given.microblock.aligned_floats % chunk_size == 0
        block = (
                chunk_size,
                plan.parallelism.parallel,
                given.microblock.aligned_floats//chunk_size)

        func.prepare(
                ["PP"] + discr.dimensions*[float_type],
                texrefs=[diff_rst_mat_texref])

        return block, func

    # data blocks -------------------------------------------------------------
    @memoize_method
    def gpu_diffmats(self, diff_op, elgroup):
        discr = self.discr
        given = self.plan.given
        d = discr.dimensions

        diff_op.matrices(elgroup)

        if given.float_type == numpy.float32:
            first_dim = given.devdata.make_valid_tex_channel_count(d)
        elif given.float_type == numpy.float64:
            first_dim = d
        else:
            assert False

        result = numpy.zeros((first_dim,
            self.plan.image_dofs_per_el,
            self.plan.preimage_dofs_per_el),
            dtype=given.float_type, order="F")
        for i, dm in enumerate(diff_op.matrices(elgroup)):
            result[i] = dm

        return gpuarray.to_gpu(result)
