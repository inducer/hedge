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
import hedge.cuda.plan
from pytools import memoize_method, memoize
import pycuda.driver as cuda
import hedge.cuda.plan




# plan ------------------------------------------------------------------------
class SMemFieldDiffExecutionPlan(hedge.cuda.plan.SMemFieldLocalOpExecutionPlan):
    def registers(self):
        return 16

    @memoize_method
    def shared_mem_use(self):
        given = self.given
        
        return (64 # parameters, block header, small extra stuff
               + given.float_size() * (
                   self.parallelism.parallel * self.given.microblock.aligned_floats))

    @staticmethod
    def plan_type():
        return "diff"

    @staticmethod
    def feature_columns():
        return ("type text",
                "parallel integer", 
                "inline integer", 
                "serial integer", 
                "chunk_size integer", 
                "max_unroll integer",
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
                0,
                0,
                lmem,
                smem,
                registers,
                self.threads(),
                )

    def make_kernel(self, discr):
        return SMemFieldDiffKernel(discr, self)




# kernel ----------------------------------------------------------------------
class SMemFieldDiffKernel(object):
    def __init__(self, discr, plan):
        self.discr = discr
        self.plan = plan

        from hedge.cuda.tools import int_ceiling
        fplan = discr.flux_plan

        self.grid = (int_ceiling(
                len(discr.blocks)*fplan.dofs_per_block()
                / self.plan.dofs_per_macroblock()),
                1)

    def benchmark(self):
        if set(["cuda_diff", "cuda_debugbuf"]) <= self.discr.debug:
            return 0

        discr = self.discr
        given = discr.given
        elgroup, = discr.element_groups

        from hedge.optemplate import DifferentiationOperator as op_class
        func = self.get_kernel(op_class, elgroup)

        def vol_empty():
            from hedge.cuda.tools import int_ceiling
            dofs = int_ceiling(
                    discr.flux_plan.dofs_per_block() * len(discr.blocks),     
                    self.plan.dofs_per_macroblock())

            import pycuda.gpuarray as gpuarray
            return gpuarray.empty((dofs,), dtype=given.float_type,
                    allocator=discr.pool.allocate)

        field = vol_empty()
        field.fill(0)

        xyz_diff = [vol_empty() for axis in range(discr.dimensions)]
        xyz_diff_gpudata = [subarray.gpudata for subarray in xyz_diff] 

        count = 20

        gpu_diffmats = self.gpu_diffmats(op_class, elgroup)

        start = cuda.Event()
        start.record()
        cuda.Context.synchronize()
        for i in range(count):
            func.prepared_call(self.grid, 
                    0, # debugbuf
                    field.gpudata,
                    *xyz_diff_gpudata)
        stop = cuda.Event()
        stop.record()
        stop.synchronize()

        return (1e-3/count * stop.time_since(start),
                func.lmem, func.smem, func.registers)

    def __call__(self, op_class, field):
        discr = self.discr
        given = discr.given

        d = discr.dimensions
        elgroup, = discr.element_groups

        func = self.get_kernel(op_class, elgroup)

        assert field.dtype == given.float_type

        use_debugbuf = set(["cuda_diff", "cuda_debugbuf"]) <= discr.debug
        if use_debugbuf:
            import pycuda.gpuarray as gpuarray
            debugbuf = gpuarray.zeros((512,), dtype=numpy.float32)
        else:
            from hedge.cuda.tools import FakeGPUArray
            debugbuf = FakeGPUArray()

        xyz_diff = [discr.volume_empty() for axis in range(d)]
        xyz_diff_gpudata = [subarray.gpudata for subarray in xyz_diff] 

        if discr.instrumented:
            discr.diff_op_timer.add_timer_callable(
                    func.prepared_timed_call(self.grid, 
                        debugbuf.gpudata, field.gpudata, *xyz_diff_gpudata))
        else:
            func.prepared_call(self.grid, 
                    debugbuf.gpudata, field.gpudata, *xyz_diff_gpudata)

        if use_debugbuf:
            copied_debugbuf = debugbuf.get()
            print "DEBUG"
            print field.shape
            #print numpy.reshape(copied_debugbuf, (len(copied_debugbuf)//16, 16))
            print copied_debugbuf
            raw_input()

        return xyz_diff

    @memoize_method
    def get_kernel(self, diff_op_cls, elgroup):
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
        given = discr.given
        fplan = discr.flux_plan

        diffmat_data = self.gpu_diffmats(diff_op_cls, elgroup)
        elgroup, = discr.element_groups

        float_type = given.float_type

        f_decl = CudaGlobal(FunctionDeclaration(Value("void", "apply_diff_mat_smem"), 
            [Pointer(POD(float_type, "debugbuf")), Pointer(POD(float_type, "field")), ]
            + [Pointer(POD(float_type, "dxyz%d" % i)) for i in dims]
            ))

        par = self.plan.parallelism
        
        rst_channels = given.devdata.make_valid_tex_channel_count(d)
        cmod = Module([
                Value("texture<float%d, 2, cudaReadModeElementType>"
                    % rst_channels, 
                    "rst_to_xyz_tex"),
                Value("texture<float%d, 2, cudaReadModeElementType>"
                    % rst_channels, 
                    "diff_rst_mat_tex"),
                Line(),
                Define("DIMENSIONS", discr.dimensions),
                Define("DOFS_PER_EL", given.dofs_per_el()),
                Define("ALIGNED_DOFS_PER_MB", given.microblock.aligned_floats),
                Define("ELS_PER_MB", given.microblock.elements),
                Define("DOFS_PER_MB", "(DOFS_PER_EL*ELS_PER_MB)"),
                Line(),
                Define("MB_DOF", "threadIdx.x"),
                Define("PAR_MB_NR", "threadIdx.y"),
                Define("EL_DOF", "(MB_DOF - mb_el*DOFS_PER_EL)"),
                Line(),
                Define("MACROBLOCK_NR", "blockIdx.x"),
                Line(),
                Define("PAR_MB_COUNT", par.parallel),
                Define("INLINE_MB_COUNT", par.inline),
                Define("SEQ_MB_COUNT", par.serial),
                Line(),
                Define("THREAD_NUM", "(MB_DOF+PAR_MB_NR*ALIGNED_DOFS_PER_MB)"),
                Line(),
                Define("GLOBAL_MB_NR_BASE", 
                    "(MACROBLOCK_NR*PAR_MB_COUNT*INLINE_MB_COUNT*SEQ_MB_COUNT)"),
                Define("GLOBAL_MB_NR", 
                    "(GLOBAL_MB_NR_BASE"
                    "+ (seq_mb_number*PAR_MB_COUNT + PAR_MB_NR)*INLINE_MB_COUNT)"),
                Define("GLOBAL_MB_DOF_BASE", "(GLOBAL_MB_NR*ALIGNED_DOFS_PER_MB)"),
                Line(),
                CudaShared(
                    ArrayOf(
                        ArrayOf(
                            ArrayOf(
                                POD(float_type, "smem_field"), 
                                "PAR_MB_COUNT"),
                            "INLINE_MB_COUNT"),
                        "ALIGNED_DOFS_PER_MB")),
                Line(),
                ])

        S = Statement
        f_body = Block([
            Initializer(Const(POD(numpy.uint16, "mb_el")),
                "MB_DOF / DOFS_PER_EL"),
            Line(),
            ])
            
        # ---------------------------------------------------------------------
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

            from pytools import flatten
            code.extend([
                Comment("everybody needs to be done with the old data"),
                S("__syncthreads()"),
                Line(),
                ]+[
                Assign("smem_field[PAR_MB_NR][%d][MB_DOF]" % inl,
                    "field[(GLOBAL_MB_NR+%d)*ALIGNED_DOFS_PER_MB + MB_DOF]" % inl)
                for inl in range(par.inline)
                ]+[
                Line(),
                Comment("all the new data must be loaded"),
                S("__syncthreads()"),
                Line(),
                Value("float%d" % rst_channels, "dmat_entries"),
                ]+[
                POD(float_type, "field_value%d" % inl)
                for inl in range(par.inline)
                ]+[
                Line(),

                If("MB_DOF < DOFS_PER_MB", Block(list(flatten([
                    Assign("dmat_entries",
                        "tex2D(diff_rst_mat_tex, EL_DOF, %d)" % j),
                    ]+[
                    Assign("field_value%d" % inl, 
                        "smem_field[PAR_MB_NR][%d][mb_el*DOFS_PER_EL+%d]" % (inl, j))
                    for inl in range(par.inline)
                    ]+[
                    Line(),
                    ]+[
                        S("d%drst%d += dmat_entries.%s * field_value%d" 
                            % (inl, axis, tex_channels[axis], inl))
                        for inl in range(par.inline)
                        for axis in dims
                        ]+[Line()]
                    for j in range(given.dofs_per_el())
                    ))+[store_code]))
                ])

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

        mod = cuda.SourceModule(cmod, 
                keep=True, 
                options=["--maxrregcount=16"]
                )
        print "diff: lmem=%d smem=%d regs=%d" % (mod.lmem, mod.smem, mod.registers)

        rst_to_xyz_texref = mod.get_texref("rst_to_xyz_tex")
        cuda.bind_array_to_texref(
                self.localop_rst_to_xyz(diff_op_cls, elgroup), 
                rst_to_xyz_texref)

        diff_rst_mat_texref = mod.get_texref("diff_rst_mat_tex")
        diff_rst_mat_texref.set_array(self.gpu_diffmats(diff_op_cls, elgroup))

        func = mod.get_function("apply_diff_mat_smem")
        func.prepare(
                ["PP"] + discr.dimensions*[float_type],
                block=(given.microblock.aligned_floats, self.plan.parallelism.parallel, 1),
                texrefs=[rst_to_xyz_texref, diff_rst_mat_texref])
        return func

    # data blocks -------------------------------------------------------------
    @memoize_method
    def gpu_diffmats(self, diff_op_cls, elgroup):
        discr = self.discr
        given = discr.given
        d = discr.dimensions

        rst_channels = given.devdata.make_valid_tex_channel_count(d)
        result = numpy.zeros((rst_channels, given.dofs_per_el(), given.dofs_per_el()),
                dtype=given.float_type, order="F")
        for i, dm in enumerate(diff_op_cls.matrices(elgroup)):
            result[i] = dm

        return cuda.make_multichannel_2d_array(result)

    @memoize_method
    def localop_rst_to_xyz(self, diff_op, elgroup):
        discr = self.discr
        given = discr.given
        d = discr.dimensions

        fplan = discr.flux_plan
        coeffs = diff_op.coefficients(elgroup)

        elgroup_indices = self.discr.elgroup_microblock_indices(elgroup)
        el_count = len(discr.blocks) * fplan.elements_per_block()

        # indexed local, el_number, global
        result_matrix = (coeffs[:,:,elgroup_indices]
                .transpose(1,0,2))
        channels = given.devdata.make_valid_tex_channel_count(d)
        add_channels = channels - result_matrix.shape[0]
        if add_channels:
            result_matrix = numpy.vstack((
                result_matrix,
                numpy.zeros((add_channels,d,el_count), dtype=result_matrix.dtype)
                ))

        assert result_matrix.shape == (channels, d, el_count)

        if discr.debug:
            def get_el_index_in_el_group(el):
                mygroup, idx = discr.group_map[el.id]
                assert mygroup is elgroup
                return idx

            for block in discr.blocks:
                i = block.number * fplan.elements_per_block()
                for mb in block.microblocks:
                    for el in mb:
                        egi = get_el_index_in_el_group(el)
                        assert egi == elgroup_indices[i]
                        assert (result_matrix[:d,:,i].T == coeffs[:,:,egi]).all()
                        i += 1

        return cuda.make_multichannel_2d_array(result_matrix)

