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




# plan ------------------------------------------------------------------------
class ChunkedDiffExecutionPlan(hedge.cuda.plan.ChunkedLocalOperatorExecutionPlan):
    def columns(self):
        return self.given.dofs_per_el() * self.given.ldis.dimensions # r,s,t

    def registers(self):
        return 16

    def fetch_buffer_chunks(self):
        return 0

    def make_kernel(self, discr):
        return DiffKernel(discr)





def make_plan(given):
    def generate_plans():
        from hedge.cuda.tools import int_ceiling

        chunk_sizes = range(given.microblock.align_size, 
                given.microblock.elements*given.dofs_per_el()+1, 
                given.microblock.align_size)

        from hedge.cuda.plan import Parallelism

        for pe in range(1,32):
            from hedge.cuda.tools import int_ceiling
            localop_par = Parallelism(pe, 256//pe)
            for chunk_size in chunk_sizes:
                yield ChunkedDiffExecutionPlan(given, localop_par, chunk_size)

    from hedge.cuda.plan import optimize_plan
    return optimize_plan(
            generate_plans,
            lambda plan: plan.parallelism.total()
            )




# kernel ----------------------------------------------------------------------
class DiffKernel(object):
    def __init__(self, discr):
        self.discr = discr

        from hedge.cuda.tools import int_ceiling
        fplan = discr.flux_plan
        lplan = discr.diff_plan

        self.grid = (lplan.chunks_per_microblock(), 
                    int_ceiling(
                        fplan.dofs_per_block()*len(discr.blocks)/
                        lplan.dofs_per_macroblock()))

    def __call__(self, field):
        discr = self.discr
        given = discr.given
        lplan = discr.diff_plan

        d = discr.dimensions
        elgroup, = discr.element_groups

        func, field_texref = self.get_kernel(op.__class__, elgroup)

        assert field.dtype == given.float_type

        field.bind_to_texref(field_texref)

        xyz_diff = [discr.volume_empty() for axis in range(d)]

        #debugbuf = gpuarray.zeros((512,), dtype=numpy.float32)
        args = [subarray.gpudata for subarray in xyz_diff]+[
                self.ex.diff_kernel.gpu_diffmats(op.__class__, elgroup).device_memory,
                #debugbuf,
                ]

        if discr.instrumented:
            kernel_time = func.prepared_timed_call(
                    self.ex.diff_kernel.grid, *args)
            discr.diff_op_timer.add_time(kernel_time)
            discr.diff_op_counter.add(discr.dimensions)
            discr.flop_counter.add(
                    # r,s,t diff
                    2 # mul+add
                    * discr.dimensions
                    * len(discr.nodes)
                    * given.dofs_per_el()

                    # x,y,z rescale
                    +2 # mul+add
                    * discr.dimensions**2
                    * len(discr.nodes)
                    )
        else:
            func.prepared_call(self.ex.diff_kernel.grid, *args)

        if False:
            copied_debugbuf = debugbuf.get()
            print "DEBUG"
            #print numpy.reshape(copied_debugbuf, (len(copied_debugbuf)//16, 16))
            print copied_debugbuf[:100].reshape((10,10))
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
        lplan = discr.diff_plan

        diffmat_data = self.gpu_diffmats(diff_op_cls, elgroup)
        elgroup, = discr.element_groups

        float_type = given.float_type

        f_decl = CudaGlobal(FunctionDeclaration(Value("void", "apply_diff_mat"), 
            [Pointer(POD(float_type, "dxyz%d" % i)) for i in dims]
            + [
                Pointer(POD(numpy.uint8, "gmem_diff_rst_mat")),
                #Pointer(POD(float_type, "debugbuf")),
                ]
            ))

        rst_channels = discr.devdata.make_valid_tex_channel_count(d)
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
                Define("CHUNK_DOF_COUNT", lplan.chunk_size),
                Define("MB_CHUNK_COUNT", lplan.chunks_per_microblock()),
                Define("MB_DOF_COUNT", given.microblock.aligned_floats),
                Define("MB_EL_COUNT", given.microblock.elements),
                Define("PAR_MB_COUNT", lplan.parallelism.p),
                Define("SEQ_MB_COUNT", lplan.parallelism.s),
                Line(),
                Define("THREAD_NUM", "(CHUNK_DOF+PAR_MB_NR*CHUNK_DOF_COUNT)"),
                Define("COALESCING_THREAD_COUNT", "(PAR_MB_COUNT*CHUNK_DOF_COUNT)"),
                Line(),
                Define("MB_DOF_BASE", "(MB_CHUNK*CHUNK_DOF_COUNT)"),
                Define("MB_DOF", "(MB_DOF_BASE+CHUNK_DOF)"),
                Define("GLOBAL_MB_NR_BASE", "(MACROBLOCK_NR*PAR_MB_COUNT*SEQ_MB_COUNT)"),
                Line(),
                Define("DIFFMAT_CHUNK_FLOATS", diffmat_data.block_floats),
                Define("DIFFMAT_CHUNK_BYTES", "(DIFFMAT_CHUNK_FLOATS*%d)"
                     % given.float_size()),
                Define("DIFFMAT_COLUMNS", diffmat_data.matrix_columns),
                Line(),
                CudaShared(ArrayOf(POD(float_type, "smem_diff_rst_mat"), 
                    "DIFFMAT_COLUMNS*CHUNK_DOF_COUNT")),
                Line(),
                ])

        S = Statement
        f_body = Block()
            
        f_body.extend_log_block("calculate responsibility data", [
            Initializer(POD(numpy.uint8, "mb_el"),
                "MB_DOF/DOFS_PER_EL"),
            ])

        from hedge.cuda.tools import get_load_code
        f_body.extend(
            get_load_code(
                dest="smem_diff_rst_mat",
                base="gmem_diff_rst_mat + MB_CHUNK*DIFFMAT_CHUNK_BYTES",
                bytes="DIFFMAT_CHUNK_BYTES",
                descr="load diff mat chunk")
            +[S("__syncthreads()")])

        # ---------------------------------------------------------------------
        def get_scalar_diff_code(matrix_row, dest_pattern):
            code = []
            for axis in dims:
                code.append(
                    Initializer(POD(float_type, "drst%d" % axis), 0))

            code.append(Line())

            def get_mat_entry(row, col, axis):
                return ("smem_diff_rst_mat["
                        "%(row)s*DIFFMAT_COLUMNS + %(axis)s*DOFS_PER_EL"
                        "+%(col)s"
                        "]" % {"row":row, "col":col, "axis":axis}
                        )

            tex_channels = ["x", "y", "z", "w"]
            from pytools import flatten
            code.extend(
                    [POD(float_type, "field_value"),
                        Line(),
                        ]
                    +list(flatten( [
                        Assign("field_value", 
                            "tex1Dfetch(field_tex, global_mb_dof_base"
                            "+mb_el*DOFS_PER_EL+%d)" % j
                            ),
                        Line(),
                        ]
                        +[
                        S("drst%d += %s * field_value" 
                            % (axis, get_mat_entry(matrix_row, j, axis)))
                        for axis in dims
                        ]+[Line()]
                        for j in range(given.dofs_per_el())
                        ))
                    )

            store_code = Block()
            for glob_axis in dims:
                store_code.append(Block([
                    Initializer(Value("float%d" % rst_channels, "rst_to_xyz"),
                        "tex2D(rst_to_xyz_tex, %d, global_mb_nr*MB_EL_COUNT+mb_el)" 
                        % glob_axis
                        ),
                    Assign(
                        dest_pattern % glob_axis,
                        " + ".join(
                            "rst_to_xyz.%s"
                            "*"
                            "drst%d" % (tex_channels[loc_axis], loc_axis)
                            for loc_axis in dims
                            )
                        )
                    ])
                        )

            code.append(If("MB_DOF < DOFS_PER_EL*MB_EL_COUNT", store_code))

            return code

        f_body.extend([
            For("unsigned short seq_mb_number = 0",
                "seq_mb_number < SEQ_MB_COUNT",
                "++seq_mb_number",
                Block([
                    Initializer(POD(numpy.uint32, "global_mb_nr"),
                        "GLOBAL_MB_NR_BASE + seq_mb_number*PAR_MB_COUNT + PAR_MB_NR"),
                    Initializer(POD(numpy.uint32, "global_mb_dof_base"),
                        "global_mb_nr*MB_DOF_COUNT"),
                    Line(),
                    ]+
                    get_scalar_diff_code(
                        "CHUNK_DOF",
                        "dxyz%d[global_mb_dof_base+MB_DOF]")
                    )
                )
            ])

        # finish off ----------------------------------------------------------
        cmod.append(FunctionBody(f_decl, f_body))

        mod = cuda.SourceModule(cmod, 
                keep=True, 
                #options=["--maxrregcount=10"]
                )
        print "diff: lmem=%d smem=%d regs=%d" % (mod.lmem, mod.smem, mod.registers)

        rst_to_xyz_texref = mod.get_texref("rst_to_xyz_tex")
        cuda.bind_array_to_texref(
                self.localop_rst_to_xyz(diff_op_cls, elgroup), 
                rst_to_xyz_texref)

        field_texref = mod.get_texref("field_tex")

        func = mod.get_function("apply_diff_mat")
        func.prepare(
                discr.dimensions*[float_type] + ["P"],
                block=(lplan.chunk_size, lplan.parallelism.p, 1),
                texrefs=[field_texref, rst_to_xyz_texref])
        return func, field_texref

    # data blocks -------------------------------------------------------------
    @memoize_method
    def gpu_diffmats(self, diff_op_cls, elgroup):
        discr = self.discr
        given = discr.given
        lplan = discr.diff_plan

        columns = given.dofs_per_el()*discr.dimensions
        additional_columns = 0
        # avoid smem fetch bank conflicts by ensuring odd col count
        if columns % 2 == 0:
            columns += 1
            additional_columns += 1

        block_floats = self.discr.devdata.align_dtype(
                columns*lplan.chunk_size, given.float_size())

        vstacked_matrices = [
                numpy.vstack(given.microblock.elements*(m,))
                for m in diff_op_cls.matrices(elgroup)
                ]

        chunks = []

        from pytools import single_valued
        for chunk_start in range(0, given.microblock.elements*given.dofs_per_el(), lplan.chunk_size):
            matrices = [
                m[chunk_start:chunk_start+lplan.chunk_size] 
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

    @memoize_method
    def localop_rst_to_xyz(self, diff_op, elgroup):
        discr = self.discr
        d = discr.dimensions

        fplan = discr.flux_plan
        coeffs = diff_op.coefficients(elgroup)

        elgroup_indices = self.discr.elgroup_microblock_indices(elgroup)
        el_count = len(discr.blocks) * fplan.elements_per_block()

        # indexed local, el_number, global
        result_matrix = (coeffs[:,:,elgroup_indices]
                .transpose(1,0,2))
        channels = discr.devdata.make_valid_tex_channel_count(d)
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
