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
from hedge.cuda.tools import FakeGPUArray
import hedge.cuda.plan 




# plan ------------------------------------------------------------------------
class FluxLiftingExecutionPlan(hedge.cuda.plan.ChunkedMatrixLocalOpExecutionPlan):
    def __init__(self, given, parallelism, chunk_size, use_prefetch_branch):
        hedge.cuda.plan.ChunkedMatrixLocalOpExecutionPlan.__init__(
                self, given, parallelism, chunk_size)

        self.use_prefetch_branch = use_prefetch_branch

    def columns(self):
        return self.given.face_dofs_per_el()

    def registers(self):
        return 12 + par.inline

    def fetch_buffer_chunks(self):
        return 1

    def __str__(self):
        return "%s prefetch_branch=%s" % (
                hedge.cuda.plan.ChunkedMatrixLocalOpExecutionPlan.__str__(self),
                self.use_prefetch_branch)

    def make_kernel(self, discr):
        return FluxLocalKernel(discr, self)



def make_plan(discr, given):
    def generate_plans():
        from hedge.cuda.plan import Parallelism

        for use_prefetch_branch in [True]:
        #for use_prefetch_branch in [True, False]:
            chunk_sizes = range(given.microblock.align_size, 
                    given.microblock.elements*given.dofs_per_el()+1, 
                    given.microblock.align_size)

            for pe in range(1,32):
                for inline in range(1, 5):
                    for seq in range(1, 5):
                        localop_par = Parallelism(pe, inline, seq)
                        for chunk_size in chunk_sizes:
                            yield FluxLiftingExecutionPlan(given, 
                                    localop_par, chunk_size,
                                    use_prefetch_branch)

        from hedge.cuda.fluxlocal_alt import SMemFieldFluxLocalExecutionPlan

        for pe in range(1,32):
            localop_par = Parallelism(pe, 1, 1)
            yield SMemFieldFluxLocalExecutionPlan(given, localop_par)

    def target_func(plan):
        return plan.make_kernel(discr).benchmark()

    from hedge.cuda.plan import optimize_plan
    return optimize_plan(generate_plans, target_func, maximize=False,
            desirable_occupancy=0.5, debug=True)




# kernel ----------------------------------------------------------------------
class FluxLocalKernel(object):
    def __init__(self, discr, plan):
        self.discr = discr
        self.plan = plan

        from hedge.cuda.tools import int_ceiling
        self.grid = (plan.chunks_per_microblock(), 
                int_ceiling(
                    len(discr.blocks)
                    * discr.flux_plan.dofs_per_block()
                    / plan.dofs_per_macroblock())
                )

    def benchmark(self):
        discr = self.discr
        given = discr.given
        elgroup, = discr.element_groups

        is_lift = True
        lift, fluxes_on_faces_texref = self.get_kernel(is_lift, elgroup)

        def vol_empty():
            from hedge.cuda.tools import int_ceiling
            dofs = int_ceiling(
                    discr.flux_plan.dofs_per_block() * len(discr.blocks),     
                    self.plan.dofs_per_macroblock())

            return gpuarray.empty((dofs,), dtype=given.float_type,
                    allocator=discr.pool.allocate)

        flux = vol_empty()
        fluxes_on_faces = gpuarray.empty(
                discr.flux_plan.fluxes_on_faces_shape(len(discr.blocks)), 
                dtype=given.float_type,
                allocator=discr.pool.allocate)
        fluxes_on_faces.bind_to_texref(fluxes_on_faces_texref)

        count = 20

        start = cuda.Event()
        start.record()
        cuda.Context.synchronize()
        for i in range(count):
            lift.prepared_call(
                    self.grid,
                    flux.gpudata, 
                    self.gpu_liftmat(is_lift).device_memory,
                    0)
        stop = cuda.Event()
        stop.record()
        stop.synchronize()

        return 1e-3/count * stop.time_since(start)

    def __call__(self, fluxes_on_faces, is_lift):
        discr = self.discr
        elgroup, = discr.element_groups

        lift, fluxes_on_faces_texref = \
                self.get_kernel(is_lift, elgroup)

        flux = discr.volume_empty() 
        fluxes_on_faces.bind_to_texref(fluxes_on_faces_texref)

        if set(["cuda_lift", "cuda_debugbuf"]) <= discr.debug:
            debugbuf = gpuarray.zeros((1024,), dtype=numpy.float32)
        else:
            debugbuf = FakeGPUArray()

        if discr.instrumented:
            discr.inner_flux_timer.add_timer_callable(
                    lift.prepared_timed_call(
                        self.grid,
                        flux.gpudata, 
                        self.gpu_liftmat(is_lift).device_memory,
                        debugbuf.gpudata))
        else:
            lift.prepared_call(
                    self.grid,
                    flux.gpudata, 
                    self.gpu_liftmat(is_lift).device_memory,
                    debugbuf.gpudata)

        if set(["cuda_lift", "cuda_debugbuf"]) <= discr.debug:
            copied_debugbuf = debugbuf.get()[:144*7].reshape((144,7))
            print "DEBUG"
            numpy.set_printoptions(linewidth=100)
            copied_debugbuf.shape = (144,7)
            numpy.set_printoptions(threshold=3000)

            print copied_debugbuf
            raw_input()

        return flux

    @memoize_method
    def get_kernel(self, is_lift, elgroup):
        from hedge.cuda.cgen import \
                Pointer, POD, Value, ArrayOf, Const, \
                Module, FunctionDeclaration, FunctionBody, Block, \
                Comment, Line, \
                CudaShared, CudaConstant, CudaGlobal, Static, \
                Define, \
                Constant, Initializer, If, For, Statement, Assign, \
                ArrayInitializer
                
        discr = self.discr
        d = discr.dimensions
        dims = range(d)
        given = discr.given

        liftmat_data = self.gpu_liftmat(is_lift)

        float_type = given.float_type

        f_decl = CudaGlobal(FunctionDeclaration(Value("void", "apply_lift_mat"), 
            [
                Pointer(POD(float_type, "flux")),
                Pointer(POD(numpy.uint8, "gmem_lift_mat")),
                Pointer(POD(float_type, "debugbuf")),
                ]
            ))

        cmod = Module([
                Value("texture<float, 1, cudaReadModeElementType>", 
                    "fluxes_on_faces_tex"),
                ])
        if is_lift:
            cmod.append(
                Value("texture<float, 1, cudaReadModeElementType>",
                    "inverse_jacobians_tex"),
                )

        par = self.plan.parallelism

        cmod.extend([
                Line(),
                Define("DIMENSIONS", discr.dimensions),
                Define("DOFS_PER_EL", given.dofs_per_el()),
                Define("FACES_PER_EL", given.faces_per_el()),
                Define("DOFS_PER_FACE", given.dofs_per_face()),
                Define("FACE_DOFS_PER_EL", "(DOFS_PER_FACE*FACES_PER_EL)"),
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
                Define("ALIGNED_FACE_DOFS_PER_MB", given.aligned_face_dofs_per_microblock()),
                Define("MB_EL_COUNT", given.microblock.elements),
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
                Define("GLOBAL_MB_NR_BASE", "(MACROBLOCK_NR*PAR_MB_COUNT*"
                    "INLINE_MB_COUNT*SEQ_MB_COUNT)"),
                Line(),
                Define("LIFTMAT_COLUMNS", liftmat_data.matrix_columns),
                Define("LIFTMAT_CHUNK_FLOATS", liftmat_data.block_floats),
                Define("LIFTMAT_CHUNK_BYTES", 
                    "(LIFTMAT_CHUNK_FLOATS*%d)" % given.float_size()),

                Line(),
                CudaShared(ArrayOf(POD(float_type, "smem_lift_mat"), 
                    "LIFTMAT_CHUNK_FLOATS")),
                CudaShared(
                    ArrayOf(
                        ArrayOf(
                            ArrayOf(
                                POD(float_type, "dof_buffer"), 
                                "PAR_MB_COUNT"),
                            "INLINE_MB_COUNT"),
                        "DOFS_PER_CHUNK"),
                    ),
                CudaShared(POD(numpy.uint16, "chunk_start_el")),
                CudaShared(POD(numpy.uint16, "chunk_stop_el")),
                CudaShared(POD(numpy.uint16, "chunk_el_count")),
                Line(),
                ArrayInitializer(
                        CudaConstant(
                            ArrayOf(
                                POD(numpy.uint16, "chunk_start_el_lookup"),
                            "CHUNKS_PER_MB")),
                        [(chk*self.plan.chunk_size)//given.dofs_per_el()
                            for chk in range(self.plan.chunks_per_microblock())]
                        ),
                ArrayInitializer(
                        CudaConstant(
                            ArrayOf(
                                POD(numpy.uint16, "chunk_stop_el_lookup"),
                            "CHUNKS_PER_MB")),
                        [min(given.microblock.elements, 
                            (chk*self.plan.chunk_size+self.plan.chunk_size-1)
                                //given.dofs_per_el()+1)
                            for chk in range(self.plan.chunks_per_microblock())]
                        ),
                ])

        S = Statement
        f_body = Block()
            
        f_body.extend_log_block("calculate this dof's element", [
            Initializer(POD(numpy.uint8, "dof_el"),
                "MB_DOF/DOFS_PER_EL"),
            Line(),])

        if self.plan.use_prefetch_branch:
            f_body.extend_log_block("calculate chunk responsibility data", [
                If("THREAD_NUM==0",
                    Block([
                        Assign("chunk_start_el", "chunk_start_el_lookup[MB_CHUNK]"),
                        Assign("chunk_stop_el", "chunk_stop_el_lookup[MB_CHUNK]"),
                        Assign("chunk_el_count", "chunk_stop_el-chunk_start_el")
                        ])
                    ),
                S("__syncthreads()")
                ])

        from hedge.cuda.tools import get_load_code
        f_body.extend(
            get_load_code(
                dest="smem_lift_mat",
                base=("gmem_lift_mat + MB_CHUNK*LIFTMAT_CHUNK_BYTES"),
                bytes="LIFTMAT_CHUNK_BYTES",
                descr="load lift mat chunk")
            +[S("__syncthreads()")]
            )

        # ---------------------------------------------------------------------
        def get_batched_fetch_mat_mul_code(el_fetch_count):
            result = []
            dofs = range(given.face_dofs_per_el())

            for load_chunk_start in range(0, given.face_dofs_per_el(),
                    self.plan.chunk_size):
                result.extend(
                        Assign(
                            "dof_buffer[PAR_MB_NR][%d][CHUNK_DOF]" % inl,
                            "tex1Dfetch(fluxes_on_faces_tex, "
                            "global_mb_facedof_base"
                            " + %d*ALIGNED_FACE_DOFS_PER_MB"
                            " + (chunk_start_el)*FACE_DOFS_PER_EL + %d + CHUNK_DOF)"
                            % (inl, load_chunk_start)
                            )
                        for inl in range(par.inline)
                        )
            
                result.extend([
                        S("__syncthreads()"),
                        Line(),
                        ])

                for dof in dofs[load_chunk_start:load_chunk_start+self.plan.chunk_size]:
                    for inl in range(par.inline):
                        result.append(
                                S("result%d += "
                                    "smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + %d]"
                                    "*"
                                    "dof_buffer[PAR_MB_NR][%d][%d]"
                                    % (inl, dof, inl, dof-load_chunk_start))
                                )
                result.append(Line())
            return result

        def get_direct_tex_mat_mul_code():
            from pytools import flatten
            return (
                    [POD(float_type, "fof%d" % inl) for inl in range(par.inline)]
                    + [POD(float_type, "lm"), Line()]
                    + list(flatten([
                    [
                        Assign("fof%d" % inl,
                            "tex1Dfetch(fluxes_on_faces_tex, "
                            "global_mb_facedof_base"
                            " + %(inl)d * ALIGNED_FACE_DOFS_PER_MB"
                            " + dof_el*FACE_DOFS_PER_EL+%(j)d)"
                            % {"j":j, "inl":inl, "row": "CHUNK_DOF"},)
                        for inl in range(par.inline)
                        ]+[
                        Assign("lm",
                            "smem_lift_mat["
                            "%(row)s*LIFTMAT_COLUMNS + %(j)s]"
                            % {"j":j, "row": "CHUNK_DOF"},
                            )
                        ]+[
                        S("result%(inl)d += fof%(inl)d*lm" % {"inl":inl})
                        for inl in range(par.inline)
                        ]
                    for j in range(
                        given.dofs_per_face()*given.faces_per_el())
                    ]))+[ Line(), ])

        def get_mat_mul_code(el_fetch_count):
            if el_fetch_count == 1:
                return get_batched_fetch_mat_mul_code(el_fetch_count)
            else:
                return get_direct_tex_mat_mul_code()

        def lift_outer_loop(fetch_count):
            if is_lift:
                inv_jac_multiplier = ("tex1Dfetch(inverse_jacobians_tex,"
                        "(global_mb_nr+%d)*MB_EL_COUNT+dof_el)")
            else:
                inv_jac_multiplier = "1"

            return For("unsigned short seq_mb_number = 0",
                "seq_mb_number < SEQ_MB_COUNT",
                "++seq_mb_number",
                Block([
                    Initializer(POD(numpy.uint32, "global_mb_nr"),
                        "GLOBAL_MB_NR_BASE + (seq_mb_number*PAR_MB_COUNT + PAR_MB_NR)*INLINE_MB_COUNT"),
                    Initializer(POD(numpy.uint32, "global_mb_dof_base"),
                        "global_mb_nr*ALIGNED_DOFS_PER_MB"),
                    Initializer(POD(numpy.uint32, "global_mb_facedof_base"),
                        "global_mb_nr*ALIGNED_FACE_DOFS_PER_MB"),
                    Line(),
                    ]+[
                    Initializer(POD(float_type, "result%d" % inl), 0)
                    for inl in range(par.inline)
                    ]+[ Line() ]
                    +get_mat_mul_code(fetch_count)
                    +[
                    If("MB_DOF < DOFS_PER_EL*MB_EL_COUNT",
                        Block([
                            Assign(
                                "flux[global_mb_dof_base"
                                " + %d*ALIGNED_DOFS_PER_MB"
                                " + MB_DOF]" % inl,
                                "result%d * %s" % (inl, (inv_jac_multiplier % inl))
                                )
                            for inl in range(par.inline)
                            ])
                        )
                    ])
                )

        if self.plan.use_prefetch_branch:
            from hedge.cuda.cgen import make_multiple_ifs
            f_body.append(make_multiple_ifs([
                    ("chunk_el_count == %d" % fetch_count,
                        lift_outer_loop(fetch_count))
                    for fetch_count in 
                    range(1, self.plan.max_elements_touched_by_chunk()+1)]
                    ))
        else:
            f_body.append(lift_outer_loop(0))

        # finish off ----------------------------------------------------------
        cmod.append(FunctionBody(f_decl, f_body))

        mod = cuda.SourceModule(cmod, 
                keep=True, 
                #options=["--maxrregcount=12"]
                )
        print "lift: lmem=%d smem=%d regs=%d" % (mod.lmem, mod.smem, mod.registers)

        fluxes_on_faces_texref = mod.get_texref("fluxes_on_faces_tex")
        texrefs = [fluxes_on_faces_texref]

        if is_lift:
            inverse_jacobians_texref = mod.get_texref("inverse_jacobians_tex")
            self.inverse_jacobians_tex(elgroup).bind_to_texref(
                    inverse_jacobians_texref)
            texrefs.append(inverse_jacobians_texref)

        func = mod.get_function("apply_lift_mat")
        func.prepare(
                "PPP", 
                block=(self.plan.chunk_size, self.plan.parallelism.parallel, 1),
                texrefs=texrefs)

        return func, fluxes_on_faces_texref

    @memoize_method
    def gpu_liftmat(self, is_lift):
        discr = self.discr
        given = discr.given

        columns = given.face_dofs_per_el()
        # avoid smem fetch bank conflicts by ensuring odd col count
        if columns % 2 == 0:
            columns += 1

        block_floats = given.devdata.align_dtype(
                columns*self.plan.chunk_size, given.float_size())

        if is_lift:
            mat = given.ldis.lifting_matrix()
        else:
            mat = given.ldis.multi_face_mass_matrix()

        vstacked_matrix = numpy.vstack(
                given.microblock.elements*(mat,)
                )

        if vstacked_matrix.shape[1] < columns:
            vstacked_matrix = numpy.hstack((
                vstacked_matrix,
                numpy.zeros((
                    vstacked_matrix.shape[0],
                    columns-vstacked_matrix.shape[1]
                    ))
                ))
                
        chunks = [
                buffer(numpy.asarray(
                    vstacked_matrix[
                        chunk_start:chunk_start+self.plan.chunk_size],
                    dtype=given.float_type,
                    order="C"))
                for chunk_start in range(
                    0, given.microblock.elements*given.dofs_per_el(), 
                    self.plan.chunk_size)
                ]
        
        from hedge.cuda.tools import pad_and_join

        from pytools import Record
        class GPULiftMatrices(Record): pass

        return GPULiftMatrices(
                device_memory=cuda.to_device(
                    pad_and_join(chunks, block_floats*given.float_size())),
                block_floats=block_floats,
                matrix_columns=columns,
                )

    # data blocks -------------------------------------------------------------
    @memoize_method
    def inverse_jacobians_tex(self, elgroup):
        ij = elgroup.inverse_jacobians[
                    self.discr.elgroup_microblock_indices(elgroup)]
        return gpuarray.to_gpu(
                ij.astype(self.discr.given.float_type))

