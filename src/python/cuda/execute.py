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
from pytools import memoize_method, memoize
import hedge.optemplate
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import pymbolic.mapper.stringifier




# structures ------------------------------------------------------------------
@memoize
def flux_header_struct():
    from hedge.cuda.cgen import Struct, POD

    return Struct("flux_header", [
        POD(numpy.uint16, "els_in_block"),
        POD(numpy.uint16, "same_facepairs_end"),
        POD(numpy.uint16, "diff_facepairs_end"),
        POD(numpy.uint16, "bdry_facepairs_end"),
        ])

@memoize
def face_pair_struct(float_type, dims):
    from hedge.cuda.cgen import Struct, POD, ArrayOf
    return Struct("face_pair", [
        POD(float_type, "h", ),
        POD(float_type, "order"),
        POD(float_type, "face_jacobian"),
        ArrayOf(POD(float_type, "normal"), dims),

        POD(numpy.uint32, "a_base"),
        POD(numpy.uint32, "b_base"),

        POD(numpy.uint16, "a_ilist_index"),
        POD(numpy.uint16, "b_ilist_index"), 
        POD(numpy.uint16, "b_write_ilist_index"), 
        POD(numpy.uint8, "a_flux_number"),
        POD(numpy.uint8, "b_flux_number_and_bdry_flag"), 
        POD(numpy.uint16, "a_dest"), 
        POD(numpy.uint16, "b_dest"), 
        ])



# flux to code mapper ---------------------------------------------------------
class FluxToCodeMapper(pymbolic.mapper.stringifier.StringifyMapper):
    def __init__(self, flip_normal):
        def float_mapper(x):
            if isinstance(x, float):
                return "%sf" % repr(x)
            else:
                return repr(x)

        pymbolic.mapper.stringifier.StringifyMapper.__init__(self, float_mapper)
        self.flip_normal = flip_normal

    def map_normal(self, expr, enclosing_prec):
        if self.flip_normal:
            sign = "-"
        else:
            sign = ""
        return "%sfpair->normal[%d]" % (sign, expr.axis)

    def map_penalty_term(self, expr, enclosing_prec):
        return ("pow(fpair->order*fpair->order/fpair->h, %r)" 
                % expr.power)

    def map_if_positive(self, expr, enclosing_prec):
        from pymbolic.mapper.stringifier import PREC_NONE
        return "(%s > 0 ? %s : %s)" % (
                self.rec(expr.criterion, PREC_NONE),
                self.rec(expr.then, PREC_NONE),
                self.rec(expr.else_, PREC_NONE),
                )




# exec mapper -----------------------------------------------------------------
class ExecutionMapper(hedge.optemplate.Evaluator,
        hedge.optemplate.BoundOpMapperMixin, 
        hedge.optemplate.LocalOpReducerMixin):

    def __init__(self, context, executor):
        hedge.optemplate.Evaluator.__init__(self, context)
        self.ex = executor

        self.diff_xyz_cache = {}

    def print_error_structure(self, computed, reference, diff):
        discr = self.ex.discr

        norm_ref = la.norm(reference)
        struc = ""

        numpy.set_printoptions(precision=2, linewidth=130, suppress=True)
        for block in discr.blocks:
            i_el = 0
            for mb in block.microblocks:
                for el in mb:
                    s = discr.find_el_range(el.id)
                    relerr = la.norm(diff[s])/norm_ref
                    if relerr > 1e-4:
                        struc += "*"
                        if True:
                            print "block %d, el %d, global el #%d, rel.l2err=%g" % (
                                    block.number, i_el, el.id, relerr)
                            print computed[s]
                            print reference[s]
                            print diff[s]
                            raw_input()
                    elif numpy.isnan(relerr):
                        struc += "N"
                        if False:
                            print "block %d, el %d, global el #%d, rel.l2err=%g" % (
                                    block.number, i_el, el.id, relerr)
                            print computed[s]
                            print reference[s]
                            print diff[s]
                            raw_input()
                    else:
                        if numpy.max(numpy.abs(reference[s])) == 0:
                            struc += "0"
                        else:
                            if False:
                                print "block %d, el %d, global el #%d, rel.l2err=%g" % (
                                        block.number, i_el, el.id, relerr)
                                print computed[s]
                                print reference[s]
                                print diff[s]
                                raw_input()
                            struc += "."
                    i_el += 1
                struc += " "
            struc += "\n"
        print
        print struc

    def map_diff_base(self, op, field_expr, out=None):
        try:
            xyz_diff = self.diff_xyz_cache[op.__class__, field_expr]
        except KeyError:
            pass
        else:
            print "HIT"
            return xyz_diff[op.xyz_axis]

        discr = self.ex.discr
        d = discr.dimensions

        eg, = discr.element_groups
        func, texrefs, field_texref = self.ex.get_diff_kernel(op.__class__, eg)

        fplan = discr.flux_plan
        lplan = fplan.diff_plan()

        field = self.rec(field_expr)
        assert field.dtype == discr.flux_plan.float_type

        field.bind_to_texref(field_texref)
        
        from hedge.cuda.tools import int_ceiling
        kwargs = {
                "block": (lplan.chunk_size, lplan.parallelism.p, 1),
                "grid": (fplan.mb_chunks, 
                    int_ceiling(
                        fplan.dofs_per_block()*len(discr.blocks)/
                        lplan.dofs_per_macroblock())
                    ),
                "time_kernel": discr.instrumented,
                "texrefs": texrefs,
                }

        #debugbuf = gpuarray.zeros((512,), dtype=numpy.float32)

        xyz_diff = [discr.volume_empty() for axis in range(d)]

        elgroup, = discr.element_groups
        args = xyz_diff+[
                self.ex.gpu_diffmats(op.__class__, eg).device_memory,
                #debugbuf,
                ]

        kernel_time = func(*args, **kwargs)
        if discr.instrumented:
            discr.diff_op_timer.add_time(kernel_time)
            discr.diff_op_counter.add(discr.dimensions)

        if False:
            copied_debugbuf = debugbuf.get()
            print "DEBUG"
            #print numpy.reshape(copied_debugbuf, (len(copied_debugbuf)//16, 16))
            print copied_debugbuf[:100].reshape((10,10))
            raw_input()
        
        if discr.debug:
            f = discr.volume_from_gpu(field)
            dx = discr.volume_from_gpu(xyz_diff[0])
            
            test_discr = discr.test_discr
            real_dx = test_discr.nabla[0].apply(f.astype(numpy.float64))
            
            diff = dx - real_dx

            rel_err_norm = la.norm(diff)/la.norm(real_dx)
            print "diff", rel_err_norm
            if rel_err_norm > 5e-5:
                self.print_error_structure(dx, real_dx, diff)
            assert rel_err_norm < 5e-5

        self.diff_xyz_cache[op.__class__, field_expr] = xyz_diff
        return xyz_diff[op.xyz_axis]

    def map_whole_domain_flux(self, op, field_expr, out=None):
        field = self.rec(field_expr)
        discr = self.ex.discr

        eg, = discr.element_groups
        fdata = self.ex.flux_with_temp_data(op, eg)
        fplan = discr.flux_plan
        lplan = fplan.flux_lifting_plan()

        gather, gather_texrefs, field_texref, bfield_texref = \
                self.ex.get_flux_gather_kernel(op)
        lift, lift_texrefs, fluxes_on_faces_texref = \
                self.ex.get_flux_local_kernel(op)

        flux = discr.volume_empty() 
        bfield = None
        for boundary in op.boundaries:
            if bfield is None:
                bfield = self.rec(boundary.bfield_expr)
            else:
                bfield = bfield + self.rec(boundary.bfield_expr)
            
        assert field.dtype == fplan.float_type
        assert bfield.dtype == fplan.float_type

        debugbuf = gpuarray.zeros((512,), dtype=numpy.float32)

        from hedge.cuda.tools import int_ceiling
        fluxes_on_faces = gpuarray.empty(
                (int_ceiling(
                    len(discr.blocks)
                    * fplan.aligned_face_dofs_per_microblock()
                    * fplan.microblocks_per_block(),
                    lplan.parallelism.total()
                    * fplan.aligned_face_dofs_per_microblock()
                    ),),
                dtype=fplan.float_type)

        # gather phase --------------------------------------------------------
        gather_args = [
                debugbuf, 
                fluxes_on_faces, 
                fdata.device_memory,
                self.ex.index_list_global_data().device_memory,
                ]

        gather_kwargs = {
                "texrefs": gather_texrefs, 
                "block": (discr.flux_plan.dofs_per_face(), 
                    fplan.parallel_faces, 1),
                "grid": (len(discr.blocks), 1),
                "time_kernel": discr.instrumented,
                }

        field.bind_to_texref(field_texref)
        bfield.bind_to_texref(bfield_texref)

        kernel_time = gather(*gather_args, **gather_kwargs)

        if discr.instrumented:
            discr.inner_flux_timer.add_time(kernel_time)
            discr.inner_flux_counter.add()

        if False:
            fof = fluxes_on_faces.get()
            print numpy.reshape(fof[:20*15], (20,15))
            raw_input()

        if False:
            copied_debugbuf = debugbuf.get()
            print "DEBUG"
            numpy.set_printoptions(linewidth=100)
            print numpy.reshape(copied_debugbuf, (32, 16))
            #print copied_debugbuf
            raw_input()

        # lift phase ----------------------------------------------------------
        lift_args = [
                flux, 
                self.ex.gpu_liftmat().device_memory,
                debugbuf,
                ]

        lift_kwargs = {
                "texrefs": lift_texrefs, 
                "block": (lplan.chunk_size, lplan.parallelism.p, 1),
                "grid": (fplan.mb_chunks, 
                    int_ceiling(
                        fplan.dofs_per_block()*len(discr.blocks)/
                        lplan.dofs_per_macroblock())
                    ),
                "time_kernel": discr.instrumented,
                }

        fluxes_on_faces.bind_to_texref(fluxes_on_faces_texref)

        kernel_time = lift(*lift_args, **lift_kwargs)

        if discr.instrumented:
            discr.inner_flux_timer.add_time(kernel_time)
            discr.inner_flux_counter.add()

        if False:
            copied_debugbuf = debugbuf.get()
            print "DEBUG"
            numpy.set_printoptions(linewidth=100)
            print copied_debugbuf
            print eg.inverse_jacobians[
                    self.ex.elgroup_microblock_indices(eg)][:500]
            raw_input()

        # verification --------------------------------------------------------
        if discr.debug:
            cot = discr.test_discr.compile(op.flux_optemplate)
            ctx = {field_expr.name: 
                    discr.volume_from_gpu(field).astype(numpy.float64)
                    }
            for boundary in op.boundaries:
                ctx[boundary.bfield_expr.name] = \
                        discr.test_discr.boundary_zeros(boundary.tag)
            true_flux = cot(**ctx)
            
            copied_flux = discr.volume_from_gpu(flux)

            diff = copied_flux-true_flux

            norm_true = la.norm(true_flux)

            if False:
                self.print_error_structure(copied_flux, true_flux, diff)
                raw_input()

            print "flux", la.norm(diff)/norm_true
            assert la.norm(diff)/norm_true < 1e-6

        if False:
            copied_bfield = bfield.get()
            face_len = discr.flux_plan.ldis.face_node_count()
            aligned_face_len = discr.devdata.align_dtype(face_len, 4)
            for elface in discr.mesh.tag_to_boundary.get('inflow', []):
                face_stor = discr.face_storage_map[elface]
                bdry_stor = face_stor.opposite
                gpu_base = bdry_stor.gpu_bdry_index_in_floats
                print gpu_base, copied_bfield[gpu_base:gpu_base+aligned_face_len]
                raw_input()

        return flux




class OpTemplateWithEnvironment(object):
    def __init__(self, discr, optemplate):
        self.discr = discr

        from hedge.optemplate import OperatorBinder, InverseMassContractor, \
                FluxDecomposer
        from pymbolic.mapper.constant_folder import CommutativeConstantFoldingMapper
        from hedge.cuda.optemplate import BoundaryCombiner

        self.optemplate = (
                BoundaryCombiner(discr)(
                    InverseMassContractor()(
                        CommutativeConstantFoldingMapper()(
                            FluxDecomposer()(
                                OperatorBinder()(
                                    optemplate))))))

    def __call__(self, **vars):
        return ExecutionMapper(vars, self)(self.optemplate)




    # helpers -----------------------------------------------------------------
    def get_load_code(self, dest, base, bytes, word_type=numpy.uint32,
            descr=None):
        from hedge.cuda.cgen import \
                Pointer, POD, Value, ArrayOf, Const, \
                Comment, Block, Line, \
                Constant, Initializer, If, For, Statement, Assign

        from hedge.cuda.cgen import dtype_to_ctype
        copy_dtype = numpy.dtype(word_type)
        copy_dtype_str = dtype_to_ctype(copy_dtype)

        code = []
        if descr is not None:
            code.append(Comment(descr))

        code.extend([
            Block([
                Constant(Pointer(POD(copy_dtype, "load_base")), 
                    ("(%s *) (%s)" % (copy_dtype_str, base))),
                For("unsigned word_nr = THREAD_NUM", 
                    "word_nr*sizeof(int) < (%s)" % bytes, 
                    "word_nr += COALESCING_THREAD_COUNT",
                    Statement("((%s *) (%s))[word_nr] = load_base[word_nr]"
                        % (copy_dtype_str, dest))
                    ),
                ]),
            Line(),
            ])

        return code




    # diff kernel -------------------------------------------------------------
    @memoize_method
    def get_diff_kernel(self, diff_op_cls, elgroup):
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
        fplan = discr.flux_plan
        lplan = fplan.diff_plan()

        diffmat_data = self.gpu_diffmats(diff_op_cls, elgroup)
        elgroup, = discr.element_groups

        float_type = fplan.float_type

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
                Define("DOFS_PER_EL", fplan.dofs_per_el()),
                Line(),
                Define("CHUNK_DOF", "threadIdx.x"),
                Define("PAR_MB_NR", "threadIdx.y"),
                Line(),
                Define("MB_CHUNK", "blockIdx.x"),
                Define("MACROBLOCK_NR", "blockIdx.y"),
                Line(),
                Define("CHUNK_DOF_COUNT", lplan.chunk_size),
                Define("MB_CHUNK_COUNT", fplan.mb_chunks),
                Define("MB_DOF_COUNT", "(MB_CHUNK_COUNT*CHUNK_DOF_COUNT)"),
                Define("MB_EL_COUNT", fplan.mb_elements),
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
                     % fplan.float_size),
                Define("DIFFMAT_COLUMNS", diffmat_data.matrix_columns),
                Line(),
                CudaShared(ArrayOf(POD(float_type, "smem_diff_rst_mat"), 
                    "DIMENSIONS*DOFS_PER_EL*CHUNK_DOF_COUNT")),
                Line(),
                ])

        S = Statement
        f_body = Block()
            
        f_body.extend_log_block("calculate responsibility data", [
            Initializer(POD(numpy.uint8, "mb_el"),
                "MB_DOF/DOFS_PER_EL"),
            ])

        f_body.extend(
            self.get_load_code(
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
                            #"int_dofs[PAR_MB_NR][chunk_el*DOFS_PER_EL+%d]" % (j)
                            "tex1Dfetch(field_tex, "
                            "global_mb_dof_base"
                            "+mb_el*DOFS_PER_EL+%d)" % j
                            ),
                        Line(),
                        ]
                        +[
                        S("drst%d += %s * field_value" 
                            % (axis, get_mat_entry(matrix_row, j, axis)))
                        for axis in dims
                        ]+[Line()]
                        for j in range(fplan.dofs_per_el())
                        ))
                    )

            for glob_axis in dims:
                code.append(Block([
                    Initializer(Value("float%d" % rst_channels, "rst_to_xyz"),
                        "tex2D(rst_to_xyz_tex, %d, global_mb_nr*MB_EL_COUNT+mb_el)" % glob_axis
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
                    ]))
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
        print "lmem=%d smem=%d regs=%d" % (mod.lmem, mod.smem, mod.registers)

        rst_to_xyz_texref = mod.get_texref("rst_to_xyz_tex")
        cuda.bind_array_to_texref(
                self.localop_rst_to_xyz(diff_op_cls, elgroup), 
                rst_to_xyz_texref)

        field_texref = mod.get_texref("field_tex")
        texrefs = [field_texref, rst_to_xyz_texref]

        return mod.get_function("apply_diff_mat"), texrefs, field_texref




    # diff kernel -------------------------------------------------------------
    @memoize_method
    def get_flux_local_kernel(self, elgroup):
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
        fplan = discr.flux_plan
        lplan = fplan.flux_lifting_plan()

        liftmat_data = self.gpu_liftmat()
        elgroup, = discr.element_groups

        float_type = fplan.float_type

        f_decl = CudaGlobal(FunctionDeclaration(Value("void", "apply_lift_mat"), 
            [
                Pointer(POD(float_type, "flux")),
                Pointer(POD(numpy.uint8, "gmem_lift_mat")),
                Pointer(POD(float_type, "debugbuf")),
                ]
            ))

        rst_channels = discr.devdata.make_valid_tex_channel_count(d)
        cmod = Module([
                Value("texture<float, 1, cudaReadModeElementType>",
                    "inverse_jacobians_tex"),
                Value("texture<float, 1, cudaReadModeElementType>", 
                    "fluxes_on_faces_tex"),
                Line(),
                Define("DIMENSIONS", discr.dimensions),
                Define("DOFS_PER_EL", fplan.dofs_per_el()),
                Define("FACES_PER_EL", fplan.faces_per_el()),
                Define("DOFS_PER_FACE", fplan.dofs_per_face()),
                Define("FACE_DOFS_PER_EL", "(DOFS_PER_FACE*FACES_PER_EL)"),
                Line(),
                Define("CHUNK_DOF", "threadIdx.x"),
                Define("PAR_MB_NR", "threadIdx.y"),
                Line(),
                Define("MB_CHUNK", "blockIdx.x"),
                Define("MACROBLOCK_NR", "blockIdx.y"),
                Line(),
                Define("CHUNK_DOF_COUNT", lplan.chunk_size),
                Define("MB_CHUNK_COUNT", fplan.mb_chunks),
                Define("MB_DOF_COUNT", "(MB_CHUNK_COUNT*CHUNK_DOF_COUNT)"),
                Define("MB_FACEDOF_COUNT", fplan.aligned_face_dofs_per_microblock()),
                Define("MB_EL_COUNT", fplan.mb_elements),
                Define("PAR_MB_COUNT", lplan.parallelism.p),
                Define("SEQ_MB_COUNT", lplan.parallelism.s),
                Define("ELEMENTS_TOUCHED_BY_CHUNK", lplan.max_elements_touched_by_chunk()),
                Line(),
                Define("THREAD_NUM", "(CHUNK_DOF+PAR_MB_NR*CHUNK_DOF_COUNT)"),
                Define("COALESCING_THREAD_COUNT", "(PAR_MB_COUNT*CHUNK_DOF_COUNT)"),
                Line(),
                Define("MB_DOF_BASE", "(MB_CHUNK*CHUNK_DOF_COUNT)"),
                Define("MB_DOF", "(MB_DOF_BASE+CHUNK_DOF)"),
                Define("GLOBAL_MB_NR_BASE", "(MACROBLOCK_NR*PAR_MB_COUNT*SEQ_MB_COUNT)"),
                Line(),
                Define("LIFTMAT_COLUMNS", liftmat_data.matrix_columns),
                Define("LIFTMAT_CHUNK_FLOATS", liftmat_data.block_floats),
                Define("LIFTMAT_CHUNK_BYTES", 
                    "(LIFTMAT_CHUNK_FLOATS*%d)" % fplan.float_size),
                Define("DOF_LOCAL_EL", "(dof_el-chunk_start_el)"),

                Line(),
                CudaShared(ArrayOf(POD(float_type, "smem_lift_mat"), 
                    "LIFTMAT_CHUNK_FLOATS")),
                CudaShared(
                    ArrayOf(
                        ArrayOf(
                            ArrayOf(
                                POD(float_type, "dof_buffer"), 
                                "PAR_MB_COUNT"),
                            "ELEMENTS_TOUCHED_BY_CHUNK"),
                        "CHUNK_DOF_COUNT"),
                    ),
                CudaShared(POD(numpy.uint16, "chunk_start_el")),
                CudaShared(POD(numpy.uint16, "chunk_stop_el")),
                CudaShared(POD(numpy.uint16, "chunk_el_count")),
                Line(),
                ])

        S = Statement
        f_body = Block()
            
        f_body.extend_log_block("calculate responsibility data", [
            Initializer(POD(numpy.uint8, "dof_el"),
                "MB_DOF/DOFS_PER_EL"),
            Line(),
            Assign("chunk_start_el", "MB_DOF_BASE/DOFS_PER_EL"),
            Assign("chunk_stop_el",
                "min(MB_EL_COUNT, (MB_DOF_BASE+CHUNK_DOF_COUNT-1)/DOFS_PER_EL+1)"),
            Assign("chunk_el_count", "chunk_stop_el-chunk_start_el")
            ])

        f_body.extend(
            self.get_load_code(
                dest="smem_lift_mat",
                base=("gmem_lift_mat + MB_CHUNK*LIFTMAT_CHUNK_BYTES"),
                bytes="LIFTMAT_CHUNK_BYTES",
                descr="load lift mat chunk")
            )

        # ---------------------------------------------------------------------
        def get_mat_mul_code(fetch_count):
            result = []
            dofs = range(fplan.face_dofs_per_el())
            for load_chunk_start in range(0, fplan.face_dofs_per_el(),
                    lplan.chunk_size):
                result.extend(
                        Assign(
                            "dof_buffer[PAR_MB_NR][%d][CHUNK_DOF]"
                            % fetch_el,
                            "tex1Dfetch(fluxes_on_faces_tex, "
                            "global_mb_facedof_base"
                            "+(chunk_start_el+%d)*FACE_DOFS_PER_EL+%d+CHUNK_DOF)"
                            % (fetch_el, load_chunk_start)
                            )
                        for fetch_el in range(fetch_count))
            
                result.extend([
                        S("__syncthreads()"),
                        Line(),
                        ])

                for dof in dofs[load_chunk_start:load_chunk_start+lplan.chunk_size]:
                    result.append(
                            S("result += "
                                "smem_lift_mat[CHUNK_DOF*LIFTMAT_COLUMNS + %d]"
                                "*"
                                "dof_buffer[PAR_MB_NR][DOF_LOCAL_EL][%d]"
                                % (dof, dof-load_chunk_start))
                            )
                result.append(Line())
            return result

        from hedge.cuda.cgen import make_multiple_ifs
        f_body.append(make_multiple_ifs([
                ("chunk_el_count == %d" % fetch_count,
                    For("unsigned short seq_mb_number = 0",
                        "seq_mb_number < SEQ_MB_COUNT",
                        "++seq_mb_number",
                        Block([
                            Initializer(POD(numpy.uint32, "global_mb_nr"),
                                "GLOBAL_MB_NR_BASE + seq_mb_number*PAR_MB_COUNT + PAR_MB_NR"),
                            Initializer(POD(numpy.uint32, "global_mb_dof_base"),
                                "global_mb_nr*MB_DOF_COUNT"),
                            Initializer(POD(numpy.uint32, "global_mb_facedof_base"),
                                "global_mb_nr*MB_FACEDOF_COUNT"),
                            Line(),
                            Initializer(POD(float_type, "result"), 0),
                            Line(),
                            ]
                            +get_mat_mul_code(fetch_count)+[
                            Assign(
                                "flux[global_mb_dof_base+MB_DOF]",
                                "result*tex1Dfetch(inverse_jacobians_tex,"
                                "global_mb_nr*MB_EL_COUNT+dof_el)"
                                )
                            ])
                        )
                    )
                for fetch_count in 
                range(1, lplan.max_elements_touched_by_chunk()+1)]
                ))

        # finish off ----------------------------------------------------------
        cmod.append(FunctionBody(f_decl, f_body))

        mod = cuda.SourceModule(cmod, 
                keep=True, 
                #options=["--maxrregcount=10"]
                )
        print "lmem=%d smem=%d regs=%d" % (mod.lmem, mod.smem, mod.registers)

        inverse_jacobians_texref = mod.get_texref("inverse_jacobians_tex")
        self.inverse_jacobians_tex(elgroup).bind_to_texref(
                inverse_jacobians_texref)

        fluxes_on_faces_texref = mod.get_texref("fluxes_on_faces_tex")
        texrefs = [fluxes_on_faces_texref, inverse_jacobians_texref]

        return (mod.get_function("apply_lift_mat"), 
                texrefs, 
                fluxes_on_faces_texref)




    # flux kernel -------------------------------------------------------------
    @memoize_method
    def get_flux_gather_kernel(self, wdflux):
        from hedge.cuda.cgen import \
                Pointer, POD, Value, ArrayOf, Const, \
                Module, FunctionDeclaration, FunctionBody, Block, \
                Comment, Line, \
                CudaShared, CudaGlobal, Static, \
                Define, Pragma, \
                Constant, Initializer, If, For, Statement, Assign, While
                
        discr = self.discr
        fplan = discr.flux_plan
        d = discr.dimensions
        dims = range(d)

        elgroup, = discr.element_groups
        flux_with_temp_data = self.flux_with_temp_data(wdflux, elgroup)

        float_type = fplan.float_type

        f_decl = CudaGlobal(FunctionDeclaration(Value("void", "apply_flux"), 
            [
                Pointer(POD(float_type, "debugbuf")),
                Pointer(POD(float_type, "gmem_fluxes_on_faces")),
                #Pointer(POD(float_type, "flux")),
                Pointer(POD(numpy.uint8, "gmem_data")),
                Pointer(POD(numpy.uint8, "gmem_index_lists")),
                ]
            ))

        cmod = Module([
                Value("texture<float, 1, cudaReadModeElementType>", 
                    "field_tex"),
                Value("texture<float, 1, cudaReadModeElementType>", 
                    "bfield_tex"),
                flux_header_struct(),
                face_pair_struct(float_type, discr.dimensions),
                Line(),
                Define("DIMENSIONS", discr.dimensions),
                Define("DOFS_PER_EL", fplan.dofs_per_el()),
                Define("DOFS_PER_FACE", fplan.dofs_per_face()),
                Line(),
                Define("CONCURRENT_FACES", fplan.parallel_faces),
                Define("BLOCK_MB_COUNT", fplan.mbs_per_block),
                Line(),
                Define("FACEDOF_NR", "threadIdx.x"),
                Define("BLOCK_FACE", "threadIdx.y"),
                Line(),
                Define("THREAD_NUM", "(FACEDOF_NR + BLOCK_FACE*DOFS_PER_FACE)"),
                Define("THREAD_COUNT", "(DOFS_PER_FACE*CONCURRENT_FACES)"),
                Define("COALESCING_THREAD_COUNT", "(THREAD_COUNT & ~0xf)"),
                Line(),
                Define("DATA_BLOCK_SIZE", flux_with_temp_data.block_bytes),
                Define("ALIGNED_FACE_DOFS_PER_MB", fplan.aligned_face_dofs_per_microblock()),
                Define("ALIGNED_FACE_DOFS_PER_BLOCK", 
                    "(ALIGNED_FACE_DOFS_PER_MB*BLOCK_MB_COUNT)"),
                Line(),
                Line(),
                ] + self.index_list_global_data().code + [
                Line(),
                flux_with_temp_data.struct,
                Line(),
                CudaShared(
                    ArrayOf(Value("index_list_entry_t", "smem_index_lists"),
                        "INDEX_LISTS_LENGTH")),
                CudaShared(Value("flux_data", "data")),
                CudaShared(ArrayOf(POD(float_type, "smem_fluxes_on_faces"),
                    "ALIGNED_FACE_DOFS_PER_MB*BLOCK_MB_COUNT"
                    )),
                Line(),
                ])

        S = Statement
        f_body = Block()
            
        f_body.extend(self.get_load_code(
            dest="smem_index_lists",
            base="gmem_index_lists",
            bytes="sizeof(index_list_entry_t)*INDEX_LISTS_LENGTH",
            descr="load index list data")
            )

        f_body.extend(self.get_load_code(
            dest="&data",
            base="gmem_data + blockIdx.x*DATA_BLOCK_SIZE",
            bytes="sizeof(flux_data)",
            descr="load face_pair data")
            +[ S("__syncthreads()"), Line() ])

        def flux_coeff_getter(flux_number_expr, prefix, flip_normal, internal_only):
            from hedge.cuda.cgen import make_multiple_ifs
            from pymbolic.mapper.stringifier import PREC_NONE
            if internal_only:
                int_coeff, ext_coeff = wdflux.fluxes[wdflux.interior_flux_number]
                return [
                        Initializer(
                            POD(float_type, "%sint_coeff" % prefix),
                            FluxToCodeMapper(flip_normal)(int_coeff, PREC_NONE),
                            ),
                        Initializer(
                            POD(float_type, "%sext_coeff" % prefix),
                            FluxToCodeMapper(flip_normal)(ext_coeff, PREC_NONE),
                            )
                        ]
            else:
                return [
                    POD(float_type, "%sint_coeff" % prefix),
                    POD(float_type, "%sext_coeff" % prefix),
                    make_multiple_ifs(
                        [
                        ("(%s) == %d" % (flux_number_expr, flux_nr),
                            Block([
                                Assign("%sint_coeff" % prefix, 
                                    FluxToCodeMapper(flip_normal)(int_coeff, PREC_NONE),
                                    ),
                                Assign("%sext_coeff" % prefix, 
                                    FluxToCodeMapper(flip_normal)(ext_coeff, PREC_NONE),
                                    ),
                                ])
                            )
                        for flux_nr, (int_coeff, ext_coeff)
                        in enumerate(wdflux.fluxes)
                        ],
                        base= Block([
                            Assign("%sint_coeff" % prefix, 0),
                            Assign("%sext_coeff" % prefix, 0),
                            ])
                        ),
                    ]

        def get_flux_code(is_bdry, is_twosided):
            flux_code = Block([])

            flux_code.extend([
                Initializer(Pointer(
                    Value("face_pair", "fpair")),
                    "data.facepairs+fpair_nr"),
                Initializer(Pointer(Value(
                    "index_list_entry_t", "a_ilist")),
                    "smem_index_lists + fpair->a_ilist_index"
                    ),
                Initializer(Pointer(Value(
                    "index_list_entry_t", "b_ilist")),
                    "smem_index_lists + fpair->b_ilist_index"
                    ),
                Initializer(
                    POD(float_type, "a_value"),
                    "tex1Dfetch(field_tex, fpair->a_base + a_ilist[FACEDOF_NR])"
                    ),
                ])

            if is_bdry:
                flux_code.extend([
                    Initializer(
                        POD(float_type, "b_value"),
                        "tex1Dfetch(bfield_tex, fpair->b_base + b_ilist[FACEDOF_NR])"
                        ),
                    ])
            else:
                flux_code.extend([
                    Initializer(
                        POD(float_type, "b_value"),
                        "tex1Dfetch(field_tex, fpair->b_base + b_ilist[FACEDOF_NR])"
                        ),
                    ])

            flux_code.extend(
                    flux_coeff_getter("fpair->a_flux_number", "a_", 
                        flip_normal=False, internal_only=not is_bdry))

            if is_twosided:
                flux_code.extend(
                    flux_coeff_getter("fpair->b_flux_number_and_bdry_flag >> 1", 
                        "b_", flip_normal=True, internal_only=not is_bdry)
                    +[
                    Initializer(Pointer(Value(
                        "index_list_entry_t", "b_write_ilist")),
                        "smem_index_lists + fpair->b_write_ilist_index"
                        ),
                    ])

            flux_code.extend([
                Assign(
                    "smem_fluxes_on_faces[fpair->a_dest+FACEDOF_NR]",
                    "fpair->face_jacobian*("
                    "a_int_coeff*a_value+a_ext_coeff*b_value"
                    ")"),
                ])

            if is_twosided:
                flux_code.extend([
                    Assign(
                        "smem_fluxes_on_faces[fpair->b_dest+b_write_ilist[FACEDOF_NR]]",
                        "fpair->face_jacobian*("
                        "b_int_coeff*b_value+b_ext_coeff*a_value"
                        ")"
                        ),
                    ])

            flux_code.append(S("fpair_nr += CONCURRENT_FACES"))

            return flux_code

        f_body.extend_log_block("compute the fluxes", [
            Initializer(POD(numpy.uint16, "fpair_nr"), "BLOCK_FACE"),
            Comment("fluxes for dual-sided (intra-block) interior face pairs"),
            While("fpair_nr < data.header.same_facepairs_end",
                get_flux_code(is_bdry=False, is_twosided=True)
                ),
            Line(),
            Comment("work around nvcc assertion failure"),
            S("fpair_nr+=1"),
            S("fpair_nr-=1"),
            Line(),
            Comment("fluxes for single-sided (inter-block) interior face pairs"),
            While("fpair_nr < data.header.diff_facepairs_end",
                get_flux_code(is_bdry=False, is_twosided=False)
                ),
            Line(),
            Comment("fluxes for single-sided boundary face pairs"),
            While("fpair_nr < data.header.bdry_facepairs_end",
                get_flux_code(is_bdry=True, is_twosided=False)
                ),
            ])

        f_body.extend_log_block("store the fluxes", [
            S("__syncthreads()"),
            Line(),
            For("unsigned word_nr = THREAD_NUM", 
                "word_nr < ALIGNED_FACE_DOFS_PER_MB*BLOCK_MB_COUNT", 
                "word_nr += COALESCING_THREAD_COUNT",
                Assign(
                    "gmem_fluxes_on_faces[blockIdx.x*ALIGNED_FACE_DOFS_PER_BLOCK+word_nr]",
                    "smem_fluxes_on_faces[word_nr]")
                ),
            ])

        # finish off ----------------------------------------------------------
        cmod.append(FunctionBody(f_decl, f_body))

        mod = cuda.SourceModule(cmod, 
                keep=True, 
                options=["--maxrregcount=12"]
                )
        print "lmem=%d smem=%d regs=%d" % (mod.lmem, mod.smem, mod.registers)

        field_texref = mod.get_texref("field_tex")
        bfield_texref = mod.get_texref("bfield_tex")
        texrefs = [field_texref, bfield_texref]

        return mod.get_function("apply_flux"), texrefs, field_texref, bfield_texref

    # gpu data blocks ---------------------------------------------------------
    @memoize_method
    def gpu_diffmats(self, diff_op_cls, elgroup):
        discr = self.discr
        fplan = discr.flux_plan
        lplan = fplan.diff_plan()

        columns = fplan.dofs_per_el()*discr.dimensions
        additional_columns = 0
        # avoid smem fetch bank conflicts by ensuring odd col count
        if columns % 2 == 0:
            columns += 1
            additional_columns += 1

        block_floats = self.discr.devdata.align_dtype(
                columns*lplan.chunk_size, fplan.float_size)

        vstacked_matrices = [
                numpy.vstack(fplan.mb_elements*(m,))
                for m in diff_op_cls.matrices(elgroup)
                ]

        chunks = []

        from pytools import single_valued
        for chunk_start in range(0, fplan.mb_elements*fplan.dofs_per_el(), lplan.chunk_size):
            matrices = [
                m[chunk_start:chunk_start+lplan.chunk_size] 
                for m in vstacked_matrices]

            matrices.append(
                numpy.zeros((single_valued(m.shape[0] for m in matrices), 
                    additional_columns))
                )

            diffmats = numpy.asarray(
                    numpy.hstack(matrices),
                    dtype=self.discr.flux_plan.float_type,
                    order="C")
            chunks.append(buffer(diffmats))
        
        from pytools import Record
        from hedge.cuda.tools import pad_and_join
        return Record(
                device_memory=cuda.to_device(
                    pad_and_join(chunks, block_floats*fplan.float_size)),
                block_floats=block_floats,
                matrix_columns=columns)

    @memoize_method
    def gpu_liftmat(self):
        discr = self.discr
        fplan = discr.flux_plan
        lplan = fplan.flux_lifting_plan()

        columns = fplan.face_dofs_per_el()
        # avoid smem fetch bank conflicts by ensuring odd col count
        if columns % 2 == 0:
            columns += 1

        block_floats = self.discr.devdata.align_dtype(
                columns*lplan.chunk_size, fplan.float_size)

        vstacked_matrix = numpy.vstack(
                fplan.mb_elements*(fplan.ldis.lifting_matrix(),)
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
                        chunk_start:chunk_start+lplan.chunk_size],
                    dtype=self.discr.flux_plan.float_type,
                    order="C"))
                for chunk_start in range(
                    0, fplan.mb_elements*fplan.dofs_per_el(), 
                    lplan.chunk_size)
                ]
        
        from pytools import Record
        from hedge.cuda.tools import pad_and_join
        return Record(
                device_memory=cuda.to_device(
                    pad_and_join(chunks, block_floats*fplan.float_size)),
                block_floats=block_floats,
                matrix_columns=columns,
                )

        
    @memoize_method
    def elgroup_microblock_indices(self, elgroup):
        def get_el_index_in_el_group(el):
            mygroup, idx = discr.group_map[el.id]
            assert mygroup is elgroup
            return idx

        discr = self.discr
        fplan = discr.flux_plan

        el_count = len(discr.blocks) * fplan.elements_per_block()
        elgroup_indices = numpy.zeros((el_count,), dtype=numpy.intp)

        for block in discr.blocks:
            block_elgroup_indices = [ get_el_index_in_el_group(el) 
                    for mb in block.microblocks 
                    for el in mb]
            offset = block.number * fplan.elements_per_block()
            elgroup_indices[offset:offset+len(block_elgroup_indices)] = \
                    block_elgroup_indices

        return elgroup_indices

    @memoize_method
    def localop_rst_to_xyz(self, diff_op, elgroup):
        discr = self.discr
        d = discr.dimensions

        fplan = discr.flux_plan
        coeffs = diff_op.coefficients(elgroup)

        elgroup_indices = self.elgroup_microblock_indices(elgroup)
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

    @memoize_method
    def inverse_jacobians_tex(self, elgroup):
        ij = elgroup.inverse_jacobians[
                    self.elgroup_microblock_indices(elgroup)]
        return gpuarray.to_gpu(
                ij.astype(self.discr.flux_plan.float_type))

    @memoize_method
    def flux_inverse_jacobians(self, elgroup):
        discr = self.discr
        d = discr.dimensions

        fplan = discr.flux_plan

        floats_per_block = fplan.elements_per_block()
        bytes_per_block = floats_per_block*fplan.float_size

        inv_jacs = elgroup.inverse_jacobians

        blocks = []
        
        def get_el_index_in_el_group(el):
            mygroup, idx = discr.group_map[el.id]
            assert mygroup is elgroup
            return idx

        from hedge.cuda.tools import pad
        for block in discr.blocks:
            block_elgroup_indices = numpy.fromiter(
                    (get_el_index_in_el_group(el) 
                        for mb in block.microblocks
                        for el in mb
                        ),
                    dtype=numpy.intp)

            block_inv_jacs = (inv_jacs[block_elgroup_indices].copy().astype(fplan.float_type))
            blocks.append(pad(str(buffer(block_inv_jacs)), bytes_per_block))
                
        from hedge.cuda.cgen import POD, ArrayOf
        return blocks, ArrayOf(
                POD(fplan.float_type, "inverse_jacobians"),
                floats_per_block)

    @memoize_method
    def flux_with_temp_data(self, wdflux, elgroup):
        discr = self.discr
        fplan = discr.flux_plan
        headers = []
        fp_blocks = []

        INVALID_DEST = (1<<16)-1

        from hedge.cuda.discretization import GPUBoundaryFaceStorage

        fp_struct = face_pair_struct(discr.flux_plan.float_type, discr.dimensions)

        def find_elface_dest(el_face):
            elface_dofs = face_dofs*ldis.face_count()
            num_in_block = discr.find_number_in_block(el_face[0])
            mb_index, index_in_mb = divmod(num_in_block,  fplan.mb_elements)
            return (mb_index * fplan.aligned_face_dofs_per_microblock()
                    + index_in_mb * elface_dofs
                    + el_face[1]*face_dofs)

        outf = open("el_faces.txt", "w")
        for block in discr.blocks:
            ldis = block.local_discretization
            el_dofs = ldis.node_count()
            face_dofs = ldis.face_node_count()

            faces_todo = set((el,face_nbr)
                    for mb in block.microblocks
                    for el in mb
                    for face_nbr in range(ldis.face_count()))
            same_fp_structs = []
            diff_fp_structs = []
            bdry_fp_structs = []

            while faces_todo:
                elface = faces_todo.pop()

                a_face = discr.face_storage_map[elface]
                b_face = a_face.opposite

                print>>outf, "block %d el %d (global: %d) face %d" % (
                        block.number, discr.find_number_in_block(a_face.el_face[0]),
                        elface[0].id, elface[1]),
                        
                if isinstance(b_face, GPUBoundaryFaceStorage):
                    # boundary face
                    b_base = b_face.gpu_bdry_index_in_floats
                    a_flux_number = wdflux.boundary_elface_to_flux_number(
                            a_face.el_face)
                    b_flux_number = len(wdflux.fluxes) # invalid
                    b_load_from_bdry = 1
                    b_write_index_list = 0 # doesn't matter
                    b_dest = INVALID_DEST
                    print>>outf, "bdy%d" % a_flux_number

                    fp_structs = bdry_fp_structs
                else:
                    # interior face
                    b_base = discr.find_el_gpu_index(b_face.el_face[0])

                    a_flux_number = wdflux.interior_flux_number
                    b_flux_number = wdflux.interior_flux_number
                    b_load_from_bdry = 0

                    if b_face.native_block == a_face.native_block:
                        # same block
                        faces_todo.remove(b_face.el_face)
                        b_write_index_list = a_face.opp_write_index_list_id
                        b_dest = find_elface_dest(b_face.el_face)

                        fp_structs = same_fp_structs

                        print>>outf, "same el %d (global: %d) face %d" % (
                                discr.find_number_in_block(b_face.el_face[0]), 
                                b_face.el_face[0].id, b_face.el_face[1])
                    else:
                        # different block
                        b_write_index_list = 0 # doesn't matter
                        b_dest = INVALID_DEST

                        fp_structs = diff_fp_structs

                        print>>outf, "diff"

                fp_structs.append(
                        fp_struct.make(
                            h=a_face.face_pair_side.h,
                            order=a_face.face_pair_side.order,
                            face_jacobian=a_face.face_pair_side.face_jacobian,
                            normal=a_face.face_pair_side.normal,

                            a_base=discr.find_el_gpu_index(a_face.el_face[0]),
                            b_base=b_base,

                            a_ilist_index= \
                                    a_face.global_int_flux_index_list_id*face_dofs,
                            b_ilist_index= \
                                    a_face.global_ext_flux_index_list_id*face_dofs,

                            a_flux_number=a_flux_number,
                            b_flux_number_and_bdry_flag=\
                                    (b_flux_number << 1) + b_load_from_bdry,
                            b_write_ilist_index= \
                                    b_write_index_list*face_dofs,

                            a_dest=find_elface_dest(a_face.el_face),
                            b_dest=b_dest
                            ))

            headers.append(flux_header_struct().make(
                    els_in_block=len(block.el_number_map),
                    same_facepairs_end=\
                            len(same_fp_structs),
                    diff_facepairs_end=\
                            len(same_fp_structs)+len(diff_fp_structs),
                    bdry_facepairs_end=\
                            len(same_fp_structs)+len(diff_fp_structs)\
                            +len(bdry_fp_structs),
                    ))
            fp_blocks.append(same_fp_structs+diff_fp_structs+bdry_fp_structs)

        from hedge.cuda.cgen import Value
        from hedge.cuda.tools import make_superblocks

        return make_superblocks(
                discr.devdata, "flux_data",
                [
                    (headers, Value(flux_header_struct().tpname, "header")),
                    ],
                [ (fp_blocks, Value(fp_struct.tpname, "facepairs")), ])

    @memoize_method
    def index_list_global_data(self):
        discr = self.discr

        from pytools import single_valued
        ilist_length = single_valued(len(il) for il in discr.index_lists)

        if ilist_length > 256:
            tp = numpy.uint16
        else:
            tp = numpy.uint8

        from hedge.cuda.cgen import ArrayInitializer, ArrayOf, \
                Typedef, POD, Value, CudaConstant, Define

        from pytools import flatten, Record
        flat_ilists = numpy.array(
                list(flatten(discr.index_lists)),
                dtype=tp)
        return Record(
                code=[
                    Define("INDEX_LISTS_LENGTH", len(flat_ilists)),
                    Typedef(POD(tp, "index_list_entry_t")),
                    ],
                device_memory=cuda.to_device(flat_ilists)
                )
