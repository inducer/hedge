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
import hedge.optemplate
import pycuda.driver as cuda




class ExecutionMapper(hedge.optemplate.Evaluator,
        hedge.optemplate.BoundOpMapperMixin, 
        hedge.optemplate.LocalOpReducerMixin):
    def __init__(self, context, executor):
        hedge.optemplate.Evaluator.__init__(self, context)
        self.ex = executor

    def map_diff_base(self, op, field_expr, out=None):
        field = self.rec(field_expr)

        ii = self.ex.localop_indexing_info()
        print ii.__dict__
        func, texrefs = self.ex.get_diff_kernel()

        discr = self.ex.discr
        d = discr.dimensions

        lop_par = discr.plan.find_localop_par()
        rst_diff = [discr.volume_zeros() for axis in range(d)]
        xyz_diff = [discr.volume_zeros() for axis in range(d)]
        args = rst_diff+xyz_diff+[field, ii.headers, ii.facedups]

        kwargs = {
                "shared": discr.plan.shared_mem_use(),
                "texrefs": texrefs, 
                "block_shape": (discr.plan.dofs_per_el(), lop_par.p, 1),
                "grid": (len(discr.blocks), 1)
                }

        func(*args, **kwargs)

        print "TSCHUES"
        import sys
        sys.exit(1)




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

    # code generation ---------------------------------------------------------
    @memoize_method
    def get_diff_kernel(self):
        from hedge.cuda.cgen import \
                Pointer, POD, Value, ArrayOf, Const, \
                Module, FunctionDeclaration, FunctionBody, Block, \
                Comment, Line, \
                CudaShared, CudaGlobal, Static, \
                Define, \
                Constant, Initializer, If, For, Statement
                
        discr = self.discr
        d = discr.dimensions
        dims = range(d)

        lop_par = discr.plan.find_localop_par()
        ind_info = self.localop_indexing_info()
        print ind_info.__dict__
        ilist_data = self.index_list_data()

        texref_names = ["diff_rst%d_matrix" % i for i in dims]

        f_decl = CudaGlobal(FunctionDeclaration(Value("void", "apply_diff_mat"), 
            [Pointer(POD(numpy.float32, "drst%d" % i)) for i in dims]
            + [Pointer(POD(numpy.float32, "dxyz%d" % i)) for i in dims]
            +
            [
                Pointer(POD(numpy.float32, "field")),
                Pointer(Value("localop_block_header", "block_headers")),
                Pointer(POD(numpy.uint32, "gmem_ind_info")),
                ]
            ))

        cmod = Module(
            [Value("texture<float, 2, cudaReadModeElementType>", name)
                for name in texref_names]
            +[
                self.localop_block_header_struct(),
                self.localop_facedup_struct(),
                Define("DOFS_PER_EL", discr.plan.dofs_per_el()),
                Define("DOFS_BLOCK_BASE", "(blockIdx.x*BLOCK_DOFS)"),
                Define("BLOCK_DOFS", discr.int_dof_floats + discr.ext_dof_floats),
                Define("ILIST_LENGTH", ilist_data.single_ilist_length),
                Define("LOP_IND_INFO_BLOCK_SIZE", ind_info.block_size),
                Line(),
                ilist_data.initializer,
                Line(),
                CudaShared(Value("localop_block_header", "block_header")),
                CudaShared(ArrayOf(Value("localop_facedup", "facedups"), 
                    ind_info.max_facedups_in_block)),
                CudaShared(ArrayOf(POD(numpy.float32, "int_dofs"), 
                    discr.int_dof_floats)),
                Line(),
                ])

        S = Statement
        f_body = Block()
        f_body.extend_log_block("Variable initializations", [
            Constant(POD(numpy.uint16, "dofs_per_el"), "blockDim.x"),
            Constant(POD(numpy.uint16, "concurrent_els"), "blockDim.y"),
            Constant(POD(numpy.uint16, "el_dof"), "threadIdx.x"),
            Constant(POD(numpy.uint16, "block_el"), "threadIdx.y"),
            Constant(POD(numpy.uint16, "thread_count"), "dofs_per_el*concurrent_els"),
            Constant(POD(numpy.uint16, "thread_num"), "block_el*dofs_per_el + el_dof"),
            ])
            
        f_body.extend_log_block("load block header in thread 0", [
            If("thread_num == 0",
                S("block_header = block_headers[blockIdx.x]")),
            S("__syncthreads()"),
            ])

        f_body.extend_log_block("load internal dofs", [
            For("unsigned dof_nr = thread_num", 
                "dof_nr < block_header.els_in_block*DOFS_PER_EL", 
                "dof_nr += thread_count",
                S("int_dofs[dof_nr] = field[DOFS_BLOCK_BASE+dof_nr]"),
                ),
            S("__syncthreads()"),
            ])

        from hedge.cuda.cgen import dtype_to_ctype
        copy_dtype = numpy.dtype(numpy.int32)
        copy_dtype_str = dtype_to_ctype(copy_dtype)
        f_body.extend_log_block("load indexing info", [
            Constant(Pointer(POD(copy_dtype, "facedups_block_base")), 
                ("(%s *)" % copy_dtype_str)+
                "(gmem_ind_info + blockIdx.x*LOP_IND_INFO_BLOCK_SIZE)"),
            For("unsigned facedup_word_nr = thread_num", 
                "facedup_word_nr*sizeof(int) < "
                "sizeof(localop_facedup)*block_header.facedups_in_block", 
                "facedup_word_nr += thread_count",
                S("((%s *) facedups)[facedup_word_nr] = facedups_block_base[facedup_word_nr]"
                    % copy_dtype_str)
                ),
            S("__syncthreads()"),
            ])
        # ---------------------------------------------------------------------
        for axis in dims:
            f_body.extend([
                    Comment("perform local diff along axis %d" % axis),
                    For("unsigned el_nr = block_el",
                        "el_nr < block_header.els_in_block",
                        "el_nr += concurrent_els", Block([
                            Initializer(POD(numpy.float32, "accum"), 0),
                            For("unsigned j = 0", "j < DOFS_PER_EL", "++j",
                                S("accum += tex2D(diff_rst%d_matrix, el_dof, j)"
                                    "* int_dofs[el_nr*DOFS_PER_EL+el_dof]" % axis)
                                ),
                            S("drst%d[DOFS_BLOCK_BASE+el_nr*DOFS_PER_EL+el_dof]"
                                "= accum" % axis),
                            ])
                        ),

                    Line(),
                    ])
            
        cmod.append(FunctionBody(f_decl, f_body))

        mod = cuda.SourceModule(cmod, keep=True)
        texrefs = [mod.get_texref(name) for name in texref_names]
        for texref, dmat in zip(texrefs, discr.plan.ldis.differentiation_matrices()):
            cuda.matrix_to_texref(dmat, texref)

        return mod.get_function("apply_diff_mat"), texrefs

    # gpu data blocks ---------------------------------------------------------
    @memoize_method
    def get_diffmat_array(self, diff_op_cls, elgroup, axis):
        return cuda.matrix_to_array(diff_op_cls.matrices(elgroup)[axis])

    @memoize_method
    def localop_block_header_struct(self):
        from hedge.cuda.cgen import Struct, POD

        return Struct("localop_block_header", [
            POD(numpy.uint16, "els_in_block"),
            POD(numpy.uint16, "facedups_in_block"),
            ])

    @memoize_method
    def localop_facedup_struct(self):
        from hedge.cuda.cgen import Struct, POD

        return Struct("localop_facedup", [
            POD(numpy.uint16, "smem_face_base"),
            POD(numpy.uint8, "smem_face_ilist_number"),
            POD(numpy.uint8, "reserved"),
            POD(numpy.uint32, "dup_global_base"),
            ])

    @memoize_method
    def localop_rst_to_xyz(self):
        pass

    @memoize_method
    def localop_indexing_info(self):
        discr = self.discr
        headers = []
        blocks = []
        facedup_counts = []

        from hedge.cuda.discretization import GPUInteriorFaceStorage
        for block in discr.blocks:
            ldis = block.local_discretization
            el_dofs = ldis.node_count()
            face_dofs = ldis.node_count()
            facedup_count = 0 

            block_data = ""
            for extface in block.ext_faces_from_me:
                if not isinstance(extface, GPUInteriorFaceStorage):
                    continue
                facedup_count += 1
                block_data += self.localop_facedup_struct().make(
                        smem_face_base=el_dofs*extface.native_block_el_num,
                        smem_face_ilist_number=extface.native_index_list_id,
                        reserved=0,
                        dup_global_base=extface.dup_global_base)

            headers.append(self.localop_block_header_struct().make(
                        els_in_block=len(block.elements),
                        facedups_in_block=facedup_count))
            blocks.append(block_data)
            facedup_counts.append(facedup_count)

        block_size = discr.devdata.align(max(len(b) for b in blocks))
        from hedge.cuda.tools import pad_and_join
        from pytools import Record
        return Record(
                block_size=block_size, 
                max_facedups_in_block=max(facedup_counts),
                headers=cuda.to_device("".join(headers)),
                facedups=cuda.to_device(pad_and_join(blocks, block_size)),
                )


    @memoize_method
    def flux_indexing_info(self):
        discr = self.discr

        block_len = discr.plan.indexing_smem()

        INVALID_U8 = (1<<8) - 1
        INVALID_U16 = (1<<16) - 1
        INVALID_U32 = (1<<32) - 1

        headers = []
        blocks = []

        from hedge.cuda.discretization import GPUBoundaryFaceStorage

        for block in discr.blocks:
            ldis = block.local_discretization
            el_dofs = ldis.node_count()

            faces_todo = set((el,face_nbr)
                    for el in block.elements
                    for face_nbr in range(ldis.face_count()))
            fp_blocks = []

            while faces_todo:
                elface = faces_todo.pop()

                int_face = discr.face_storage_map[elface]
                opp = int_face.opposite

                if isinstance(opp, GPUBoundaryFaceStorage):
                    # boundary face
                    b_base = INVALID_U16
                    bdry_flux_number = 1
                    b_global_base = opp.gpu_bdry_index_in_floats
                    b_ilist_number = INVALID_U8
                else:
                    # interior face
                    b_base = opp.native_block_el_num*el_dofs
                    bdry_flux_number = 0
                    if opp.native_block == int_face.native_block:
                        # same block
                        faces_todo.remove(opp.el_face)
                        b_global_base = INVALID_U32
                        b_ilist_number = opp.native_index_list_id
                    else:
                        # different block
                        b_global_base = opp.dup_global_base
                        b_ilist_number = INVALID_U8

                fp_blocks.append(
                        discr.plan.get_face_pair_struct().make(
                            h=int_face.flux_face.h,
                            order=int_face.flux_face.order,
                            face_jacobian=int_face.flux_face.face_jacobian,
                            normal=int_face.flux_face.normal,
                            a_base=int_face.native_block_el_num*el_dofs,
                            b_base=b_base,
                            a_ilist_number=int_face.native_index_list_id,
                            b_ilist_number=b_ilist_number,
                            bdry_flux_number=bdry_flux_number,
                            reserved=0,
                            b_global_base=b_global_base,
                            ))

            headers.append(discr.plan.get_block_header_struct().make(
                    els_in_block=len(block.elements),
                    face_pairs_in_block=len(fp_blocks)
                    ))
            blocks.append("".join(fp_blocks))

        # make sure the indexing_smem estimate is achieved
        assert max(len(b) for b in blocks) == block_len

        from hedge.cuda.tools import pad_and_join
        facepair_blocks = pad_and_join(blocks)
        assert len(facepair_blocks) == block_len*len(discr.blocks)

        from pytools import Record
        return Record(
            headers="".join(headers),
            facepair_blocks=cuda.to_device(facepair_blocks))

    @memoize_method
    def index_list_data(self):
        discr = self.discr

        from pytools import single_valued
        ilist_length = single_valued(len(il) for il in discr.index_lists)

        if ilist_length > 256:
            tp = numpy.uint16
        else:
            tp = numpy.uint8

        from hedge.cuda.cgen import ArrayInitializer, ArrayOf, POD, CudaConstant

        from pytools import flatten, Record
        flat_ilists = list(flatten(discr.index_lists))
        return Record(
                single_ilist_length=ilist_length,
                initializer=ArrayInitializer(
                    CudaConstant(
                        ArrayOf(POD(tp, "index_lists"))),
                    flat_ilists
                    )
                )
