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




# structures ------------------------------------------------------------------
@memoize
def flux_block_header_struct():
    from hedge.cuda.cgen import Struct, POD

    return Struct("flux_block_header", [
        POD(numpy.int16, "els_in_block"),
        POD(numpy.int16, "face_pairs_in_block"),
        ])

@memoize
def flux_face_pair_struct(dims):
    from hedge.cuda.cgen import Struct, POD, ArrayOf

    return Struct("flux_face_pair", [
        POD(numpy.float32, "h", ),
        POD(numpy.float32, "order"),
        POD(numpy.float32, "face_jacobian"),
        ArrayOf(POD(numpy.float32, "normal"), dims),
        POD(numpy.uint16, "a_base"),
        POD(numpy.uint16, "b_base"),
        POD(numpy.uint8, "a_ilist_number"),
        POD(numpy.uint8, "b_ilist_number"),
        POD(numpy.uint8, "bdry_flux_number"), # 0 if not on boundary
        POD(numpy.uint8, "reserved"),
        POD(numpy.uint32, "b_global_base"),

        # memory handling here deserves a comment.
        # Interior face (bdry_flux_number==0) dofs are duplicated if they cross
        # a block boundary. The flux results for these dofs are written out 
        # to b_global_base in addition to their local location.
        #
        # Boundary face (bdry_flux_number!=0) dofs are read from b_global_base
        # linearly (not (!) using b_global_ilist_number) into the extface
        # space at b_base. They are not written out again.
        ])

@memoize
def localop_block_header_struct():
    from hedge.cuda.cgen import Struct, POD

    return Struct("localop_block_header", [
        POD(numpy.uint16, "els_in_block"),
        POD(numpy.uint16, "facedups_in_block"),
        ])

@memoize
def localop_facedup_struct():
    from hedge.cuda.cgen import Struct, POD

    return Struct("localop_facedup", [
        POD(numpy.uint16, "smem_face_ilist_number"),
        POD(numpy.uint16, "smem_element_number"),
        POD(numpy.uint32, "dup_global_base"),
        ])




# exec mapper -----------------------------------------------------------------
class ExecutionMapper(hedge.optemplate.Evaluator,
        hedge.optemplate.BoundOpMapperMixin, 
        hedge.optemplate.LocalOpReducerMixin):

    def __init__(self, context, executor):
        hedge.optemplate.Evaluator.__init__(self, context)
        self.ex = executor

        self.diff_xyz_cache = {}

    def map_diff_base(self, op, field_expr, out=None):
        try:
            xyz_diff = self.diff_xyz_cache[op.__class__, field_expr]
        except KeyError:
            pass
        else:
            print "HIT"
            return xyz_diff[op.xyz_axis]

        field = self.rec(field_expr)

        discr = self.ex.discr
        d = discr.dimensions

        ii = self.ex.localop_indexing_info()
        eg, = discr.element_groups
        func, texref = self.ex.get_bound_diff_kernel(op.__class__, eg)

        lop_par = discr.plan.find_localop_par()
        
        kwargs = {
                "shared": discr.plan.localop_shared_mem_use(),
                "texrefs": [texref], 
                "block": (discr.plan.dofs_per_el(), lop_par.p, 1),
                "grid": (len(discr.blocks), 1)
                }

        xyz_diff = [discr.volume_zeros() for axis in range(d)]
        elgroup, = discr.element_groups
        args = xyz_diff+[
                field, 
                ii.headers, ii.facedups, 
                self.ex.localop_rst_to_xyz(op, elgroup)
                ]

        func(*args, **kwargs)

        if False:
            f = discr.volume_from_gpu(field)
            dx = discr.volume_from_gpu(xyz_diff[0], check=True)
            
            test_discr = discr.test_discr
            real_dx = test_discr.nabla[0].apply(f.astype(numpy.float64))

            print la.norm(dx-real_dx)/la.norm(real_dx)
        
        self.diff_xyz_cache[op.__class__, field_expr] = xyz_diff
        return xyz_diff[op.xyz_axis]

    def map_whole_domain_flux(self, op, field_expr, out=None):
        print op
        print field_expr

        field = self.rec(field_expr)
        discr = self.ex.discr

        ii = self.ex.flux_indexing_info()
        func, texref = self.ex.get_flux_kernel(op)

        flux_par = discr.plan.flux_par
        
        kwargs = {
                "shared": discr.plan.flux_shared_mem_use(),
                "texrefs": [texref], 
                "block": (discr.plan.dofs_per_face(), 
                    discr.plan.faces_per_el(),
                    flux_par.p),
                "grid": (len(discr.blocks), 1)
                }

        flux = discr.volume_zeros() 
        elgroup, = discr.element_groups
        args = [flux, field, ii.headers, ii.facepair_blocks]

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




    # diff kernel -------------------------------------------------------------
    @memoize_method
    def get_diff_kernel(self):
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

        lop_par = discr.plan.find_localop_par()
        ind_info = self.localop_indexing_info()
        rst2xyz_coeffs_size_unaligned = d*d*lop_par.total()
        elgroup, = discr.element_groups

        f_decl = CudaGlobal(FunctionDeclaration(Value("void", "apply_diff_mat"), 
            [Pointer(POD(numpy.float32, "dxyz%d" % i)) for i in dims]
            + [
                Pointer(POD(numpy.float32, "field")),
                Pointer(Value("localop_block_header", "block_headers")),
                Pointer(POD(numpy.uint8, "gmem_ind_info")),
                Pointer(POD(numpy.float32, "gmem_rst_to_xyz")),
                ]
            ))

        cmod = Module([
                Value("texture<float%d, 2, cudaReadModeElementType>" 
                    % self.diffmat_channels(), "diff_rst_matrices"),
                localop_block_header_struct(),
                localop_facedup_struct(),
                Define("EL_DOF", "threadIdx.x"),
                Define("BLOCK_EL", "threadIdx.y"),
                Define("DOFS_PER_EL", discr.plan.dofs_per_el()),
                Define("CONCURRENT_ELS", lop_par.p),
                Define("DOFS_BLOCK_BASE", "(blockIdx.x*BLOCK_DOFS)"),
                Define("THREAD_NUM", "(BLOCK_EL*DOFS_PER_EL + EL_DOF)"),
                Define("THREAD_COUNT", "(DOFS_PER_EL*CONCURRENT_ELS)"),
                Define("BLOCK_DOFS", discr.int_dof_floats + discr.ext_dof_floats),
                Define("LOP_IND_INFO_BLOCK_SIZE", ind_info.block_size),
                Define("RST2XYZ_BLOCK_SIZE", 
                    self.localop_floats_per_rst_to_xyz_block()),
                Line(),
                Comment("face-related stuff"),
                Define("DOFS_PER_FACE", discr.plan.dofs_per_face()),
                Define("CONCURRENT_FACES", 
                    discr.plan.dofs_per_el()*lop_par.p
                    //discr.plan.dofs_per_face()),
                Line(),
                self.index_list_data(),
                Line(),
                CudaShared(Value("localop_block_header", "block_header")),
                CudaShared(ArrayOf(Value("localop_facedup", "facedups"), 
                    ind_info.max_facedups_in_block)),
                CudaShared(ArrayOf(POD(numpy.float32, "rst_to_xyz_coefficients"), 
                    rst2xyz_coeffs_size_unaligned)),
                CudaShared(ArrayOf(POD(numpy.float32, "int_dofs"), 
                    discr.int_dof_floats)),
                Line(),
                ])

        S = Statement
        f_body = Block()
            
        f_body.extend_log_block("load block header in thread 0", [
            If("THREAD_NUM == 0",
                S("block_header = block_headers[blockIdx.x]")),
            S("__syncthreads()"),
            ])

        f_body.extend_log_block("load internal dofs", [
            For("unsigned dof_nr = THREAD_NUM", 
                "dof_nr < block_header.els_in_block*DOFS_PER_EL", 
                "dof_nr += THREAD_COUNT",
                S("int_dofs[dof_nr] = field[DOFS_BLOCK_BASE+dof_nr]"),
                ),
            ])

        from hedge.cuda.cgen import dtype_to_ctype
        copy_dtype = numpy.dtype(numpy.int32)
        copy_dtype_str = dtype_to_ctype(copy_dtype)
        f_body.extend_log_block("load indexing info", [
            Constant(Pointer(POD(copy_dtype, "facedups_block_base")), 
                ("(%s *)" % copy_dtype_str)+
                "(gmem_ind_info + blockIdx.x*LOP_IND_INFO_BLOCK_SIZE)"),
            For("unsigned facedup_word_nr = THREAD_NUM", 
                "facedup_word_nr*sizeof(int) < "
                "sizeof(localop_facedup)*block_header.facedups_in_block", 
                "facedup_word_nr += THREAD_COUNT",
                S("((%s *) facedups)[facedup_word_nr] = facedups_block_base[facedup_word_nr]"
                    % copy_dtype_str)
                ),
            ])
        # ---------------------------------------------------------------------
        f_body.extend_log_block("load rst_to_xyz_coefficients" , [
            Initializer(POD(numpy.int32, "r2x_base"),
                "blockIdx.x * RST2XYZ_BLOCK_SIZE"),
            For("unsigned i = THREAD_NUM", 
                "i < %d" % (d*d*lop_par.total()),
                "i += THREAD_COUNT",
                S("rst_to_xyz_coefficients[i] = "
                    "gmem_rst_to_xyz[r2x_base+i]")),
            S("__syncthreads()"),
            ])

        def get_scalar_diff_code(el_nr, dest_dof, dest_pattern):
            code = []
            for axis in dims:
                code.append(
                    Initializer(POD(numpy.float32, "drst%d" % axis), 0))

            code.append(Line())

            tex_channels = ["x", "y", "z", "w"]
            code.append(
                For("unsigned j = 0", "j < DOFS_PER_EL", "++j", Block([
                    S("float%d diff_ij = tex2D(diff_rst_matrices, (%s) , j)" 
                        % (self.diffmat_channels(), dest_dof)),
                    Initializer(POD(numpy.float32, "field_value"),
                        "int_dofs[(%s)*DOFS_PER_EL+j]" % el_nr),
                    Line(),
                    ]+[
                    S("drst%d += diff_ij.%s * field_value" 
                        % (axis, tex_channels[axis]))
                    for axis in dims
                    ]))
                )

            code.append(Line())

            for glob_axis in dims:
                code.append(
                    Assign(
                        dest_pattern % glob_axis,
                        " + ".join(
                            "rst_to_xyz_coefficients[%d + %d*(%s)]"
                            "*"
                            "drst%d" % (
                                d*glob_axis+loc_axis, 
                                d*d,
                                el_nr,
                                loc_axis)
                            for loc_axis in dims
                            )
                        ))
            return code

        # global diff on volume -----------------------------------------------
        f_body.extend_log_block("perform global diff on volume", [
            For("unsigned base_el = BLOCK_EL",
                "base_el < block_header.els_in_block",
                "base_el += CONCURRENT_ELS", 
                Block(get_scalar_diff_code(
                    "base_el",
                    "EL_DOF",
                    "dxyz%d[DOFS_BLOCK_BASE+base_el*DOFS_PER_EL+EL_DOF]" 
                    ))
                )])

        # global diff on duplicated faces -------------------------------------
        f_body.extend_log_block("perform global diff on dup faces", [
            Initializer(Const(POD(numpy.int16, "block_face")),
                "THREAD_NUM / DOFS_PER_FACE"),
            Initializer(Const(POD(numpy.int16, "face_dof")),
                "THREAD_NUM - DOFS_PER_FACE*block_face"),
            For("unsigned face_nr = block_face",
                "face_nr < block_header.facedups_in_block",
                "face_nr += CONCURRENT_FACES", Block([
                    Initializer(POD(numpy.uint16, "face_el_dof"),
                        "index_lists["
                        "facedups[face_nr].smem_face_ilist_number*DOFS_PER_FACE"
                        "+face_dof"
                        "]"),
                    Initializer(POD(numpy.uint16, "face_el_nr"),
                        "facedups[face_nr].smem_element_number"),
                    Initializer(POD(numpy.uint32, "tgt_dof"),
                        "facedups[face_nr].dup_global_base+face_dof"),
                    Line(),
                    ]+get_scalar_diff_code("face_el_nr", "face_el_dof",
                        "dxyz%d[tgt_dof]" )
                    )
                )
            ])
            
        # finish off ----------------------------------------------------------
        cmod.append(FunctionBody(f_decl, f_body))

        mod = cuda.SourceModule(cmod, 
                #keep=True, 
                #options=["--maxrregcount=12"]
                )

        return (mod.get_function("apply_diff_mat"), 
                mod.get_texref("diff_rst_matrices"))

    @memoize_method
    def get_bound_diff_kernel(self, diff_op_cls, elgroup):
        kernel, texref = self.get_diff_kernel()
        cuda.bind_array_to_texref(
                self.diffmat_array(diff_op_cls, elgroup),
                texref)

        return kernel, texref




    # flux kernel -------------------------------------------------------------
    @memoize_method
    def get_flux_kernel(self, wdflux):
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

        lop_par = discr.plan.find_localop_par()
        ind_info = self.flux_indexing_info()
        rst2xyz_coeffs_size_unaligned = d*d*lop_par.total()
        elgroup, = discr.element_groups

        f_decl = CudaGlobal(FunctionDeclaration(Value("void", "apply_flux"), 
            [
                Pointer(POD(numpy.float32, "flux")),
                Pointer(POD(numpy.float32, "field")),
                Pointer(Value("flux_block_header", "block_headers")),
                Pointer(POD(numpy.uint8, "gmem_ind_info")),
                ]
            ))

        cmod = Module([
                Value("texture<float, 2, cudaReadModeElementType>", "lift_matrix"),
                flux_block_header_struct(),
                flux_facepair_struct(),
                Define("FACE_DOF", "threadIdx.x"),
                Define("EL_FACE", "threadIdx.y"),
                Define("BLOCK_EL", "threadIdx.z"),
                Define("DOFS_PER_FACE", discr.plan.dofs_per_face()),
                Define("CONCURRENT_FACES", discr.plan.faces_per_el()*flux_par.p),
                Define("DOFS_BLOCK_BASE", "(blockIdx.x*BLOCK_DOFS)"),
                Define("THREAD_NUM", "(BLOCK_EL*DOFS_PER_EL + FACE_DOF)"),
                Define("THREAD_COUNT", "(DOFS_PER_FACE*CONCURRENT_FACES)"),
                Define("BLOCK_DOFS", discr.int_dof_floats + discr.ext_dof_floats),
                Line(),
                Comment("element-related stuff"),
                Define("DOFS_PER_EL", discr.plan.dofs_per_el()),
                Line(),
                self.index_list_data(),
                Line(),
                CudaShared(Value("flux_block_header", "block_header")),
                CudaShared(ArrayOf(Value("flux_face_pair", "face_pairs"), 
                    ind_info.max_face_pairs_in_block)),
                CudaShared(ArrayOf(POD(numpy.float32, "int_dofs"), 
                    discr.int_dof_floats)),
                CudaShared(ArrayOf(POD(numpy.float32, "ext_dofs"), 
                    discr.ext_dof_floats)),
                Line(),
                ])

        S = Statement
        f_body = Block()
            
        f_body.extend_log_block("load block header in thread 0", [
            If("THREAD_NUM == 0",
                S("block_header = block_headers[blockIdx.x]")),
            S("__syncthreads()"),
            ])

        f_body.extend_log_block("load internal dofs", [
            For("unsigned dof_nr = THREAD_NUM", 
                "dof_nr < block_header.els_in_block*DOFS_PER_EL", 
                "dof_nr += THREAD_COUNT",
                S("int_dofs[dof_nr] = field[DOFS_BLOCK_BASE+dof_nr]"),
                ),
            ])

        from hedge.cuda.cgen import dtype_to_ctype
        copy_dtype = numpy.dtype(numpy.int32)
        copy_dtype_str = dtype_to_ctype(copy_dtype)
        f_body.extend_log_block("load indexing info", [
            Constant(Pointer(POD(copy_dtype, "facedups_block_base")), 
                ("(%s *)" % copy_dtype_str)+
                "(gmem_ind_info + blockIdx.x*LOP_IND_INFO_BLOCK_SIZE)"),
            For("unsigned facedup_word_nr = THREAD_NUM", 
                "facedup_word_nr*sizeof(int) < "
                "sizeof(localop_facedup)*block_header.facedups_in_block", 
                "facedup_word_nr += THREAD_COUNT",
                S("((%s *) facedups)[facedup_word_nr] = facedups_block_base[facedup_word_nr]"
                    % copy_dtype_str)
                ),
            ])
            
        # finish off ----------------------------------------------------------
        cmod.append(FunctionBody(f_decl, f_body))

        mod = cuda.SourceModule(cmod, 
                keep=True, 
                #options=["--maxrregcount=12"]
                )

        return (mod.get_function("apply_flux"), 
                mod.get_texref("lift_matrix"))

    # gpu data blocks ---------------------------------------------------------
    def diffmat_channels(self):
        return min(ch
                for ch in [1,2,4]
                if ch >= self.discr.dimensions)

    @memoize_method
    def diffmat_array(self, diff_op_cls, elgroup):
        diffmats = diff_op_cls.matrices(elgroup)[:]
        channel_count = self.diffmat_channels()

        from pytools import single_valued
        diffmat_shape = single_valued(dm.shape for dm in diffmats)
        while channel_count > len(diffmats):
            diffmats.append(
                    numpy.zeros(diffmat_shape, dtype=numpy.float32))
        
        from pytools import Record
        return cuda.make_multichannel_2d_array(diffmats)

    def localop_floats_per_rst_to_xyz_block(self):
        discr = self.discr
        d = discr.dimensions
        return discr.devdata.align_dtype(
                d*d*discr.plan.find_localop_par().total(), 
                self.discr.plan.float_size)

    @memoize_method
    def localop_rst_to_xyz(self, diff_op, elgroup):
        discr = self.discr

        floats_per_block = self.localop_floats_per_rst_to_xyz_block()
        bytes_per_block = floats_per_block*discr.plan.float_size

        coeffs = diff_op.coefficients(elgroup)

        blocks = []
        
        def get_el_index_in_el_group(el):
            mygroup, idx = discr.group_map[el.id]
            assert mygroup is elgroup
            return idx

        for block in discr.blocks:
            block_elgroup_indices = numpy.fromiter(
                    (get_el_index_in_el_group(el) for el in block.elements),
                    dtype=numpy.intp)

            flattened = (coeffs[:,:,block_elgroup_indices]
                .transpose(2,0,1).flatten().astype(numpy.float32))
            blocks.append(flattened)
                
            #numpy.set_printoptions(precision=3)
            #print flattened
            #print coeffs[:,:,block_elgroup_indices]
            #raw_input()

        from hedge.cuda.tools import pad_and_join
        return cuda.to_device(
                    pad_and_join((str(buffer(s)) for s in blocks), 
                        bytes_per_block))

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
                #assert extface.native_block is block
                assert isinstance(extface, GPUInteriorFaceStorage)
                facedup_count += 1
                block_data += localop_facedup_struct().make(
                        smem_face_ilist_number=extface.native_index_list_id,
                        smem_element_number=extface.native_block_el_num,
                        dup_global_base=extface.dup_global_base)

            #raw_input()

            headers.append(localop_block_header_struct().make(
                        els_in_block=len(block.elements),
                        facedups_in_block=facedup_count))
            blocks.append(block_data)
            facedup_counts.append(facedup_count)

        block_size = discr.devdata.align(max(len(b) for b in blocks))

        assert block_size < discr.plan.localop_indexing_smem()

        from hedge.cuda.tools import pad_and_join
        all_headers = "".join(headers)
        all_blocks = pad_and_join(blocks, block_size)
        
        from pytools import Record
        return Record(
                block_size=block_size, 
                max_facedups_in_block=max(facedup_counts),
                headers=cuda.to_device(all_headers),
                facedups=cuda.to_device(all_blocks),
                )


    @memoize_method
    def flux_indexing_info(self):
        discr = self.discr

        block_len = discr.plan.flux_indexing_smem()

        INVALID_U8 = (1<<8) - 1
        INVALID_U16 = (1<<16) - 1
        INVALID_U32 = (1<<32) - 1

        headers = []
        blocks = []
        fp_counts = []

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
                        flux_face_pair_struct(discr.dimensions).make(
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

            headers.append(flux_block_header_struct().make(
                    els_in_block=len(block.elements),
                    face_pairs_in_block=len(fp_blocks)
                    ))
            blocks.append("".join(fp_blocks))
            fp_counts.append(len(fp_blocks))

        # make sure the indexing_smem estimate is achieved
        assert max(len(b) for b in blocks) == block_len

        from hedge.cuda.tools import pad_and_join
        facepair_blocks = pad_and_join(blocks)
        assert len(facepair_blocks) == block_len*len(discr.blocks)

        assert max(fp_counts) == discr.plan.face_pair_count()
        from pytools import Record
        return Record(
            headers="".join(headers),
            face_pair_blocks=cuda.to_device(facepair_blocks),
            max_face_pairs_in_block=max(fp_counts)
            )

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
        return ArrayInitializer(
                    CudaConstant(
                        ArrayOf(POD(tp, "index_lists"))),
                    flat_ilists
                    )
