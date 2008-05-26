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
def block_header_struct():
    from hedge.cuda.cgen import Struct, POD

    return Struct("block_header", [
        POD(numpy.uint16, "els_in_block"),
        POD(numpy.uint16, "facedups_in_block"),
        POD(numpy.uint16, "boundaries_in_block"),
        POD(numpy.uint16, "reserved"),
        ])

@memoize
def flux_face_properties_struct(float_type, dims):
    from hedge.cuda.cgen import Struct, POD, ArrayOf

    return Struct("flux_face_properties", [
        POD(float_type, "h", ),
        POD(float_type, "order"),
        POD(float_type, "face_jacobian"),
        ArrayOf(POD(float_type, "normal"), dims),
        ])

@memoize
def flux_face_location_struct():
    from hedge.cuda.cgen import Struct, POD, ArrayOf

    return Struct("flux_face_location", [
        POD(numpy.uint16, "prop_block_number_and_side"), # lsb==1: flip normal
        POD(numpy.uint16, "a_base"),
        POD(numpy.uint16, "b_base"),

        POD(numpy.uint8, "a_ilist_number"),
        POD(numpy.uint8, "b_ilist_number_or_flux"), 
        # msb==1: use identity ilist (0), bits 0..6 specify flux number
        ])

@memoize
def facedup_struct():
    from hedge.cuda.cgen import Struct, POD

    return Struct("facedup", [
        POD(numpy.uint16, "smem_element_number"),
        POD(numpy.uint8, "face_number"),
        POD(numpy.uint8, "smem_face_ilist_number"),
        POD(numpy.uint32, "dup_global_base"),
        ])

@memoize
def boundary_load_struct():
    from hedge.cuda.cgen import Struct, POD

    return Struct("boundary_load", [
        POD(numpy.uint32, "global_base"),
        POD(numpy.uint16, "smem_base"),
        POD(numpy.uint16, "reserved"),
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

        localop_data = self.ex.localop_data()
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
                field, localop_data.device_memory,
                self.ex.localop_rst_to_xyz(op, elgroup),
                ]

        func(*args, **kwargs)

        if True:
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

        ii = self.ex.flux_aux_info()
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
                    "word_nr += THREAD_COUNT",
                    Statement("((%s *) (%s))[word_nr] = load_base[word_nr]"
                        % (copy_dtype_str, dest))
                    ),
                ]),
            Line(),
            ])

        return code

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
        lop_data = self.localop_data()
        rst2xyz_coeffs_size_unaligned = d*d*lop_par.total()
        elgroup, = discr.element_groups

        float_type = discr.plan.float_type

        f_decl = CudaGlobal(FunctionDeclaration(Value("void", "apply_diff_mat"), 
            [Pointer(POD(float_type, "dxyz%d" % i)) for i in dims]
            + [
                Pointer(POD(float_type, "field")),
                Pointer(POD(numpy.uint8, "gmem_data")),
                Pointer(POD(float_type, "gmem_rst_to_xyz")),
                ]
            ))

        cmod = Module([
                Value("texture<float%d, 2, cudaReadModeElementType>" 
                    % self.diffmat_channels(), "diff_rst_matrices"),
                block_header_struct(),
                facedup_struct(),
                Define("EL_DOF", "threadIdx.x"),
                Define("BLOCK_EL", "threadIdx.y"),
                Define("DOFS_PER_EL", discr.plan.dofs_per_el()),
                Define("CONCURRENT_ELS", lop_par.p),
                Define("DOFS_BLOCK_BASE", "(blockIdx.x*BLOCK_DOF_COUNT)"),
                Define("THREAD_NUM", "(BLOCK_EL*DOFS_PER_EL + EL_DOF)"),
                Define("THREAD_COUNT", "(DOFS_PER_EL*CONCURRENT_ELS)"),
                Define("BLOCK_DOF_COUNT", discr.block_dof_count()),
                Define("DATA_BLOCK_SIZE", lop_data.block_size),
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
                lop_data.struct,
                CudaShared(Value("localop_data", "data")),
                CudaShared(ArrayOf(POD(float_type, "rst_to_xyz_coefficients"), 
                    rst2xyz_coeffs_size_unaligned)),
                CudaShared(ArrayOf(POD(float_type, "int_dofs"), 
                    discr.int_dof_floats)),
                Line(),
                ])

        S = Statement
        f_body = Block()
            
        f_body.extend(self.get_load_code(
            dest="&data",
            base="gmem_data + blockIdx.x*DATA_BLOCK_SIZE",
            bytes="sizeof(localop_data)",
            descr="load localop_data"))

        f_body.append( S("__syncthreads()"))

        f_body.extend_log_block("load internal dofs", [
            For("unsigned dof_nr = THREAD_NUM", 
                "dof_nr < data.header.els_in_block*DOFS_PER_EL", 
                "dof_nr += THREAD_COUNT",
                S("int_dofs[dof_nr] = field[DOFS_BLOCK_BASE+dof_nr]"),
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

        # ---------------------------------------------------------------------
        def get_scalar_diff_code(el_nr, dest_dof, dest_pattern):
            code = []
            for axis in dims:
                code.append(
                    Initializer(POD(float_type, "drst%d" % axis), 0))

            code.append(Line())

            tex_channels = ["x", "y", "z", "w"]
            code.append(
                For("unsigned j = 0", "j < DOFS_PER_EL", "++j", Block([
                    S("float%d diff_ij = tex2D(diff_rst_matrices, (%s) , j)" 
                        % (self.diffmat_channels(), dest_dof)),
                    Initializer(POD(float_type, "field_value"),
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
                "base_el < data.header.els_in_block",
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
                "face_nr < data.header.facedups_in_block",
                "face_nr += CONCURRENT_FACES", Block([
                    Initializer(POD(numpy.uint16, "face_el_dof"),
                        "index_lists["
                        "data.facedups[face_nr].smem_face_ilist_number*DOFS_PER_FACE"
                        "+face_dof"
                        "]"),
                    Initializer(POD(numpy.uint16, "face_el_nr"),
                        "data.facedups[face_nr].smem_element_number"),
                    Initializer(POD(numpy.uint32, "tgt_dof"),
                        "data.facedups[face_nr].dup_global_base+face_dof"),
                    Line(),
                    ]+get_scalar_diff_code("face_el_nr", "face_el_dof",
                        "dxyz%d[tgt_dof]" )
                    )
                )
            ])
            
        # finish off ----------------------------------------------------------
        cmod.append(FunctionBody(f_decl, f_body))

        mod = cuda.SourceModule(cmod, 
                keep=True, 
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

        flux_par = discr.plan.find_localop_par()
        facedups = self.facedup_data()
        aux_info = self.flux_aux_info()
        elgroup, = discr.element_groups

        float_type = discr.plan.float_type

        f_decl = CudaGlobal(FunctionDeclaration(Value("void", "apply_flux"), 
            [
                Pointer(POD(float_type, "flux")),
                Pointer(POD(float_type, "field")),
                Pointer(Value("block_header", "block_headers")),
                Pointer(POD(numpy.uint8, "gmem_face_properties")),
                Pointer(POD(numpy.uint8, "gmem_face_locations")),
                Pointer(POD(numpy.uint8, "gmem_facedups")),
                ]
            ))

        cmod = Module([
                Value("texture<float, 2, cudaReadModeElementType>", "lift_matrix"),
                block_header_struct(),
                facedup_struct(),
                flux_face_properties_struct(float_type, discr.dimensions),
                flux_face_location_struct(),
                boundary_load_struct(),
                Define("FACE_DOF", "threadIdx.x"),
                Define("EL_FACE", "threadIdx.y"),
                Define("BLOCK_EL", "threadIdx.z"),
                Define("DOFS_PER_EL", discr.plan.dofs_per_el()),
                Define("DOFS_PER_FACE", discr.plan.dofs_per_face()),
                Define("CONCURRENT_ELS", flux_par.p),
                Define("DOFS_BLOCK_BASE", "(blockIdx.x*BLOCK_DOF_COUNT)"),
                Define("THREAD_NUM", "(BLOCK_EL*DOFS_PER_EL + FACE_DOF)"),
                Define("THREAD_COUNT", "(DOFS_PER_EL*CONCURRENT_ELS)"),
                Define("BLOCK_DOF_COUNT", discr.int_dof_floats + discr.ext_dof_floats),
                Line(),
                Comment("element-related stuff"),
                Define("DOFS_PER_EL", discr.plan.dofs_per_el()),
                Line(),
                self.index_list_data(),
                Line(),
                Struct("flux_aux_data", [
                    Value("block_header", "header"),
                    ArrayOf(Value("facedup", "facedups"),
                        XXX),
                    ArrayOf(Value("flux_face_properties", "faceprops"), 
                        aux_info.properties.max_per_block),
                    ArrayOf(Value("flux_face_location", "facelocs"), 
                        aux_info.location.max_per_block),
                    ArrayOf(Value("boundary_load", "bdry_loads"), 
                        aux_info.boundary_load.max_per_block),
                    ]),
                CudaShared(Value("flux_aux_data", "data")),
                CudaShared(ArrayOf(POD(float_type, "dofs"),
                    "BLOCK_DOF_COUNT")),
                Line(),
                ])

        S = Statement
        f_body = Block()
            
        f_body.extend_log_block("load block header in thread 0", [
            If("THREAD_NUM == 0",
                S("header = block_headers[blockIdx.x]")),
            S("__syncthreads()"),
            ])

        f_body.extend_log_block("load dofs", [
            For("unsigned dof_nr = THREAD_NUM", 
                "dof_nr < BLOCK_DOF_COUNT", 
                "dof_nr += THREAD_COUNT",
                S("dofs[dof_nr] = field[DOFS_BLOCK_BASE+dof_nr]"),
                ),
            ])

        f_body.extend(self.get_load_code(
            dest="facedups",
            base="gmem_facedups + blockIdx.x*FACEDUP_BLOCK_SIZE",
            bytes="sizeof(facedup)*header.facedups_in_block",
            descr="load facedups"))

        f_body.extend(self.get_load_code(
            dest="faceprops",
            base="gmem_face_properties + blockIdx.x*FACEPROP_BLOCK_SIZE",
            bytes="sizeof(flux_face_properties)*MAX_FACEPROPS_PER_BLOCK",
            descr="load face properties"))

        f_body.extend(self.get_load_code(
            dest="facelocs",
            base="gmem_face_locations + blockIdx.x*FACELOC_BLOCK_SIZE",
            bytes="sizeof(flux_face_location)*MAX_FACELOCS_PER_BLOCK",
            descr="load face locations"))
            
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
                    numpy.zeros(diffmat_shape, dtype=self.discr.plan.float_type))
        
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
                .transpose(2,0,1).flatten().astype(discr.plan.float_type))
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
    def facedup_blocks(self):
        discr = self.discr
        headers = []
        blocks = []

        from hedge.cuda.discretization import GPUInteriorFaceStorage
        for block in discr.blocks:
            ldis = block.local_discretization
            el_dofs = ldis.node_count()
            face_dofs = ldis.node_count()
            facedup_count = 0 
            block_boundary_count = 0

            structs = []
            for extface in block.ext_faces_from_me:
                assert extface.native_block is block
                assert isinstance(extface, GPUInteriorFaceStorage)
                if isinstance(extface.opposite, GPUInteriorFaceStorage):
                    facedup_count += 1
                    structs.append(facedup_struct().make(
                            smem_element_number=extface.native_block_el_num,
                            face_number=extface.el_face[1],
                            smem_face_ilist_number=extface.native_index_list_id,
                            dup_global_base=extface.dup_global_base))
                else:
                    block_boundary_count += 1

            headers.append(block_header_struct().make(
                        els_in_block=len(block.elements),
                        facedups_in_block=facedup_count,
                        boundaries_in_block=block_boundary_count,
                        reserved=0,
                        ))
            blocks.append(structs)

        return headers, blocks

    @memoize_method
    def localop_data(self):
        discr = self.discr
        headers, blocks = self.facedup_blocks()

        from hedge.cuda.tools import make_superblocks
        return make_superblocks(
                discr.devdata, "localop_data",
                block_header_struct(),
                headers,
                [("facedups", blocks, facedup_struct())])

    @memoize_method
    def flux_aux_info(self):
        discr = self.discr

        f_prop_blocks = []
        f_loc_blocks = []
        bdry_load_blocks = []

        from hedge.cuda.discretization import GPUBoundaryFaceStorage

        for block in discr.blocks:
            ldis = block.local_discretization
            el_dofs = ldis.node_count()
            face_dofs = ldis.face_node_count()

            f_prop_structs = []
            f_loc_structs = []
            bdry_load_structs = []
            fstorage_to_props_number = {}

            def get_face_props_number(fstorage):
                try:
                    return fstorage_to_props_number[fstorage]
                except KeyError:
                    result = len(f_prop_structs) << 1
                    fstorage_to_props_number[fstorage] = result
                    fstorage_to_props_number[fstorage.opposite] = result | 1

                    flux_face = fstorage.flux_face
                    f_prop_structs.append(flux_face_properties_struct(
                        discr.plan.float_type, discr.dimensions).make(
                            h=flux_face.h,
                            order=flux_face.order,
                            face_jacobian=flux_face.face_jacobian,
                            normal=flux_face.normal,
                            ))
                    return result

            for el in block.elements:
                for face_nbr in range(ldis.face_count()):
                    elface = el, face_nbr

                    int_face = discr.face_storage_map[elface]
                    opp = int_face.opposite

                    if isinstance(opp, GPUBoundaryFaceStorage):
                        # boundary face
                        b_base = discr.int_dof_floats+opp.dup_ext_face_number*face_dofs
                        b_ilist_number_or_flux = 1<<7

                        bdry_load_blocks.append(boundary_load_struct().make(
                            global_base=opp.gpu_bdry_index_in_floats,
                            smem_base=b_base,
                            reserved=0,
                            ))
                    else:
                        # interior face
                        if opp.native_block == int_face.native_block:
                            # same block
                            b_base = opp.native_block_el_num*el_dofs
                            b_ilist_number_or_flux = opp.native_index_list_id
                        else:
                            # different block
                            b_base = discr.int_dof_floats+opp.dup_ext_face_number*face_dofs
                            b_ilist_number_or_flux = 0

                    f_loc_structs.append(
                            flux_face_location_struct().make(
                                prop_block_number_and_side=get_face_props_number(int_face),
                                a_base=int_face.native_block_el_num*el_dofs,
                                b_base=b_base,
                                a_ilist_number=int_face.native_index_list_id,
                                b_ilist_number_or_flux=b_ilist_number_or_flux,
                                ))

            f_prop_blocks.append(f_prop_structs)
            f_loc_blocks.append(f_loc_structs)
            bdry_load_blocks.append(bdry_load_structs)

        from hedge.cuda.tools import make_blocks
        from pytools import Record
        result = Record(
                location=make_blocks(discr.devdata, f_loc_blocks),
                properties=make_blocks(discr.devdata, f_prop_blocks),
                boundary_load=make_blocks(discr.devdata, bdry_load_blocks),
                )

        assert result.properties.block_size <= discr.plan.flux_properties_smem()
        assert result.location.block_size <= discr.plan.flux_location_smem()
        
        return result

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
