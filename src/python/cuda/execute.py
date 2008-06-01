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
def block_header_struct():
    from hedge.cuda.cgen import Struct, POD

    return Struct("block_header", [
        POD(numpy.uint16, "els_in_block"),
        POD(numpy.uint16, "facedup_writes_in_block"),
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
        POD(numpy.uint16, "prop_block_number_and_flux_and_side"), 
        # bit 0: flip normal
        # bit 1..3: flux_number
        # bit 4..: prop_block_number
        POD(numpy.uint16, "a_base"),
        POD(numpy.uint16, "b_base"),

        POD(numpy.uint8, "a_ilist_number"),
        POD(numpy.uint8, "b_ilist_number"), 
        ])

@memoize
def facedup_read_struct():
    from hedge.cuda.cgen import Struct, POD

    return Struct("facedup_read", [
        POD(numpy.uint16, "smem_base"),
        POD(numpy.uint8, "reserved"),
        POD(numpy.uint8, "global_ilist_number"),
        POD(numpy.uint32, "global_base"),
        ])
    
@memoize
def facedup_write_struct():
    from hedge.cuda.cgen import Struct, POD

    return Struct("facedup_write", [
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



# flux to code mapper ---------------------------------------------------------
class FluxToCodeMapper(pymbolic.mapper.stringifier.StringifyMapper):
    def map_normal(self, expr, enclosing_prec):
        return "normal[%d]" % expr.axis

    def map_penalty_term(self, expr, enclosing_prec):
        return "pow(fprops->order*fprops->order/fprops->h, %r)" % expr.power

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
        localop_data = self.ex.localop_data(op, eg)
        func, texref = self.ex.get_diff_kernel(op.__class__, eg)

        lop_par = discr.plan.find_localop_par()
        
        kwargs = {
                "texrefs": [texref], 
                "block": (discr.plan.dofs_per_el(), lop_par.p, 1),
                "grid": (len(discr.blocks), 1)
                }

        field = self.rec(field_expr)
        assert field.dtype == discr.plan.float_type

        xyz_diff = [discr.volume_empty() for axis in range(d)]
        elgroup, = discr.element_groups
        args = xyz_diff+[field, localop_data.device_memory]

        func(*args, **kwargs)

        if False:
            f = discr.volume_from_gpu(field)
            dx = discr.volume_from_gpu(xyz_diff[0], check=True)
            
            test_discr = discr.test_discr
            real_dx = test_discr.nabla[0].apply(f.astype(numpy.float64))
            
            diff = dx - real_dx

            rel_err_norm = la.norm(diff)/la.norm(real_dx)
            assert rel_err_norm < 1e-5
        
        self.diff_xyz_cache[op.__class__, field_expr] = xyz_diff
        return xyz_diff[op.xyz_axis]

    def map_whole_domain_flux(self, op, field_expr, out=None):
        field = self.rec(field_expr)
        discr = self.ex.discr

        eg, = discr.element_groups
        fdata = self.ex.flux_data(op, eg)
        mod, func, texrefs = self.ex.get_flux_kernel(op)

        debugbuf = gpuarray.zeros((512,), dtype=numpy.float32)

        flux_par = discr.plan.flux_par
        
        #print "lmem=%d smem=%d regs=%d" % (mod.lmem, mod.smem, mod.registers)
        kwargs = {
                "texrefs": texrefs, 
                "block": (discr.plan.dofs_per_el(), flux_par.p, 1),
                "grid": (len(discr.blocks), 1)
                }

        flux = discr.volume_zeros() 
        bfield = None
        for boundary in op.boundaries:
            if bfield is None:
                bfield = self.rec(boundary.bfield_expr)
            else:
                bfield = bfield + self.rec(boundary.bfield_expr)
            
        assert field.dtype == discr.plan.float_type
        assert bfield.dtype == discr.plan.float_type

        args = [debugbuf, flux, field, bfield, fdata.device_memory]

        func(*args, **kwargs)

        if False:
            copied_debugbuf = debugbuf.get()
            print "DEBUG"
            print numpy.reshape(copied_debugbuf, (len(copied_debugbuf)//16, 16))

        if False:
            cot = discr.test_discr.compile(op.flux_optemplate)
            ctx = {field_expr.name: 
                    discr.volume_from_gpu(field, check=True).astype(numpy.float64)
                    }
            for boundary in op.boundaries:
                ctx[boundary.bfield_expr.name] = \
                        discr.test_discr.boundary_zeros(boundary.tag)
            true_flux = cot(**ctx)
            
            copied_flux = discr.volume_from_gpu(flux, check=True)

            diff = copied_flux-true_flux

            norm_true = la.norm(true_flux)

            if False:
                numpy.seterr(all="ignore")
                struc = ""
                for block in discr.blocks:
                    for i_el, el in enumerate(block.elements):
                        s = discr.find_el_range(el.id)
                        relerr = la.norm(diff[s])/norm_true
                        if relerr > 1e-4:
                            struc += "*"
                            if True:
                                print "block %d, el %d, global el #%d, rel.l2err=%g" % (
                                        block.number, i_el, el.id, relerr)
                                print copied_flux[s]
                                print true_flux[s]
                                print diff[s]
                                raw_input()
                        elif numpy.isnan(relerr):
                            struc += "N"
                            if True:
                                print "block %d, el %d, global el #%d, rel.l2err=%g" % (
                                        block.number, i_el, el.id, relerr)
                                print copied_flux[s]
                                print true_flux[s]
                                print diff[s]
                                raw_input()
                        else:
                            if numpy.max(numpy.abs(true_flux[s])) == 0:
                                struc += "0"
                            else:
                                struc += "."
                    struc += "\n"
                print
                print struc

            print la.norm(diff)/norm_true
            assert la.norm(diff)/norm_true < 1e-6

        if False:
            copied_bfield = bfield.get()
            face_len = discr.plan.ldis.face_node_count()
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

        lop_par = discr.plan.find_localop_par()
        lop_data = self.localop_data(diff_op_cls, elgroup)
        rst2xyz_coeffs_size_unaligned = d*d*lop_par.total()
        elgroup, = discr.element_groups

        float_type = discr.plan.float_type

        f_decl = CudaGlobal(FunctionDeclaration(Value("void", "apply_diff_mat"), 
            [Pointer(POD(float_type, "dxyz%d" % i)) for i in dims]
            + [
                Pointer(POD(float_type, "field")),
                Pointer(POD(numpy.uint8, "gmem_data")),
                ]
            ))

        cmod = Module([
                Value("texture<float%d, 2, cudaReadModeElementType>" 
                    % self.diffmat_channels(), "diff_rst_matrices"),
                block_header_struct(),
                facedup_write_struct(),
                Define("EL_DOF", "threadIdx.x"),
                Define("BLOCK_EL", "threadIdx.y"),
                Define("DOFS_PER_EL", discr.plan.dofs_per_el()),
                Define("CONCURRENT_ELS", lop_par.p),
                Define("DOFS_BLOCK_BASE", "(blockIdx.x*BLOCK_DOF_COUNT)"),
                Define("THREAD_NUM", "(BLOCK_EL*DOFS_PER_EL + EL_DOF)"),
                Define("THREAD_COUNT", "(DOFS_PER_EL*CONCURRENT_ELS)"),
                Define("BLOCK_DOF_COUNT", discr.block_dof_count()),
                Define("DATA_BLOCK_SIZE", lop_data.block_size),
                Line(),
                Comment("face-related stuff"),
                Define("DOFS_PER_FACE", discr.plan.dofs_per_face()),
                Define("CONCURRENT_FACES", 
                    discr.plan.dofs_per_el()*lop_par.p
                    //discr.plan.dofs_per_face()),
                Line(),
                ] + self.index_list_data() + [
                Line(),
                lop_data.struct,
                CudaShared(Value("localop_data", "data")),
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
            descr="load localop_data")
            +[ S("__syncthreads()"), Line() ])

        f_body.extend_log_block("load internal dofs", [
            For("unsigned dof_nr = THREAD_NUM", 
                "dof_nr < data.header.els_in_block*DOFS_PER_EL", 
                "dof_nr += THREAD_COUNT",
                S("int_dofs[dof_nr] = field[DOFS_BLOCK_BASE+dof_nr]"),
                ),
            ])

        f_body.extend([
            S("__syncthreads()"),
            Line()
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
                For("unsigned short j = 0", "j < DOFS_PER_EL", "++j", Block([
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
                            "data.rst_to_xyz_coefficients[%d + %d*(%s)]"
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
            For("unsigned short face_nr = block_face",
                "face_nr < data.header.facedup_writes_in_block",
                "face_nr += CONCURRENT_FACES", Block([
                    Initializer(POD(numpy.uint16, "face_el_dof"),
                        "const_index_lists["
                        "data.facedup_writes[face_nr].smem_face_ilist_number*DOFS_PER_FACE"
                        "+face_dof"
                        "]"),
                    Initializer(POD(numpy.uint16, "face_el_nr"),
                        "data.facedup_writes[face_nr].smem_element_number"),
                    Initializer(POD(numpy.uint32, "tgt_dof"),
                        "data.facedup_writes[face_nr].dup_global_base+face_dof"),
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

        texref = mod.get_texref("diff_rst_matrices")

        cuda.bind_array_to_texref(
                self.diffmat_array(diff_op_cls, elgroup),
                texref)

        return mod.get_function("apply_diff_mat"), texref




    # flux kernel -------------------------------------------------------------
    @memoize_method
    def get_flux_kernel(self, wdflux):
        from hedge.cuda.cgen import \
                Pointer, POD, Value, ArrayOf, Const, \
                Module, FunctionDeclaration, FunctionBody, Block, \
                Comment, Line, \
                CudaShared, CudaGlobal, Static, \
                Define, Pragma, \
                Constant, Initializer, If, For, Statement, Assign
                
        discr = self.discr
        d = discr.dimensions
        dims = range(d)

        flux_par = discr.plan.find_localop_par()
        elgroup, = discr.element_groups
        flux_data = self.flux_data(wdflux, elgroup)

        float_type = discr.plan.float_type

        f_decl = CudaGlobal(FunctionDeclaration(Value("void", "apply_flux"), 
            [
                Pointer(POD(float_type, "debugbuf")),
                Pointer(POD(float_type, "flux")),
                Pointer(POD(float_type, "field")),
                Pointer(POD(float_type, "bfield")),
                Pointer(POD(numpy.uint8, "gmem_data")),
                ]
            ))

        coalesced_dofs_per_face = discr.devdata.coalesce(
                discr.plan.dofs_per_face())

        cmod = Module([
                Value("texture<float, 2, cudaReadModeElementType>", 
                    "lift_matrix_tex"),
                block_header_struct(),
                facedup_write_struct(),
                flux_face_properties_struct(float_type, discr.dimensions),
                flux_face_location_struct(),
                boundary_load_struct(),
                Line(),
                Define("EL_DOF", "threadIdx.x"),
                Define("BLOCK_EL", "threadIdx.y"),
                Define("DOFS_PER_EL", discr.plan.dofs_per_el()),
                Define("CONCURRENT_ELS", flux_par.p),
                Define("DOFS_BLOCK_BASE", "(blockIdx.x*BLOCK_DOF_COUNT)"),
                Define("THREAD_NUM", "(BLOCK_EL*DOFS_PER_EL + EL_DOF)"),
                Define("THREAD_COUNT", "(DOFS_PER_EL*CONCURRENT_ELS)"),
                Define("BLOCK_DOF_COUNT", discr.block_dof_count()),
                Define("DATA_BLOCK_SIZE", flux_data.block_size),
                Line(),
                Comment("face-related stuff"),
                Define("DOFS_PER_FACE", discr.plan.dofs_per_face()),
                Define("FACES_PER_EL", discr.plan.faces_per_el()),
                Define("COALESCED_DOFS_PER_FACE", 
                    coalesced_dofs_per_face),
                Define("CONCURRENT_FACES", 
                    discr.plan.dofs_per_el()*flux_par.p
                    //discr.plan.dofs_per_face()),
                Define("CONCURRENT_COALESCED_FACES", 
                    discr.plan.dofs_per_el()*flux_par.p
                    //coalesced_dofs_per_face),
                Line(),
                ] + self.index_list_data() + [
                Line(),
                self.lift_matrix_initializer(discr.plan.ldis),
                Line(),
                flux_data.struct,
                Line(),
                CudaShared(POD(discr.plan.float_type, "dummy_dest")),
                #CudaShared(ArrayOf(
                    #POD(numpy.uint32, "shared_index_lists"),
                    #"INDEX_LISTS_LENGTH")),
                CudaShared(Value("flux_data", "data")),
                CudaShared(ArrayOf(POD(float_type, "dofs"),
                    "BLOCK_DOF_COUNT")),
                Line(),
                ])

        S = Statement
        f_body = Block()
            
        f_body.extend(self.get_load_code(
            dest="&data",
            base="gmem_data + blockIdx.x*DATA_BLOCK_SIZE",
            bytes="sizeof(flux_data)",
            descr="load flux_data")
            +[ S("__syncthreads()"), Line() ])

        f_body.extend_log_block("load dofs", [
            For("unsigned dof_nr = THREAD_NUM", 
                "dof_nr < BLOCK_DOF_COUNT", 
                "dof_nr += THREAD_COUNT",
                S("dofs[dof_nr] = field[DOFS_BLOCK_BASE+dof_nr]"),
                ),
            Line(),
            Comment("avoid write-after-write hazard with boundary load"),
            S("__syncthreads()"),
            ])

        f_body.extend_log_block("load boundary dofs", [Block([
            Initializer(Const(POD(numpy.int16, "block_face")),
                "THREAD_NUM / COALESCED_DOFS_PER_FACE"),
            Initializer(Const(POD(numpy.int16, "face_dof")),
                "THREAD_NUM - COALESCED_DOFS_PER_FACE*block_face"),
            If("face_dof < DOFS_PER_FACE && block_face < CONCURRENT_COALESCED_FACES",
                Block([
                    For("unsigned bdry_nr = block_face",
                        "bdry_nr < data.header.boundaries_in_block",
                        "bdry_nr += CONCURRENT_COALESCED_FACES",
                        Block([
                            Assign(
                                "dofs[data.boundary_loads[bdry_nr].smem_base + face_dof]",
                                "bfield[data.boundary_loads[bdry_nr].global_base + face_dof]"),
                            ])
                        )
                    ])
                )
            ]),
            Line(),
            S("__syncthreads()"),
            ])

        if False:
            f_body.extend(self.get_load_code(
                dest="shared_index_lists",
                base="const_index_lists",
                bytes="sizeof(const_index_lists)",
                descr="load index lists into shared")
                +[ S("__syncthreads()"), Line() ])

        if False:
            f_body.extend([
                Block([
                    For("unsigned word_nr = THREAD_NUM", 
                        "word_nr < INDEX_LISTS_LENGTH", 
                        "word_nr += THREAD_COUNT",
                        Statement("shared_index_lists[word_nr] = const_index_lists[word_nr]")
                        ),
                    ]),
                Line(),
                S("__syncthreads()"),
                ])
            
        # actually do the flux computation ------------------------------------
        def flux_coefficients_getter():
            from hedge.cuda.cgen import make_multiple_ifs
            from pymbolic.mapper.stringifier import PREC_NONE
            return [
                Initializer(
                    POD(numpy.uint8, "flux_number"),
                    "(floc->prop_block_number_and_flux_and_side>>1) & 7"),
                ArrayOf(POD(discr.plan.float_type, "normal"),
                    discr.dimensions),
                If("floc->prop_block_number_and_flux_and_side & 1",
                    Block([Assign("normal[%d]" % i, "-fprops->normal[%d]" % i)
                        for i in range(discr.dimensions)]),
                    Block([Assign("normal[%d]" % i, "fprops->normal[%d]" % i)
                        for i in range(discr.dimensions)])
                    ),
                make_multiple_ifs(
                    [
                    ("flux_number == %d" % flux_nr,
                        Block([
                            Assign("int_coeff", 
                                FluxToCodeMapper(repr)(int_coeff, PREC_NONE),
                                ),
                            Assign("ext_coeff", 
                                FluxToCodeMapper(repr)(ext_coeff, PREC_NONE),
                                ),
                            ])
                        )
                    for flux_nr, (int_coeff, ext_coeff)
                    in enumerate(wdflux.fluxes)
                    ],
                    base= Block([
                        Assign("int_coeff", 0),
                        Assign("ext_coeff", 0),
                        ])
                    ),
                S("int_coeff *= fprops->face_jacobian"),
                S("ext_coeff *= fprops->face_jacobian"),
                ]

        def add_up_flux_on_faces(**kwargs):
            face_loop_body = Block([
                Initializer(Pointer(Value(
                    "flux_face_location", "floc")),
                    "data.face_locations+((%(el_nr)s)*FACES_PER_EL+face_nr)" % kwargs),
                Initializer(Pointer(Value(
                    "flux_face_properties", "fprops")),
                    "data.face_properties"
                    "+(floc->prop_block_number_and_flux_and_side>>4)"),
                POD(discr.plan.float_type, "int_coeff"),
                POD(discr.plan.float_type, "ext_coeff"),
                ]+flux_coefficients_getter()+[
                Initializer(Pointer(Value(
                    "index_list_entry_t", "ilist_a")),
                    "const_index_lists + floc->a_ilist_number*DOFS_PER_FACE"
                    ),
                Initializer(Pointer(Value(
                    "index_list_entry_t", "ilist_b")),
                    "const_index_lists + floc->b_ilist_number*DOFS_PER_FACE"
                    ),
                Initializer(POD(numpy.uint16, "a_base"), "floc->a_base"),
                Initializer(POD(numpy.uint16, "b_base"), "floc->b_base"),
                For("unsigned short facedof_nr = 0",
                    "facedof_nr < DOFS_PER_FACE",
                    "++facedof_nr",
                    S("result += "
                        "tex2D(lift_matrix_tex, (%(el_dof)s), facedof_nr+face_nr*DOFS_PER_FACE)"
                        #"lift_matrix[(%(el_dof)s)][facedof_nr+face_nr*DOFS_PER_FACE]"
                        "* ("
                        "int_coeff*dofs[a_base + ilist_a[facedof_nr]]"
                        "+"
                        "ext_coeff*dofs[b_base + ilist_b[facedof_nr]]"
                        ")" % kwargs),
                    )
                ])

            return For("unsigned short face_nr = 0",
                            "face_nr < FACES_PER_EL",
                            "++face_nr", face_loop_body)

        f_body.extend_log_block("compute the fluxes", [
                For("unsigned short el_nr = BLOCK_EL",
                    "el_nr < data.header.els_in_block",
                    "el_nr += CONCURRENT_ELS",
                    Block([
                        Initializer(POD(discr.plan.float_type, "result"), "0"),
                        add_up_flux_on_faces(
                            el_dof="EL_DOF",
                            el_nr="el_nr",
                            ),
                        Assign(
                            "flux[DOFS_BLOCK_BASE+el_nr*DOFS_PER_EL+EL_DOF]",
                            #"shared_debug",
                            "data.inverse_jacobians[el_nr] * result"),
                        ])
                    )
                ])

        f_body.extend_log_block("compute the fluxes on dup faces", [
            Initializer(Const(POD(numpy.int16, "block_face")),
                "THREAD_NUM / DOFS_PER_FACE"),
            Initializer(Const(POD(numpy.int16, "face_dof")),
                "THREAD_NUM - DOFS_PER_FACE*block_face"),
            For("unsigned short dest_face_nr = block_face",
                "dest_face_nr < data.header.facedup_writes_in_block",
                "dest_face_nr += CONCURRENT_FACES", Block([
                    Initializer(POD(numpy.uint16, "face_el_dof"),
                        "const_index_lists["
                        "data.facedup_writes[dest_face_nr].smem_face_ilist_number*DOFS_PER_FACE"
                        "+face_dof"
                        "]"),
                    Initializer(POD(discr.plan.float_type, "result"), "0"),
                    add_up_flux_on_faces(
                        el_dof="face_el_dof",
                        el_nr="data.facedup_writes[dest_face_nr].smem_element_number",
                        ),
                    Assign(
                        "flux[data.facedup_writes[dest_face_nr].dup_global_base+face_dof]",
                        #"shared_debug",
                        "data.inverse_jacobians[data.facedup_writes[dest_face_nr].smem_element_number]"
                        "* result"),
                    ])
                )
            ])

        # finish off ----------------------------------------------------------
        cmod.append(FunctionBody(f_decl, f_body))

        mod = cuda.SourceModule(cmod, 
                keep=True, 
                #options=["--maxrregcount=16"]
                )

        texref = mod.get_texref("lift_matrix_tex")
        if wdflux.is_lift:
            cuda.matrix_to_texref(discr.plan.ldis.lifting_matrix(), texref)
        else:
            cuda.matrix_to_texref(discr.plan.ldis.multi_face_mass_matrix(), texref)
        texrefs = [texref]

        return mod, mod.get_function("apply_flux"), texrefs

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

    @memoize_method
    def localop_rst_to_xyz(self, diff_op, elgroup):
        discr = self.discr
        d = discr.dimensions

        floats_per_block = d*d*discr.plan.find_localop_par().total()
        bytes_per_block = floats_per_block*discr.plan.float_size

        coeffs = diff_op.coefficients(elgroup)

        blocks = []
        
        def get_el_index_in_el_group(el):
            mygroup, idx = discr.group_map[el.id]
            assert mygroup is elgroup
            return idx

        from hedge.cuda.tools import pad
        for block in discr.blocks:
            block_elgroup_indices = numpy.fromiter(
                    (get_el_index_in_el_group(el) for el in block.elements),
                    dtype=numpy.intp)

            flattened = (coeffs[:,:,block_elgroup_indices]
                .transpose(2,0,1).flatten().astype(discr.plan.float_type))
            blocks.append(pad(str(buffer(flattened)), bytes_per_block))
                
        from hedge.cuda.cgen import POD, ArrayOf
        return blocks, ArrayOf(
                POD(discr.plan.float_type, "rst_to_xyz_coefficients"),
                floats_per_block)

    @memoize_method
    def flux_inverse_jacobians(self, elgroup):
        discr = self.discr
        d = discr.dimensions

        floats_per_block = discr.plan.flux_par.total()
        bytes_per_block = floats_per_block*discr.plan.float_size

        inv_jacs = elgroup.inverse_jacobians

        blocks = []
        
        def get_el_index_in_el_group(el):
            mygroup, idx = discr.group_map[el.id]
            assert mygroup is elgroup
            return idx

        from hedge.cuda.tools import pad
        for block in discr.blocks:
            block_elgroup_indices = numpy.fromiter(
                    (get_el_index_in_el_group(el) for el in block.elements),
                    dtype=numpy.intp)

            block_inv_jacs = (inv_jacs[block_elgroup_indices].copy().astype(discr.plan.float_type))
            blocks.append(pad(str(buffer(block_inv_jacs)), bytes_per_block))
                
        from hedge.cuda.cgen import POD, ArrayOf
        return blocks, ArrayOf(
                POD(discr.plan.float_type, "inverse_jacobians"),
                floats_per_block)

    @memoize_method
    def facedup_blocks(self):
        discr = self.discr
        headers = []
        read_blocks = []
        write_blocks = []

        from hedge.cuda.discretization import GPUInteriorFaceStorage
        for block in discr.blocks:
            ldis = block.local_discretization
            el_dofs = ldis.node_count()
            face_dofs = ldis.node_count()
            block_boundary_count = 0

            read_structs = []
            write_structs = []
            for extface in block.ext_faces_from_me:
                assert extface.native_block is block
                assert isinstance(extface, GPUInteriorFaceStorage)
                if isinstance(extface.opposite, GPUInteriorFaceStorage):
                    write_structs.append(facedup_write_struct().make(
                            smem_element_number=extface.native_block_el_num,
                            face_number=extface.el_face[1],
                            smem_face_ilist_number=extface.native_index_list_id,
                            dup_global_base=extface.dup_global_base))
                else:
                    block_boundary_count += 1

            headers.append(block_header_struct().make(
                        els_in_block=len(block.elements),
                        facedup_writes_in_block=len(write_structs),
                        boundaries_in_block=block_boundary_count,
                        reserved=0,
                        ))
            read_blocks.append(read_structs)
            write_blocks.append(write_structs)

        return headers, read_blocks, write_blocks

    @memoize_method
    def localop_data(self, op, elgroup):
        discr = self.discr
        headers, read_facedups, write_facedups = self.facedup_blocks()

        from hedge.cuda.cgen import Value
        from hedge.cuda.tools import make_superblocks

        return make_superblocks(
                discr.devdata, "localop_data",
                [(headers, Value(block_header_struct().tpname, "header")),
                    self.localop_rst_to_xyz(op, elgroup)
                    ],
                [(write_facedups, Value(facedup_write_struct().tpname, "facedup_writes"))],
                    )

    @memoize_method
    def flux_data(self, wdflux, elgroup):
        discr = self.discr

        f_prop_blocks = []
        f_loc_blocks = []
        bdry_load_blocks = []

        from hedge.cuda.discretization import GPUBoundaryFaceStorage

        outf = open("el_faces.txt", "w")
        for block in discr.blocks:
            ldis = block.local_discretization
            el_dofs = ldis.node_count()
            face_dofs = ldis.face_node_count()

            f_prop_structs = []
            f_loc_structs = []
            bdry_load_structs = []
            fstorage_to_props_number = {}

            def get_face_props_number(fstorage, flux_number):
                try:
                    result = fstorage_to_props_number[fstorage] 
                except KeyError:
                    result = len(f_prop_structs) << 4
                    fstorage_to_props_number[fstorage] = result
                    fstorage_to_props_number[fstorage.opposite] = result | 1

                    flux_face = fstorage.face_pair_side
                    f_prop_structs.append(flux_face_properties_struct(
                        discr.plan.float_type, discr.dimensions).make(
                            h=flux_face.h,
                            order=flux_face.order,
                            face_jacobian=flux_face.face_jacobian,
                            normal=flux_face.normal,
                            ))

                assert 0 <= flux_number < 1<<3
                return result | flux_number << 1

            for i_el, el in enumerate(block.elements):
                print>>outf, "block %d el %d (global:%d):" % (block.number, i_el, el.id),
                for face_nbr in range(ldis.face_count()):
                    elface = el, face_nbr

                    int_face = discr.face_storage_map[elface]
                    opp = int_face.opposite

                    if isinstance(opp, GPUBoundaryFaceStorage):
                        # boundary face
                        flux_number = wdflux.elface_to_flux_number(
                                int_face.el_face)

                        b_base = discr.int_dof_floats+opp.dup_ext_face_number*face_dofs

                        bdry_load_structs.append(boundary_load_struct().make(
                            global_base=opp.gpu_bdry_index_in_floats,
                            smem_base=b_base,
                            reserved=0,
                            ))
                        print>>outf, "bdy%d" % flux_number,
                    else:
                        # interior face
                        flux_number = wdflux.interior_flux_number

                        if opp.native_block == int_face.native_block:
                            # same block
                            b_base = opp.native_block_el_num*el_dofs
                            print>>outf, "same",
                        else:
                            # different block
                            b_base = discr.int_dof_floats+opp.dup_ext_face_number*face_dofs
                            print>>outf, "diff",

                    f_loc_structs.append(
                            flux_face_location_struct().make(
                                prop_block_number_and_flux_and_side
                                =get_face_props_number(int_face, flux_number),
                                a_base=int_face.native_block_el_num*el_dofs,
                                b_base=b_base,
                                a_ilist_number=int_face.int_flux_index_list_id,
                                b_ilist_number=int_face.ext_flux_index_list_id,
                                ))
                print>>outf

            f_prop_blocks.append(f_prop_structs)
            f_loc_blocks.append(f_loc_structs)
            bdry_load_blocks.append(bdry_load_structs)
        
        headers, facedup_read_blocks, facedup_write_blocks = \
                self.facedup_blocks()

        from hedge.cuda.cgen import Value
        from hedge.cuda.tools import make_superblocks

        return make_superblocks(
                discr.devdata, "flux_data",
                [
                    (headers, Value(block_header_struct().tpname, "header")),
                    self.flux_inverse_jacobians(elgroup),
                    ],
                [
                    (facedup_write_blocks, 
                        Value(facedup_write_struct().tpname, "facedup_writes")),
                    (f_prop_blocks, 
                        Value(flux_face_properties_struct(
                            discr.plan.float_type, discr.dimensions
                            ).tpname, 
                            "face_properties")),
                    (f_loc_blocks, 
                        Value(flux_face_location_struct().tpname, 
                            "face_locations")),
                    (bdry_load_blocks, 
                        Value(boundary_load_struct().tpname, 
                            "boundary_loads")),
                    ])

    def const_matrix_initializer(self, matrix, name):
        from hedge.cuda.cgen import ArrayInitializer, ArrayOf, \
                Typedef, POD, CudaConstant

        return ArrayInitializer(
                CudaConstant(
                        ArrayOf(
                            ArrayOf(
                                POD(self.discr.plan.float_type, name),
                                matrix.shape[0],
                                ),
                            matrix.shape[1]
                            )
                        ),
                ["{ %s }" % ", ".join(repr(entry) for entry in row)
                    for row in matrix]
                )

    @memoize_method
    def lift_matrix_initializer(self, ldis):
        return self.const_matrix_initializer(
                ldis.lifting_matrix(), "lift_matrix")

    @memoize_method
    def index_list_data(self):
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
        flat_ilists = list(flatten(discr.index_lists))
        return [
                Define("INDEX_LISTS_LENGTH", len(flat_ilists)),
                Typedef(POD(tp, "index_list_entry_t")),
                ArrayInitializer(
                    CudaConstant(
                        ArrayOf(Value(
                            "index_list_entry_t", 
                            "const_index_lists"),
                        "INDEX_LISTS_LENGTH")),
                    flat_ilists
                    )
                ]
