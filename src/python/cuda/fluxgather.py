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
        POD(numpy.uint8, "boundary_id"),
        POD(numpy.uint8, "pad"), 
        POD(numpy.uint16, "a_dest"), 
        POD(numpy.uint16, "b_dest"), 
        ])



# flux to code mapper ---------------------------------------------------------
class FluxToCodeMapper(pymbolic.mapper.stringifier.StringifyMapper):
    def __init__(self, flip_normal=False):
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

    def map_power(self, expr, enclosing_prec):
        from pymbolic.mapper.stringifier import PREC_NONE
        from pymbolic.primitives import is_constant
        if is_constant(expr.exponent):
            if expr.exponent == 0:
                return "1"
            elif expr.exponent == 1:
                return self.rec(expr.base, enclosing_prec)
            elif expr.exponent == 2:
                return self.rec(expr.base*expr.base, enclosing_prec)
            else:
                return ("pow(%s, %s)" 
                        % (self.rec(expr.base, PREC_NONE), 
                        self.rec(expr.exponent, PREC_NONE)))
                return self.rec(expr.base*expr.base, enclosing_prec)
        else:
            return ("pow(%s, %s)" 
                    % (self.rec(expr.base, PREC_NONE), 
                    self.rec(expr.exponent, PREC_NONE)))

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





# flux gather kernel ----------------------------------------------------------
class FluxGatherKernel:
    def __init__(self, discr):
        self.discr = discr

    @memoize_method
    def get_kernel(self, wdflux):
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

        cmod = Module()

        for dep_expr in wdflux.all_deps:
            cmod.append(
                Value("texture<float, 1, cudaReadModeElementType>", 
                    "%s_tex" % wdflux.short_name(dep_expr)))

        cmod.extend([
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
            
        from hedge.cuda.tools import get_load_code
        f_body.extend(get_load_code(
            dest="smem_index_lists",
            base="gmem_index_lists",
            bytes="sizeof(index_list_entry_t)*INDEX_LISTS_LENGTH",
            descr="load index list data")
            )

        f_body.extend(get_load_code(
            dest="&data",
            base="gmem_data + blockIdx.x*DATA_BLOCK_SIZE",
            bytes="sizeof(flux_data)",
            descr="load face_pair data")
            +[ S("__syncthreads()"), Line() ])

        def bdry_flux_writer():
            def get_field(flux_rec, is_interior):
                if is_interior:
                    return ("tex1Dfetch(%s_tex, a_index)" 
                        % flux_rec.field_short_name)
                else:
                    return ("tex1Dfetch(%s_tex, b_index)" 
                        % flux_rec.bfield_short_name)

            from hedge.cuda.cgen import make_multiple_ifs
            from pymbolic.mapper.stringifier import PREC_NONE

            flux_write_code = Block([
                    Initializer(POD(float_type, "flux"), 0)
                    ])

            from pytools import flatten
            from hedge.tools import is_zero

            flux_write_code.append(
                make_multiple_ifs([
                    ("(fpair->boundary_id) == %d" % (bdry_id),
                        Block(list(flatten([
                            S("flux += /*%s*/ (%s) * %s"
                                % (is_interior and "int" or "ext",
                                    FluxToCodeMapper()(coeff, PREC_NONE),
                                    get_field(flux_rec, is_interior=is_interior)))
                                for is_interior, coeff, field_expr in [
                                    (True, flux_rec.int_coeff, flux_rec.field_expr),
                                    (False, flux_rec.ext_coeff, flux_rec.bfield_expr),
                                    ]
                                if not is_zero(field_expr)
                            ]
                            for flux_rec in fluxes
                            ))))
                    for bdry_id, fluxes in wdflux.bdry_id_to_fluxes.iteritems()
                    ])
                )

            flux_write_code.append(
                    Assign(
                        "smem_fluxes_on_faces[fpair->a_dest+FACEDOF_NR]",
                        "fpair->face_jacobian*flux"))

            return flux_write_code

        def int_flux_writer(is_twosided):
            def get_field(flux_rec, is_interior, flipped):
                if is_interior ^ flipped:
                    prefix = "a"
                else:
                    prefix = "b"

                return ("tex1Dfetch(%s_tex, %s_index)"
                        % (wdflux.short_name(flux_rec.field_expr), 
                            prefix))

            from hedge.cuda.cgen import make_multiple_ifs
            from pymbolic.mapper.stringifier import PREC_NONE

            flux_write_code = Block([
                Initializer(POD(float_type, "a_flux"), 0)
                ])

            if is_twosided:
                flux_write_code.append(
                        Initializer(POD(float_type, "b_flux"), 0))
                prefixes = ["a", "b"]
                flip_values = [False, True]
            else:
                prefixes = ["a"]
                flip_values = [False]

            flux_write_code.append(Line())

            for int_rec in wdflux.interiors:
                for prefix, is_flipped in zip(prefixes, flip_values):
                    for label, coeff in zip(
                            ["int", "ext"], 
                            [int_rec.int_coeff, int_rec.ext_coeff]):
                        flux_write_code.append(
                            S("%s_flux += /*%s*/ (%s) * %s"
                                % (prefix, label,
                                    FluxToCodeMapper(is_flipped)(coeff, PREC_NONE),
                                    get_field(int_rec, 
                                        is_interior=label =="int", 
                                        flipped=is_flipped)
                                    )))

            flux_write_code.append(Line())

            flux_write_code.append(
                    Assign(
                        "smem_fluxes_on_faces[fpair->a_dest+FACEDOF_NR]",
                        "fpair->face_jacobian*a_flux"))

            if is_twosided:
                flux_write_code.extend([
                    Initializer(Pointer(Value(
                        "index_list_entry_t", "b_write_ilist")),
                        "smem_index_lists + fpair->b_write_ilist_index"
                        ),
                    Assign(
                        "smem_fluxes_on_faces[fpair->b_dest+b_write_ilist[FACEDOF_NR]]",
                        "fpair->face_jacobian*b_flux")
                    ])

            return flux_write_code

        def get_flux_code(flux_writer):
            flux_code = Block([])

            from hedge.cuda.cgen import MaybeUnused

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
                    MaybeUnused(POD(numpy.uint32, "a_index")),
                    "fpair->a_base + a_ilist[FACEDOF_NR]"),
                Initializer(
                    MaybeUnused(POD(numpy.uint32, "b_index")),
                    "fpair->b_base + b_ilist[FACEDOF_NR]"),
                Line(),
                flux_writer(),
                Line(),
                S("fpair_nr += CONCURRENT_FACES")
                ])

            return flux_code

        f_body.extend_log_block("compute the fluxes", [
            Initializer(POD(numpy.uint16, "fpair_nr"), "BLOCK_FACE"),
            Comment("fluxes for dual-sided (intra-block) interior face pairs"),
            While("fpair_nr < data.header.same_facepairs_end",
                get_flux_code(lambda: int_flux_writer(True))
                ),
            Line(),
            Comment("work around nvcc assertion failure"),
            S("fpair_nr+=1"),
            S("fpair_nr-=1"),
            Line(),
            Comment("fluxes for single-sided (inter-block) interior face pairs"),
            While("fpair_nr < data.header.diff_facepairs_end",
                get_flux_code(lambda: int_flux_writer(False))
                ),
            Line(),
            Comment("fluxes for single-sided boundary face pairs"),
            While("fpair_nr < data.header.bdry_facepairs_end",
                get_flux_code(bdry_flux_writer)
                ),
            ])

        f_body.extend_log_block("store the fluxes", [
            S("__syncthreads()"),
            Line(),
            For("unsigned word_nr = THREAD_NUM", 
                "word_nr < ALIGNED_FACE_DOFS_PER_MB*BLOCK_MB_COUNT", 
                "word_nr += COALESCING_THREAD_COUNT",
                Block([
                    Assign(
                        "gmem_fluxes_on_faces[blockIdx.x*ALIGNED_FACE_DOFS_PER_BLOCK+word_nr]",
                        "smem_fluxes_on_faces[word_nr]"),
                    ])
                )
            ])

        # finish off ----------------------------------------------------------
        cmod.append(FunctionBody(f_decl, f_body))

        mod = cuda.SourceModule(cmod, 
                keep=True, 
                options=["--maxrregcount=16"]
                )
        print "flux: lmem=%d smem=%d regs=%d" % (mod.lmem, mod.smem, mod.registers)

        expr_to_texture_map = dict(
                (dep_expr, mod.get_texref(
                    "%s_tex" % wdflux.short_name(dep_expr)))
                for dep_expr in wdflux.all_deps)

        texrefs = expr_to_texture_map.values()

        return mod.get_function("apply_flux"), texrefs, expr_to_texture_map

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
            mb_index, index_in_mb = divmod(num_in_block,  fplan.microblock.elements)
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
                    boundary_id = wdflux.boundary_elface_to_bdry_id(
                            a_face.el_face)
                    b_write_index_list = 0 # doesn't matter
                    b_dest = INVALID_DEST
                    print>>outf, "bdy%d" % boundary_id

                    fp_structs = bdry_fp_structs
                else:
                    # interior face
                    b_base = discr.find_el_gpu_index(b_face.el_face[0])
                    boundary_id = 0

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

                            boundary_id=boundary_id,
                            pad=0,
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

        from pytools import flatten
        flat_ilists = numpy.array(
                list(flatten(discr.index_lists)),
                dtype=tp)

        from pytools import Record
        class GPUIndexLists(Record): pass

        return GPUIndexLists(
                code=[
                    Define("INDEX_LISTS_LENGTH", len(flat_ilists)),
                    Typedef(POD(tp, "index_list_entry_t")),
                    ],
                device_memory=cuda.to_device(flat_ilists)
                )
