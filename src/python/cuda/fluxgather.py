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
from pytools import memoize_method, memoize, Record
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import pymbolic.mapper.stringifier
import hedge.cuda.plan




class GPUIndexLists(Record): pass




# structures ------------------------------------------------------------------
@memoize
def flux_header_struct():
    from hedge.cuda.cgen import GenerableStruct, POD

    return GenerableStruct("flux_header", [
        POD(numpy.uint16, "same_facepairs_end"),
        POD(numpy.uint16, "diff_facepairs_end"),
        POD(numpy.uint16, "bdry_facepairs_end"),
        ], align_bytes=4)

@memoize
def face_pair_struct(float_type, dims):
    from hedge.cuda.cgen import GenerableStruct, POD, ArrayOf
    return GenerableStruct("face_pair", [
        POD(float_type, "h", ),
        POD(float_type, "order"),
        POD(float_type, "face_jacobian"),
        ArrayOf(POD(float_type, "normal"), dims),

        POD(numpy.uint32, "a_base"),
        POD(numpy.uint32, "b_base"),

        POD(numpy.uint16, "a_ilist_index"),
        POD(numpy.uint16, "b_ilist_index"), 
        POD(numpy.uint16, "b_write_ilist_index"), 
        POD(numpy.uint8, "boundary_bitmap"),
        POD(numpy.uint8, "pad"), 
        POD(numpy.uint16, "a_dest"), 
        POD(numpy.uint16, "b_dest"), 
        ], align_bytes=4, aligned_prime_to=[2])




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





# plan ------------------------------------------------------------------------
class FluxGatherPlan(hedge.cuda.plan.ExecutionPlan):
    def __init__(self, given, 
            parallel_faces, mbs_per_block, flux_count,
            direct_store=False, max_face_pair_count=None,
            ):
        hedge.cuda.plan.ExecutionPlan.__init__(self, given)
        self.parallel_faces = parallel_faces
        self.mbs_per_block = mbs_per_block
        self.flux_count = flux_count
        self.direct_store = direct_store

        self.max_face_pair_count = max_face_pair_count

    def copy(self, given=None,
            parallel_faces=None, mbs_per_block=None, flux_count=None,
            direct_store=None, max_face_pair_count=None):
        def default_if_none(a, default):
            if a is None:
                return default
            else:
                return a

        return self.__class__(
                default_if_none(given, self.given),
                default_if_none(parallel_faces, self.parallel_faces),
                default_if_none(mbs_per_block, self.mbs_per_block),
                default_if_none(flux_count, self.flux_count),
                default_if_none(direct_store, self.direct_store),
                default_if_none(max_face_pair_count, self.max_face_pair_count),
                )

    def microblocks_per_block(self):
        return self.mbs_per_block

    def elements_per_block(self):
        return self.microblocks_per_block()*self.given.microblock.elements

    def dofs_per_block(self):
        return self.microblocks_per_block()*self.given.microblock.aligned_floats

    @memoize_method
    def estimate_extface_count(self):
        d = self.given.ldis.dimensions

        # How many equivalent cubes would I need to tesselate the same space
        # as the elements in my thread block?
        from pytools import factorial
        equiv_cubes = self.elements_per_block() / factorial(d)

        # If these cubes in turn formed a perfect macro-cube, how long would
        # its side be?
        macrocube_side = equiv_cubes ** (1/d)

        # What total face area does the macro-cube have?
        macrocube_face_area = 2*d * macrocube_side ** (d-1)

        # How many of my faces do I need to tesselate this face area?
        return macrocube_face_area * factorial(d-1)

    @memoize_method
    def face_pair_count(self):
        if self.max_face_pair_count is None:
            from hedge.cuda.tools import int_ceiling
            ext_face_count = int_ceiling(self.estimate_extface_count())
            int_face_count = (self.elements_per_block() * self.given.faces_per_el() - 
                    ext_face_count)
            return (int_face_count+1) // 2 + ext_face_count
        else:
            return self.max_face_pair_count

    @memoize_method
    def shared_mem_use(self):
        from hedge.cuda.fluxgather import face_pair_struct
        d = self.given.ldis.dimensions

        if self.given.dofs_per_face() > 255:
            index_lists_entry_size = 2
        else:
            index_lists_entry_size = 1

        result = (128 # parameters, block header, small extra stuff
                + len(face_pair_struct(self.given.float_type, d))
                * self.face_pair_count())

        if not self.direct_store:
            result += (self.given.aligned_face_dofs_per_microblock()
                * self.flux_count
                * self.microblocks_per_block()
                * self.given.float_size())

        return result

    def threads_per_face(self):
        dpf = self.given.dofs_per_face()

        devdata = self.given.devdata
        if dpf % devdata.smem_granularity >= devdata.smem_granularity // 2:
            from hedge.cuda.tools import int_ceiling
            return int_ceiling(dpf, devdata.smem_granularity)
        else:
            return dpf

    def threads(self):
        return self.parallel_faces*self.threads_per_face()

    def registers(self):
        return 51

    def __str__(self):
        result = ("%s pfaces=%d mbs_per_block=%d mb_elements=%d" % (
            hedge.cuda.plan.ExecutionPlan.__str__(self),
            self.parallel_faces,
            self.mbs_per_block,
            self.given.microblock.elements,
            ))

        if self.direct_store:
            result += " direct_store"

        return result

    def make_kernel(self, discr, elface_to_bdry_bitmap, fluxes):
        return FluxGatherKernel(discr, self, elface_to_bdry_bitmap, fluxes)





def make_plan(discr, given, tune_for):
    from hedge.cuda.execute import OpTemplateWithEnvironment
    from hedge.cuda.optemplate import FluxCollector
    fluxes = FluxCollector()(
            OpTemplateWithEnvironment.compile_optemplate(
                discr.mesh, tune_for))
    flux_count = len(fluxes)

    from hedge.cuda.plan import optimize_plan

    def generate_valid_plans():
        #for direct_store in [True, False]:
        for direct_store in [False]:
            for parallel_faces in range(1,32):
                for mbs_per_block in range(1,8):
                    flux_plan = FluxGatherPlan(given, parallel_faces, 
                            mbs_per_block, flux_count, direct_store=direct_store)
                    if flux_plan.invalid_reason() is None:
                        yield flux_plan

    def target_func(plan):
        return plan.make_kernel(discr, elface_to_bdry_bitmap=None,
                fluxes=fluxes).benchmark()

    return optimize_plan(
            generate_valid_plans, target_func,
            maximize=False,
            debug="cuda_gather_plan" in discr.debug)




# flux gather kernel ----------------------------------------------------------
class FluxGatherKernel:
    def __init__(self, discr, plan, elface_to_bdry_bitmap, fluxes):
        self.discr = discr
        self.plan = plan
        self.elface_to_bdry_bitmap = elface_to_bdry_bitmap
        self.fluxes = fluxes

        interior_deps_set = set()
        boundary_deps_set = set()
        for f in fluxes:
            interior_deps_set.update(set(f.interior_deps))
            boundary_deps_set.update(set(f.boundary_deps))

        self.interior_deps = list(interior_deps_set)
        self.boundary_deps = list(boundary_deps_set)
        self.all_deps = list(interior_deps_set|boundary_deps_set)

    def benchmark(self):
        discr = self.discr
        given = self.plan.given

        from hedge.cuda.tools import int_ceiling
        block_count = int_ceiling(
                len(discr.mesh.elements)/self.plan.elements_per_block())
        all_fluxes_on_faces = [gpuarray.empty(
            (block_count * self.plan.microblocks_per_block()
                * given.aligned_face_dofs_per_microblock(),),
                dtype=given.float_type,
                allocator=discr.pool.allocate)
                for i in range(len(self.fluxes))]

        field = gpuarray.empty(
                (self.plan.dofs_per_block() * block_count,), 
                dtype=given.float_type,
                allocator=discr.pool.allocate)

        fdata = self.fake_flux_face_data_block(block_count)
        ilist_data = self.fake_index_list_data()

        gather, texref_map = self.get_kernel(fdata, ilist_data,
                for_benchmark=True)

        for dep_expr in self.all_deps:
            field.bind_to_texref(texref_map[dep_expr])

        if "cuda_fastbench" in discr.debug:
            count = 1
        else:
            count = 20

        start = cuda.Event()
        start.record()
        cuda.Context.synchronize()
        for i in range(count):
            try:
                gather.prepared_call(
                        (block_count, 1),
                        0, 
                        fdata.device_memory,
                        *tuple(fof.gpudata for fof in all_fluxes_on_faces)
                        )
            except cuda.LaunchError:
                return None

        stop = cuda.Event()
        stop.record()
        stop.synchronize()

        return 1e-3/count * stop.time_since(start)

    def __call__(self, eval_dependency, lift_plan):
        discr = self.discr
        given = self.plan.given
        elgroup, = discr.element_groups

        all_fluxes_on_faces = [gpuarray.empty(
                given.fluxes_on_faces_shape(lift_plan),
                dtype=given.float_type,
                allocator=discr.pool.allocate)
                for i in range(len(self.fluxes))]

        fdata = self.flux_face_data_block(elgroup)
        ilist_data = self.index_list_data()

        gather, texref_map = self.get_kernel(fdata, ilist_data,
                for_benchmark=False)

        for dep_expr in self.all_deps:
            dep_field = eval_dependency(dep_expr)
            assert dep_field.dtype == given.float_type
            dep_field.bind_to_texref(texref_map[dep_expr])

        if set(["cuda_flux", "cuda_debugbuf"]) <= discr.debug:
            debugbuf = gpuarray.zeros((512,), dtype=numpy.float32)
        else:
            from hedge.cuda.tools import FakeGPUArray
            debugbuf = FakeGPUArray()

        if discr.instrumented:
            discr.flux_gather_timer.add_timer_callable(gather.prepared_timed_call(
                    (len(discr.blocks), 1),
                    debugbuf.gpudata, 
                    fdata.device_memory,
                    *tuple(fof.gpudata for fof in all_fluxes_on_faces)
                    ))

            from pytools import product
            discr.gmem_bytes_gather.add(
                    len(discr.blocks) * fdata.block_bytes
                    +
                    given.float_size()
                    * (
                        # fetch
                        len(self.fluxes)
                        * 2*fdata.fp_count
                        * given.dofs_per_face()

                        # store
                        + len(discr.blocks) 
                        * len(self.fluxes) 
                        * self.plan.microblocks_per_block()
                        * given.aligned_face_dofs_per_microblock()
                        ))
        else:
            gather.prepared_call(
                    (len(discr.blocks), 1),
                    debugbuf.gpudata, 
                    fdata.device_memory,
                    *tuple(fof.gpudata for fof in all_fluxes_on_faces)
                    )

        if set(["cuda_flux", "cuda_debugbuf"]) <= discr.debug:
            copied_debugbuf = debugbuf.get()
            print "DEBUG", len(discr.blocks)
            numpy.set_printoptions(linewidth=100)
            print numpy.reshape(copied_debugbuf, (32, 16))
            #print copied_debugbuf
            raw_input()

        return zip(self.fluxes, all_fluxes_on_faces)

    @memoize_method
    def get_kernel(self, fdata, ilist_data, for_benchmark):
        from hedge.cuda.cgen import \
                Pointer, POD, Value, ArrayOf, Const, \
                Module, FunctionDeclaration, FunctionBody, Block, \
                Comment, Line, \
                CudaShared, CudaGlobal, Static, MaybeUnused, \
                Define, Pragma, \
                Constant, Initializer, If, For, Statement, Assign, While
                
        discr = self.discr
        given = self.plan.given
        fplan = self.plan
        d = discr.dimensions
        dims = range(d)

        elgroup, = discr.element_groups

        float_type = given.float_type

        f_decl = CudaGlobal(FunctionDeclaration(Value("void", "apply_flux"), 
            [
                Pointer(POD(float_type, "debugbuf")),
                Pointer(POD(numpy.uint8, "gmem_facedata")),
                ]+[
                Pointer(POD(float_type, "gmem_fluxes_on_faces%d" % flux_nr))
                for flux_nr in range(len(self.fluxes))
                ]
            ))

        cmod = Module()

        from hedge.cuda.optemplate import WholeDomainFluxOperator as WDFlux
        short_name = WDFlux.short_name

        for dep_expr in self.all_deps:
            cmod.append(
                Value("texture<float, 1, cudaReadModeElementType>", 
                    "%s_tex" % short_name(dep_expr)))

        cmod.extend([
            flux_header_struct(),
            face_pair_struct(float_type, discr.dimensions),
            Line(),
            Define("DIMENSIONS", discr.dimensions),
            Define("DOFS_PER_EL", given.dofs_per_el()),
            Define("DOFS_PER_FACE", given.dofs_per_face()),
            Define("THREADS_PER_FACE", fplan.threads_per_face()),
            Line(),
            Define("CONCURRENT_FACES", fplan.parallel_faces),
            Define("BLOCK_MB_COUNT", fplan.mbs_per_block),
            Line(),
            Define("FACEDOF_NR", "threadIdx.x"),
            Define("BLOCK_FACE", "threadIdx.y"),
            Line(),
            Define("FLUX_COUNT", fplan.flux_count),
            Line(),
            Define("THREAD_NUM", "(FACEDOF_NR + BLOCK_FACE*THREADS_PER_FACE)"),
            Define("THREAD_COUNT", "(THREADS_PER_FACE*CONCURRENT_FACES)"),
            Define("COALESCING_THREAD_COUNT", 
                "(THREAD_COUNT < 0x10 ? THREAD_COUNT : THREAD_COUNT & ~0xf)"),
            Line(),
            Define("DATA_BLOCK_SIZE", fdata.block_bytes),
            Define("ALIGNED_FACE_DOFS_PER_MB", given.aligned_face_dofs_per_microblock()),
            Define("ALIGNED_FACE_DOFS_PER_BLOCK", 
                "(ALIGNED_FACE_DOFS_PER_MB*BLOCK_MB_COUNT)"),
            Line(),
            Define("FOF_BLOCK_BASE", "(blockIdx.x*ALIGNED_FACE_DOFS_PER_BLOCK)"),
            Line(),
            ] + ilist_data.code + [
            Line(),
            Value("texture<index_list_entry_t, 1, cudaReadModeElementType>", 
                "tex_index_lists"),
            Line(),
            fdata.struct,
            Line(),
            CudaShared(Value("flux_data", "data")),
            ])

        if not fplan.direct_store:
            cmod.extend([
                CudaShared(
                    ArrayOf(
                        ArrayOf(
                            POD(float_type, "smem_fluxes_on_faces"),
                            "FLUX_COUNT"),
                        "ALIGNED_FACE_DOFS_PER_MB*BLOCK_MB_COUNT")
                    ),
                Line(),
                ])

        S = Statement
        f_body = Block()
            
        from hedge.cuda.tools import get_load_code

        f_body.extend(get_load_code(
            dest="&data",
            base="gmem_facedata + blockIdx.x*DATA_BLOCK_SIZE",
            bytes="sizeof(flux_data)",
            descr="load face_pair data")
            +[ S("__syncthreads()"), Line() ])

        def gen_store(flux_nr, index, what):
            if fplan.direct_store:
                return Assign(
                        "gmem_fluxes_on_faces%d[FOF_BLOCK_BASE + %s]" % (flux_nr, index),
                        what)
            else:
                return Assign(
                        "smem_fluxes_on_faces[%d][%s]" % (flux_nr, index),
                        what)

        def bdry_flux_writer():
            from hedge.cuda.cgen import make_multiple_ifs
            from pymbolic.mapper.stringifier import PREC_NONE

            flux_write_code = Block([POD(float_type, "flux") ])

            from pytools import flatten
            from hedge.tools import is_zero

            for dep in self.interior_deps:
                flux_write_code.append(
                        Initializer(
                            MaybeUnused(POD(float_type, "val_a_%s" 
                                % short_name(dep))),
                            "tex1Dfetch(%s_tex, a_index)" % short_name(dep)))
            for dep in self.boundary_deps:
                flux_write_code.append(
                        Initializer(
                            MaybeUnused(POD(float_type, "val_b_%s" 
                                % short_name(dep))),
                            "tex1Dfetch(%s_tex, b_index)" % short_name(dep)))

            for flux_nr, wdflux in enumerate(self.fluxes):
                flux_write_code.extend([
                    Line(),
                    Assign("flux", 0),
                    Line(),
                    ])

                flux_write_code.extend(
                        If("(fpair->boundary_bitmap) & (1 << %d)" % (bdry_id),
                            Block(list(flatten([
                                S("flux += /*%s*/ (%s) * val_%s_%s"
                                    % (is_interior and "int" or "ext",
                                        FluxToCodeMapper()(coeff, PREC_NONE),
                                        is_interior and "a" or "b",
                                        short_name(field_expr)))
                                    for is_interior, coeff, field_expr in [
                                        (True, flux_rec.int_coeff, flux_rec.field_expr),
                                        (False, flux_rec.ext_coeff, flux_rec.bfield_expr),
                                        ]
                                    if not is_zero(field_expr)
                                    ]
                                    for flux_rec in fluxes
                                    ))))
                            for bdry_id, fluxes in wdflux.bdry_id_to_fluxes.iteritems()
                    )

                flux_write_code.append(
                            gen_store(flux_nr, "fpair->a_dest+FACEDOF_NR",
                                "fpair->face_jacobian*flux"))

            return flux_write_code

        def int_flux_writer(is_twosided):
            def get_field(flux_rec, is_interior, flipped):
                if is_interior ^ flipped:
                    prefix = "a"
                else:
                    prefix = "b"

                return ("val_%s_%s" % (prefix, short_name(flux_rec.field_expr)))

            from hedge.cuda.cgen import make_multiple_ifs
            from pymbolic.mapper.stringifier import PREC_NONE

            flux_write_code = Block([POD(float_type, "a_flux") ])
            
            zero_flux_code = [Assign("a_flux", 0)]

            if is_twosided:
                flux_write_code.append(POD(float_type, "b_flux"))
                zero_flux_code.append(Assign("b_flux", 0))
                prefixes = ["a", "b"]
                flip_values = [False, True]
            else:
                prefixes = ["a"]
                flip_values = [False]

            flux_write_code.append(Line())

            for dep in self.interior_deps:
                for side in ["a", "b"]:
                    flux_write_code.append(
                            Initializer(
                                MaybeUnused(POD(float_type, "val_%s_%s" 
                                    % (side, short_name(dep)))),
                                "tex1Dfetch(%s_tex, %s_index)"
                                % (short_name(dep), side)))

            for flux_nr, wdflux in enumerate(self.fluxes):
                flux_write_code.extend([Line()]+zero_flux_code+[Line()])

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
                        gen_store(flux_nr, "fpair->a_dest+FACEDOF_NR",
                            "fpair->face_jacobian*a_flux"))

                if is_twosided:
                    flux_write_code.append(
                            gen_store(flux_nr, 
                                "fpair->b_dest+tex1Dfetch(tex_index_lists, "
                                "fpair->b_write_ilist_index + FACEDOF_NR)",
                                "fpair->face_jacobian*b_flux"))

            return flux_write_code

        def get_flux_code(flux_writer):
            flux_code = Block([])

            flux_code.extend([
                Initializer(Pointer(
                    Value("face_pair", "fpair")),
                    "data.facepairs+fpair_nr"),
                Initializer(
                    MaybeUnused(POD(numpy.uint32, "a_index")),
                    "fpair->a_base + tex1Dfetch(tex_index_lists, "
                    "fpair->a_ilist_index + FACEDOF_NR)"),
                Initializer(
                    MaybeUnused(POD(numpy.uint32, "b_index")),
                    "fpair->b_base + tex1Dfetch(tex_index_lists, "
                    "fpair->b_ilist_index + FACEDOF_NR)"),
                Line(),
                flux_writer(),
                Line(),
                S("fpair_nr += CONCURRENT_FACES")
                ])

            return flux_code

        flux_computation = Block([
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

        f_body.extend_log_block("compute the fluxes", 
                [If("FACEDOF_NR < DOFS_PER_FACE", flux_computation)])

        f_body.extend([
            Line(),
            S("__syncthreads()"),
            Line()
            ])

        if not fplan.direct_store:
            for flux_nr in range(len(self.fluxes)):
                f_body.extend_log_block("store flux number %d" % flux_nr, [
                    For("unsigned word_nr = THREAD_NUM", 
                        "word_nr < ALIGNED_FACE_DOFS_PER_MB*BLOCK_MB_COUNT", 
                        "word_nr += COALESCING_THREAD_COUNT",
                        Block([
                            Assign(
                                "gmem_fluxes_on_faces%d"
                                "[FOF_BLOCK_BASE+word_nr]"
                                % flux_nr,
                                "smem_fluxes_on_faces[%d][word_nr]" % flux_nr),
                            ])
                        )
                    ])

        # finish off ----------------------------------------------------------
        cmod.append(FunctionBody(f_decl, f_body))

        if not for_benchmark and "cuda_dumpkernels" in discr.debug:
            open("flux_gather.cu", "w").write(str(cmod))

        from pycuda.tools import allow_user_edit
        mod = cuda.SourceModule(
                #allow_user_edit(cmod, "kernel.cu", "the flux kernel"), 
                cmod,
                keep=True, 
                options=["--maxrregcount=%d" % self.plan.max_registers()]
                )
        if "cuda_flux" in discr.debug:
            print "flux: lmem=%d smem=%d regs=%d" % (mod.lmem, mod.smem, mod.registers)

        expr_to_texture_map = dict(
                (dep_expr, mod.get_texref(
                    "%s_tex" % short_name(dep_expr)))
                for dep_expr in self.all_deps)

        index_list_texref = mod.get_texref("tex_index_lists")
        index_list_texref.set_address(
                ilist_data.device_memory,
                ilist_data.bytes)
        index_list_texref.set_format(
                cuda.dtype_to_array_format(ilist_data.type), 1)
        index_list_texref.set_flags(cuda.TRSF_READ_AS_INTEGER)

        func = mod.get_function("apply_flux")
        func.prepare(
                (2+len(self.all_deps))*"P",
                block=(fplan.threads_per_face(), 
                    fplan.parallel_faces, 1),
                texrefs=expr_to_texture_map.values()
                + [index_list_texref])

        return func, expr_to_texture_map

    # data blocks -------------------------------------------------------------
    @memoize_method
    def flux_face_data_block(self, elgroup):
        discr = self.discr
        given = self.plan.given
        fplan = discr.flux_plan
        headers = []
        fp_blocks = []

        INVALID_DEST = (1<<16)-1

        from hedge.cuda.discretization import GPUBoundaryFaceStorage

        fp_struct = face_pair_struct(given.float_type, discr.dimensions)
        fh_struct = flux_header_struct()

        def find_elface_dest(el_face):
            elface_dofs = face_dofs*ldis.face_count()
            num_in_block = discr.find_number_in_block(el_face[0])
            mb_index, index_in_mb = divmod(num_in_block,  given.microblock.elements)
            return (mb_index * given.aligned_face_dofs_per_microblock()
                    + index_in_mb * elface_dofs
                    + el_face[1]*face_dofs)

        int_fp_count, ext_fp_count, bdry_fp_count = 0, 0, 0

        for block in discr.blocks:
            ldis = block.local_discretization
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

                if isinstance(b_face, GPUBoundaryFaceStorage):
                    # boundary face
                    b_base = b_face.gpu_bdry_index_in_floats
                    boundary_bitmap = self.elface_to_bdry_bitmap.get(a_face.el_face, 0)
                    b_write_index_list = 0 # doesn't matter
                    b_dest = INVALID_DEST

                    fp_structs = bdry_fp_structs
                    bdry_fp_count += 1
                else:
                    # interior face
                    b_base = discr.find_el_gpu_index(b_face.el_face[0])
                    boundary_bitmap = 0

                    if b_face.native_block == a_face.native_block:
                        # same block
                        faces_todo.remove(b_face.el_face)
                        b_write_index_list = a_face.opp_write_index_list_id
                        b_dest = find_elface_dest(b_face.el_face)

                        fp_structs = same_fp_structs
                        int_fp_count += 1
                    else:
                        # different block
                        b_write_index_list = 0 # doesn't matter
                        b_dest = INVALID_DEST

                        fp_structs = diff_fp_structs
                        ext_fp_count += 1

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

                            boundary_bitmap=boundary_bitmap,
                            pad=0,
                            b_write_ilist_index= \
                                    b_write_index_list*face_dofs,

                            a_dest=find_elface_dest(a_face.el_face),
                            b_dest=b_dest
                            ))

            headers.append(fh_struct.make(
                    same_facepairs_end=\
                            len(same_fp_structs),
                    diff_facepairs_end=\
                            len(same_fp_structs)+len(diff_fp_structs),
                    bdry_facepairs_end=\
                            len(same_fp_structs)+len(diff_fp_structs)
                            +len(bdry_fp_structs),
                    ))
            fp_blocks.append(
                    same_fp_structs
                    +diff_fp_structs
                    +bdry_fp_structs)

        #print len(same_fp_structs), len(diff_fp_structs), len(bdry_fp_structs)

        from hedge.cuda.cgen import Value, POD
        from hedge.cuda.tools import make_superblocks

        return make_superblocks(
                given.devdata, "flux_data",
                [(headers, Value(flux_header_struct().tpname, "header"))],
                [(fp_blocks, Value(fp_struct.tpname, "facepairs"))],
                extra_fields={
                    "int_fp_count": int_fp_count,
                    "ext_fp_count": ext_fp_count,
                    "bdry_fp_count": bdry_fp_count,
                    "fp_count": int_fp_count+ext_fp_count+bdry_fp_count,
                    }
                )

    @memoize_method
    def fake_flux_face_data_block(self, block_count):
        discr = self.discr
        given = self.plan.given

        fp_struct = face_pair_struct(given.float_type, discr.dimensions)

        min_headers = []
        min_fp_blocks = []

        from random import randrange, choice

        face_dofs = given.dofs_per_face()

        mp_count = discr.device.get_attribute(
                    cuda.device_attribute.MULTIPROCESSOR_COUNT)

        for block_nr in range(mp_count):
            fp_structs = []

            faces = [(mb_nr, mb_el_nr, face_nr)
                    for mb_nr in range(self.plan.microblocks_per_block())
                    for mb_el_nr in range(given.microblock.elements)
                    for face_nr in range(given.faces_per_el())]

            def draw_base():
                mb_nr, mb_el_nr, face_nr = choice(faces)
                return (block_nr * given.microblock.aligned_floats
                        * self.plan.microblocks_per_block()
                        + mb_nr * given.microblock.aligned_floats
                        + mb_el_nr * given.dofs_per_el())

            def draw_dest():
                mb_nr, mb_el_nr, face_nr = choice(faces)
                return (mb_nr * given.aligned_face_dofs_per_microblock()
                        + mb_el_nr * face_dofs * given.faces_per_el()
                        + face_nr * face_dofs)

            for i in range(self.plan.face_pair_count()):
                fp_structs.append(
                        fp_struct.make(
                            h=0.5, order=2, face_jacobian=0.5,
                            normal=discr.dimensions*[0.1],

                            a_base=draw_base(), b_base=draw_base(),

                            a_ilist_index=randrange(self.FAKE_INDEX_LIST_COUNT)*face_dofs,
                            b_ilist_index=randrange(self.FAKE_INDEX_LIST_COUNT)*face_dofs,

                            boundary_bitmap=1,
                            pad=0,
                            b_write_ilist_index=randrange(self.FAKE_INDEX_LIST_COUNT)*face_dofs,

                            a_dest=draw_dest(), b_dest=draw_dest()
                            ))

            total_ext_face_count = int(self.plan.estimate_extface_count())
            bdry_count = min(total_ext_face_count, 
                    randrange(1+int(total_ext_face_count/6)))
            diff_count = total_ext_face_count-bdry_count

            min_headers.append(flux_header_struct().make(
                    same_facepairs_end=len(fp_structs)-total_ext_face_count,
                    diff_facepairs_end=diff_count,
                    bdry_facepairs_end=bdry_count))
            min_fp_blocks.append(fp_structs)

        dups = block_count//mp_count + 1
        headers = (min_headers * dups)[:block_count]
        fp_blocks = (min_fp_blocks * dups)[:block_count]

        from hedge.cuda.cgen import Value
        from hedge.cuda.tools import make_superblocks

        return make_superblocks(
                given.devdata, "flux_data",
                [(headers, Value(flux_header_struct().tpname, "header")) ],
                [(fp_blocks, Value(fp_struct.tpname, "facepairs"))]
                )

    # index lists -------------------------------------------------------------
    FAKE_INDEX_LIST_COUNT = 30

    @memoize_method
    def index_list_data(self):
        return self.index_list_backend(self.discr.index_lists)

    @memoize_method
    def fake_index_list_data(self):
        ilists = []
        from random import shuffle
        for i in range(self.FAKE_INDEX_LIST_COUNT):
            ilist = range(self.plan.given.dofs_per_face())
            shuffle(ilist)
            ilists.append(ilist)

        return self.index_list_backend(ilists)

    def index_list_backend(self, ilists):
        from pytools import single_valued
        ilist_length = single_valued(len(il) for il in ilists)
        assert ilist_length == self.plan.given.dofs_per_face()

        if ilist_length > 256:
            tp = numpy.uint16
        else:
            tp = numpy.uint8

        from hedge.cuda.cgen import Typedef, POD, Value, Define

        from pytools import flatten
        flat_ilists = numpy.array(
                list(flatten(ilists)),
                dtype=tp)

        return GPUIndexLists(
                type=tp,
                code=[
                    Define("INDEX_LISTS_LENGTH", len(flat_ilists)),
                    Typedef(POD(tp, "index_list_entry_t")),
                    ],
                device_memory=cuda.to_device(flat_ilists),
                bytes=flat_ilists.size*flat_ilists.itemsize,
                )

