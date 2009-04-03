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
from pytools import memoize_method, memoize, Record
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import pymbolic.mapper.stringifier
import hedge.backends.cuda.plan
from pymbolic.mapper.c_code import CCodeMapper
from hedge.flux import FluxIdentityMapper




class GPUIndexLists(Record): pass




# structures ------------------------------------------------------------------
@memoize
def flux_header_struct():
    from codepy.cgen import GenerableStruct, POD

    return GenerableStruct("flux_header", [
        POD(numpy.uint16, "same_facepairs_end"),
        POD(numpy.uint16, "diff_facepairs_end"),
        POD(numpy.uint16, "bdry_facepairs_end"),
        ], align_bytes=4)

@memoize
def face_pair_struct(float_type, dims):
    from codepy.cgen import GenerableStruct, POD, ArrayOf
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
        POD(numpy.uint16, "boundary_bitmap"),
        POD(numpy.uint16, "a_dest"), 
        POD(numpy.uint16, "b_dest"), 
        ], 
        align_bytes=4, 

        # ensure that adjacent face_pair instances# ensure that adjacent face_pair 
        # instances can be accessed without bank conflicts.
        aligned_prime_to=[2], 
        )




# flux to code mapper ---------------------------------------------------------
class FluxConcretizer(FluxIdentityMapper):
    def __init__(self, int_field_expr, ext_field_expr, dep_to_index):
        self.int_field_expr = int_field_expr
        self.ext_field_expr = ext_field_expr
        self.dep_to_index = dep_to_index

    def map_field_component(self, expr):
        if expr.is_local:
            prefix = "a"
            f_expr = self.int_field_expr
        else:
            prefix = "b"
            f_expr = self.ext_field_expr

        from hedge.backends.cuda.optemplate import WholeDomainFluxOperator

        from hedge.tools import is_obj_array, is_zero
        from pymbolic import var
        if is_obj_array(f_expr):
            f_expr = f_expr[expr.index]
            if is_zero(f_expr):
                return 0
            return var("val_%s_field%d" % (prefix, self.dep_to_index[f_expr]))
        else:
            assert expr.index == 0, repr(f_expr)
            if is_zero(f_expr):
                return 0
            return var("val_%s_field%d" % (prefix, self.dep_to_index[f_expr]))




class FluxToCodeMapper(CCodeMapper):
    def __init__(self):
        def float_mapper(x):
            if isinstance(x, float):
                return "value_type(%s)" % repr(x)
            else:
                return repr(x)

        CCodeMapper.__init__(self, float_mapper, reverse=False)

    def map_normal(self, expr, enclosing_prec):
        return "fpair->normal[%d]" % expr.axis

    def map_penalty_term(self, expr, enclosing_prec):
        return ("pow(fpair->order*fpair->order/fpair->h, %r)" 
                % expr.power)






def flux_to_code(f2cm, is_flipped, int_field_expr, ext_field_expr, 
        dep_to_index, flux, prec):
    if is_flipped:
        from hedge.flux import FluxFlipper
        flux = FluxFlipper()(flux)

    return f2cm(FluxConcretizer(int_field_expr, ext_field_expr, dep_to_index)(flux), prec)




# plan ------------------------------------------------------------------------
class ExecutionPlan(hedge.backends.cuda.plan.ExecutionPlan):
    def __init__(self, given, 
            parallel_faces, mbs_per_block, flux_count,
            direct_store, partition_data):
        hedge.backends.cuda.plan.ExecutionPlan.__init__(self, given)
        self.parallel_faces = parallel_faces
        self.mbs_per_block = mbs_per_block
        self.flux_count = flux_count
        self.direct_store = direct_store

        self.partition_data = partition_data

    def microblocks_per_block(self):
        return self.mbs_per_block

    def elements_per_block(self):
        return self.microblocks_per_block()*self.given.microblock.elements

    def dofs_per_block(self):
        return self.microblocks_per_block()*self.given.microblock.aligned_floats

    @memoize_method
    def face_pair_count(self):
        return self.partition_data.max_face_pair_count

    @memoize_method
    def shared_mem_use(self):
        from hedge.backends.cuda.fluxgather import face_pair_struct
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
            from hedge.backends.cuda.tools import int_ceiling
            return int_ceiling(dpf, devdata.smem_granularity)
        else:
            return dpf

    def threads(self):
        return self.parallel_faces*self.threads_per_face()

    def registers(self):
        return 51

    def __str__(self):
        result = ("%s pfaces=%d mbs_per_block=%d mb_elements=%d" % (
            hedge.backends.cuda.plan.ExecutionPlan.__str__(self),
            self.parallel_faces,
            self.mbs_per_block,
            self.given.microblock.elements,
            ))

        if self.direct_store:
            result += " direct_store"

        return result

    def make_kernel(self, discr, executor, fluxes):
        return Kernel(discr, self, executor, fluxes)





def make_plan(discr, given, tune_for):
    from hedge.backends.cuda.execute import Executor
    from hedge.backends.cuda.optemplate import FluxCollector
    if tune_for is not None:
        fbatch1 = Executor.get_first_flux_batch(discr.mesh, tune_for)
        if fbatch1 is not None:
            fluxes = list(fbatch1.flux_exprs)
            flux_count = len(fluxes)
        else:
            fluxes = None
    else:
        fluxes = None

    if fluxes is None:
        # a reasonable guess?
        flux_count = discr.dimensions

    def generate_valid_plans():
        #for direct_store in [True, False]:
        for direct_store in [False]:
            for parallel_faces in range(1,32):
                for mbs_per_block in range(1,8):
                    flux_plan = ExecutionPlan(given, parallel_faces, 
                            mbs_per_block, flux_count, 
                            direct_store=direct_store,
                            partition_data=discr._get_partition_data(
                                mbs_per_block*given.microblock.elements))
                    if flux_plan.invalid_reason() is None:
                        yield flux_plan

    def target_func(plan):
        if tune_for is None:
            return 0
        else:
            return plan.make_kernel(discr, executor=None,
                    fluxes=fluxes).benchmark()

    from hedge.backends.cuda.plan import optimize_plan

    return optimize_plan(
            "gather",
            generate_valid_plans, target_func,
            maximize=False,
            debug_flags=discr.debug)




# flux gather kernel ----------------------------------------------------------
class Kernel:
    def __init__(self, discr, plan, executor, fluxes):
        self.discr = discr
        self.plan = plan
        self.executor = executor

        assert isinstance(fluxes, list)
        self.fluxes = fluxes

        interior_deps_set = set()
        boundary_int_deps_set = set()
        boundary_ext_deps_set = set()
        self.dep_to_tag = {}
        for f in fluxes:
            interior_deps_set.update(set(f.interior_deps))
            boundary_int_deps_set.update(set(f.boundary_int_deps))
            boundary_ext_deps_set.update(set(f.boundary_ext_deps))
            self.dep_to_tag.update(f.dep_to_tag)

        self.interior_deps = list(interior_deps_set)
        self.boundary_int_deps = list(boundary_int_deps_set)
        self.boundary_ext_deps = list(boundary_ext_deps_set)
        self.all_deps = list(
                interior_deps_set
                | boundary_int_deps_set
                | boundary_ext_deps_set
                )

        self.dep_to_index = dict((dep, i) for i, dep in enumerate(self.all_deps))

    def benchmark(self):
        discr = self.discr
        given = self.plan.given

        from hedge.backends.cuda.tools import int_ceiling
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

            from hedge.tools import is_zero
            if is_zero(dep_field):
                if dep_expr in self.dep_to_tag:
                    dep_field = discr.boundary_zeros(self.dep_to_tag[dep_expr])
                else:
                    dep_field = discr.volume_zeros()

            assert dep_field.dtype == given.float_type
            dep_field.bind_to_texref(texref_map[dep_expr])

        if set(["cuda_flux", "cuda_debugbuf"]) <= discr.debug:
            debugbuf = gpuarray.zeros((512,), dtype=numpy.float32)
        else:
            from hedge.backends.cuda.tools import FakeGPUArray
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
            from hedge.tools import get_rank, wait_for_keypress
            if get_rank(discr) == 0:
                copied_debugbuf = debugbuf.get()
                print "DEBUG", len(discr.blocks)
                numpy.set_printoptions(linewidth=130)
                print numpy.reshape(copied_debugbuf, (32, 16))
                #print copied_debugbuf

                wait_for_keypress(discr)

        if "cuda_flux" in discr.debug and False:
            from hedge.tools import get_rank, wait_for_keypress
            if get_rank(discr) == 0:
                for fof in all_fluxes_on_faces:
                    numpy.set_printoptions(linewidth=130, precision=2, threshold=10**6)
                    print fof.get()
                    wait_for_keypress(discr)
                #print "B", [la.norm(fof.get()) for fof in all_fluxes_on_faces]
            

        return all_fluxes_on_faces

    @memoize_method
    def get_kernel(self, fdata, ilist_data, for_benchmark):
        from codepy.cgen import \
                Pointer, POD, Value, ArrayOf, Const, Typedef, \
                Module, FunctionDeclaration, FunctionBody, Block, \
                Comment, Line, \
                Static, MaybeUnused, \
                Define, Pragma, \
                Constant, Initializer, If, For, Statement, Assign, While
                
        from codepy.cgen.cuda import CudaShared, CudaGlobal

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

        from hedge.backends.cuda.optemplate import WholeDomainFluxOperator as WDFlux

        for dep_expr in self.all_deps:
            cmod.extend([
                Comment(str(dep_expr)),
                Value("texture<float, 1, cudaReadModeElementType>", 
                    "field%d_tex" % self.dep_to_index[dep_expr])
                ])

        if fplan.flux_count != len(self.fluxes):
            from warnings import warn
            warn("Flux count in flux execution plan different from actual flux count.\n"
                    "You may want to specify the tune_for= kwarg in the Discretization\n"
                    "constructor.")

        cmod.extend([
            Line(),
            Typedef(POD(float_type, "value_type")),
            Line(),
            flux_header_struct(),
            Line(),
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
            Define("FLUX_COUNT", len(self.fluxes)),
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
            
        from hedge.backends.cuda.tools import get_load_code

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
            from codepy.cgen import make_multiple_ifs
            from pymbolic.mapper.stringifier import PREC_NONE

            flux_write_code = Block([MaybeUnused(POD(float_type, "flux"))])

            from pytools import flatten
            from hedge.tools import is_zero

            for dep in self.boundary_int_deps:
                flux_write_code.append(
                        Initializer(
                            MaybeUnused(POD(float_type, "val_a_field%d" 
                                % self.dep_to_index[dep])),
                            "tex1Dfetch(field%d_tex, a_index)" % self.dep_to_index[dep]))
            for dep in self.boundary_ext_deps:
                flux_write_code.append(
                        Initializer(
                            MaybeUnused(POD(float_type, "val_b_field%d" 
                                % self.dep_to_index[dep])),
                            "tex1Dfetch(field%s_tex, b_index)" % self.dep_to_index[dep]))

            f2cm = FluxToCodeMapper()

            fluxes_by_bdry_number = {}
            for flux_nr, wdflux in enumerate(self.fluxes):
                for tag, fluxes in wdflux.tag_to_fluxes.iteritems():
                    if for_benchmark:
                        bdry_number = 0
                    else:
                        bdry_number = self.executor.boundary_tag_to_number[tag]

                    fluxes_by_bdry_number.setdefault(bdry_number, [])\
                            .append((flux_nr, fluxes))

            # FIXME
            # This generates incorrect code for overlapping tags. While the
            # data structure supports tag overlapping by allocating one bit
            # per tag, this code only evaluates the last flux in an overlap
            # correctly.

            flux_sub_codes = []
            for bdry_number, nrs_and_fluxes in fluxes_by_bdry_number.iteritems():
                bblock = []
                for flux_nr, fluxes in nrs_and_fluxes:
                    from hedge.mesh import TAG_ALL
                    bblock.extend(
                            [Assign("flux", 0),]
                            + [S("flux += "+
                                flux_to_code(f2cm, is_flipped=False,
                                    int_field_expr=flux_rec.bpair.field,
                                    ext_field_expr=flux_rec.bpair.bfield,
                                    dep_to_index=self.dep_to_index, 
                                    flux=flux_rec.flux_expr, prec=PREC_NONE))
                                for flux_rec in fluxes]
                            + [
                            gen_store(flux_nr, "fpair->a_dest+FACEDOF_NR",
                                "fpair->face_jacobian * flux"),]
                            )

                flux_sub_codes.extend([
                    Line(),
                    Comment(nrs_and_fluxes[0][1][0].bpair.tag),
                    If("(fpair->boundary_bitmap) & (1 << %d)" % (bdry_number),
                        Block(bblock)),
                    ])

            flux_write_code.extend(
                    Initializer(
                        Value("value_type", f2cm.cse_prefix+str(i)), cse)
                    for i, cse in f2cm.cses)
            flux_write_code.extend(flux_sub_codes)

            return flux_write_code

        def int_flux_writer(is_twosided):
            def get_field(flux_rec, is_interior, flipped):
                if is_interior ^ flipped:
                    prefix = "a"
                else:
                    prefix = "b"

                return ("val_%s_field%d" % (prefix, self.dep_to_index[flux_rec.field_expr]))

            from codepy.cgen import make_multiple_ifs
            from pymbolic.mapper.stringifier import PREC_NONE

            flux_write_code = Block([])
            
            flux_var_decl = [Initializer( POD(float_type, "a_flux"), 0)]

            if is_twosided:
                flux_var_decl.append(Initializer(POD(float_type, "b_flux"), 0))
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
                                MaybeUnused(POD(float_type, "val_%s_field%d" 
                                    % (side, self.dep_to_index[dep]))),
                                "tex1Dfetch(field%d_tex, %s_index)"
                                % (self.dep_to_index[dep], side)))

            f2cm = FluxToCodeMapper()

            flux_sub_codes = []
            for flux_nr, wdflux in enumerate(self.fluxes):
                my_flux_block = Block(flux_var_decl)

                for int_rec in wdflux.interiors:
                    for prefix, is_flipped in zip(prefixes, flip_values):
                        my_flux_block.append(
                                S("%s_flux += %s"
                                    % (prefix, 
                                        flux_to_code(f2cm, is_flipped,
                                            int_rec.field_expr, 
                                            int_rec.field_expr, 
                                            self.dep_to_index,
                                            int_rec.flux_expr, PREC_NONE),
                                        )))

                my_flux_block.append(Line())

                my_flux_block.append(
                        gen_store(flux_nr, "fpair->a_dest+FACEDOF_NR",
                            "fpair->face_jacobian*a_flux"))

                if is_twosided:
                    my_flux_block.append(
                            gen_store(flux_nr, 
                                "fpair->b_dest+tex1Dfetch(tex_index_lists, "
                                "fpair->b_write_ilist_index + FACEDOF_NR)",
                                "fpair->face_jacobian*b_flux"))
                flux_sub_codes.append(my_flux_block)

            flux_write_code.extend(
                    Initializer(
                        Value("value_type", f2cm.cse_prefix+str(i)), cse)
                    for i, cse in f2cm.cses)
            flux_write_code.extend(flux_sub_codes)

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
            f_body.extend_log_block("store fluxes", [
                For("unsigned word_nr = THREAD_NUM", 
                    "word_nr < ALIGNED_FACE_DOFS_PER_MB*BLOCK_MB_COUNT", 
                    "word_nr += COALESCING_THREAD_COUNT",
                    Block([
                        Assign(
                            "gmem_fluxes_on_faces%d"
                            "[FOF_BLOCK_BASE+word_nr]"
                            % flux_nr,
                            "smem_fluxes_on_faces[%d][word_nr]" % flux_nr)
                        for flux_nr in range(len(self.fluxes))
                        ])
                    )
                ])

        # finish off ----------------------------------------------------------
        cmod.append(FunctionBody(f_decl, f_body))

        if not for_benchmark and "cuda_dumpkernels" in discr.debug:
            open("flux_gather.cu", "w").write(str(cmod))

        from pycuda.tools import allow_user_edit
        mod = SourceModule(
                #allow_user_edit(cmod, "kernel.cu", "the flux kernel"), 
                cmod,
                keep="cuda_keep_kernels" in discr.debug, 
                options=["--maxrregcount=%d" % self.plan.max_registers()]
                )
        if "cuda_flux" in discr.debug:
            print "flux: lmem=%d smem=%d regs=%d" % (mod.lmem, mod.smem, mod.registers)

        expr_to_texture_map = dict(
                (dep_expr, mod.get_texref(
                    "field%d_tex" % self.dep_to_index[dep_expr]))
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
                (2+len(self.fluxes))*"P",
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

        from hedge.backends.cuda import GPUBoundaryFaceStorage

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
                    boundary_bitmap = self.executor.elface_to_bdry_bitmap.get(
                            a_face.el_face, 0)
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

        from codepy.cgen import Value, POD
        from hedge.backends.cuda.tools import make_superblocks

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

            def bound_int(low, x, hi):
                return int(min(max(low, x), hi))

            from random import gauss
            pdata = self.plan.partition_data
            fp_count = bound_int(
                    0,
                    gauss(
                        pdata.face_pair_avg,
                        (pdata.max_face_pair_count-pdata.face_pair_avg)/2),
                    pdata.max_face_pair_count)


            for i in range(fp_count):
                fp_structs.append(
                        fp_struct.make(
                            h=0.5, order=2, face_jacobian=0.5,
                            normal=discr.dimensions*[0.1],

                            a_base=draw_base(), b_base=draw_base(),

                            a_ilist_index=randrange(self.FAKE_INDEX_LIST_COUNT)*face_dofs,
                            b_ilist_index=randrange(self.FAKE_INDEX_LIST_COUNT)*face_dofs,

                            boundary_bitmap=1,
                            b_write_ilist_index=randrange(self.FAKE_INDEX_LIST_COUNT)*face_dofs,

                            a_dest=draw_dest(), b_dest=draw_dest()
                            ))

            total_ext_face_count = bound_int(0, 
                pdata.ext_face_avg + randrange(-1,2), 
                fp_count)

            bdry_count = min(total_ext_face_count, 
                    randrange(1+int(round(total_ext_face_count/6))))
            diff_count = total_ext_face_count-bdry_count

            min_headers.append(flux_header_struct().make(
                    same_facepairs_end=len(fp_structs)-total_ext_face_count,
                    diff_facepairs_end=diff_count,
                    bdry_facepairs_end=bdry_count))
            min_fp_blocks.append(fp_structs)

        dups = block_count//mp_count + 1
        headers = (min_headers * dups)[:block_count]
        fp_blocks = (min_fp_blocks * dups)[:block_count]

        from codepy.cgen import Value
        from hedge.backends.cuda.tools import make_superblocks

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

        from codepy.cgen import Typedef, POD, Value, Define

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

