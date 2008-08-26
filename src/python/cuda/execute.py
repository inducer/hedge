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




# exec mapper -----------------------------------------------------------------
class ExecutionMapper(hedge.optemplate.Evaluator,
        hedge.optemplate.BoundOpMapperMixin, 
        hedge.optemplate.LocalOpReducerMixin):

    def __init__(self, context, executor):
        hedge.optemplate.Evaluator.__init__(self, context)
        self.ex = executor

        self.diff_xyz_cache = {}

    def get_vec_structure(self, vec, point_size, chunk_size, block_size,
            other_char=lambda snippet: "."):
        result = ""
        for block in range(len(vec) // block_size):
            struc = ""
            for chunk in range(block_size//chunk_size):
                for point in range(chunk_size//point_size):
                    offset = block*block_size + chunk*chunk_size + point*point_size
                    snippet = vec[offset:offset+point_size]

                    if numpy.isnan(snippet).any():
                        struc += "N"
                    elif (snippet == 0).any():
                        struc += "0"
                    else:
                        struc += other_char(snippet)

                struc += " "
            result += struc + "\n"
        return result
            
    def print_error_structure(self, computed, reference, diff):
        discr = self.ex.discr

        norm_ref = la.norm(reference)
        struc = ""

        if norm_ref == 0:
            norm_ref = 1

        numpy.set_printoptions(precision=2, linewidth=130, suppress=True)
        for block in discr.blocks:
            i_el = 0
            for mb in block.microblocks:
                for el in mb:
                    s = discr.find_el_range(el.id)
                    relerr = relative_error(diff[s])/norm_ref
                    if relerr > 1e-4:
                        struc += "*"
                        if False:
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

    def map_chunk_diff_base(self, op, field_expr, out=None):
        discr = self.ex.discr
        fplan = discr.flux_plan
        lplan = fplan.diff_plan()

        d = discr.dimensions

        eg, = discr.element_groups
        func, texrefs, field_texref = self.ex.diff_kernel.get_kernel(op.__class__, eg)

        field = self.rec(field_expr)
        assert field.dtype == discr.flux_plan.float_type

        field.bind_to_texref(field_texref)
        
        from hedge.cuda.tools import int_ceiling
        kwargs = {
                "block": (lplan.chunk_size, lplan.parallelism.p, 1),
                "grid": (lplan.chunks_per_microblock(), 
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
                self.ex.diff_kernel.gpu_diffmats(op.__class__, eg).device_memory,
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

        return xyz_diff

    def map_diff_base(self, op, field_expr, out=None):
        try:
            return self.diff_xyz_cache[op.__class__, field_expr][op.xyz_axis]
        except KeyError:
            pass

        discr = self.ex.discr
        fplan = discr.flux_plan
        lplan = fplan.diff_plan()

        xyz_diff = self.map_chunk_diff_base(op, field_expr, out)
        
        if discr.debug:
            field = self.rec(field_expr)
            f = discr.volume_from_gpu(field)
            dx = discr.volume_from_gpu(xyz_diff[0])
            
            test_discr = discr.test_discr
            real_dx = test_discr.nabla[0].apply(f.astype(numpy.float64))
            
            diff = dx - real_dx

            from hedge.tools import relative_error
            rel_err_norm = relative_error(la.norm(diff), la.norm(real_dx))
            print "diff", rel_err_norm
            if not (rel_err_norm < 5e-5):
                self.print_error_structure(dx, real_dx, diff)
            assert rel_err_norm < 5e-5

        self.diff_xyz_cache[op.__class__, field_expr] = xyz_diff
        return xyz_diff[op.xyz_axis]

    def map_whole_domain_flux(self, wdflux, out=None):
        discr = self.ex.discr

        eg, = discr.element_groups
        fdata = self.ex.fluxgather_kernel.flux_with_temp_data(wdflux, eg)
        fplan = discr.flux_plan
        lplan = fplan.flux_lifting_plan()

        gather, gather_texrefs, texref_map = \
                self.ex.fluxgather_kernel.get_kernel(wdflux)
        lift, lift_texrefs, fluxes_on_faces_texref = \
                self.ex.fluxlocal_kernel.get_kernel(wdflux.is_lift, eg)

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
                dtype=fplan.float_type,
                allocator=discr.pool.allocate)

        # gather phase --------------------------------------------------------
        for dep_expr in wdflux.all_deps:
            dep_field = self.rec(dep_expr)
            assert dep_field.dtype == fplan.float_type
            dep_field.bind_to_texref(texref_map[dep_expr])

        gather_args = [
                debugbuf, 
                fluxes_on_faces, 
                fdata.device_memory,
                self.ex.fluxgather_kernel.index_list_global_data().device_memory,
                ]

        gather_kwargs = {
                "texrefs": gather_texrefs, 
                "block": (discr.flux_plan.dofs_per_face(), 
                    fplan.parallel_faces, 1),
                "grid": (len(discr.blocks), 1),
                "time_kernel": discr.instrumented,
                }

        kernel_time = gather(*gather_args, **gather_kwargs)

        if discr.instrumented:
            discr.inner_flux_timer.add_time(kernel_time)
            discr.inner_flux_counter.add()

        if False:
            copied_debugbuf = debugbuf.get()
            print "DEBUG", len(discr.blocks)
            numpy.set_printoptions(linewidth=100)
            print numpy.reshape(copied_debugbuf, (32, 16))
            #print copied_debugbuf
            raw_input()

        if discr.debug:
            useful_size = (len(discr.blocks)
                    * fplan.aligned_face_dofs_per_microblock()
                    * fplan.microblocks_per_block())
            fof = fluxes_on_faces.get()

            fof = fof[:useful_size]

            have_used_nans = False
            for i_b, block in enumerate(discr.blocks):
                offset = (fplan.aligned_face_dofs_per_microblock()
                        *fplan.microblocks_per_block())
                size = (len(block.el_number_map)
                        *fplan.dofs_per_face()
                        *fplan.faces_per_el())
                if numpy.isnan(la.norm(fof[offset:offset+size])):
                    have_used_nans = True

            if have_used_nans:
                struc = ( fplan.dofs_per_face(),
                        fplan.dofs_per_face()*fplan.faces_per_el(),
                        fplan.aligned_face_dofs_per_microblock(),
                        )

                print self.get_vec_structure(fof, *struc)

            assert not have_used_nans

        # lift phase ----------------------------------------------------------
        flux = discr.volume_empty() 

        lift_args = [
                flux, 
                self.ex.fluxlocal_kernel.gpu_liftmat(wdflux.is_lift).device_memory,
                debugbuf,
                ]

        lift_kwargs = {
                "texrefs": lift_texrefs, 
                "block": (lplan.chunk_size, lplan.parallelism.p, 1),
                "grid": (lplan.chunks_per_microblock(), 
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
        if discr.debug and False:
            cot = discr.test_discr.compile(wdflux.flux_optemplate)
            ctx = {field_expr.name: 
                    discr.volume_from_gpu(field).astype(numpy.float64)
                    }
            for boundary in wdflux.boundaries:
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

        # instantiate kernels
        from hedge.cuda.diff import DiffKernel
        from hedge.cuda.fluxgather import FluxGatherKernel
        from hedge.cuda.fluxlocal import FluxLocalKernel

        self.diff_kernel = DiffKernel(discr)
        self.fluxlocal_kernel = FluxLocalKernel(discr)
        self.fluxgather_kernel = FluxGatherKernel(discr)

        # compile the optemplate
        from hedge.optemplate import OperatorBinder, InverseMassContractor, \
                FluxDecomposer, BCToFluxRewriter
        from pymbolic.mapper.constant_folder import CommutativeConstantFoldingMapper
        from hedge.cuda.optemplate import BoundaryCombiner

        optemplate = (
                BoundaryCombiner(discr)(
                    InverseMassContractor()(
                        CommutativeConstantFoldingMapper()(
                            FluxDecomposer()(
                                BCToFluxRewriter()(
                                    OperatorBinder()(
                                        optemplate)))))))

        def compile_vec_expr(expr):
            from pycuda.vector_expr import CompiledVectorExpression
            return CompiledVectorExpression(
                    expr, 
                    type_getter=lambda expr: (True, self.discr.default_scalar_type),
                    result_dtype=self.discr.default_scalar_type,
                    allocator=self.discr.pool.allocate)

        if isinstance(optemplate, numpy.ndarray):
            assert optemplate.dtype == object
            self.compiled_vec_expr = [compile_vec_expr(subexpr) for subexpr in optemplate]
        else:
            self.compiled_vec_expr = compile_vec_expr(optemplate)

    def __call__(self, **vars):
        ex_mapper = ExecutionMapper(vars, self)
        if isinstance(self.compiled_vec_expr, list):
            return numpy.array([
                ce(ex_mapper) for ce in self.compiled_vec_expr],
                dtype=object)
        else:
            return ce(ex_mapper)

    # gpu data blocks ---------------------------------------------------------
    @memoize_method
    def _unused_flux_inverse_jacobians(self, elgroup):
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


