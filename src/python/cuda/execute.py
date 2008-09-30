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

        self.diff_xyz_cache = {} # map field expr to its dx,dy,dz
        self.flux_cache = {} # map WholeDomainFluxOperator instance to flux

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
            
    def print_error_structure(self, computed, reference, diff,
            eventful_only=False, detail=True):
        discr = self.ex.discr

        norm_ref = la.norm(reference)
        struc_lines = []

        if norm_ref == 0:
            norm_ref = 1

        from hedge.tools import relative_error
        numpy.set_printoptions(precision=2, linewidth=130, suppress=True)
        for block in discr.blocks:
            add_lines = []
            struc_line  = "%7d " % (block.number * discr.flux_plan.dofs_per_block())
            i_el = 0
            eventful = False
            for mb in block.microblocks:
                for el in mb:
                    s = discr.find_el_range(el.id)
                    relerr = relative_error(la.norm(diff[s]), norm_ref)
                    if relerr > 1e-4:
                        eventful = True
                        struc_line += "*"
                        if detail:
                            print "block %d, el %d, global el #%d, rel.l2err=%g" % (
                                    block.number, i_el, el.id, relerr)
                            print computed[s]
                            print reference[s]
                            print diff[s]
                            print diff[s]/norm_ref
                            print la.norm(diff[s]), norm_ref
                            raw_input()
                    elif numpy.isnan(diff[s]).any():
                        eventful = True
                        struc_line += "N"
                        add_lines.append(str(diff[s]))
                        
                        if detail:
                            print "block %d, el %d, global el #%d, rel.l2err=%g" % (
                                    block.number, i_el, el.id, relerr)
                            print computed[s]
                            print reference[s]
                            print diff[s]
                            raw_input()
                    else:
                        if numpy.max(numpy.abs(reference[s])) == 0:
                            struc_line += "0"
                        else:
                            if False:
                                print "block %d, el %d, global el #%d, rel.l2err=%g" % (
                                        block.number, i_el, el.id, relerr)
                                print computed[s]
                                print reference[s]
                                print diff[s]
                                raw_input()
                            struc_line += "."
                    i_el += 1
                struc_line += " "
            if (not eventful_only) or eventful:
                struc_lines.append(struc_line)
                if detail:
                    struc_lines.extend(add_lines)
        print
        print "\n".join(struc_lines)

    def map_diff_base(self, op, field_expr):
        try:
            return self.diff_xyz_cache[op.__class__, field_expr][op.xyz_axis]
        except KeyError:
            pass

        discr = self.ex.discr

        if discr.instrumented:
            given = discr.given
            discr.diff_op_counter.add(discr.dimensions)
            discr.flop_counter.add(
                    # r,s,t diff
                    2 # mul+add
                    * discr.dimensions
                    * len(discr.nodes)
                    * given.dofs_per_el()

                    # x,y,z rescale
                    +2 # mul+add
                    * discr.dimensions**2
                    * len(discr.nodes)
                    )

        field = self.rec(field_expr)
        xyz_diff = self.ex.diff_kernel(op.__class__, field)
        
        if set(["cuda_diff", "cuda_compare"]) <= discr.debug:
            field = self.rec(field_expr)
            f = discr.volume_from_gpu(field)
            assert not numpy.isnan(f).any(), "Initial field contained NaNs."
            cpu_xyz_diff = [discr.volume_from_gpu(xd) for xd in xyz_diff]
            dx = cpu_xyz_diff[0]

            test_discr = discr.test_discr
            real_dx = test_discr.nabla[0].apply(f.astype(numpy.float64))
            
            diff = dx - real_dx

            for i, xd in enumerate(cpu_xyz_diff):
                if numpy.isnan(xd).any():
                    self.print_error_structure(xd, xd, xd-xd,
                            eventful_only=False, detail=False)
                    assert False, "Resulting field %d contained NaNs." % i
            
            from hedge.tools import relative_error
            rel_err_norm = relative_error(la.norm(diff), la.norm(real_dx))
            print "diff", rel_err_norm
            if not (rel_err_norm < 5e-5):
                self.print_error_structure(dx, real_dx, diff,
                        eventful_only=False, detail=False)

            assert rel_err_norm < 5e-5

        self.diff_xyz_cache[op.__class__, field_expr] = xyz_diff
        return xyz_diff[op.xyz_axis]

    def map_whole_domain_flux(self, wdflux, out=None):
        discr = self.ex.discr
        given = discr.given
        eg, = discr.element_groups

        # gather phase --------------------------------------------------------
        try:
            fluxes_on_faces = self.flux_cache[wdflux]
        except KeyError:
            computed_fluxes = self.ex.fluxgather_kernel(self.rec)

            for f_wdflux, f_fluxes_on_faces in computed_fluxes:
                self.flux_cache[f_wdflux] = f_fluxes_on_faces

            fluxes_on_faces = self.flux_cache[wdflux]

            # count flops
            if discr.instrumented:
                discr.flop_counter.add(
                        len(computed_fluxes)
                        * given.dofs_per_face()
                        * given.faces_per_el()
                        * len(discr.mesh.elements)
                        * (1 # facejac-mul
                            + 2 * # int+ext
                            3*len(wdflux.all_deps) # const-mul, normal-mul, add
                            )
                        )

        if discr.debug & set(["cuda_lift", "cuda_flux"]):
            fplan = discr.flux_plan

            useful_size = (len(discr.blocks)
                    * given.aligned_face_dofs_per_microblock()
                    * fplan.microblocks_per_block())
            fof = fluxes_on_faces.get()

            fof = fof[:useful_size]

            have_used_nans = False
            for i_b, block in enumerate(discr.blocks):
                offset = i_b*(given.aligned_face_dofs_per_microblock()
                        *fplan.microblocks_per_block())
                size = (len(block.el_number_map)
                        *given.dofs_per_face()
                        *given.faces_per_el())
                if numpy.isnan(la.norm(fof[offset:offset+size])).any():
                    have_used_nans = True

            if have_used_nans:
                struc = ( given.dofs_per_face(),
                        given.dofs_per_face()*given.faces_per_el(),
                        given.aligned_face_dofs_per_microblock(),
                        )

                print self.get_vec_structure(fof, *struc)
                raise RuntimeError("Detected used NaNs in flux gather output.")

            assert not have_used_nans

        # lift phase ----------------------------------------------------------
        flux = self.ex.fluxlocal_kernel(fluxes_on_faces, wdflux.is_lift)

        # count flops
        if discr.instrumented:
            discr.inner_flux_counter.add()
            discr.flop_counter.add(
                    2 # mul+add
                    * given.dofs_per_face()
                    * given.faces_per_el()
                    * given.dofs_per_el()
                    * len(discr.mesh.elements)
                    )

        # verification --------------------------------------------------------
        if "cuda_lift" in discr.debug:
            cuda.Context.synchronize()
            print "NANCHECK"
            copied_flux = discr.volume_from_gpu(flux)
            contains_nans = numpy.isnan(copied_flux).any()
            if contains_nans:
                self.print_error_structure(
                        copied_flux, copied_flux, copied_flux-copied_flux,
                        eventful_only=True)
            assert not contains_nans, "Resulting flux contains NaNs."

        return flux




class OpTemplateWithEnvironment(object):
    def __init__(self, discr, optemplate):
        self.discr = discr

        # build a boundary tag bitmap
        from hedge.cuda.optemplate import BoundaryTagCollector
        boundary_tag_to_number = {}
        for btag in BoundaryTagCollector()(optemplate):
            boundary_tag_to_number.setdefault(btag, 
                    len(boundary_tag_to_number))

        elface_to_bdry_bitmap = {}
        for btag, bdry_number in boundary_tag_to_number.iteritems():
            bdry_bit = 1 << bdry_number
            for elface in discr.mesh.tag_to_boundary.get(btag, []):
                elface_to_bdry_bitmap[elface] = (
                        elface_to_bdry_bitmap.get(elface, 0) | bdry_bit)

        # compile the optemplate
        from hedge.cuda.optemplate import FluxCollector
        optemplate = self.compile_optemplate(discr.mesh, optemplate)
        fluxes = FluxCollector()(optemplate)

        # build the kernels 
        self.diff_kernel = self.discr.diff_plan.make_kernel(discr)
        self.fluxlocal_kernel = self.discr.fluxlocal_plan.make_kernel(discr)
        self.fluxgather_kernel = self.discr.flux_plan.make_kernel(discr, 
                elface_to_bdry_bitmap, fluxes)

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

    @staticmethod
    def compile_optemplate(mesh, optemplate):
        from pymbolic.mapper.constant_folder import CommutativeConstantFoldingMapper
        from hedge.optemplate import OperatorBinder, InverseMassContractor, \
                FluxDecomposer, BCToFluxRewriter
        from hedge.cuda.optemplate import \
                BoundaryCombiner, BoundaryTagCollector, FluxCollector

        return BoundaryCombiner(mesh)(
                InverseMassContractor()(
                    CommutativeConstantFoldingMapper()(
                        FluxDecomposer()(
                            BCToFluxRewriter()(
                                OperatorBinder()(
                                    optemplate))))))

    # actual execution --------------------------------------------------------
    def __call__(self, **vars):
        ex_mapper = ExecutionMapper(vars, self)
        if isinstance(self.compiled_vec_expr, list):
            return numpy.array([
                ce(ex_mapper) for ce in self.compiled_vec_expr],
                dtype=object)
        else:
            return self.compiled_vec_expr(ex_mapper)
