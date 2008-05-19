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



from pytools import memoize_method
import hedge.optemplate




class ExecutionMapper(hedge.optemplate.Evaluator,
        hedge.optemplate.BoundOpMapperMixin, 
        hedge.optemplate.LocalOpReducerMixin):
    def __init__(self, context, executor):
        hedge.optemplate.Evaluator.__init__(self, context)
        self.executor = executor




class OpTemplateWithEnvironment(object):
    def __init__(self, discr, optemplate):
        self.discr = discr

        from hedge.optemplate import OperatorBinder, InverseMassContractor, \
                FluxDecomposer
        from pymbolic.mapper.constant_folder import CommutativeConstantFoldingMapper
        from hedge.cuda.optemplate import BoundaryCombiner

        self.optemplate = (
                BoundaryCombiner(self.discr)(
                    InverseMassContractor()(
                        CommutativeConstantFoldingMapper()(
                            FluxDecomposer()(
                                OperatorBinder()(
                                    optemplate))))))

    @memoize_method
    def indexing_info(self):
        result = ""
        block_len = discr.plan.indexing_smem()
        block_dofs = discr.int_dof_floats + discr.ext_dof_floats

        INVALID_U8 = (1<<8) - 1
        INVALID_U16 = (1<<16) - 1
        INVALID_U32 = (1<<32) - 1

        block_lengths = []

        for block in discr.blocks:
            ldis = block.local_discretization
            el_dofs = ldis.node_count()

            faces_todo = set((el,face_nbr)
                    for el in block.elements
                    for face_nbr in range(ldis.face_count()))
            fp_blocks = []

            bf = isame = idiff = 0
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
                    bf += 1
                else:
                    # interior face
                    b_base = opp.native_block_el_num*el_dofs
                    bdry_flux_number = 0
                    if opp.native_block == int_face.native_block:
                        # same block
                        faces_todo.remove(opp.el_face)
                        b_global_base = INVALID_U32
                        b_ilist_number = opp.native_index_list_id
                        isame += 1
                    else:
                        # different block
                        b_global_base = (
                                opp.native_block_el_num*el_dofs
                                + block_dofs*opp.native_block.number)
                        b_ilist_number = INVALID_U8
                        idiff += 1

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

            bheader = discr.plan.get_block_header_struct().make(
                    els_in_block=len(block.elements),
                    face_pairs_in_block=len(fp_blocks)
                    )
            block_data = bheader + "".join(fp_blocks)

            # take care of alignment
            missing_bytes = block_len - len(block_data)
            assert missing_bytes >= 0
            block_data = block_data + "\x00"*missing_bytes
            block_lengths.append(len(block_data))

            result += block_data

        # make sure the indexing_smem estimate is achieved
        assert max(block_lengths) == discr.plan.indexing_smem()
        assert len(result) == block_len*len(discr.blocks)
        return cuda.to_device(result)

    def execute(self, vars):
        return ExecutionMapper(vars, self)(self.optemplate)
