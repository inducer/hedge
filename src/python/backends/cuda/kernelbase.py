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
from pytools import memoize_method, Record
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray




class DiffKernelBase(object):
    class RstToXyzInfo(Record):
        pass

    @memoize_method
    def fake_localop_rst_to_xyz(self):
        discr = self.discr
        given = self.plan.given
        d = discr.dimensions

        el_count = given.block_count * given.elements_per_block()
        channels = given.devdata.make_valid_tex_channel_count(d)

        return self.RstToXyzInfo(
                gpu_data=gpuarray.to_gpu(
                    numpy.ones((channels, d, el_count), 
                        dtype=given.float_type, order="F")),
                channels=channels)

    @memoize_method
    def localop_rst_to_xyz(self, diff_op, elgroup):
        discr = self.discr
        given = discr.given
        d = discr.dimensions

        coeffs = diff_op.coefficients(elgroup)

        elgroup_indices = self.discr.elgroup_microblock_indices(elgroup)
        el_count = given.block_count * given.elements_per_block()

        # indexed local, el_number, global
        result_matrix = (coeffs[:,:,elgroup_indices]
                .transpose(1,0,2)).astype(given.float_type)
        channels = given.devdata.make_valid_tex_channel_count(d)
        add_channels = channels - result_matrix.shape[0]
        if add_channels:
            result_matrix = numpy.vstack((
                result_matrix,
                numpy.zeros((add_channels,d,el_count), dtype=result_matrix.dtype)
                ))

        assert result_matrix.shape == (channels, d, el_count)

        if "cuda_diff" in discr.debug:
            def get_el_index_in_el_group(el):
                mygroup, idx = discr.group_map[el.id]
                assert mygroup is elgroup
                return idx

            for block in discr.blocks:
                i = block.number * given.elements_per_block()
                for mb in block.microblocks:
                    for el in mb:
                        egi = get_el_index_in_el_group(el)
                        assert egi == elgroup_indices[i]
                        assert (result_matrix[:d,:,i].T == coeffs[:,:,egi]).all()
                        i += 1

        return self.RstToXyzInfo(
                gpu_data=gpuarray.to_gpu(
                    numpy.asarray(result_matrix, order="F")),
                channels=channels)




def fake_elwise_scaling(given):
    el_count = given.block_count * given.elements_per_block()
    ij = numpy.ones((el_count,), dtype=given.float_type)
    return gpuarray.to_gpu(ij)




# FIXME remove me
class FluxLocalKernelBase(object):
    @memoize_method
    def fake_inverse_jacobians_tex(self):
        given = self.plan.given
        el_count = given.block_count * given.elements_per_block()
        ij = numpy.ones((el_count,), dtype=given.float_type)
        return gpuarray.to_gpu(ij)

    @memoize_method
    def inverse_jacobians_tex(self, elgroup):
        ij = elgroup.inverse_jacobians[
                    self.discr.elgroup_microblock_indices(elgroup)]
        return gpuarray.to_gpu(
                ij.astype(self.plan.given.float_type))



