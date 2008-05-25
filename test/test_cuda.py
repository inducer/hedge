# Hedge - the Hybrid'n'Easy DG Environment
# Copyright (C) 2007 Andreas Kloeckner
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import division




import numpy
import numpy.linalg as la
import unittest
import pycuda.driver as cuda
from hedge.discr_precompiled import Discretization




class TestHedge(unittest.TestCase):
    def test_from_to_gpu(self):
        from hedge.mesh import make_cylinder_mesh, make_ball_mesh, make_box_mesh

        mesh = make_cylinder_mesh(max_volume=0.004, 
                periodic=False, radial_subdivisions=32)
        from hedge.cuda.discretization import Discretization
        discr = Discretization(mesh, order=4, init_cuda=False, debug=True)
        a = numpy.arange(len(discr), dtype=numpy.float32)
        a_gpu = discr.volume_to_gpu(a)
        a_copy = discr.volume_from_gpu(a_gpu)
        diff = a - a_copy
        assert la.norm(diff) < 1e-10 * la.norm(a)




if __name__ == '__main__':
    cuda.init()
    assert cuda.Device.count() >= 1
    ctx = cuda.Device(0).make_context()
    unittest.main()
