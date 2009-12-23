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




def have_pycuda():
    try:
        import pycuda
        return True
    except:
        return False




if have_pycuda():
    from pycuda.tools import mark_cuda_test

    @mark_cuda_test
    def test_from_to_gpu():
        import pycuda.driver as cuda
        from hedge.mesh import make_cylinder_mesh, make_ball_mesh, make_box_mesh

        mesh = make_cylinder_mesh(max_volume=0.004, 
                periodic=False, radial_subdivisions=32)
        from hedge.backends.cuda import Discretization
        discr = Discretization(mesh, order=4, init_cuda=False, 
                debug=["cuda_no_plan"])
        a = numpy.arange(len(discr), dtype=numpy.float32)
        a_gpu = discr.convert_volume(a, discr.compute_kind)
        a_copy = discr.convert_volume(a_gpu, "numpy")
        diff = a - a_copy
        assert la.norm(diff) < 1e-10 * la.norm(a)
