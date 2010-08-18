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

    @mark_cuda_test
    def test_cuda_volume_quadrature():
        from hedge.mesh.generator import make_rect_mesh
        mesh = make_rect_mesh(a=(-1,-1),b=(1,1),max_area=0.08)

        from hedge.backends import guess_run_context
        cpu_rcon = guess_run_context(['jit'])
        gpu_rcon = guess_run_context(['cuda'])

        order = 4
        quad_min_degrees = {"quad": 3*order}

        cpu_discr, gpu_discr = [
                rcon.make_discretization(mesh, order=order,
                    default_scalar_type=numpy.float64, 
                    debug=["cuda_no_plan", "cuda_no_microblock", ],
                    quad_min_degrees=quad_min_degrees
                    )
                for rcon in [cpu_rcon, gpu_rcon]]

        from math import sin, cos
        def f(x, el):
            return sin(x[0])*cos(x[1])

        cpu_field, gpu_field = [
                discr.interpolate_volume_function(f)
                for discr in [cpu_discr, gpu_discr]]

        def make_optemplate():
            from hedge.optemplate.operators import QuadratureGridUpsampler
            from hedge.optemplate import Field, make_stiffness_t

            u = Field("u")
            qu = QuadratureGridUpsampler("quad")(u)

            return make_stiffness_t(2)[0](Field("intercept")(qu))

        saved_vectors = []
        def intercept(x):
            saved_vectors.append(x)
            return x

        for discr in [cpu_discr, gpu_discr]:
            discr.add_function("intercept", intercept)

        opt = make_optemplate()
        cpu_bound, gpu_bound = [discr.compile(make_optemplate())
                for discr in [cpu_discr, gpu_discr]]

        cpu_result = cpu_bound(u=cpu_field)
        gpu_result = gpu_bound(u=gpu_field)

        cpu_ivec, gpu_ivec = saved_vectors
        gpu_ivec_on_host = gpu_ivec.get()[gpu_discr._gpu_volume_embedding("quad")]
        ierr = cpu_ivec-gpu_ivec_on_host
        assert la.norm(ierr) < 5e-15

        gpu_result_on_host = gpu_discr.convert_volume(gpu_result, kind="numpy")
        err = cpu_result-gpu_result_on_host
        assert la.norm(err) < 2e-14

        cpu_discr.close()
        gpu_discr.close()

