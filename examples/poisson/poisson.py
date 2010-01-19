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
from hedge.tools import Reflection, Rotation




def main(write_output=True):
    from hedge.data import GivenFunction, ConstantGivenFunction

    from hedge.backends import guess_run_context
    rcon = guess_run_context()

    dim = 2

    def boundary_tagger(fvi, el, fn, points):
        from math import atan2, pi
        normal = el.face_normals[fn]
        if -90/180*pi < atan2(normal[1], normal[0]) < 90/180*pi:
            return ["neumann"]
        else:
            return ["dirichlet"]

    if dim == 2:
        if rcon.is_head_rank:
            from hedge.mesh.generator import make_disk_mesh
            mesh = make_disk_mesh(r=0.5, boundary_tagger=boundary_tagger,
                    max_area=1e-2)
    elif dim == 3:
        if rcon.is_head_rank:
            from hedge.mesh.generator import make_ball_mesh
            mesh = make_ball_mesh(max_volume=0.0001,
                    boundary_tagger=lambda fvi, el, fn, points:
                    ["dirichlet"])
    else:
        raise RuntimeError, "bad number of dimensions"

    if rcon.is_head_rank:
        print "%d elements" % len(mesh.elements)
        mesh_data = rcon.distribute_mesh(mesh)
    else:
        mesh_data = rcon.receive_mesh()

    discr = rcon.make_discretization(mesh_data, order=5, 
            debug=[])

    def dirichlet_bc(x, el):
        from math import sin
        return sin(10*x[0])

    def rhs_c(x, el):
        if la.norm(x) < 0.1:
            return 1000
        else:
            return 0

    def my_diff_tensor():
        result = numpy.eye(dim)
        result[0,0] = 0.1
        return result

    try:
        from hedge.models.poisson import PoissonOperator
        from hedge.second_order import \
                IPDGSecondDerivative, LDGSecondDerivative, \
                StabilizedCentralSecondDerivative
        from hedge.mesh import TAG_NONE, TAG_ALL
        op = PoissonOperator(discr.dimensions, 
                diffusion_tensor=my_diff_tensor(),

                #dirichlet_tag="dirichlet",
                #neumann_tag="neumann", 

                dirichlet_tag=TAG_ALL,
                neumann_tag=TAG_NONE, 

                #dirichlet_tag=TAG_ALL,
                #neumann_tag=TAG_NONE, 

                dirichlet_bc=GivenFunction(dirichlet_bc),
                neumann_bc=ConstantGivenFunction(-10),

                scheme=StabilizedCentralSecondDerivative(),
                #scheme=LDGSecondDerivative(),
                #scheme=IPDGSecondDerivative(),
                )
        bound_op = op.bind(discr)

        from hedge.iterative import parallel_cg
        u = -parallel_cg(rcon, -bound_op, 
                bound_op.prepare_rhs(discr.interpolate_volume_function(rhs_c)), 
                debug=20, tol=5e-4,
                dot=discr.nodewise_dot_product,
                x=discr.volume_zeros())

        if write_output:
            from hedge.visualization import SiloVisualizer, VtkVisualizer
            vis = VtkVisualizer(discr, rcon)
            visf = vis.make_file("fld")
            vis.add_data(visf, [ ("sol", discr.convert_volume(u, kind="numpy")), ])
            visf.close()
    finally:
        discr.close()





if __name__ == "__main__":
    main()




# entry points for py.test ----------------------------------------------------
from pytools.test import mark_test
@mark_test.long
def test_poisson():
    main(write_output=False)
