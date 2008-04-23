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




def main() :
    from hedge.element import TriangularElement, TetrahedralElement
    from hedge.parallel import guess_parallelization_context
    from hedge.data import GivenFunction, ConstantGivenFunction

    pcon = guess_parallelization_context()

    dim = 2

    def boundary_tagger(fvi, el, fn):
        from math import atan2, pi
        normal = el.face_normals[fn]
        if -10/180*pi < atan2(normal[1], normal[0]) < 10/180*pi:
            return ["neumann"]
        else:
            return ["dirichlet"]

    if dim == 2:
        if pcon.is_head_rank:
            from hedge.mesh import make_disk_mesh
            mesh = make_disk_mesh(r=0.5, boundary_tagger=boundary_tagger,
                    max_area=1e-2)
        el_class = TriangularElement
    elif dim == 3:
        if pcon.is_head_rank:
            from hedge.mesh import make_ball_mesh
            mesh = make_ball_mesh(max_volume=0.0001)
        el_class = TetrahedralElement
    else:
        raise RuntimeError, "bad number of dimensions"

    if pcon.is_head_rank:
        print "%d elements" % len(mesh.elements)
        mesh_data = pcon.distribute_mesh(mesh)
    else:
        mesh_data = pcon.receive_mesh()

    discr = pcon.make_discretization(mesh_data, el_class(5))

    def dirichlet_bc(x):
        from math import sin
        return sin(10*x[0])

    def rhs_c(x):
        if la.norm(x) < 0.1:
            return 1000
        else:
            return 0

    def my_diff_tensor():
        result = numpy.eye(dim)
        result[0,0] = 0.1
        return result

    from hedge.operators import WeakPoissonOperator
    op = WeakPoissonOperator(discr, 
            diffusion_tensor=ConstantGivenFunction(my_diff_tensor()),

            dirichlet_tag="dirichlet",
            neumann_tag="neumann", 

            dirichlet_bc=GivenFunction(dirichlet_bc),
            neumann_bc=ConstantGivenFunction(-10),
            )

    from hedge.tools import parallel_cg
    u = -parallel_cg(pcon, -op, op.prepare_rhs(GivenFunction(rhs_c)), 
            debug=True, tol=1e-10)

    from hedge.visualization import SiloVisualizer, VtkVisualizer
    vis = VtkVisualizer(discr, pcon)
    visf = vis.make_file("fld")
    vis.add_data(visf, [ ("sol", u), ])
    visf.close()





if __name__ == "__main__":
    main()

