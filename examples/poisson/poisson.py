
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




import pylinear.array as num
import pylinear.operator as operator
import pylinear.computation as comp
from hedge.tools import Rotation, dot




def main() :
    from hedge.element import \
            TriangularElement, \
            TetrahedralElement
    from hedge.operators import StrongLaplacianOperator
    from hedge.mesh import \
            make_disk_mesh, \
            make_regular_square_mesh, \
            make_square_mesh, \
            make_ball_mesh
    from hedge.visualization import SiloVisualizer, VtkVisualizer
    from math import sin, cos, pi, exp, sqrt
    from hedge.parallel import guess_parallelization_context

    pcon = guess_parallelization_context()

    dim = 2

    def boundary_tagger(fvi, el, fn):
        if el.face_normals[fn][0] > 0:
            return ["dirichlet"]
        else:
            return ["neumann"]

    if dim == 2:
        if pcon.is_head_rank:
            mesh = make_disk_mesh(r=0.5, boundary_tagger=boundary_tagger)
            #mesh = make_regular_square_mesh(
                    #n=9, periodicity=(True,True))
            #mesh = make_regular_square_mesh(n=9)
            #mesh = make_square_mesh(max_area=0.008)
            #mesh.transform(Rotation(pi/8))
        el_class = TriangularElement
    elif dim == 3:
        if pcon.is_head_rank:
            mesh = make_ball_mesh(max_volume=0.001)
        el_class = TetrahedralElement
    else:
        raise RuntimeError, "bad number of dimensions"

    if pcon.is_head_rank:
        print "%d elements" % len(mesh.elements)
        mesh_data = pcon.distribute_mesh(mesh)
    else:
        mesh_data = pcon.receive_mesh()

    discr = pcon.make_discretization(mesh_data, el_class(3))
    vis = VtkVisualizer(discr, pcon)

    def u0(x):
        if comp.norm_2(x) < 0.2:
            #return exp(-100*x*x)
            return 1
        else:
            return 0

    def coeff(x):
        if x[0] < 0:
            return 0.25
        else:
            return 1

    def dirichlet_bc(t, x):
        return 0

    def neumann_bc(t, x):
        return 2


    import pymbolic
    v_x = pymbolic.var("x")
    sol = pymbolic.parse("math.sin(x[0]**2*x[1]**2)")
    sol_c = pymbolic.compile(sol, variables=["x"])
    rhs = pymbolic.simplify(pymbolic.laplace(sol, [v_x[0], v_x[1]]))
    rhs_c = pymbolic.compile(rhs, variables=["x"])

    op = StrongLaplacianOperator(discr, 
            #coeff=coeff,
            dirichlet_tag=None,
            dirichlet_bc=lambda t, x: sol_c(x),
            neumann_tag="empty", 
            #neumann_bc=neumann_bc,
            stabilisation=1e-2
            )

    class MyOperator(operator.Operator(num.Float64)):
        def size1(self):
            return len(discr)

        def size2(self):
            return len(discr)

        def apply(self, before, after):
            after[:] = op.rhs(0, before)

    a_inv = operator.BiCGSTABOperator.make(MyOperator(), 4000, 1e-10)
    #a_inv.debug_level = 1

    results = comp.operator_eigenvectors(a_inv, 5, 
            #which=comp.LARGEST_MAGNITUDE
            )
    scalars = []
    for i, (value,vector) in enumerate(results):
        print i, value
        scalars.append(("ev%d" % i, vector))

    visf = vis.make_file("eigenvectors" % i)
    vis.add_data(visf, scalars)
    visf.close()

    sol_v = discr.interpolate_volume_function(sol_c)
    rhs_v = discr.interpolate_volume_function(rhs_c)

    visf = vis.make_file("fld")
    vis.add_data(visf, [
        #("u", u), 
        ("sol", sol_v), 
        ("rhs2", op.rhs(0, sol_v)), 
        ("rhs", rhs_v), 
        ])
    visf.close()

    u = a_inv(rhs_v)

    visf = vis.make_file("fld")
    vis.add_data(visf, [
        ("u", u), 
        ("rhs", rhs_v), 
        ])
    visf.close()



if __name__ == "__main__":
    #import cProfile as profile
    #profile.run("main()", "wave2d.prof")
    main()

