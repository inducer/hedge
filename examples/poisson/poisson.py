
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
import pylinear.array as num
import pylinear.operator as operator
import pylinear.computation as comp
from hedge.tools import Reflection, Rotation, dot




def main() :
    from hedge.element import \
            TriangularElement, \
            TetrahedralElement
    from hedge.operators import WeakPoissonOperator
    from hedge.mesh import \
            make_disk_mesh, \
            make_regular_square_mesh, \
            make_square_mesh, \
            make_ball_mesh, \
            TAG_ALL, TAG_NONE
    from hedge.visualization import SiloVisualizer, VtkVisualizer
    from math import sin, cos, pi, exp, sqrt
    from hedge.parallel import guess_parallelization_context

    pcon = guess_parallelization_context()

    dim = 2

    def boundary_tagger(fvi, el, fn):
        from math import atan2, pi
        normal = el.face_normals[fn]
        if -10/180*pi < atan2(normal[1], normal[0]) < 10/180*pi:
            return ["dirichlet"]
        else:
            return ["neumann"]

    def boundary_tagger_2(fvi, el, fn):
        if el.face_normals[fn][0] > 0:
            return ["dirichlet"]
        else:
            return ["neumann"]

    if dim == 2:
        if pcon.is_head_rank:
            mesh = make_disk_mesh(r=0.5, boundary_tagger=boundary_tagger)
            #mesh = make_regular_square_mesh(n=3, boundary_tagger=boundary_tagger)
            #mesh = make_regular_square_mesh(n=9)
            #mesh = make_square_mesh(max_area=0.03, boundary_tagger=boundary_tagger)
            #mesh.transform(Reflection(0,2))
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

    discr = pcon.make_discretization(mesh_data, el_class(2))
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
        return -2

    import pymbolic
    v_x = pymbolic.var("x")
    sol = pymbolic.parse("math.sin(x[0]**2*x[1]**2)")
    #sol = pymbolic.parse("(x[0]**2+-0.25)*(x[1]**2+-0.25)")
    sol_c = pymbolic.compile(sol, variables=["x"])
    #rhs = pymbolic.simplify(pymbolic.laplace(sol, [v_x[0], v_x[1]]))
    #rhs_c = pymbolic.compile(rhs, variables=["x"])
    #print sol,rhs

    def rhs_c(x):
        return 0

    op = WeakPoissonOperator(discr, 
            #coeff=coeff,
            dirichlet_tag="dirichlet",
            #dirichlet_bc=lambda t, x: 0,
            dirichlet_bc=lambda t, x: 0,
            neumann_tag="neumann", 
            neumann_bc=lambda t, x: -1,
            ldg=False
            )

    def l2_norm(v):
        return sqrt(v*(discr.mass_operator*v))

    def matrix_rep(op):
        h,w = op.shape
        mat = num.zeros(op.shape)
        for j in range(w):
            mat[:,j] = op(num.unit_vector(w, j))
        return mat

    if False:
        mat = matrix_rep(op)
        print comp.norm_frobenius(mat-mat.T)
        #print comp.eigenvalues(mat)
        print mat.shape
    
    if False:
        from hedge.discretization import PylinearOpWrapper
        results = comp.operator_eigenvectors(-op, 20, PylinearOpWrapper(discr.mass_operator),
                which=comp.SMALLEST_MAGNITUDE
                )
        scalars = []
        for i, (value,vector) in enumerate(results):
            print i, value, l2_norm(vector.real)
            scalars.append(("ev%d" % i, vector.real))
        print 

        visf = vis.make_file("eigenvectors")
        vis.add_data(visf, scalars)
        visf.close()
        return

    sol_v = discr.interpolate_volume_function(sol_c)
    rhs_v = discr.interpolate_volume_function(rhs_c)

    #a_inv = operator.BiCGSTABOperator.make(StiffnessOperator(), 40000, 1e-10)
    a_inv = operator.CGOperator.make(-op, 40000, 1e-10)
    a_inv.debug_level = 1

    u = -a_inv(op.prepare_rhs(rhs_v))

    v_ones = 1+discr.volume_zeros()

    from hedge.discretization import generate_ones_on_boundary
    visf = vis.make_file("fld")
    vis.add_data(visf, [
        ("sol", u), 
        ("truesol", sol_v), 
        ("rhs2", discr.inverse_mass_operator* op(sol_v)), 
        ("rhs", rhs_v), 
        ("dir", generate_ones_on_boundary(discr, "dirichlet")), 
        ("neu", generate_ones_on_boundary(discr, "neumann")), 
        ])
    visf.close()




if __name__ == "__main__":
    #import cProfile as profile
    #profile.run("main()", "wave2d.prof")
    main()

