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
    from hedge.data import GivenFunction, ConstantGivenFunction

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
            mesh = make_disk_mesh(r=0.5, boundary_tagger=boundary_tagger,
                    max_area=1e-2)
            #mesh = make_regular_square_mesh(n=3, boundary_tagger=boundary_tagger, periodicity=(True,False))
            #mesh = make_regular_square_mesh(n=9)
            #mesh = make_square_mesh(max_area=0.1, boundary_tagger=boundary_tagger)
            #mesh.transform(Reflection(0,2))
        el_class = TriangularElement
    elif dim == 3:
        if pcon.is_head_rank:
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
    vis = VtkVisualizer(discr, pcon)

    def u0(x):
        if comp.norm_2(x) < 0.2:
            #return exp(-100*x*x)
            return 1
        else:
            return 0

    def dirichlet_bc(x):
        if x[0] > 0:
            return 0
        else:
            return 1

    def rhs_c(x):
        if la.norm(x) < 0.1:
            return 1000
        else:
            return 0

    class DiffTensor:
        shape = (dim, dim)

        def __call__(self, x):
            result = num.identity(dim)
            result[0,0] = 20*abs(x[0]-0.5)
            return result

    def my_diff_tensor():
        result = numpy.eye(dim)
        result[0,0] = 0.1
        return result

    op = WeakPoissonOperator(discr, 
            diffusion_tensor=ConstantGivenFunction(my_diff_tensor()),
            #diffusion_tensor=GivenFunction(DiffTensor()),

            #dirichlet_tag="dirichlet",
            #neumann_tag="neumann", 

            dirichlet_tag=TAG_ALL,
            neumann_tag=TAG_NONE,

            #dirichlet_bc=GivenFunction(dirichlet_bc),
            dirichlet_bc=ConstantGivenFunction(0),
            neumann_bc=ConstantGivenFunction(-10),
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
        results = comp.operator_eigenvectors(-op, 30, PylinearOpWrapper(discr.mass_operator),
                which=comp.SMALLEST_MAGNITUDE
                )
        scalars = []
        for i, (value,vector) in enumerate(results):
            print i, value, l2_norm(vector.real)
            scalars.append(("ev%03d" % i, vector.real))
        print 

        visf = vis.make_file("eigenvectors")
        vis.add_data(visf, scalars)
        visf.close()
        return


    #a_inv = operator.BiCGSTABOperator.make(-op, 40000, 1e-10)
    #a_inv = operator.CGOperator.make(-op, 40000, 1e-4)
    #a_inv.debug_level = 1
    #u = -a_inv(op.prepare_rhs(GivenFunction(rhs_c)))

    numpy.seterr('raise')
    if False:
        u = GivenFunction(rhs_c).volume_interpolant(discr)

        N = 30000
        from time import time
        start = time()
        
        #z = num.arange(17)
        #u = numpy.array([z,z])
        #u = ArithmeticList([z,z])
        #u = [z,z]
        for i in xrange(N):
            op.op(u)
            #op.grad(u)

        print (time()-start)/(N)
        return

    from hedge.tools import parallel_cg
    u = -parallel_cg(pcon, -op, op.prepare_rhs(GivenFunction(rhs_c)), 
            debug=True, tol=1e-10)
    print len(u)

    from hedge.discretization import ones_on_boundary
    visf = vis.make_file("fld")
    vis.add_data(visf, [
        ("sol", u), 
        ("dir", ones_on_boundary(discr, "dirichlet")), 
        ("neu", ones_on_boundary(discr, "neumann")), 
        ])
    visf.close()





if __name__ == "__main__":
    #import cProfile as profile
    #profile.run("main()", "poisson.prof")
    main()

