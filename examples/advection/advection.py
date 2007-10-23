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
import pylinear.computation as comp
from hedge.tools import dot




def main() :
    from hedge.element import \
            TriangularElement, \
            TetrahedralElement
    from hedge.timestep import RK4TimeStepper
    from hedge.mesh import \
            make_disk_mesh, \
            make_square_mesh, \
            make_regular_square_mesh, \
            make_single_element_mesh, \
            make_ball_mesh, \
            make_box_mesh
    from hedge.discretization import Discretization, ones_on_boundary
    from hedge.visualization import SiloVisualizer, VtkVisualizer
    from hedge.tools import dot
    from pytools.arithmetic_container import ArithmeticList
    from pytools.stopwatch import Job
    from math import sin, cos, pi, sqrt
    from hedge.parallel import \
            guess_parallelization_context, \
            reassemble_volume_field
    from time import time
    from hedge.operators import StrongAdvectionOperator, WeakAdvectionOperator
    from hedge.data import TimeDependentGivenFunction

    def u_analytic(x, t):
        return sin(3*(a*x+t))

    def boundary_tagger(vertices, el, face_nr):
        if el.face_normals[face_nr] * a > 0:
            return ["inflow"]
        else:
            return ["outflow"]

    pcon = guess_parallelization_context()

    dim = 2
    periodic = False
    if dim == 2:
        a = num.array([1,0])
        if pcon.is_head_rank:
            #mesh = make_square_mesh(boundary_tagger=boundary_tagger, max_area=0.1)
            #mesh = make_square_mesh(boundary_tagger=boundary_tagger, max_area=0.2)
            #mesh = make_regular_square_mesh(a=-r, b=r, boundary_tagger=boundary_tagger, n=3)
            #mesh = make_single_element_mesh(boundary_tagger=boundary_tagger)
            #mesh = make_disk_mesh(r=pi, boundary_tagger=boundary_tagger, max_area=0.5)
            mesh = make_disk_mesh(boundary_tagger=boundary_tagger)
        el_class = TriangularElement
    elif dim == 3:
        a = num.array([0,0,0.5])
        if pcon.is_head_rank:
            if periodic:
                mesh = make_box_mesh(dimensions=(1,1,2*pi/3),
                        periodic=periodic, max_volume=0.01)
            else:
                mesh = make_box_mesh(max_volume=0.01, 
                        boundary_tagger=boundary_tagger)
                #mesh = make_ball_mesh(boundary_tagger=boundary_tagger)
        el_class = TetrahedralElement
    else:
        raise RuntimeError, "bad number of dimensions"

    if pcon.is_head_rank:
        mesh_data = pcon.distribute_mesh(mesh)
    else:
        mesh_data = pcon.receive_mesh()

    discr = pcon.make_discretization(mesh_data, el_class(3))
    vis = SiloVisualizer(discr, pcon)
    #vis = VtkVisualizer(discr, "fld", pcon)
    op = StrongAdvectionOperator(discr, a, 
            inflow_u=TimeDependentGivenFunction(u_analytic))
    #op = WeakAdvectionOperator(discr, a, 
            #inflow_u=TimeDependentGivenFunction(u_analytic))

    print "%d elements" % len(discr.mesh.elements)

    #silo = SiloFile("bdry.silo")
    #vis.add_to_silo(silo,
            #[("outflow", ones_on_boundary(discr, "outflow")), 
                #("inflow", ones_on_boundary(discr, "inflow"))])
    #return 

    u = discr.interpolate_volume_function(lambda x: u_analytic(x, 0))

    dt = discr.dt_factor(comp.norm_2(a))/2
    stepfactor = 10
    nsteps = int(1/dt)

    stepper = RK4TimeStepper()
    start_step = time()
    for step in range(nsteps):
        if step % stepfactor == 0:
            now = time()
            print "timestep %d, t=%f, l2=%f, secs=%f" % (
                    step, dt*step, sqrt(u*(op.mass*u)), now-start_step)
            start_step = now

        t = step*dt
        visf = vis.make_file("fld-%04d" % step)
        vis.add_data(visf, [
                    ("u", u), 
                    #("u_true", u_true), 
                    ], 
                    #expressions=[("error", "u-u_true")]
                    time=t, 
                    step=step
                    )
        visf.close()

        u = stepper(u, t, dt, op.rhs)

        #u_true = discr.interpolate_volume_function(
                #lambda x: u_analytic(t, x))

    vis.close()


if __name__ == "__main__":
    import cProfile as profile
    profile.run("main()", "advec.prof")
    #main()


