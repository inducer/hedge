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
import pylinear.computation as comp




def main() :
    from hedge.element import \
            TriangularElement, \
            TetrahedralElement
    from hedge.timestep import RK4TimeStepper
    from hedge.discretization import Discretization, ones_on_boundary
    from hedge.visualization import SiloVisualizer, VtkVisualizer
    from hedge.tools import dot, mem_checkpoint
    from pytools.arithmetic_container import ArithmeticList
    from pytools.stopwatch import Job
    from math import sin, cos, pi, sqrt
    from hedge.parallel import \
            guess_parallelization_context, \
            reassemble_volume_field
    from math import floor

    def f(x):
        return sin(pi*x)
        #if int(floor(x)) % 2 == 0:
            #return 1
        #else:
            #return 0

    def u_analytic(x, t):
        return f((a*x/norm_a+t*norm_a))

    def boundary_tagger(vertices, el, face_nr):
        if el.face_normals[face_nr] * a > 0:
            return ["inflow"]
        else:
            return ["outflow"]

    pcon = guess_parallelization_context()

    dim = 3

    job = Job("mesh")
    if dim == 2:
        a = num.array([-1,0])
        if pcon.is_head_rank:
            from hedge.mesh import \
                    make_disk_mesh, \
                    make_square_mesh, \
                    make_rect_mesh, \
                    make_regular_square_mesh, \
                    make_regular_rect_mesh, \
                    make_single_element_mesh
        
            #mesh = make_square_mesh(max_area=0.0003, boundary_tagger=boundary_tagger)
            #mesh = make_regular_square_mesh(a=-r, b=r, boundary_tagger=boundary_tagger, n=3)
            #mesh = make_single_element_mesh(boundary_tagger=boundary_tagger)
            #mesh = make_disk_mesh(r=pi, boundary_tagger=boundary_tagger, max_area=0.5)
            #mesh = make_disk_mesh(boundary_tagger=boundary_tagger)
            mesh = make_rect_mesh(
                    (-0.5, -0.5),
                    (10, 0.5),
                    max_area=0.3,
                    boundary_tagger=boundary_tagger)
        el_class = TriangularElement
    elif dim == 3:
        a = num.array([0,0,0.3])
        if pcon.is_head_rank:
            from hedge.mesh import make_cylinder_mesh, make_ball_mesh, make_box_mesh

            mesh = make_cylinder_mesh(max_volume=0.004, boundary_tagger=boundary_tagger,
                    periodic=False, radial_subdivisions=32)
            #mesh = make_box_mesh(dimensions=(1,1,2*pi/3), max_volume=0.01,
                    #boundary_tagger=boundary_tagger)
            #mesh = make_box_mesh(max_volume=0.01, boundary_tagger=boundary_tagger)
            #mesh = make_ball_mesh(boundary_tagger=boundary_tagger)
            #mesh = make_cylinder_mesh(max_volume=0.01, boundary_tagger=boundary_tagger)
        el_class = TetrahedralElement
    else:
        raise RuntimeError, "bad number of dimensions"

    norm_a = comp.norm_2(a)

    if pcon.is_head_rank:
        mesh_data = pcon.distribute_mesh(mesh)
    else:
        mesh_data = pcon.receive_mesh()
    job.done()

    job = Job("discretization")
    mesh_data = mesh_data.reordered_by("cuthill")
    discr = pcon.make_discretization(mesh_data, el_class(7))
    vis_discr = discr
    job.done()

    dt = discr.dt_factor(2*norm_a)
    nsteps = int(20/dt)

    if pcon.is_head_rank:
        print "%d elements, dt=%g, nsteps=%d" % (
                len(discr.mesh.elements),
                dt,
                nsteps)

    vis = SiloVisualizer(vis_discr, pcon)
    #vis = VtkVisualizer(vis_discr, pcon, "fld")

    # operator setup ----------------------------------------------------------
    from hedge.data import \
            ConstantGivenFunction, \
            TimeConstantGivenFunction, \
            TimeDependentGivenFunction
    from hedge.operators import StrongAdvectionOperator, WeakAdvectionOperator
    op = StrongAdvectionOperator(discr, a, 
            inflow_u=TimeConstantGivenFunction(ConstantGivenFunction()),
            #inflow_u=TimeDependentGivenFunction(u_analytic)),
            flux_type="lf")

    #from sizer import scanner
    #objs = scanner.Objects()
    #import code
    #code.interact(local = {'objs': objs})

    #from pyrticle._internal import ShapeFunction
    #sf = ShapeFunction(0.25, 2)

    def gauss_hump(x):
        from math import exp
        rsquared = (x*x)/(0.1**2)
        return exp(-rsquared)
    def gauss2_hump(x):
        from math import exp
        rsquared = (x*x)/(0.1**2)
        return exp(-rsquared)-0.5*exp(-rsquared/2)

    u = discr.interpolate_volume_function(lambda x: u_analytic(x, 0))
    #u = discr.interpolate_volume_function(sf)

    stepper = RK4TimeStepper()

    # diagnostics setup -------------------------------------------------------
    from pytools.log import LogManager, \
            add_general_quantities, \
            add_simulation_quantities, \
            add_run_info

    logmgr = LogManager("advection.dat", pcon.communicator)
    add_run_info(logmgr)
    add_general_quantities(logmgr)
    add_simulation_quantities(logmgr, dt)
    discr.add_instrumentation(logmgr)

    from pytools.log import IntervalTimer
    vis_timer = IntervalTimer("t_vis", "Time spent visualizing")
    logmgr.add_quantity(vis_timer)
    stepper.add_instrumentation(logmgr)

    from hedge.log import Integral, L1Norm, L2Norm, VariableGetter
    logmgr.add_quantity(Integral(VariableGetter(locals(), "u"), discr))
    logmgr.add_quantity(L1Norm(VariableGetter(locals(), "u"), discr))
    logmgr.add_quantity(L2Norm(VariableGetter(locals(), "u"), discr))

    logmgr.add_watches(["step.max", "t_sim.max", "l2_u", "t_step.max"])

    # timestep loop -----------------------------------------------------------
    for step in range(nsteps):
        logmgr.tick()

        t = step*dt

        if step % 5 == 0:
            vis_timer.start()
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
            vis_timer.stop()

        u = stepper(u, t, dt, op.rhs)

        #u_true = discr.interpolate_volume_function(
                #lambda x: u_analytic(t, x))

    vis.close()

    logmgr.tick()
    logmgr.save()


if __name__ == "__main__":
    #import cProfile as profile
    #profile.run("main()", "advec.prof")
    main()


