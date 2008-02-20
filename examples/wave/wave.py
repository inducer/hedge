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
from pytools.arithmetic_container import ArithmeticList, join_fields
from hedge.tools import dot




def main() :
    from hedge.element import \
            IntervalElement, \
            TriangularElement, \
            TetrahedralElement
    from hedge.timestep import RK4TimeStepper, AdamsBashforthTimeStepper
    from hedge.mesh import \
            make_disk_mesh, \
            make_regular_square_mesh, \
            make_square_mesh, \
            make_ball_mesh
    from hedge.visualization import SiloVisualizer, VtkVisualizer
    from pytools.stopwatch import Job
    from math import sin, cos, pi, exp, sqrt
    from hedge.parallel import guess_parallelization_context

    pcon = guess_parallelization_context()

    dim = 1

    if dim == 1:
        if pcon.is_head_rank:
            from hedge.mesh import make_uniform_1d_mesh
            mesh = make_uniform_1d_mesh(-1, 1, 50)

        el_class = IntervalElement
    elif dim == 2:
        if pcon.is_head_rank:
            mesh = make_disk_mesh(max_area=1e-3)
            #mesh = make_regular_square_mesh(
                    #n=9, periodicity=(True,True))
            #mesh = make_square_mesh(max_area=0.008)
            #mesh.transform(Rotation(pi/8))
        el_class = TriangularElement
    elif dim == 3:
        if pcon.is_head_rank:
            mesh = make_ball_mesh(max_volume=0.0005)
        el_class = TetrahedralElement
    else:
        raise RuntimeError, "bad number of dimensions"

    if pcon.is_head_rank:
        print "%d elements" % len(mesh.elements)
        mesh_data = pcon.distribute_mesh(mesh)
    else:
        mesh_data = pcon.receive_mesh()

    discr = pcon.make_discretization(mesh_data, el_class(3))
    stepper = RK4TimeStepper()
    #stepper = AdamsBashforthTimeStepper(1)
    #vis = VtkVisualizer(discr, pcon, "fld")
    vis = SiloVisualizer(discr, pcon)

    def source_u(x):
        return exp(-x*x*1024)

    source_u_vec = discr.interpolate_volume_function(source_u)

    def source_vec_getter(t):
        if t > 1e-2:
            return discr.volume_zeros()
        else:
            return source_u_vec

    from hedge.operators import StrongWaveOperator
    from hedge.mesh import TAG_ALL, TAG_NONE
    op = StrongWaveOperator(1, discr, 
            source_vec_getter,
            dirichlet_tag=TAG_NONE,
            neumann_tag=TAG_NONE,
            radiation_tag=TAG_ALL,
            flux_type="central",
            )

    fields = join_fields(discr.volume_zeros(),
            [discr.volume_zeros() for i in range(discr.dimensions)])
    #fields = join_fields(
            #discr.interpolate_volume_function(lambda x: sin(x[0])),
            #[discr.volume_zeros() for i in range(discr.dimensions)]) # v

    dt = discr.dt_factor(op.max_eigenvalue()) / 2
    nsteps = int(3/dt)
    if pcon.is_head_rank:
        print "dt", dt
        print "nsteps", nsteps

    # diagnostics setup -------------------------------------------------------
    from pytools.log import LogManager, \
            add_general_quantities, \
            add_simulation_quantities, \
            add_run_info

    logmgr = LogManager("wave.dat", pcon.communicator)
    add_run_info(logmgr)
    add_general_quantities(logmgr)
    add_simulation_quantities(logmgr, dt)
    discr.add_instrumentation(logmgr)

    from pytools.log import IntervalTimer
    vis_timer = IntervalTimer("t_vis", "Time spent visualizing")
    logmgr.add_quantity(vis_timer)
    stepper.add_instrumentation(logmgr)

    from hedge.log import Integral, L1Norm, L2Norm, VariableGetter
    u_getter = VariableGetter(locals(), "fields", 0)
    logmgr.add_quantity(L1Norm(u_getter, discr, name="l1_u"))
    logmgr.add_quantity(L2Norm(u_getter, discr, name="l2_u"))

    logmgr.add_watches(["step.max", "t_sim.max", "l2_u", "t_step.max"])

    # timestep loop -----------------------------------------------------------
    for step in range(nsteps):
        logmgr.tick()

        t = step*dt

        fields = stepper(fields, t, dt, op.rhs)

        if step % 1 == 0:
            def vlmz(f): return discr.volumize_boundary_field(f, TAG_ALL)
            visf = vis.make_file("fld-%04d" % step)

            vis.add_data(visf,
                    [
                        ("u", fields[0]),
                        ("v", fields[1:]), 
                    ],
                    time=t,
                    scale_factor=2e1,
                    step=step)
            visf.close()

    vis.close()

    logmgr.tick()
    logmgr.save()

if __name__ == "__main__":
    #import cProfile as profile
    #profile.run("main()", "wave2d.prof")
    main()

