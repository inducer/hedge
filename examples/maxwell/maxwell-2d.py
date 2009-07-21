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




def main(write_output=True):
    from math import sqrt, pi, exp

    from hedge.backends import guess_run_context
    rcon = guess_run_context()

    epsilon0 = 8.8541878176e-12 # C**2 / (N m**2)
    mu0 = 4*pi*1e-7 # N/A**2.
    epsilon = 1*epsilon0
    mu = 1*mu0

    cylindrical = False
    periodic = False

    from hedge.mesh import make_disk_mesh
    mesh = make_disk_mesh(r=0.5)

    if rcon.is_head_rank:
        mesh_data = rcon.distribute_mesh(mesh)
    else:
        mesh_data = rcon.receive_mesh()

    class CurrentSource:
        shape = (3,)

        def __call__(self, x, el):
            return [0,0,exp(-80*la.norm(x))]

    order = 3
    discr = rcon.make_discretization(mesh_data, order=order)

    from hedge.visualization import VtkVisualizer
    if write_output:
        vis = VtkVisualizer(discr, rcon, "em-%d" % order)

    dt = discr.dt_factor(1/sqrt(mu*epsilon))
    final_time = dt*200
    nsteps = int(final_time/dt)+1
    dt = final_time/nsteps

    if rcon.is_head_rank:
        print "order %d" % order
        print "dt", dt
        print "nsteps", nsteps
        print "#elements=", len(mesh.elements)

    from hedge.mesh import TAG_ALL, TAG_NONE
    from hedge.models.em import TMMaxwellOperator
    from hedge.data import GivenFunction, TimeIntervalGivenFunction
    op = TMMaxwellOperator(epsilon, mu, flux_type=1,
            current=TimeIntervalGivenFunction(
                GivenFunction(CurrentSource()), off_time=final_time/10),
            absorb_tag=TAG_ALL, pec_tag=TAG_NONE)
    fields = op.assemble_eh(discr=discr)

    from hedge.timestep import RK4TimeStepper
    stepper = RK4TimeStepper()
    from time import time
    last_tstep = time()
    t = 0

    # diagnostics setup ---------------------------------------------------
    from pytools.log import LogManager, add_general_quantities, \
            add_simulation_quantities, add_run_info

    if write_output:
        log_file_name = "maxwell-%d.dat" % order
    else:
        log_file_name = None

    logmgr = LogManager(log_file_name, "w", rcon.communicator)
    add_run_info(logmgr)
    add_general_quantities(logmgr)
    add_simulation_quantities(logmgr, dt)
    discr.add_instrumentation(logmgr)
    stepper.add_instrumentation(logmgr)

    from pytools.log import IntervalTimer
    vis_timer = IntervalTimer("t_vis", "Time spent visualizing")
    logmgr.add_quantity(vis_timer)

    from hedge.log import EMFieldGetter, add_em_quantities
    field_getter = EMFieldGetter(discr, op, lambda: fields)
    add_em_quantities(logmgr, op, field_getter)

    logmgr.add_watches(["step.max", "t_sim.max", 
        ("W_field", "W_el+W_mag"), "t_step.max"])

    # timestep loop -------------------------------------------------------
    rhs = op.bind(discr)

    try:
        for step in range(nsteps):
            logmgr.tick()

            if step % 10 == 0 and write_output:
                e, h = op.split_eh(fields)
                visf = vis.make_file("em-%d-%04d" % (order, step))
                vis.add_data(visf,
                        [ ("e", e), ("h", h), ],
                        time=t, step=step
                        )
                visf.close()

            fields = stepper(fields, t, dt, rhs)
            t += dt

        assert discr.norm(fields) < 0.03
    finally:
        if write_output:
            vis.close()

        logmgr.close()
        discr.close()

if __name__ == "__main__":
    import cProfile as profile
    #profile.run("main()", "wave2d.prof")
    main()




# entry points for py.test ----------------------------------------------------
from pytools.test import mark_test
@mark_test(long=True)
def test_maxwell_2d():
    main(write_output=False)
