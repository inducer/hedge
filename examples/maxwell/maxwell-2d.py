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

"Maxwell's equation example with fixed material coefficients"


from __future__ import division
import numpy.linalg as la


def main(write_output=True):
    from math import sqrt, pi, exp
    from os.path import join

    from hedge.backends import guess_run_context
    rcon = guess_run_context()

    epsilon0 = 8.8541878176e-12 # C**2 / (N m**2)
    mu0 = 4*pi*1e-7 # N/A**2.
    epsilon = 1*epsilon0
    mu = 1*mu0

    output_dir = "maxwell-2d"
    import os
    if not os.access(output_dir, os.F_OK):
        os.makedirs(output_dir)
    
    from hedge.mesh.generator import make_disk_mesh
    mesh = make_disk_mesh(r=0.5, max_area=1e-3)

    if rcon.is_head_rank:
        mesh_data = rcon.distribute_mesh(mesh)
    else:
        mesh_data = rcon.receive_mesh()

    class CurrentSource:
        shape = (3,)

        def __call__(self, x, el):
            return [0,0,exp(-80*la.norm(x))]

    order = 3
    final_time = 1e-8
    discr = rcon.make_discretization(mesh_data, order=order,
            debug=["cuda_no_plan"])

    from hedge.visualization import VtkVisualizer
    if write_output:
        vis = VtkVisualizer(discr, rcon, join(output_dir, "em-%d" % order))

    if rcon.is_head_rank:
        print "order %d" % order
        print "#elements=", len(mesh.elements)

    from hedge.mesh import TAG_ALL, TAG_NONE
    from hedge.models.em import TMMaxwellOperator
    from hedge.data import make_tdep_given, TimeIntervalGivenFunction
    op = TMMaxwellOperator(epsilon, mu, flux_type=1,
            current=TimeIntervalGivenFunction(
                make_tdep_given(CurrentSource()), off_time=final_time/10),
            absorb_tag=TAG_ALL, pec_tag=TAG_NONE)
    fields = op.assemble_eh(discr=discr)

    from hedge.timestep import LSRK4TimeStepper
    stepper = LSRK4TimeStepper()
    from time import time
    last_tstep = time()
    t = 0

    # diagnostics setup ---------------------------------------------------
    from pytools.log import LogManager, add_general_quantities, \
            add_simulation_quantities, add_run_info

    if write_output:
        log_file_name = join(output_dir, "maxwell-%d.dat" % order)
    else:
        log_file_name = None

    logmgr = LogManager(log_file_name, "w", rcon.communicator)
    add_run_info(logmgr)
    add_general_quantities(logmgr)
    add_simulation_quantities(logmgr)
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
        from hedge.timestep import times_and_steps
        step_it = times_and_steps(
                final_time=final_time, logmgr=logmgr,
                max_dt_getter=lambda t: op.estimate_timestep(discr,
                    stepper=stepper, t=t, fields=fields))

        for step, t, dt in step_it:
            if step % 10 == 0 and write_output:
                e, h = op.split_eh(fields)
                visf = vis.make_file(join(output_dir, "em-%d-%04d" % (order, step)))
                vis.add_data(visf,
                        [
                            ("e", discr.convert_volume(e, "numpy")),
                            ("h", discr.convert_volume(h, "numpy")),
                            ],
                        time=t, step=step
                        )
                visf.close()

            fields = stepper(fields, t, dt, rhs)

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
@mark_test.long
def test_maxwell_2d():
    main(write_output=False)
