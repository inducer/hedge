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

def main(write_output=True, order=6):
    from hedge.data import TimeConstantGivenFunction, \
            GivenFunction
    from os.path import join
    from hedge.backends import guess_run_context
    rcon = guess_run_context()

    dim = 3
    output_dir = "octahedron"
    
    import os
    if not os.access(output_dir, os.F_OK):
        os.makedirs(output_dir)

    if rcon.is_head_rank:
        from hedge.mesh.reader.gmsh import read_gmsh
        mesh = read_gmsh("octahedron.msh", 
                boundary_tagger=lambda x,y,z,w: ["traction"])

    if rcon.is_head_rank:
        print "%d elements" % len(mesh.elements)
        mesh_data = rcon.distribute_mesh(mesh)
    else:
        mesh_data = rcon.receive_mesh()

    class Displacement:
        shape = (3,)
        def __call__(self, x, el):
            R = x[0] + x[1] + x[2]
            return [-R/30, -R/30, -R/30]
    
    final_time = 3
    
    discr = rcon.make_discretization(mesh_data, order=order, 
            debug=[])

    from hedge.visualization import VtkVisualizer
    if write_output:
        vis = VtkVisualizer(discr, rcon, join(output_dir, "test-%d" % order))
        
    if rcon.is_head_rank:
        print "order %d" % order
        print "#elements=", len(mesh.elements)
 
    from hedge.mesh import TAG_NONE, TAG_ALL
    from hedge.models.solid_mechanics import SolidMechanicsOperator
    from hedge.models.solid_mechanics.constitutive_laws import NeoHookean
    
    material = NeoHookean(50, 10, 0.3)
    
    op = SolidMechanicsOperator(material, 
            init_displacement=GivenFunction(Displacement()),
            dimensions=discr.dimensions)
    fields = op.assemble_vars(discr=discr)
    
    from hedge.timestep import LSRK4TimeStepper
    stepper = LSRK4TimeStepper()
    from time import time
    last_tsep = time()
    t = 0

    # diagnostics setup -------------------------------------------------
    from pytools.log import LogManager, add_general_quantities, \
            add_simulation_quantities, add_run_info
    if write_output:
        log_file_name = join(output_dir, "oct-%d.dat" % order)
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
    logmgr.add_watches(["step.max", "t_sim.max", "t_step.max"])
    
    p_calc = op.bind_stress_calculator(discr)
    rhs = op.bind(discr)

    try:
        from hedge.timestep import times_and_steps
        step_it = times_and_steps(
                final_time=final_time, logmgr=logmgr,
                max_dt_getter=lambda t: op.estimate_timestep(discr,
                    stepper=stepper, t=t, fields=fields))

        for step, t, dt in step_it:
            u, v = op.split_vars(fields)
            P    = p_calc(u)
            if step % 5 == 0 and write_output:
                visf = vis.make_file(join(output_dir, "oct-%d-%04d" % (order, step)))
                vis.add_data(visf,
                    [
                        ("u", discr.convert_volume(u, "numpy")),
                        ("v", discr.convert_volume(v, "numpy")),
                        ("P", discr.convert_volume(P, "numpy"))
                        ],
                    time=t, step=step
                    )
                visf.close()
            
            fields = stepper(fields, t, dt, rhs)
    finally:
        if write_output:
            vis.close()
        logmgr.close()
        discr.close()


if __name__ == "__main__":
    main()




# entry points for py.test ----------------------------------------------------
from pytools.test import mark_test
@mark_test.long
def test_solid():
    main(write_output=False)
