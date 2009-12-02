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
from hedge.mesh import TAG_ALL, TAG_NONE




def main(write_output=True, 
        flux_type_arg="upwind", dtype=numpy.float64, debug=[]):
    from pytools.stopwatch import Job
    from math import sin, cos, pi, exp, sqrt

    from hedge.backends import guess_run_context
    rcon = guess_run_context()

    if rcon.is_head_rank:
        from hedge.mesh.reader.gmsh import generate_gmsh
        mesh = generate_gmsh(GEOMETRY, 2,
                allow_internal_boundaries=True)

        print "%d elements" % len(mesh.elements)
        mesh_data = rcon.distribute_mesh(mesh)
    else:
        mesh_data = rcon.receive_mesh()

    discr = rcon.make_discretization(mesh_data, order=4, debug=debug,
            default_scalar_type=dtype)
    from hedge.timestep import RK4TimeStepper
    stepper = RK4TimeStepper(dtype=dtype)

    from hedge.visualization import VtkVisualizer
    if write_output:
        vis = VtkVisualizer(discr, rcon, "fld")

    def source_u(x, el):
        return exp(-numpy.dot(x, x)*128)

    from hedge.models.wave import StrongWaveOperator
    from hedge.mesh import TAG_ALL, TAG_NONE
    from hedge.data import \
            make_tdep_given, \
            TimeHarmonicGivenFunction, \
            TimeIntervalGivenFunction

    op = StrongWaveOperator(-1, discr.dimensions, 
            source_f=TimeIntervalGivenFunction(
                TimeHarmonicGivenFunction(
                    make_tdep_given(source_u), omega=10),
                0, 1),
            dirichlet_tag=TAG_ALL,
            neumann_tag=TAG_NONE,
            radiation_tag=TAG_NONE,
            flux_type=flux_type_arg
            )

    from hedge.tools import join_fields
    fields = join_fields(discr.volume_zeros(dtype=dtype),
            [discr.volume_zeros(dtype=dtype) for i in range(discr.dimensions)])

    # diagnostics setup -------------------------------------------------------
    from pytools.log import LogManager, \
            add_general_quantities, \
            add_simulation_quantities, \
            add_run_info

    if write_output:
        log_file_name = "wiggly.dat"
    else:
        log_file_name = None

    logmgr = LogManager(log_file_name, "w", rcon.communicator)
    add_run_info(logmgr)
    add_general_quantities(logmgr)
    add_simulation_quantities(logmgr)
    discr.add_instrumentation(logmgr)

    stepper.add_instrumentation(logmgr)

    logmgr.add_watches(["step.max", "t_sim.max", "t_step.max"])

    # timestep loop -----------------------------------------------------------
    rhs = op.bind(discr)
    try:
        from hedge.timestep import times_and_steps
        step_it = times_and_steps(
                final_time=4, logmgr=logmgr,
                max_dt_getter=lambda t: op.estimate_timestep(discr,
                    stepper=stepper, t=t, fields=fields))

        for step, t, dt in step_it:
            if step % 10 == 0 and write_output:
                visf = vis.make_file("fld-%04d" % step)

                vis.add_data(visf,
                        [
                            ("u", fields[0]),
                            ("v", fields[1:]), 
                        ],
                        time=t,
                        step=step)
                visf.close()

            fields = stepper(fields, t, dt, rhs)

        assert discr.norm(fields) < 1
        assert fields[0].dtype == dtype

    finally:
        if write_output:
            vis.close()

        logmgr.close()
        discr.close()

GEOMETRY = """
w = 1;
dx = 0.2;
ch_width = 0.2;
rows = 4;

Point(0) = {0,0,0};
Point(1) = {w,0,0};

bottom_line = newl;
Line(bottom_line) = {0,1};

left_pts[] = { 0 };
right_pts[] = { 1 };

left_pts[] = { };
emb_lines[] = {};

For row In {1:rows}
  If (row % 2 == 0)
    // left
    rp = newp; Point(rp) = {w,dx*row, 0};
    right_pts[] += {rp};

    mp = newp; Point(mp) = {ch_width,dx*row, 0};
    emb_line = newl; Line(emb_line) = {mp,rp};
    emb_lines[] += {emb_line};
  EndIf
  If (row % 2)
    // right
    lp = newp; Point(lp) = {0,dx*row, 0};
    left_pts[] += {lp};

    mp = newp; Point(mp) = { w-ch_width,dx*row, 0};
    emb_line = newl; Line(emb_line) = {mp,lp};
    emb_lines[] += {emb_line};
  EndIf
EndFor

lep = newp; Point(lep) = {0,(rows+1)*dx,0};
rep = newp; Point(rep) = {w,(rows+1)*dx,0};
top_line = newl; Line(top_line) = {lep,rep};

left_pts[] += { lep };
right_pts[] += { rep };

lines[] = {bottom_line};

For i In {0:#right_pts[]-2}
  l = newl; Line(l) = {right_pts[i], right_pts[i+1]};
  lines[] += {l};
EndFor

lines[] += {-top_line};

For i In {#left_pts[]-1:0:-1}
  l = newl; Line(l) = {left_pts[i], left_pts[i-1]};
  lines[] += {l};
EndFor

Line Loop (1) = lines[];

Plane Surface (1) = {1};
Physical Surface(1) = {1};

For i In {0:#emb_lines[]-1}
  Line { emb_lines[i] } In Surface { 1 };
EndFor

boundary_lines[] = {};
boundary_lines[] += lines[];
boundary_lines[] += emb_lines[];
Physical Line ("boundary") = boundary_lines[];

Mesh.CharacteristicLengthFactor = 0.4;
"""

if __name__ == "__main__":
    main()





