# Hedge - the Hybrid'n'Easy DG Environment
# Copyright (C) 2007 Andreas Kloeckner
# Adapted 2010 by David Powell
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

"""This example is for maxwell's equations in a 2D rectangular cavity
with inhomogeneous dielectric filling"""

from __future__ import division
import numpy

CAVITY_GEOMETRY = """
// a rectangular cavity with a dielectric in one region

lc = 10e-3;
height = 50e-3;
air_width = 100e-3;
dielectric_width = 50e-3;

Point(1) = {0, 0, 0, lc};
Point(2) = {air_width, 0, 0, lc};
Point(3) = {air_width+dielectric_width, 0, 0, lc};
Point(4) = {air_width+dielectric_width, height, 0, lc};
Point(5) = {air_width, height, 0, lc};
Point(6) = {0, height, 0, lc};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 5};
Line(5) = {5, 6};
Line(6) = {6, 1};
Line(7) = {2, 5};

Line Loop(1) = {1, 7, 5, 6};
Line Loop(2) = {2, 3, 4, -7};

Plane Surface(1) = {1};
Plane Surface(2) = {2};

Physical Surface("vacuum") = {1};
Physical Surface("dielectric") = {2};
"""


def main(write_output=True, allow_features=None, flux_type_arg=1,
        bdry_flux_type_arg=None, extra_discr_args={}):
    from math import sqrt, pi
    from hedge.models.em import TEMaxwellOperator

    from hedge.backends import guess_run_context
    rcon = guess_run_context(allow_features)

    epsilon0 = 8.8541878176e-12 # C**2 / (N m**2)
    mu0 = 4*pi*1e-7 # N/A**2.
    c = 1/sqrt(mu0*epsilon0)

    materials = {"vacuum" : (epsilon0, mu0),
                 "dielectric" : (2*epsilon0, mu0)}

    output_dir = "2d_cavity"

    import os
    if not os.access(output_dir, os.F_OK):
        os.makedirs(output_dir)

    # should no tag raise an error or default to free space?
    def eps_val(x, el):
        for key in materials.keys():
            if el in material_elements[key]:
                return materials[key][0]
        raise ValueError, "Element does not belong to any material"

    def mu_val(x, el):
        for key in materials.keys():
            if el in material_elements[key]:
                return materials[key][1]
        raise ValueError, "Element does not belong to any material"

    # geometry of cavity
    d = 100e-3
    a = 150e-3

    # analytical frequency and transverse wavenumbers of resonance
    f0 = 9.0335649907522321e8
    h = 2*pi*f0/c
    l = -h*sqrt(2)

    # substitute the following and change materials for a homogeneous cavity
    #h = pi/a
    #l =-h

    def initial_val(discr):
        # the initial solution for the TE_10-like mode
        def initial_Hz(x, el):
            from math import cos, sin
            if el in material_elements["vacuum"]:
                return h*cos(h*x[0])
            else:
                return -l*sin(h*d)/sin(l*(a-d))*cos(l*(a-x[0]))

        from hedge.tools import make_obj_array
        result_zero = discr.volume_zeros(kind="numpy", dtype=numpy.float64)
        H_z = make_tdep_given(initial_Hz).volume_interpolant(0, discr)
        return make_obj_array([result_zero, result_zero, H_z])

    if rcon.is_head_rank:
        from hedge.mesh.reader.gmsh import generate_gmsh
        mesh = generate_gmsh(CAVITY_GEOMETRY, 2, force_dimension=2)
        mesh_data = rcon.distribute_mesh(mesh)
    else:
        mesh_data = rcon.receive_mesh()

    # Work out which elements belong to each material
    material_elements = {}
    for key in materials.keys():
        material_elements[key] = set(mesh_data.tag_to_elements[key])

    order = 3
    #extra_discr_args.setdefault("debug", []).append("cuda_no_plan")
    #extra_discr_args.setdefault("debug", []).append("dump_optemplate_stages")

    from hedge.data import make_tdep_given
    from hedge.mesh import TAG_ALL

    op = TEMaxwellOperator(epsilon=make_tdep_given(eps_val), mu=make_tdep_given(mu_val), \
            flux_type=flux_type_arg, \
            bdry_flux_type=bdry_flux_type_arg, dimensions=2, pec_tag=TAG_ALL)
    # op = TEMaxwellOperator(epsilon=epsilon0, mu=mu0,
            # flux_type=flux_type_arg, \
            # bdry_flux_type=bdry_flux_type_arg, dimensions=2, pec_tag=TAG_ALL)

    discr = rcon.make_discretization(mesh_data, order=order,
            tune_for=op.op_template(),
            **extra_discr_args)

    # create the initial solution
    fields = initial_val(discr)

    from hedge.visualization import VtkVisualizer
    if write_output:
        from os.path import join
        vis = VtkVisualizer(discr, rcon, join(output_dir, "cav-%d" % order))

    # monitor the solution at a point to find the resonant frequency
    try:
        point_getter = discr.get_point_evaluator(numpy.array([75e-3, 25e-3, 0])) #[0.25, 0.25, 0.25]))
    except RuntimeError:
        point_getter = None

    if rcon.is_head_rank:
        print "---------------------------------------------"
        print "order %d" % order
        print "---------------------------------------------"
        print "#elements=", len(mesh.elements)

    from hedge.timestep.runge_kutta import LSRK4TimeStepper
    stepper = LSRK4TimeStepper(dtype=discr.default_scalar_type, rcon=rcon)
    #from hedge.timestep.dumka3 import Dumka3TimeStepper
    #stepper = Dumka3TimeStepper(3, dtype=discr.default_scalar_type, rcon=rcon)

    # diagnostics setup ---------------------------------------------------
    from pytools.log import LogManager, add_general_quantities, \
            add_simulation_quantities, add_run_info

    if write_output:
        from os.path import join
        log_file_name = join(output_dir, "cavity-%d.dat" % order)
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

    #from hedge.log import EMFieldGetter, add_em_quantities
    #field_getter = EMFieldGetter(discr, op, lambda: fields)
    #add_em_quantities(logmgr, op, field_getter)

    logmgr.add_watches(
            ["step.max", "t_sim.max",
                #("W_field", "W_el+W_mag"),
                "t_step.max"]
            )

    # timestep loop -------------------------------------------------------
    rhs = op.bind(discr)
    final_time = 10e-9

    if point_getter is not None:
        from os.path import join
        pointfile = open(join(output_dir, "point.txt"), "wt")
        done_dt = False
    try:
        from hedge.timestep import times_and_steps
        from os.path import join
        step_it = times_and_steps(
                final_time=final_time, logmgr=logmgr,
                max_dt_getter=lambda t: op.estimate_timestep(discr,
                    stepper=stepper, t=t, fields=fields))

        for step, t, dt in step_it:
            if step % 10 == 0 and write_output:
                sub_timer = vis_timer.start_sub_timer()
                e, h = op.split_eh(fields)
                visf = vis.make_file(join(output_dir, "cav-%d-%04d") % (order, step))
                vis.add_data(visf,
                        [
                            ("e",
                                discr.convert_volume(e, kind="numpy")),
                            ("h",
                                discr.convert_volume(h, kind="numpy")),],
                        time=t, step=step
                        )
                visf.close()
                sub_timer.stop().submit()

            fields = stepper(fields, t, dt, rhs)
            if point_getter is not None:
                val = point_getter(fields)
                #print val
                if not done_dt:
                    pointfile.write("#%g\n" % dt)
                    done_dt = True
                pointfile.write("%g\n" %val[0])

    finally:
        if write_output:
            vis.close()

        logmgr.close()
        discr.close()

        if point_getter is not None:
            pointfile.close()







# entry point -----------------------------------------------------------------
if __name__ == "__main__":
    main(write_output=True)
