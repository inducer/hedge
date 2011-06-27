# Hedge - the Hybrid'n'Easy DG Environment
# Copyright (C) 2009 Andreas Stock
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



def main(write_output=True, flux_type_arg="central", use_quadrature=True,
        final_time=20):
    from math import sin, cos, pi, sqrt

    from hedge.backends import guess_run_context
    rcon = guess_run_context()

    # mesh setup --------------------------------------------------------------
    if rcon.is_head_rank:
        #from hedge.mesh.generator import make_disk_mesh
        #mesh = make_disk_mesh()
        from hedge.mesh.generator import make_rect_mesh
        mesh = make_rect_mesh(a=(-1,-1),b=(1,1),max_area=0.008)

    if rcon.is_head_rank:
        mesh_data = rcon.distribute_mesh(mesh)
    else:
        mesh_data = rcon.receive_mesh()

    # space-time-dependent-velocity-field -------------------------------------
    # simple vortex
    class TimeDependentVField:
        """ `TimeDependentVField` is a callable expecting `(x, t)` representing space and time

        `x` is of the length of the spatial dimension and `t` is the time."""
        shape = (2,)

        def __call__(self, pt, el, t):
            x, y = pt
            # Correction-Factor to make the speed zero on the on the boundary
            #fac = (1-x**2)*(1-y**2)
            fac = 1.
            return numpy.array([-y*fac, x*fac]) * cos(pi*t)

    class VField:
        """ `VField` is a callable expecting `(x)` representing space

        `x` is of the length of the spatial dimension."""
        shape = (2,)

        def __call__(self, pt, el):
            x, y = pt
            # Correction-Factor to make the speed zero on the on the boundary
            #fac = (1-x**2)*(1-y**2)
            fac = 1.
            return numpy.array([-y*fac, x*fac])

    # space-time-dependent State BC (optional)-----------------------------------
    class TimeDependentBc_u:
        """ space and time dependent BC for state u"""
        def __call__(self, pt, el, t):
            x, y = pt
            if t <= 0.5:
                if x > 0:
                    return 1
                else:
                    return 0
            else:
                return 0

    class Bc_u:
        """ Only space dependent BC for state u"""
        def __call__(seld, pt, el):
            x, y = pt
            if x > 0:
                return 1
            else:
                return 0


    # operator setup ----------------------------------------------------------
    # In the operator setup it is possible to switch between a only space
    # dependent velocity field `VField` or a time and space dependent
    # `TimeDependentVField`.
    # For `TimeDependentVField`: advec_v=TimeDependentGivenFunction(VField())
    # For `VField`: advec_v=TimeConstantGivenFunction(GivenFunction(VField()))
    # Same for the Bc_u Function! If you don't define Bc_u then the BC for u = 0.

    from hedge.data import \
            ConstantGivenFunction, \
            TimeConstantGivenFunction, \
            TimeDependentGivenFunction, \
            GivenFunction
    from hedge.models.advection import VariableCoefficientAdvectionOperator
    op = VariableCoefficientAdvectionOperator(mesh.dimensions,
        #advec_v=TimeDependentGivenFunction(
        #    TimeDependentVField()),
        advec_v=TimeConstantGivenFunction(
            GivenFunction(VField())),
        #bc_u_f=TimeDependentGivenFunction(
        #    TimeDependentBc_u()),
        bc_u_f=TimeConstantGivenFunction(
            GivenFunction(Bc_u())),
        flux_type=flux_type_arg)

    # discretization setup ----------------------------------------------------
    order = 5
    if use_quadrature:
        quad_min_degrees = {"quad": 3*order}
    else:
        quad_min_degrees = {}

    discr = rcon.make_discretization(mesh_data, order=order,
            default_scalar_type=numpy.float64, 
            debug=["cuda_no_plan"],
            quad_min_degrees=quad_min_degrees,
            tune_for=op.op_template(),

            )
    vis_discr = discr

    # visualization setup -----------------------------------------------------
    from hedge.visualization import VtkVisualizer
    if write_output:
        vis = VtkVisualizer(vis_discr, rcon, "fld")

    # initial condition -------------------------------------------------------
    if True:
        def initial(pt, el):
            # Gauss pulse
            from math import exp
            x = (pt-numpy.array([0.3, 0.5]))*8
            return exp(-numpy.dot(x, x))
    else:
        def initial(pt, el):
            # Rectangle
            x, y = pt
            if abs(x) < 0.5 and abs(y) < 0.2:
                return 2
            else:
                return 1

    u = discr.interpolate_volume_function(initial)

    # timestep setup ----------------------------------------------------------
    from hedge.timestep.runge_kutta import LSRK4TimeStepper
    stepper = LSRK4TimeStepper(
            vector_primitive_factory=discr.get_vector_primitive_factory())

    if rcon.is_head_rank:
        print "%d elements" % len(discr.mesh.elements)

    # filter setup-------------------------------------------------------------
    from hedge.discretization import ExponentialFilterResponseFunction
    from hedge.optemplate.operators import FilterOperator
    mode_filter = FilterOperator(
            ExponentialFilterResponseFunction(min_amplification=0.9,order=4))\
                    .bind(discr)

    # diagnostics setup -------------------------------------------------------
    from pytools.log import LogManager, \
            add_general_quantities, \
            add_simulation_quantities, \
            add_run_info

    if write_output:
        log_file_name = "space-dep.dat"
    else:
        log_file_name = None

    logmgr = LogManager(log_file_name, "w", rcon.communicator)
    add_run_info(logmgr)
    add_general_quantities(logmgr)
    add_simulation_quantities(logmgr)
    discr.add_instrumentation(logmgr)

    stepper.add_instrumentation(logmgr)

    from hedge.log import Integral, LpNorm
    u_getter = lambda: u
    logmgr.add_quantity(Integral(u_getter, discr, name="int_u"))
    logmgr.add_quantity(LpNorm(u_getter, discr, p=1, name="l1_u"))
    logmgr.add_quantity(LpNorm(u_getter, discr, name="l2_u"))

    logmgr.add_watches(["step.max", "t_sim.max", "l2_u", "t_step.max"])

    # Initialize v for data output:
    v = op.advec_v.volume_interpolant(0, discr)

    # timestep loop -----------------------------------------------------------
    rhs = op.bind(discr)
    try:
        from hedge.timestep import times_and_steps
        step_it = times_and_steps(
                final_time=final_time, logmgr=logmgr,
                max_dt_getter=lambda t: op.estimate_timestep(discr,
                    stepper=stepper, t=t, fields=u))

        for step, t, dt in step_it:
            if step % 10 == 0 and write_output:
                visf = vis.make_file("fld-%04d" % step)
                vis.add_data(visf, [ 
                    ("u", discr.convert_volume(u, kind="numpy")), 
                    ("v", discr.convert_volume(v, kind="numpy"))
                    ], time=t, step=step)
                visf.close()

            u = stepper(u, t, dt, rhs)

            # We're feeding in a discontinuity through the BCs.
            # Quadrature does not help with shock capturing--
            # therefore we do need to filter here, regardless
            # of whether quadrature is enabled.
            u = mode_filter(u)

        assert discr.norm(u) < 10

    finally:
        if write_output:
            vis.close()

        logmgr.close()
        discr.close()



if __name__ == "__main__":
    main()




# entry points for py.test ----------------------------------------------------
def test_var_velocity_advection():
    from pytools.test import mark_test
    mark_long = mark_test.long

    for flux_type in ["upwind", "central", "lf"]:
        for use_quadrature in [False, True]:
            descr = "variable-velocity-advection with %s flux" % flux_type
            if use_quadrature:
                descr += " and quadrature"

            yield descr, mark_long(main), False, flux_type, use_quadrature, 1
