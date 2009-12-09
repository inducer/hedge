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




def main(write_output=True, flux_type_arg="upwind"):
    from hedge.tools import mem_checkpoint
    from math import sin, cos, pi, sqrt
    from math import floor

    from hedge.backends import guess_run_context
    rcon = guess_run_context()

    def u0(x, el):
        return -sin(pi*x[0]+0.03)

    el_count = 11
    order = 5
    if rcon.is_head_rank:
        from hedge.mesh.generator import make_uniform_1d_mesh
        mesh = make_uniform_1d_mesh(-1, 1, el_count, periodic=True)

    if rcon.is_head_rank:
        mesh_data = rcon.distribute_mesh(mesh)
    else:
        mesh_data = rcon.receive_mesh()

    discr = rcon.make_discretization(mesh_data, order=order)
    #vis_discr = rcon.make_discretization(mesh_data, order=30)
    vis_discr = discr

    from hedge.discretization import Projector
    vis_proj = Projector(discr, vis_discr)

    from hedge.visualization import VtkVisualizer, SiloVisualizer
    if write_output:
        vis = SiloVisualizer(vis_discr, rcon)
        #vis = VtkVisualizer(vis_discr, rcon, "fld")

    # operator setup ----------------------------------------------------------
    from hedge.data import \
            ConstantGivenFunction, \
            TimeConstantGivenFunction, \
            TimeDependentGivenFunction
    from hedge.tools.second_order import \
            LDGSecondDerivative, \
            CentralSecondDerivative
    from hedge.models.burgers import BurgersOperator
    op = BurgersOperator(mesh.dimensions,
            viscosity_scheme=LDGSecondDerivative())

    u = discr.interpolate_volume_function(u0)

    if rcon.is_head_rank:
        print "%d elements" % len(discr.mesh.elements)

    # diagnostics setup -------------------------------------------------------
    from pytools.log import LogManager, \
            add_general_quantities, \
            add_simulation_quantities, \
            add_run_info

    if write_output:
        log_file_name = "burgers.dat"
    else:
        log_file_name = None

    logmgr = LogManager(log_file_name, "w", rcon.communicator)
    add_run_info(logmgr)
    add_general_quantities(logmgr)
    add_simulation_quantities(logmgr)
    discr.add_instrumentation(logmgr)

    from hedge.log import Integral, LpNorm
    u_getter = lambda: u
    logmgr.add_quantity(Integral(u_getter, discr, name="int_u"))
    logmgr.add_quantity(LpNorm(u_getter, discr, p=1, name="l1_u"))
    logmgr.add_quantity(LpNorm(u_getter, discr, name="l2_u"))

    logmgr.add_watches(["step.max", "t_sim.max", "l2_u", "t_step.max"])

    # timestep loop -----------------------------------------------------------
    h = 2/el_count
    from hedge.tools import PerssonPeraireDiscontinuitySensor
    sensor = PerssonPeraireDiscontinuitySensor(discr, kappa=5,
            eps0=h/order, s_0=1/order**4)

    rhs = op.bind(discr, sensor=sensor)

    from hedge.timestep import RK4TimeStepper
    from hedge.timestep.dumka3 import Dumka3TimeStepper
    #stepper = RK4TimeStepper()
    stepper = Dumka3TimeStepper(3)
    #stepper = Dumka3TimeStepper(4)

    stepper.add_instrumentation(logmgr)

    try:
        from hedge.timestep import times_and_steps
        # for visc=0.01
        #stab_fac = 0.1 # RK4
        stab_fac = 1.6 # dumka3(3), central
        #stab_fac = 3 # dumka3(4), central

        #stab_fac = 0.01 # RK4
        stab_fac = 0.5 # dumka3(3), central
        #stab_fac = 3 # dumka3(4), central

        dt = stab_fac*op.estimate_timestep(discr,
                stepper=RK4TimeStepper(), t=0, fields=u)

        step_it = times_and_steps(
                final_time=5, logmgr=logmgr, max_dt_getter=lambda t: dt)

        for step, t, dt in step_it:
            if step % 100 == 0 and write_output:
                visf = vis.make_file("fld-%04d" % step)
                vis.add_data(visf, [ ("u", vis_proj(u)), ],
                            time=t,
                            step=step
                            )
                visf.close()

            u = stepper(u, t, dt, rhs)

    finally:
        if write_output:
            vis.close()

        logmgr.save()




if __name__ == "__main__":
    main()




# entry points for py.test ----------------------------------------------------
def test_advection():
    from pytools.test import mark_test
    mark_long = mark_test.long

    for flux_type in ["upwind", "central", "lf"]:
        yield "advection with %s flux" % flux_type, \
                mark_long(main), False, flux_type
