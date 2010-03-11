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
from math import sin, cos, pi, sqrt
from pytools.test import mark_test




class ExactTestCase:
    a = 0
    b = 150
    final_time = 1000

    def u0(self, x):
        return self.u_exact(x, 0)

    def u_exact(self, x, t):
        # CAUTION: This gets the shock speed wrong as soon as the pulse
        # starts interacting with itself.

        def f(x, shock_loc):
            if x < (t-40)/4:
                return 1/4
            else:
                if t < 40:
                    if x < (3*t)/4:
                        return (x+15)/(t+20)
                    elif x < (t+80)/4:
                        return (x-30)/(t-40)
                    else:
                        return 1/4
                else:
                    if x < shock_loc:
                        return (x+15)/(t+20)
                    else:
                        return 1/4

        from math import sqrt

        shock_loc = 30*sqrt(2*t+40)/sqrt(120) + t/4 - 10
        shock_win = (shock_loc + 20) // self.b
        x += shock_win * 150

        x -= 20

        return max(f(x, shock_loc), f(x-self.b, shock_loc-self.b))

class OffCenterMigratingTestCase:
    a = -pi
    b = pi
    final_time = 10

    def u0(self, x):
        return -0.4+sin(x+0.1)


class CenteredStationaryTestCase:
    # does funny things to P-P
    a = -pi
    b = pi
    final_time = 10

    def u0(self, x):
        return -sin(x)

class OffCenterStationaryTestCase:
    # does funny things to P-P
    a = -pi
    b = pi
    final_time = 10

    def u0(self, x):
        return -sin(x+0.3)



def main(write_output=True, flux_type_arg="upwind", 
        #case = CenteredStationaryTestCase(),
        #case = OffCenterStationaryTestCase(),
        #case = OffCenterMigratingTestCase(),
        case = ExactTestCase(),
        ):
    from hedge.backends import guess_run_context
    rcon = guess_run_context()

    order = 3
    if rcon.is_head_rank:
        if True:
            from hedge.mesh.generator import make_uniform_1d_mesh
            mesh = make_uniform_1d_mesh(case.a, case.b, 20, periodic=True)
        else:
            from hedge.mesh.generator import make_rect_mesh
            print (pi*2)/(11*5*2)
            mesh = make_rect_mesh((-pi, -1), (pi, 1),
                    periodicity=(True, True),
                    subdivisions=(11,5),
                    max_area=(pi*2)/(11*5*2)
                    )

    if rcon.is_head_rank:
        mesh_data = rcon.distribute_mesh(mesh)
    else:
        mesh_data = rcon.receive_mesh()

    discr = rcon.make_discretization(mesh_data, order=order,
            quad_min_degrees={"quad": 3*order})

    if write_output:
        from hedge.visualization import VtkVisualizer
        vis = VtkVisualizer(discr, rcon, "fld")

    # operator setup ----------------------------------------------------------
    from hedge.second_order import IPDGSecondDerivative

    from hedge.models.burgers import BurgersOperator
    op = BurgersOperator(mesh.dimensions,
            viscosity_scheme=IPDGSecondDerivative())

    if rcon.is_head_rank:
        print "%d elements" % len(discr.mesh.elements)

    # exact solution ----------------------------------------------------------
    import pymbolic
    var = pymbolic.var

    u = discr.interpolate_volume_function(lambda x, el: case.u0(x[0]))

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

    from hedge.log import LpNorm
    u_getter = lambda: u
    logmgr.add_quantity(LpNorm(u_getter, discr, p=1, name="l1_u"))

    logmgr.add_watches(["step.max", "t_sim.max", "l1_u", "t_step.max"])

    # timestep loop -----------------------------------------------------------
    rhs = op.bind(discr)

    from hedge.timestep.runge_kutta import ODE45TimeStepper, LSRK4TimeStepper
    stepper = ODE45TimeStepper()

    stepper.add_instrumentation(logmgr)

    try:
        from hedge.timestep import times_and_steps
        # for visc=0.01
        #stab_fac = 0.1 # RK4
        #stab_fac = 1.6 # dumka3(3), central
        #stab_fac = 3 # dumka3(4), central

        #stab_fac = 0.01 # RK4
        stab_fac = 0.2 # dumka3(3), central
        #stab_fac = 3 # dumka3(4), central

        dt = stab_fac*op.estimate_timestep(discr,
                stepper=LSRK4TimeStepper(), t=0, fields=u)

        step_it = times_and_steps(
                final_time=case.final_time, logmgr=logmgr, max_dt_getter=lambda t: dt)
        from hedge.optemplate import  InverseVandermondeOperator
        inv_vdm = InverseVandermondeOperator().bind(discr)

        for step, t, dt in step_it:
            if step % 3 == 0 and write_output:
                if hasattr(case, "u_exact"):
                    extra_fields = [
                            ("u_exact",
                                discr.interpolate_volume_function(
                                    lambda x, el: case.u_exact(x[0], t)))]
                else:
                    extra_fields = []

                visf = vis.make_file("fld-%04d" % step)
                vis.add_data(visf, [
                    ("u", u),
                    ] + extra_fields,
                    time=t,
                    step=step)
                visf.close()

            u = stepper(u, t, dt, rhs)

        if isinstance(case, ExactTestCase):
            assert discr.norm(u, 1) < 50

    finally:
        if write_output:
            vis.close()

        logmgr.save()




if __name__ == "__main__":
    main()




# entry points for py.test ----------------------------------------------------
@mark_test.long
def test_stability():
    main(write_output=False)

