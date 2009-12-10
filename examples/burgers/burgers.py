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

    order = 4
    if rcon.is_head_rank:
        if False:
            from hedge.mesh.generator import make_uniform_1d_mesh
            mesh = make_uniform_1d_mesh(-pi, pi, 11, periodic=True)
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

    discr = rcon.make_discretization(mesh_data, order=order)
    vis_discr = rcon.make_discretization(mesh_data, order=30)
    #vis_discr = discr

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

    if rcon.is_head_rank:
        print "%d elements" % len(discr.mesh.elements)

    # exact solution ----------------------------------------------------------
    import pymbolic
    var = pymbolic.var

    #u0_expr = pymbolic.parse("-0.4+math.sin(x+0.1)")
    #u0_expr = pymbolic.parse("-math.sin(x)")
    #u0_fun = pymbolic.compile(u0_expr)

    def u0_fun(x):
        return -0.4+sin(x+0.1)

    #f_expr = var("u")**2/2
    #f = pymbolic.compile(f_expr)
    #from exact import CLawNoShockExactSolution
    #u_exact = CLawNoShockExactSolution(u0_expr, f_expr)

    u = discr.interpolate_volume_function(lambda x, el: u0_fun(x[0]))

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
    mesh_a, mesh_b = mesh.bounding_box()
    from pytools import product
    area = product(mesh_b[i] - mesh_a[i] for i in range(mesh.dimensions))
    h = sqrt(area/len(mesh.elements))
    from hedge.tools.bad_cell import (
            PerssonPeraireDiscontinuitySensor,
            ErrorEstimatingDiscontinuitySensorBase,
            DecayGatingDiscontinuitySensorBase)
    sensor = ErrorEstimatingDiscontinuitySensorBase(discr) 
    sensor2 = DecayGatingDiscontinuitySensorBase(discr, h/(order))
    #sensor = PerssonPeraireDiscontinuitySensor(discr, kappa=2,
            #eps0=h/order, s_0=numpy.log10(1/order**4))

    from pytools.log import LogQuantity
    class S_eMonitor(LogQuantity):
        """A source of loggable scalars."""
        def __init__(self):
            LogQuantity.__init__(self, "max_S_e", "1")

        def __call__(self):
            return discr.nodewise_max(sensor.capital_s_e(u))

    class SensorMonitor(LogQuantity):
        """A source of loggable scalars."""
        def __init__(self):
            LogQuantity.__init__(self, "max_viscosity", "1")

        def __call__(self):
            return discr.nodewise_max(sensor(u))

    class Smoothness(LogQuantity):
        """A source of loggable scalars."""
        def __init__(self):
            LogQuantity.__init__(self, "smoothness_fit", "1")

        def __call__(self):
            mode_coeff_histogram = {}
            for eg in discr.element_groups:
                ldis = eg.local_discretization
                vdm = ldis.vandermonde()
                for slc in eg.ranges:
                    modes = la.solve(vdm, u[slc])
                    for mid, mode_coeff in zip(
                            ldis.generate_mode_identifiers(), modes):
                        msum = sum(mid)
                        mode_coeff_histogram[msum] = (
                                mode_coeff_histogram.get(msum, 0) 
                                + mode_coeff**2)

            max_mode = max(mode_coeff_histogram.keys())+1
            mode_coeffs = numpy.sqrt(numpy.array([
                mode_coeff_histogram[msum]
                for msum in range(max_mode)]))
            return -numpy.polyfit(
                    numpy.log10(1+numpy.arange(max_mode)), 
                    numpy.log10(mode_coeffs), 1)[-2]

    #logmgr.add_quantity(S_eMonitor())
    #logmgr.add_quantity(SensorMonitor())
    #logmgr.add_quantity(Smoothness())

    rhs = op.bind(discr, sensor=sensor)
    rhs2 = op.bind(discr, sensor=sensor2)
    rhs3 = op.bind(discr)

    from hedge.timestep import RK4TimeStepper
    from hedge.timestep.dumka3 import Dumka3TimeStepper
    #stepper = RK4TimeStepper()
    stepper = Dumka3TimeStepper(3)
    #stepper = Dumka3TimeStepper(4)

    stepper.add_instrumentation(logmgr)

    u2 = u.copy()
    u3 = u.copy()

    try:
        from hedge.timestep import times_and_steps
        # for visc=0.01
        #stab_fac = 0.1 # RK4
        #stab_fac = 1.6 # dumka3(3), central
        #stab_fac = 3 # dumka3(4), central

        #stab_fac = 0.01 # RK4
        stab_fac = 0.1 # dumka3(3), central
        #stab_fac = 3 # dumka3(4), central

        dt = stab_fac*op.estimate_timestep(discr,
                stepper=RK4TimeStepper(), t=0, fields=u)

        step_it = times_and_steps(
                final_time=10, logmgr=logmgr, max_dt_getter=lambda t: dt)

        for step, t, dt in step_it:
            #if step == 129:
                #sensor.estimate_decay(u, debug=True)
            if step % 50 == 0 and write_output:
                def catchy_u_exact(x, el):
                    try:
                        return u_exact(x[0], t)
                    except RuntimeError:
                        return 1
                #u_exact_fld =  vis_discr.interpolate_volume_function(catchy_u_exact)

                visf = vis.make_file("fld-%04d" % step)
                vis.add_data(visf, [ 
                    ("u", vis_proj(u)), 
                    ("u2", vis_proj(u2)), 
                    #("u3", vis_proj(u3)), 
                    #("u_exact", u_exact_fld), 
                    #("alpha", vis_proj(sensor.estimate_decay(u)[0])), 
                    ("sensor", vis_proj(sensor(u))), 
                    ("sensor2", vis_proj(sensor2(u2))), 
                    ],
                    time=t,
                    step=step)
                visf.close()

            #u = stepper(u, t, dt, rhs)
            u2 = stepper(u2, t, dt, rhs2)
            #u3 = stepper(u3, t, dt, rhs3)

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
