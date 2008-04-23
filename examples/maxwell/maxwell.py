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




def main():
    from hedge.element import TetrahedralElement
    from hedge.timestep import RK4TimeStepper
    from hedge.mesh import make_ball_mesh, make_cylinder_mesh, make_box_mesh
    from hedge.visualization import \
            VtkVisualizer, \
            SiloVisualizer, \
            get_rank_partition
    from hedge.discretization import norm
    from hedge.tools import EOCRecorder, to_obj_array
    from math import sqrt, pi
    from analytic_solutions import \
            check_time_harmonic_solution, \
            RealPartAdapter, \
            SplitComplexAdapter, \
            CartesianAdapter, \
            CylindricalCavityMode, \
            RectangularWaveguideMode, \
            RectangularCavityMode
    from hedge.operators import MaxwellOperator
    from hedge.parallel import guess_parallelization_context

    pcon = guess_parallelization_context()

    epsilon0 = 8.8541878176e-12 # C**2 / (N m**2)
    mu0 = 4*pi*1e-7 # N/A**2.
    epsilon = 1*epsilon0
    mu = 1*mu0

    eoc_rec = EOCRecorder()

    cylindrical = False
    periodic = False

    if cylindrical:
        R = 1
        d = 2
        mode = CylindricalCavityMode(m=1, n=1, p=1,
                radius=R, height=d, 
                epsilon=epsilon, mu=mu)
        r_sol = CartesianAdapter(RealPartAdapter(mode))
        c_sol = SplitComplexAdapter(CartesianAdapter(mode))

        if pcon.is_head_rank:
            mesh = make_cylinder_mesh(radius=R, height=d, max_volume=0.01)
    else:
        if periodic:
            mode = RectangularWaveguideMode(epsilon, mu, (3,2,1))
            periodicity = (False, False, True)
        else:
            periodicity = None
        mode = RectangularCavityMode(epsilon, mu, (1,2,2))
        r_sol = RealPartAdapter(mode)
        c_sol = SplitComplexAdapter(mode)

        if pcon.is_head_rank:
            mesh = make_box_mesh(max_volume=0.01, periodicity=periodicity)

    if pcon.is_head_rank:
        mesh_data = pcon.distribute_mesh(mesh)
    else:
        mesh_data = pcon.receive_mesh()

    for order in [2,3,4,5,6]:
        discr = pcon.make_discretization(mesh_data, TetrahedralElement(order))

        vis = VtkVisualizer(discr, pcon, "em-%d" % order)

        mode.set_time(0)
        fields = to_obj_array(discr.interpolate_volume_function(r_sol))
        op = MaxwellOperator(discr, epsilon, mu, upwind_alpha=1,
                direct_flux=True)

        dt = discr.dt_factor(op.max_eigenvalue())
        final_time = 1e-9
        nsteps = int(final_time/dt)+1
        dt = final_time/nsteps

        if pcon.is_head_rank:
            print "---------------------------------------------"
            print "order %d" % order
            print "---------------------------------------------"
            print "dt", dt
            print "nsteps", nsteps
            print "#elements=", len(mesh.elements)

        stepper = RK4TimeStepper()

        # diagnostics setup ---------------------------------------------------
        from pytools.log import LogManager, add_general_quantities, \
                add_simulation_quantities, add_run_info

        logmgr = LogManager("maxwell-%d.dat" % order, "w", pcon.communicator)
        add_run_info(logmgr)
        add_general_quantities(logmgr)
        add_simulation_quantities(logmgr, dt)
        discr.add_instrumentation(logmgr)
        stepper.add_instrumentation(logmgr)

        from pytools.log import IntervalTimer
        vis_timer = IntervalTimer("t_vis", "Time spent visualizing")
        logmgr.add_quantity(vis_timer)

        from hedge.log import EMFieldGetter, add_em_quantities
        field_getter = EMFieldGetter(op, locals(), "fields")
        add_em_quantities(logmgr, op, field_getter)
        
        logmgr.add_watches(["step.max", "t_sim.max", "W_field", "t_step.max"])

        # timestep loop -------------------------------------------------------
        t = 0
        for step in range(nsteps):
            logmgr.tick()

            if True:
                vis_timer.start()
                e, h = op.split_eh(fields)
                visf = vis.make_file("em-%d-%04d" % (order, step))
                vis.add_data(visf,
                        [ ("e", e), ("h", h), ],
                        time=t, step=step
                        )
                visf.close()
                vis_timer.stop()

            fields = stepper(fields, t, dt, op.rhs)
            t += dt

        logmgr.tick()
        logmgr.save()

        numpy.seterr('raise')
        mode.set_time(t)
        true_fields = to_obj_array(discr.interpolate_volume_function(r_sol))

        eoc_rec.add_data_point(order, norm(discr, fields-true_fields))

        print
        print eoc_rec.pretty_print("P.Deg.", "L2 Error")

if __name__ == "__main__":
    main()
