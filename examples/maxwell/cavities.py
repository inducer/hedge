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




def main(write_output=True, allow_features=None, flux_type_arg=1,
        bdry_flux_type_arg=None, extra_discr_args={}):
    from hedge.element import TetrahedralElement
    from hedge.timestep import RK4TimeStepper
    from hedge.mesh import make_ball_mesh, make_cylinder_mesh, make_box_mesh
    from hedge.tools import EOCRecorder, to_obj_array
    from math import sqrt, pi
    from analytic_solutions import \
            check_time_harmonic_solution, \
            RealPartAdapter, \
            SplitComplexAdapter, \
            CylindricalFieldAdapter, \
            CylindricalCavityMode, \
            RectangularWaveguideMode, \
            RectangularCavityMode
    from hedge.models.em import MaxwellOperator

    from hedge.backends import guess_run_context
    rcon = guess_run_context(allow_features)

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
        r_sol = CylindricalFieldAdapter(RealPartAdapter(mode))
        c_sol = SplitComplexAdapter(CylindricalFieldAdapter(mode))

        if rcon.is_head_rank:
            mesh = make_cylinder_mesh(radius=R, height=d, max_volume=0.01)
    else:
        if periodic:
            mode = RectangularWaveguideMode(epsilon, mu, (3,2,1))
            periodicity = (False, False, True)
        else:
            periodicity = None
        mode = RectangularCavityMode(epsilon, mu, (1,2,2))

        if rcon.is_head_rank:
            mesh = make_box_mesh(max_volume=0.001, periodicity=periodicity)

    if rcon.is_head_rank:
        mesh_data = rcon.distribute_mesh(mesh)
    else:
        mesh_data = rcon.receive_mesh()

    for order in [1,2,3,4,5,6]:
        discr = rcon.make_discretization(mesh_data, order=order,
                **extra_discr_args)

        from hedge.visualization import VtkVisualizer
        if write_output:
            vis = VtkVisualizer(discr, rcon, "em-%d" % order)

        mode.set_time(0)
        def get_true_field():
            return discr.convert_volume(
                to_obj_array(mode(discr)
                    .real.astype(discr.default_scalar_type).copy()),
                kind=discr.compute_kind)
        fields = get_true_field()

        op = MaxwellOperator(epsilon, mu, \
                flux_type=flux_type_arg, \
                bdry_flux_type=bdry_flux_type_arg)

        dt = discr.dt_factor(op.max_eigenvalue())
        final_time = 0.5e-9
        nsteps = int(final_time/dt)+1
        dt = final_time/nsteps

        if rcon.is_head_rank:
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

        logmgr.add_watches(["step.max", "t_sim.max", "W_field", "t_step.max"])

        # timestep loop -------------------------------------------------------
        t = 0
        rhs = op.bind(discr)

        try:
            for step in range(nsteps):
                logmgr.tick()

                if step % 10 == 0 and write_output:
                    sub_timer = vis_timer.start_sub_timer()
                    e, h = op.split_eh(fields)
                    visf = vis.make_file("em-%d-%04d" % (order, step))
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
                t += dt
        finally:
            if write_output:
                vis.close()

            logmgr.close()
            discr.close()

        mode.set_time(t)

        eoc_rec.add_data_point(order, 
                discr.norm(fields-get_true_field()))

        if rcon.is_head_rank:
            print
            print eoc_rec.pretty_print("P.Deg.", "L2 Error")

    assert eoc_rec.estimate_order_of_convergence()[0,1] > 6





# entry points for py.test ----------------------------------------------------
from pytools.test import mark_test
@mark_test(long=True)
def test_maxwell_cavities():
    main(write_output=False)

@mark_test(long=True)
def test_maxwell_cavities_lf():
    main(write_output=False, flux_type_arg="lf", bdry_flux_type_arg=1)

@mark_test(mpi=True, long=True)
def test_maxwell_cavities_mpi():
    from pytools.mpi import run_with_mpi_ranks
    run_with_mpi_ranks(__file__, 2, main, 
            write_output=False, allow_features=["mpi"])





def test_cuda():
    marker = mark_test(cuda=True, long=True)

    yield "SP CUDA Maxwell", do_test_maxwell_cavities_cuda, numpy.float32
    yield "DP CUDA Maxwell", do_test_maxwell_cavities_cuda, numpy.float64




def do_test_maxwell_cavities_cuda(dtype):
    import py.test

    try:
        import pycuda.autoinit
    except ImportError:
        py.test.skip()

    main(write_output=False, allow_features=["cuda"], 
            extra_discr_args=dict(
                init_cuda=False,
                debug=["cuda_no_plan"],
                default_scalar_type=dtype,
                ))




# entry point -----------------------------------------------------------------
if __name__ == "__main__":
    from pytools.mpi import in_mpi_relaunch
    if in_mpi_relaunch():
        test_maxwell_cavities_mpi()
    else:
        main()
