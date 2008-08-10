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
    from math import sqrt, pi
    from hedge.parallel import guess_parallelization_context

    pcon = guess_parallelization_context()

    epsilon0 = 8.8541878176e-12 # C**2 / (N m**2)
    mu0 = 4*pi*1e-7 # N/A**2.
    epsilon = 1*epsilon0
    mu = 1*mu0

    dims = 2

    if pcon.is_head_rank:
        if dims == 2:
            from hedge.mesh import make_rect_mesh
            mesh = make_rect_mesh(
                    a=(-10.5,-1.5),
                    b=(10.5,1.5),
                    max_area=0.1
                    )
        elif dims == 3:
            from hedge.mesh import make_box_mesh
            mesh = make_box_mesh(
                    a=(-10.5,-1.5,-1.5),
                    b=(10.5,1.5,1.5),
                    max_volume=0.1)

    if pcon.is_head_rank:
        mesh_data = pcon.distribute_mesh(mesh)
    else:
        mesh_data = pcon.receive_mesh()

    #for order in [1,2,3,4,5,6]:
    discr = pcon.make_discretization(mesh_data, order=3)

    vis = SiloVisualizer(discr, pcon)

    class PointDipoleSource:
        shape = (3,)
        
        Q = 1
        d = 1./39
        omega = 6.284e8

        def __call__(self, x, el):
            from math import exp
            j0 = -1/op.epsilon*self.Q*self.d*self.omega
            spacedep = exp(-la.norm(x)**2*10)

            return numpy.array([0,0,j0*spacedep])

    from hedge.data import TimeHarmonicGivenFunction, GivenFunction
    tc_source = GivenFunction(PointDipoleSource())
    current = TimeHarmonicGivenFunction(
            tc_source,
            omega=PointDipoleSource.omega,
            #phase=pi/2
            )

    from hedge.mesh import TAG_ALL, TAG_NONE
    if dims == 2:
        from hedge.pde import TMMaxwellOperator as MaxwellOperator
    else:
        from hedge.pde import MaxwellOperator

    op = MaxwellOperator(discr, 
            epsilon, mu, 
            upwind_alpha=1,
            pec_tag=TAG_NONE,
            absorb_tag=TAG_ALL,
            current=current,
            )

    fields = op.assemble_fields()

    dt = discr.dt_factor(op.max_eigenvalue())
    final_time = 1e-6
    nsteps = int(final_time/dt)+1
    dt = final_time/nsteps

    if pcon.is_head_rank:
        print "dt", dt
        print "nsteps", nsteps
        print "#elements=", len(mesh.elements)

    stepper = RK4TimeStepper()

    # diagnostics setup ---------------------------------------------------
    from pytools.log import LogManager, add_general_quantities, \
            add_simulation_quantities, add_run_info

    logmgr = LogManager("dipole.dat", "w", pcon.communicator)
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

    point_timeseries = [
            (open("b-x%d-vs-time.dat" % i, "w"), 
                discr.get_point_evaluator(numpy.array([i,0,0][:dims],
                    dtype=discr.default_scalar_type)))
            for i in range(1,5)
            ]

    # timestep loop -------------------------------------------------------
    t = 0
    for step in range(nsteps):
        logmgr.tick()

        if step % 10 == 0:
            vis_timer.start()
            e, h = op.split_eh(fields)
            visf = vis.make_file("dipole-%04d" % step)
            vis.add_data(visf,
                    [ ("e", e), ("h", h), ],
                    time=t, step=step
                    )
            visf.close()
            vis_timer.stop()

            for outf, evaluator in point_timeseries:
                outf.write("%g\t%g\n" % (t, op.mu*evaluator(h[1])))
                outf.flush()

        fields = stepper(fields, t, dt, op.rhs)
        t += dt

    logmgr.tick()
    logmgr.save()



if __name__ == "__main__":
    main()
