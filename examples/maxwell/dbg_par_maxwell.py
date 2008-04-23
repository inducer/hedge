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
import pylinear.array as num
import pylinear.computation as comp




def main():
    from hedge.element import TetrahedralElement
    from hedge.timestep import RK4TimeStepper
    from hedge.mesh import make_ball_mesh, make_cylinder_mesh, make_box_mesh
    from hedge.discretization import Discretization, bind_mass_matrix
    from hedge.visualization import \
            make_silo_file, \
            SiloVisualizer, \
            get_rank_partition
    from pylo import DB_VARTYPE_VECTOR
    from hedge.tools import dot, EOCRecorder
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
    from hedge.parallel import \
            guess_parallelization_context, \
            reassemble_volume_field

    pcon = guess_parallelization_context()

    epsilon0 = 8.8541878176e-12 # C**2 / (N m**2)
    mu0 = 4*pi*1e-7 # N/A**2.
    epsilon = 1*epsilon0
    mu = 1*mu0

    eoc_rec = EOCRecorder()

    cylindrical = False
    periodic = False

    # default to "whole boundary is PEC"
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

    #for order in [1,2,3,4,5,6]:
    for order in [3]:
        discr = pcon.make_discretization(mesh_data, TetrahedralElement(order))

        vis = SiloVisualizer(discr)

        dt = discr.dt_factor(1/sqrt(mu*epsilon))
        final_time = dt*60
        nsteps = int(final_time/dt)+1
        dt = final_time/nsteps

        if pcon.is_head_rank:
            print "---------------------------------------------"
            print "order %d" % order
            print "---------------------------------------------"
            print "dt", dt
            print "nsteps", nsteps

        mass = bind_mass_matrix(discr)

        def l2_norm(field):
            return sqrt(dot(field, mass*field))

        #check_time_harmonic_solution(discr, mode, c_sol)
        #continue

        mode.set_time(0)
        fields = discr.interpolate_volume_function(r_sol)
        op = MaxwellOperator(discr, epsilon, mu, upwind_alpha=1)

        if pcon.is_head_rank:
            gdiscr = Discretization(mesh, TetrahedralElement(order))
            gmass = bind_mass_matrix(gdiscr)
            gvis = SiloVisualizer(gdiscr)
            gfields = gdiscr.interpolate_volume_function(r_sol)
            gop = MaxwellOperator(gdiscr, epsilon, mu, upwind_alpha=1)
            gstepper = RK4TimeStepper()

            def gl2_norm(field):
                return sqrt(dot(field, gmass*field))
        else:
            gdiscr = None

        gpart = reassemble_volume_field(pcon, gdiscr, discr, 
                get_rank_partition(pcon, discr))

        stepper = RK4TimeStepper()
        from time import time
        last_tstep = time()
        t = 0
        for step in range(nsteps):
            print "timestep %d, t=%g l2[e]=%g l2[h]=%g secs=%f" % (
                    step, t, l2_norm(fields[0:3]), l2_norm(fields[3:6]),
                    time()-last_tstep)
            last_tstep = time()

            if True:
                rhs = op.rhs(t, fields)
                g_ass_rhs = reassemble_volume_field(
                        pcon, gdiscr, discr, rhs)
                if pcon.is_head_rank:
                    grhs = gop.rhs(t, gfields)
                    print "l2rhs", \
                            gl2_norm(g_ass_rhs[0:3]), \
                            gl2_norm(grhs[0:3])
                    silo = make_silo_file("rhs-%04d" % step)
                    gvis.add_to_silo(silo,
                            vectors=[
                                ("par_rhs_e", g_ass_rhs[0:3]), 
                                ("par_rhs_h", g_ass_rhs[3:6]), 
                                ("ser_rhs_e", grhs[0:3]), 
                                ("ser_rhs_h", grhs[3:6]), 
                                ],
                            scalars=[("partition", gpart)
                                ],
                            expressions=[
                                ("diff_rhs_e", "par_rhs_e-ser_rhs_e", DB_VARTYPE_VECTOR),
                                ("diff_rhs_h", "par_rhs_h-ser_rhs_h", DB_VARTYPE_VECTOR),
                                ],
                            write_coarse_mesh=True,
                            time=t, step=step
                            )
                    silo.close()

            if False:
                silo = make_silo_file(pcon, "em-%04d" % step)
                vis.add_to_silo(silo,
                        vectors=[("e", fields[0:3]), 
                            ("h", fields[3:6]), ],
                        scalars=[("partition", get_rank_partition(pcon, discr))
                            ],
                        expressions=[
                            ],
                        write_coarse_mesh=True,
                        time=t, step=step
                        )
                silo.close()

            fields = stepper(fields, t, dt, op.rhs)
            if pcon.is_head_rank:
                gfields = gstepper(gfields, t, dt, gop.rhs)
            t += dt

        mode.set_time(t)
        true_fields = discr.interpolate_volume_function(r_sol)
        eoc_rec.add_data_point(order, l2_norm(fields-true_fields))

        print
        print eoc_rec.pretty_print("P.Deg.", "L2 Error")

if __name__ == "__main__":
    import cProfile as profile
    #profile.run("main()", "wave2d.prof")
    main()
