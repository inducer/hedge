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
    from hedge.element import TriangularElement
    from hedge.timestep import RK4TimeStepper
    from hedge.mesh import make_disk_mesh
    from hedge.visualization import \
            VtkVisualizer, \
            SiloVisualizer, \
            get_rank_partition
    from pylo import DB_VARTYPE_VECTOR
    from hedge.tools import dot, EOCRecorder
    from math import sqrt, pi, exp
    from analytic_solutions import \
            check_time_harmonic_solution, \
            RealPartAdapter, \
            SplitComplexAdapter, \
            CartesianAdapter, \
            CylindricalCavityMode, \
            RectangularWaveguideMode, \
            RectangularCavityMode
    from hedge.operators import TEMaxwellOperator, TMMaxwellOperator
    from hedge.parallel import guess_parallelization_context
    from hedge.data import GivenFunction, TimeIntervalGivenFunction
    from pytools.arithmetic_container import ArithmeticList

    pcon = guess_parallelization_context()

    epsilon0 = 8.8541878176e-12 # C**2 / (N m**2)
    mu0 = 4*pi*1e-7 # N/A**2.
    epsilon = 1*epsilon0
    mu = 1*mu0

    #eoc_rec = EOCRecorder()

    cylindrical = False
    periodic = False

    mesh = make_disk_mesh(r=0.5)

    if pcon.is_head_rank:
        mesh_data = pcon.distribute_mesh(mesh)
    else:
        mesh_data = pcon.receive_mesh()

    class CurrentSource:
        shape = (3,)

        def __call__(self, x):
            return [0,0,exp(-80*comp.norm_2_squared(x))]

    #for order in [1,2,3,4,5,6]:
    for order in [3]:
        discr = pcon.make_discretization(mesh_data, TriangularElement(order))

        vis = VtkVisualizer(discr, pcon, "em-%d" % order)
        #vis = SiloVisualizer(discr, pcon)

        dt = discr.dt_factor(1/sqrt(mu*epsilon))
        final_time = dt*200
        nsteps = int(final_time/dt)+1
        dt = final_time/nsteps

        if pcon.is_head_rank:
            print "---------------------------------------------"
            print "order %d" % order
            print "---------------------------------------------"
            print "dt", dt
            print "nsteps", nsteps
            print "#elements=", len(mesh.elements)

        def l2_norm(field):
            return sqrt(dot(field, discr.mass_operator*field))

        op = TMMaxwellOperator(discr, epsilon, mu, upwind_alpha=1,
                direct_flux=True,
                current=TimeIntervalGivenFunction(
                    GivenFunction(CurrentSource()), off_time=final_time/10)
                )
        fields = op.assemble_fields()

        stepper = RK4TimeStepper()
        from time import time
        last_tstep = time()
        t = 0
        for step in range(nsteps):
            e, h = op.split_eh(fields)
            print "timestep %d, t=%g l2[e]=%g l2[h]=%g secs=%f" % (
                    step, t, l2_norm(e), l2_norm(h),
                    time()-last_tstep)
            last_tstep = time()

            if True:
                visf = vis.make_file("em-%d-%04d" % (order, step))
                vis.add_data(visf,
                        [ ("e", e), ("h", h), ],
                        time=t, step=step
                        )
                visf.close()

            fields = stepper(fields, t, dt, op.rhs)
            t += dt

        #true_fields = discr.interpolate_volume_function(r_sol)
        #eoc_rec.add_data_point(order, l2_norm(fields-true_fields))

        #print
        #print eoc_rec.pretty_print("P.Deg.", "L2 Error")

if __name__ == "__main__":
    import cProfile as profile
    #profile.run("main()", "wave2d.prof")
    main()

