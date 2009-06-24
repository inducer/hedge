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




def main(write_output=True):
    from math import sin, exp, sqrt

    from hedge.mesh import make_rect_mesh
    mesh = make_rect_mesh(a=(-0.5,-0.5),b=(0.5,0.5),max_area=0.008)

    from hedge.backends.jit import Discretization

    discr = Discretization(mesh, order=4)

    from hedge.visualization import VtkVisualizer
    vis = VtkVisualizer(discr, None, "fld")

    def source_u(x, el):
        x = x - numpy.array([0.1,0.22])
        return exp(-numpy.dot(x, x)*128)

    source_u_vec = discr.interpolate_volume_function(source_u)

    def source_vec_getter(t):
        from math import sin
        return source_u_vec*sin(10*t)

    from hedge.pde import StrongWaveOperator
    from hedge.mesh import TAG_ALL, TAG_NONE
    op = StrongWaveOperator(-1, discr.dimensions, 
            source_vec_getter,
            dirichlet_tag=TAG_NONE,
            neumann_tag=TAG_NONE,
            radiation_tag=TAG_ALL,
            flux_type="upwind")

    from hedge.tools import join_fields
    fields = join_fields(discr.volume_zeros(),
            [discr.volume_zeros() for i in range(discr.dimensions)])

    # timestep loop -----------------------------------------------------------
    from hedge.timestep import RK4TimeStepper, AdamsBashforthTimeStepper
    if True:
        stepper = AdamsBashforthTimeStepper(3)
        dt = discr.dt_factor(op.max_eigenvalue(), 
                AdamsBashforthTimeStepper, 3)
    else:
        stepper = RK4TimeStepper(3)
        dt = discr.dt_factor(op.max_eigenvalue(), RK4TimeStepper)

    nsteps = int(1/dt)
    print "dt=%g nsteps=%d" % (dt, nsteps)

# timestep loop -----------------------------------------------------------
    rhs = op.bind(discr)
    for step in range(nsteps):
        t = step*dt

        if step % 50 == 0 and write_output:
            print step, t, discr.norm(fields[0])
            visf = vis.make_file("fld-%04d" % step)

            vis.add_data(visf,
                    [ ("u", fields[0]), ("v", fields[1:]), ],
                    time=t, step=step)
            visf.close()

        fields = stepper(fields, t, dt, rhs)

    vis.close()




if __name__ == "__main__":
    main()

# entry points for py.test ----------------------------------------------------
#from pytools.test import mark_test
#@mark_test(long=True)
#def test_wave_min():
#    main(write_output=False)


