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




def main() :
    from math import sin, exp, sqrt

    from hedge.mesh import make_rect_mesh
    mesh = make_rect_mesh(a=(-0.5,-0.5),b=(0.5,0.5),max_area=0.008)

    from hedge.discretization import Discretization
    from hedge.element import TriangularElement
    discr = Discretization(mesh, TriangularElement(4))

    from hedge.visualization import VtkVisualizer
    vis = VtkVisualizer(discr, None, "fld")

    def source_u(x):
        return exp(-numpy.dot(x, x)*128)

    source_u_vec = discr.interpolate_volume_function(source_u)

    def source_vec_getter(t):
        from math import sin
        return source_u_vec*sin(10*t)

    from hedge.operators import StrongWaveOperator
    from hedge.mesh import TAG_ALL, TAG_NONE
    op = StrongWaveOperator(-1, discr, 
            source_vec_getter,
            dirichlet_tag=TAG_NONE,
            neumann_tag=TAG_NONE,
            radiation_tag=TAG_ALL,
            flux_type="upwind")

    from hedge.tools import join_fields
    fields = join_fields(discr.volume_zeros(),
            [discr.volume_zeros() for i in range(discr.dimensions)])

    dt = discr.dt_factor(op.max_eigenvalue())
    nsteps = int(1/dt)

    # timestep loop -----------------------------------------------------------
    from hedge.timestep import RK4TimeStepper
    stepper = RK4TimeStepper()

    for step in range(nsteps):
        t = step*dt

        if step % 10 == 0:
            print step, t
            visf = vis.make_file("fld-%04d" % step)

            vis.add_data(visf,
                    [ ("u", fields[0]), ("v", fields[1:]), ],
                    time=t, step=step)
            visf.close()

        fields = stepper(fields, t, dt, op.rhs)

    vis.close()




if __name__ == "__main__":
    main()


