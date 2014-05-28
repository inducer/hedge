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

import numpy as np
import numpy.linalg as la
import pytest
from hedge.models.advection import StrongAdvectionOperator


def my_box_mesh(boundary_tagger):
    from hedge.mesh.generator import make_rect_mesh
    from math import pi
    return make_rect_mesh(
        b=(2*pi, 3), max_area=0.4,
        #periodicity=(True, False),
        # meshpy doesn't do periodic at the moment (5/2014)
        periodicity=(False, False),
        subdivisions=(5, 10),
        boundary_tagger=boundary_tagger,
        )


def my_rect_mesh(boundary_tagger):
    from hedge.mesh.generator import make_box_mesh
    from math import pi
    return make_box_mesh(
        (0, 0, 0), (2*pi, 2, 2), max_volume=0.4,
        #periodicity=(True, False, False),
        # meshpy doesn't do periodic at the moment (5/2014)
        periodicity=(False, False, False),
        boundary_tagger=boundary_tagger,
        )


def my_ball_mesh(boundary_tagger):
    from hedge.mesh.generator import make_ball_mesh
    from math import pi
    return make_ball_mesh(
        r=pi,
        boundary_tagger=boundary_tagger, max_volume=0.7)


def run_convergence_test_advec(dtype, flux_type, random_partition, mesh_gen,
        debug_output=False):
    """Test whether 2/3D advection actually converges"""

    from hedge.timestep import RK4TimeStepper
    from hedge.tools import EOCRecorder
    from math import sin
    from hedge.data import TimeDependentGivenFunction
    from hedge.visualization import SiloVisualizer

    from hedge.backends import guess_run_context
    rcon = guess_run_context(["mpi"])

    # note: x component must remain zero because x-periodicity is used
    v = np.array([0.0, 0.9, 0.3])

    def f(x):
        return sin(x)

    def u_analytic(x, el, t):
        return f((np.dot(-v[:dims], x)/la.norm(v[:dims])+t*la.norm(v[:dims])))

    def boundary_tagger(vertices, el, face_nr, points):
        face_normal = el.face_normals[face_nr]
        if np.dot(face_normal, v[:len(face_normal)]) < 0:
            return ["inflow"]
        else:
            return ["outflow"]

    mesh = mesh_gen(boundary_tagger)
    eoc_rec = EOCRecorder()

    if random_partition:
        # Distribute elements randomly across nodes.
        # This is bad, efficiency-wise, but it puts stress
        # on the parallel implementation, which is desired here.
        # Another main point of this is to force the code to split
        # a periodic face pair across nodes.
        from random import choice
        partition = [choice(rcon.ranks) for el in mesh.elements]
    else:
        partition = None

    for order in [1, 2, 3, 4]:
        if rcon.is_head_rank:
            mesh_data = rcon.distribute_mesh(mesh, partition)
        else:
            mesh_data = rcon.receive_mesh()

        dims = mesh.points.shape[1]

        discr = rcon.make_discretization(mesh_data, order=order,
                default_scalar_type=dtype)

        op = StrongAdvectionOperator(v[:dims],
                inflow_u=TimeDependentGivenFunction(u_analytic),
                flux_type=flux_type)
        if debug_output:
            vis = SiloVisualizer(discr, rcon)

        u = discr.interpolate_volume_function(
                lambda x, el: u_analytic(x, el, 0))
        ic = u.copy()

        if debug_output and rcon.is_head_rank:
            print "#elements=%d" % len(mesh.elements)

        test_name = "test-%s-o%d-m%s-r%s" % (
                flux_type, order, mesh_gen.__name__, random_partition)

        rhs = op.bind(discr)

        stepper = RK4TimeStepper(dtype=dtype)
        from hedge.timestep import times_and_steps
        final_time = 1
        step_it = times_and_steps(
                final_time=final_time,
                max_dt_getter=lambda t: op.estimate_timestep(discr,
                    stepper=stepper, t=t, fields=u))

        for step, t, dt in step_it:
            u = stepper(u, t, dt, rhs)

        assert u.dtype == dtype

        u_true = discr.interpolate_volume_function(
                lambda x, el: u_analytic(x, el, final_time))
        error = u-u_true
        l2_error = discr.norm(error)

        if debug_output:
            visf = vis.make_file(test_name+"-final")
            vis.add_data(visf, [
                ("u", u),
                ("u_true", u_true),
                ("ic", ic)])
            visf.close()

        eoc_rec.add_data_point(order, l2_error)

    if debug_output and rcon.is_head_rank:
        print "%s\n%s\n" % (flux_type.upper(), "-" * len(flux_type))
        print eoc_rec.pretty_print(abscissa_label="Poly. Order",
                error_label="L2 Error")

    assert eoc_rec.estimate_order_of_convergence()[0, 1] > 3
    assert eoc_rec.estimate_order_of_convergence(2)[-1, 1] > 7


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("flux_type", StrongAdvectionOperator.flux_types)
@pytest.mark.parametrize("random_partition", [True, False])
@pytest.mark.parametrize("mesh_gen", [my_box_mesh])
def test_hedge_parallel(dtype, flux_type, random_partition, mesh_gen):
    from pytools.mpi import run_with_mpi_ranks
    run_with_mpi_ranks(__file__, 2,
            run_convergence_test_advec,
                (dtype, flux_type, random_partition, mesh_gen))


if __name__ == "__main__":
    import sys
    from pytools.mpi import check_for_mpi_relaunch
    check_for_mpi_relaunch(sys.argv)

    if len(sys.argv) > 1:
        exec sys.argv[1]
    else:
        from py.test.cmdline import main
        main([__file__])
