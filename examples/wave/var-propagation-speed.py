"""Variable-coefficient wave propagation."""

from __future__ import division

__copyright__ = "Copyright (C) 2009 Andreas Kloeckner"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""


import numpy as np
from hedge.mesh import TAG_ALL, TAG_NONE


def main(write_output=True,
        dir_tag=TAG_NONE,
        neu_tag=TAG_NONE,
        rad_tag=TAG_ALL,
        flux_type_arg="upwind"):
    from math import sin, cos, pi, exp, sqrt  # noqa

    from hedge.backends import guess_run_context
    rcon = guess_run_context()

    dim = 2

    if dim == 1:
        if rcon.is_head_rank:
            from hedge.mesh.generator import make_uniform_1d_mesh
            mesh = make_uniform_1d_mesh(-10, 10, 500)
    elif dim == 2:
        from hedge.mesh.generator import make_rect_mesh
        if rcon.is_head_rank:
            mesh = make_rect_mesh(a=(-1, -1), b=(1, 1), max_area=0.003)
    elif dim == 3:
        if rcon.is_head_rank:
            from hedge.mesh.generator import make_ball_mesh
            mesh = make_ball_mesh(max_volume=0.0005)
    else:
        raise RuntimeError("bad number of dimensions")

    if rcon.is_head_rank:
        print "%d elements" % len(mesh.elements)
        mesh_data = rcon.distribute_mesh(mesh)
    else:
        mesh_data = rcon.receive_mesh()

    discr = rcon.make_discretization(mesh_data, order=4)

    from hedge.timestep.runge_kutta import LSRK4TimeStepper
    stepper = LSRK4TimeStepper()

    from hedge.visualization import VtkVisualizer
    if write_output:
        vis = VtkVisualizer(discr, rcon, "fld")

    source_center = np.array([0.7, 0.4])
    source_width = 1/16
    source_omega = 3

    import hedge.optemplate as sym
    sym_x = sym.nodes(2)
    sym_source_center_dist = sym_x - source_center

    from hedge.models.wave import VariableVelocityStrongWaveOperator
    op = VariableVelocityStrongWaveOperator(
            c=sym.If(sym.Comparison(
                np.dot(sym_x, sym_x), "<", 0.4**2),
                1, 0.5),
            dimensions=discr.dimensions,
            source=
            sym.CFunction("sin")(source_omega*sym.ScalarParameter("t"))
            * sym.CFunction("exp")(
                -np.dot(sym_source_center_dist, sym_source_center_dist)
                / source_width**2),
            dirichlet_tag=dir_tag,
            neumann_tag=neu_tag,
            radiation_tag=rad_tag,
            flux_type=flux_type_arg
            )

    from hedge.tools import join_fields
    fields = join_fields(discr.volume_zeros(),
            [discr.volume_zeros() for i in range(discr.dimensions)])

    # {{{ diagnostics setup

    from pytools.log import LogManager, \
            add_general_quantities, \
            add_simulation_quantities, \
            add_run_info

    if write_output:
        log_file_name = "wave.dat"
    else:
        log_file_name = None

    logmgr = LogManager(log_file_name, "w", rcon.communicator)
    add_run_info(logmgr)
    add_general_quantities(logmgr)
    add_simulation_quantities(logmgr)
    discr.add_instrumentation(logmgr)

    from pytools.log import IntervalTimer
    vis_timer = IntervalTimer("t_vis", "Time spent visualizing")
    logmgr.add_quantity(vis_timer)
    stepper.add_instrumentation(logmgr)

    from hedge.log import LpNorm
    u_getter = lambda: fields[0]
    logmgr.add_quantity(LpNorm(u_getter, discr, 1, name="l1_u"))
    logmgr.add_quantity(LpNorm(u_getter, discr, name="l2_u"))

    logmgr.add_watches(["step.max", "t_sim.max", "l2_u", "t_step.max"])

    # }}}

    # {{{ timestep loop

    rhs = op.bind(discr)
    try:
        from hedge.timestep.stability import \
                approximate_rk4_relative_imag_stability_region
        max_dt = (
                1/discr.compile(op.max_eigenvalue_expr())()
                * discr.dt_non_geometric_factor()
                * discr.dt_geometric_factor()
                * approximate_rk4_relative_imag_stability_region(stepper))
        if flux_type_arg == "central":
            max_dt *= 0.25

        from hedge.timestep import times_and_steps
        step_it = times_and_steps(final_time=3, logmgr=logmgr,
                max_dt_getter=lambda t: max_dt)

        for step, t, dt in step_it:
            if step % 10 == 0 and write_output:
                visf = vis.make_file("fld-%04d" % step)

                vis.add_data(visf,
                        [
                            ("u", fields[0]),
                            ("v", fields[1:]),
                        ],
                        time=t,
                        step=step)
                visf.close()

            fields = stepper(fields, t, dt, rhs)

        assert discr.norm(fields) < 1
    finally:
        if write_output:
            vis.close()

        logmgr.close()
        discr.close()

    # }}}

if __name__ == "__main__":
    main(flux_type_arg="upwind")


# entry points for py.test ----------------------------------------------------
def test_var_velocity_wave():
    from pytools.test import mark_test
    mark_long = mark_test.long

    for flux_type in ["upwind", "central"]:
        yield ("dirichlet var-v wave equation with %s flux" % flux_type,
                mark_long(main),
                False, TAG_ALL, TAG_NONE, TAG_NONE, flux_type)
    yield ("neumann var-v wave equation", mark_long(main),
            False, TAG_NONE, TAG_ALL, TAG_NONE)
    yield ("radiation-bc var-v wave equation", mark_long(main),
            False, TAG_NONE, TAG_NONE, TAG_ALL)

# vim: foldmethod=marker
