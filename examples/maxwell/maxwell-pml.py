"""Hedge is the Hybrid'n'Easy Discontinuous Galerkin Environment."""

from __future__ import division

__copyright__ = "Copyright (C) 2007 Andreas Kloeckner"

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




def make_mesh(a, b, pml_width=0.25, **kwargs):
    from meshpy.geometry import GeometryBuilder, make_circle
    geob = GeometryBuilder()

    circle_centers = [(-1.5, 0), (1.5, 0)]
    for cent in circle_centers:
        geob.add_geometry(*make_circle(1, cent))

    geob.wrap_in_box(1)
    geob.wrap_in_box(pml_width)

    mesh_mod = geob.mesher_module()
    mi = mesh_mod.MeshInfo()
    geob.set(mi)

    mi.set_holes(circle_centers)

    built_mi = mesh_mod.build(mi, **kwargs)

    def boundary_tagger(fvi, el, fn, points):
        return []

    from hedge.mesh import make_conformal_mesh_ext
    from hedge.mesh.element import Triangle
    pts = np.asarray(built_mi.points, dtype=np.float64)
    return make_conformal_mesh_ext(
            pts,
            [Triangle(i, el, pts)
                for i, el in enumerate(built_mi.elements)],
            boundary_tagger)




def main(write_output=True):
    from hedge.timestep.runge_kutta import LSRK4TimeStepper
    from math import sqrt, pi, exp

    from hedge.backends import guess_run_context
    rcon = guess_run_context()

    epsilon0 = 8.8541878176e-12 # C**2 / (N m**2)
    mu0 = 4*pi*1e-7 # N/A**2.
    epsilon = 1*epsilon0
    mu = 1*mu0

    c = 1/sqrt(mu*epsilon)

    pml_width = 0.5
    #mesh = make_mesh(a=np.array((-1,-1,-1)), b=np.array((1,1,1)),
    #mesh = make_mesh(a=np.array((-3,-3)), b=np.array((3,3)),
    mesh = make_mesh(a=np.array((-1,-1)), b=np.array((1,1)),
    #mesh = make_mesh(a=np.array((-2,-2)), b=np.array((2,2)),
            pml_width=pml_width, max_volume=0.01)

    if rcon.is_head_rank:
        mesh_data = rcon.distribute_mesh(mesh)
    else:
        mesh_data = rcon.receive_mesh()

    class Current:
        def volume_interpolant(self, t, discr):
            from hedge.tools import make_obj_array

            result = discr.volume_zeros(kind="numpy", dtype=np.float64)

            omega = 6*c
            if omega*t > 2*pi:
                return make_obj_array([result, result, result])

            x = make_obj_array(discr.nodes.T)
            r = np.sqrt(np.dot(x, x))

            idx = r<0.3
            result[idx] = (1+np.cos(pi*r/0.3))[idx] \
                    *np.sin(omega*t)**3

            result = discr.convert_volume(result, kind=discr.compute_kind,
                    dtype=discr.default_scalar_type)
            return make_obj_array([-result, result, result])

    order = 3
    discr = rcon.make_discretization(mesh_data, order=order,
            debug=["cuda_no_plan"])

    from hedge.visualization import VtkVisualizer
    if write_output:
        vis = VtkVisualizer(discr, rcon, "em-%d" % order)

    from hedge.mesh import TAG_ALL, TAG_NONE
    from hedge.data import GivenFunction, TimeHarmonicGivenFunction, TimeIntervalGivenFunction
    from hedge.models.em import MaxwellOperator
    from hedge.models.pml import \
            AbarbanelGottliebPMLMaxwellOperator, \
            AbarbanelGottliebPMLTMMaxwellOperator, \
            AbarbanelGottliebPMLTEMaxwellOperator

    op = AbarbanelGottliebPMLTEMaxwellOperator(epsilon, mu, flux_type=1,
            current=Current(),
            pec_tag=TAG_ALL,
            absorb_tag=TAG_NONE,
            add_decay=True
            )

    fields = op.assemble_ehpq(discr=discr)

    stepper = LSRK4TimeStepper()

    if rcon.is_head_rank:
        print "order %d" % order
        print "#elements=", len(mesh.elements)

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
    add_simulation_quantities(logmgr)
    discr.add_instrumentation(logmgr)
    stepper.add_instrumentation(logmgr)

    from pytools.log import IntervalTimer
    vis_timer = IntervalTimer("t_vis", "Time spent visualizing")
    logmgr.add_quantity(vis_timer)

    from hedge.log import EMFieldGetter, add_em_quantities
    field_getter = EMFieldGetter(discr, op, lambda: fields)
    add_em_quantities(logmgr, op, field_getter)

    logmgr.add_watches(["step.max", "t_sim.max", ("W_field", "W_el+W_mag"), "t_step.max"])

    from hedge.log import LpNorm
    class FieldIdxGetter:
        def __init__(self, whole_getter, idx):
            self.whole_getter = whole_getter
            self.idx = idx

        def __call__(self):
            return self.whole_getter()[self.idx]

    # timestep loop -------------------------------------------------------

    t = 0
    pml_coeff = op.coefficients_from_width(discr, width=pml_width)
    rhs = op.bind(discr, pml_coeff)

    try:
        from hedge.timestep import times_and_steps
        step_it = times_and_steps(
                final_time=4/c, logmgr=logmgr,
                max_dt_getter=lambda t: op.estimate_timestep(discr,
                    stepper=stepper, t=t, fields=fields))

        for step, t, dt in step_it:
            if step % 10 == 0 and write_output:
                e, h, p, q = op.split_ehpq(fields)
                visf = vis.make_file("em-%d-%04d" % (order, step))
                #pml_rhs_e, pml_rhs_h, pml_rhs_p, pml_rhs_q = \
                        #op.split_ehpq(rhs(t, fields))
                j = Current().volume_interpolant(t, discr)
                vis.add_data(visf, [
                    ("e", discr.convert_volume(e, "numpy")),
                    ("h", discr.convert_volume(h, "numpy")),
                    ("p", discr.convert_volume(p, "numpy")),
                    ("q", discr.convert_volume(q, "numpy")),
                    ("j", discr.convert_volume(j, "numpy")),
                    #("pml_rhs_e", pml_rhs_e),
                    #("pml_rhs_h", pml_rhs_h),
                    #("pml_rhs_p", pml_rhs_p),
                    #("pml_rhs_q", pml_rhs_q),
                    #("max_rhs_e", max_rhs_e),
                    #("max_rhs_h", max_rhs_h),
                    #("max_rhs_p", max_rhs_p),
                    #("max_rhs_q", max_rhs_q),
                    ],
                    time=t, step=step)
                visf.close()

            fields = stepper(fields, t, dt, rhs)

        _, _, energies_data = logmgr.get_expr_dataset("W_el+W_mag")
        energies = [value for tick_nbr, value in energies_data]

        assert energies[-1] < max(energies) * 1e-2

    finally:
        logmgr.close()

        if write_output:
            vis.close()

if __name__ == "__main__":
    main()




# entry points for py.test ----------------------------------------------------
from pytools.test import mark_test
@mark_test.long
def test_maxwell_pml():
    main(write_output=False)
