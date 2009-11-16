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




def make_mesh(a, b, pml_width=0.25, **kwargs):
    from meshpy.geometry import GeometryBuilder, make_box
    geob = GeometryBuilder()

    box_points, box_facets, _ = make_box(a, b)
    geob.add_geometry(box_points, box_facets)
    geob.wrap_in_box(pml_width)

    mesh_mod = geob.mesher_module()
    mi = mesh_mod.MeshInfo()
    geob.set(mi)

    built_mi = mesh_mod.build(mi, **kwargs)

    print "%d elements" % len(built_mi.elements)

    def boundary_tagger(fvi, el, fn, points):
        return []

    from hedge.mesh import make_conformal_mesh
    return make_conformal_mesh(
            built_mi.points,
            built_mi.elements, 
            boundary_tagger)




def main(write_output=True):
    from hedge.timestep import RK4TimeStepper
    from hedge.mesh import make_disk_mesh
    from pylo import DB_VARTYPE_VECTOR
    from math import sqrt, pi, exp

    from hedge.backends import guess_run_context, FEAT_CUDA
    rcon = guess_run_context()

    epsilon0 = 8.8541878176e-12 # C**2 / (N m**2)
    mu0 = 4*pi*1e-7 # N/A**2.
    epsilon = 1*epsilon0
    mu = 1*mu0

    c = 1/sqrt(mu*epsilon)

    cylindrical = False
    periodic = False

    pml_width = 0.5
    #mesh = make_mesh(a=numpy.array((-1,-1,-1)), b=numpy.array((1,1,1)), 
    #mesh = make_mesh(a=numpy.array((-3,-3)), b=numpy.array((3,3)), 
    mesh = make_mesh(a=numpy.array((-1,-1)), b=numpy.array((1,1)), 
    #mesh = make_mesh(a=numpy.array((-2,-2)), b=numpy.array((2,2)), 
            pml_width=pml_width, max_volume=0.01)

    if rcon.is_head_rank:
        mesh_data = rcon.distribute_mesh(mesh)
    else:
        mesh_data = rcon.receive_mesh()

    class Current:
        def volume_interpolant(self, t, discr):
            from hedge.tools import make_obj_array

            result = discr.volume_zeros()

            omega = 6*c
            if omega*t > 2*pi:
                return make_obj_array([result, result, result])

            x = make_obj_array(discr.nodes.T)
            r = numpy.sqrt(numpy.dot(x, x))

            idx = r<0.3
            result[idx] = (1+numpy.cos(pi*r/0.3))[idx] \
                    *numpy.sin(omega*t)**3

            return make_obj_array([-result, result, result])

    order = 3
    discr = rcon.make_discretization(mesh_data, order=order)

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

    stepper = RK4TimeStepper()

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
                from pylo import DB_VARTYPE_VECTOR, DB_VARTYPE_SCALAR
                vis.add_data(visf, [ 
                    ("e", e), 
                    ("h", h), 
                    ("p", p), 
                    ("q", q), 
                    ("j", Current().volume_interpolant(t, discr)), 
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
