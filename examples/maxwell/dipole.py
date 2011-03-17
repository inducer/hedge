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

# FIXME: This example doesn't quite do what it should any more.




from __future__ import division
import numpy
import numpy.linalg as la




def main(write_output=True, allow_features=None):
    from hedge.timestep import RK4TimeStepper
    from hedge.mesh import make_ball_mesh, make_cylinder_mesh, make_box_mesh
    from hedge.visualization import \
            VtkVisualizer, \
            SiloVisualizer, \
            get_rank_partition
    from math import sqrt, pi

    from hedge.backends import guess_run_context
    rcon = guess_run_context(allow_features)

    epsilon0 = 8.8541878176e-12 # C**2 / (N m**2)
    mu0 = 4*pi*1e-7 # N/A**2.
    epsilon = 1*epsilon0
    mu = 1*mu0

    dims = 3

    if rcon.is_head_rank:
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

    if rcon.is_head_rank:
        mesh_data = rcon.distribute_mesh(mesh)
    else:
        mesh_data = rcon.receive_mesh()

    #for order in [1,2,3,4,5,6]:
    discr = rcon.make_discretization(mesh_data, order=3)

    if write_output:
        vis = VtkVisualizer(discr, rcon, "dipole")

    from analytic_solutions import DipoleFarField, SphericalFieldAdapter
    from hedge.data import ITimeDependentGivenFunction

    sph_dipole = DipoleFarField(
            q=1, #C
            d=1/39,
            omega=2*pi*1e8,
            epsilon=epsilon0,
            mu=mu0,
            )
    cart_dipole = SphericalFieldAdapter(sph_dipole)

    class PointDipoleSource(ITimeDependentGivenFunction):
        def __init__(self):
            from pyrticle.tools import CInfinityShapeFunction
            sf = CInfinityShapeFunction(
                        0.1*sph_dipole.wavelength,
                        discr.dimensions)
            self.num_sf = discr.interpolate_volume_function(
                    lambda x, el: sf(x))
            self.vol_0 = discr.volume_zeros()

        def volume_interpolant(self, t, discr):
            from hedge.tools import make_obj_array
            return make_obj_array([
                self.vol_0,
                self.vol_0,
                sph_dipole.source_modulation(t)*self.num_sf
                ])

    from hedge.mesh import TAG_ALL, TAG_NONE
    if dims == 2:
        from hedge.models.em import TMMaxwellOperator as MaxwellOperator
    else:
        from hedge.models.em import MaxwellOperator

    op = MaxwellOperator(
            epsilon, mu,
            flux_type=1,
            pec_tag=TAG_NONE,
            absorb_tag=TAG_ALL,
            current=PointDipoleSource(),
            )

    fields = op.assemble_eh(discr=discr)

    if rcon.is_head_rank:
        print "#elements=", len(mesh.elements)

    stepper = RK4TimeStepper()

    # diagnostics setup ---------------------------------------------------
    from pytools.log import LogManager, add_general_quantities, \
            add_simulation_quantities, add_run_info

    if write_output:
        log_file_name = "dipole.dat"
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

    from pytools.log import PushLogQuantity
    relerr_e_q = PushLogQuantity("relerr_e", "1", "Relative error in masked E-field")
    relerr_h_q = PushLogQuantity("relerr_h", "1", "Relative error in masked H-field")
    logmgr.add_quantity(relerr_e_q)
    logmgr.add_quantity(relerr_h_q)

    logmgr.add_watches(["step.max", "t_sim.max", 
        ("W_field", "W_el+W_mag"), "t_step.max",
        "relerr_e", "relerr_h"])

    if write_output:
        point_timeseries = [
                (open("b-x%d-vs-time.dat" % i, "w"), 
                    open("b-x%d-vs-time-true.dat" % i, "w"), 
                    discr.get_point_evaluator(numpy.array([i,0,0][:dims],
                        dtype=discr.default_scalar_type)))
                    for i in range(1,5)
                    ]

    # timestep loop -------------------------------------------------------
    mask = discr.interpolate_volume_function(sph_dipole.far_field_mask)

    def apply_mask(field):
        from hedge.tools import log_shape
        ls = log_shape(field)
        result = discr.volume_empty(ls)
        from pytools import indices_in_shape
        for i in indices_in_shape(ls):
            result[i] = mask * field[i]

        return result

    rhs = op.bind(discr)

    t = 0
    try:
        from hedge.timestep import times_and_steps
        step_it = times_and_steps(
                final_time=1e-8, logmgr=logmgr,
                max_dt_getter=lambda t: op.estimate_timestep(discr,
                    stepper=stepper, t=t, fields=fields))

        for step, t, dt in step_it:
            if write_output and step % 10 == 0:
                sub_timer = vis_timer.start_sub_timer()
                e, h = op.split_eh(fields)
                sph_dipole.set_time(t)
                true_e, true_h = op.split_eh(
                        discr.interpolate_volume_function(cart_dipole))
                visf = vis.make_file("dipole-%04d" % step)

                mask_e = apply_mask(e)
                mask_h = apply_mask(h)
                mask_true_e = apply_mask(true_e)
                mask_true_h = apply_mask(true_h)

                from pyvisfile.silo import DB_VARTYPE_VECTOR
                vis.add_data(visf,
                        [ 
                            ("e", e), 
                            ("h", h), 
                            ("true_e", true_e), 
                            ("true_h", true_h), 
                            ("mask_e", mask_e), 
                            ("mask_h", mask_h), 
                            ("mask_true_e", mask_true_e), 
                            ("mask_true_h", mask_true_h)],
                        time=t, step=step)
                visf.close()
                sub_timer.stop().submit()

                from hedge.tools import relative_error
                relerr_e_q.push_value(
                        relative_error(
                            discr.norm(mask_e-mask_true_e),
                            discr.norm(mask_true_e)))
                relerr_h_q.push_value(
                        relative_error(
                            discr.norm(mask_h-mask_true_h),
                            discr.norm(mask_true_h)))

                if write_output:
                    for outf_num, outf_true, evaluator in point_timeseries:
                        for outf, ev_h in zip([outf_num, outf_true],
                                [h, true_h]):
                            outf.write("%g\t%g\n" % (t, op.mu*evaluator(ev_h[1])))
                            outf.flush()

            fields = stepper(fields, t, dt, rhs)

    finally:
        if write_output:
            vis.close()

        logmgr.save()
        discr.close()



if __name__ == "__main__":
    main()




from pytools.test import mark_test
@mark_test.long
def test_run():
    main(write_output=False)
