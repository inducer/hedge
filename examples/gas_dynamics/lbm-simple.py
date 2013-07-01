# Hedge - the Hybrid'n'Easy DG Environment
# Copyright (C) 2011 Andreas Kloeckner
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




def main(write_output=True, dtype=np.float32):
    from hedge.backends import guess_run_context
    rcon = guess_run_context()

    from hedge.mesh.generator import make_rect_mesh
    if rcon.is_head_rank:
        h_fac = 1
        mesh = make_rect_mesh(a=(0,0),b=(1,1), max_area=h_fac**2*1e-4,
                periodicity=(True,True),
                subdivisions=(int(70/h_fac), int(70/h_fac)))

    from hedge.models.gas_dynamics.lbm import \
            D2Q9LBMMethod, LatticeBoltzmannOperator

    op = LatticeBoltzmannOperator(
            D2Q9LBMMethod(), lbm_delta_t=0.001, nu=1e-4)

    if rcon.is_head_rank:
        print "%d elements" % len(mesh.elements)
        mesh_data = rcon.distribute_mesh(mesh)
    else:
        mesh_data = rcon.receive_mesh()

    discr = rcon.make_discretization(mesh_data, order=3,
            default_scalar_type=dtype,
            debug=["cuda_no_plan"])
    from hedge.timestep.runge_kutta import LSRK4TimeStepper
    stepper = LSRK4TimeStepper(dtype=dtype,
            #vector_primitive_factory=discr.get_vector_primitive_factory()
            )

    from hedge.visualization import VtkVisualizer
    if write_output:
        vis = VtkVisualizer(discr, rcon, "fld")

    from hedge.data import CompiledExpressionData
    def ic_expr(t, x, fields):
        from hedge.optemplate import CFunction
        from pymbolic.primitives import IfPositive
        from pytools.obj_array import make_obj_array

        tanh = CFunction("tanh")
        sin = CFunction("sin")

        rho = 1
        u0 = 0.05
        w = 0.05
        delta = 0.05

        from hedge.optemplate.primitives import make_common_subexpression as cse
        u = cse(make_obj_array([
            IfPositive(x[1]-1/2,
                u0*tanh(4*(3/4-x[1])/w),
                u0*tanh(4*(x[1]-1/4)/w)),
            u0*delta*sin(2*np.pi*(x[0]+1/4))]),
            "u")

        return make_obj_array([
            op.method.f_equilibrium(rho, alpha, u)
            for alpha in range(len(op.method))
            ])


    # timestep loop -----------------------------------------------------------
    stream_rhs = op.bind_rhs(discr)
    collision_update = op.bind(discr, op.collision_update)
    get_rho = op.bind(discr, op.rho)
    get_rho_u = op.bind(discr, op.rho_u)


    f_bar = CompiledExpressionData(ic_expr).volume_interpolant(0, discr)

    from hedge.discretization import ExponentialFilterResponseFunction
    from hedge.optemplate.operators import FilterOperator
    mode_filter = FilterOperator(
            ExponentialFilterResponseFunction(min_amplification=0.9, order=4))\
                    .bind(discr)

    final_time = 1000
    try:
        lbm_dt = op.lbm_delta_t
        dg_dt = op.estimate_timestep(discr, stepper=stepper)
        print dg_dt

        dg_steps_per_lbm_step = int(np.ceil(lbm_dt / dg_dt))
        dg_dt = lbm_dt / dg_steps_per_lbm_step

        lbm_steps = int(final_time // op.lbm_delta_t)
        for step in xrange(lbm_steps):
            t = step*lbm_dt

            if step % 100 == 0 and write_output:
                visf = vis.make_file("fld-%04d" % step)

                rho = get_rho(f_bar)
                rho_u = get_rho_u(f_bar)
                vis.add_data(visf,
                        [ ("fbar%d" %i, 
                            discr.convert_volume(f_bar_i, "numpy")) for i, f_bar_i in enumerate(f_bar)]+
                        [
                            ("rho", discr.convert_volume(rho, "numpy")),
                            ("rho_u", discr.convert_volume(rho_u, "numpy")),
                        ],
                        time=t,
                        step=step)
                visf.close()

            print "step=%d, t=%f" % (step, t)

            f_bar = collision_update(f_bar)

            for substep in range(dg_steps_per_lbm_step):
                f_bar = stepper(f_bar, t + substep*dg_dt, dg_dt, stream_rhs)

            #f_bar = mode_filter(f_bar)

    finally:
        if write_output:
            vis.close()

        discr.close()




if __name__ == "__main__":
    main(True)
