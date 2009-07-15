# Hedge - the Hybrid'n'Easy DG Environment
# Copyright (C) 2008 Andreas Kloeckner
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




class SteadyShearFlow:
    def __init__(self, gamma, mu):
        self.gamma = gamma
        self.mu = mu

    def __call__(self, t, x_vec):
        # JSH/TW Nodal DG Methods, p.326 

        rho = numpy.zeros_like(x_vec[0])
        rho.fill(1)
        rho_u = x_vec[1] * x_vec[1]
        rho_v = numpy.zeros_like(x_vec[0])
        e = (2 * self.mu * x_vec[0] + 10) / (self.gamma - 1) + x_vec[1]**4 / 2

        from hedge.tools import join_fields
        return join_fields(rho, e, rho_u, rho_v)

    def volume_interpolant(self, t, discr):
        return discr.convert_volume(
			self(t, discr.nodes.T),
			kind=discr.compute_kind)

    def boundary_interpolant(self, t, discr, tag):
        return discr.convert_boundary(
			self(t, discr.get_boundary(tag).nodes.T),
			 tag=tag, kind=discr.compute_kind)




def main():
    from hedge.backends import guess_run_context
    rcon = guess_run_context(
    ["cuda"]
    )

    gamma = 1.5
    mu = 0.01

    from hedge.tools import EOCRecorder, to_obj_array
    eoc_rec = EOCRecorder()
    
    if rcon.is_head_rank:
        from hedge.mesh import make_rect_mesh, \
                               make_centered_regular_rect_mesh
        #mesh = make_rect_mesh((0,-5), (10,5), max_area=0.15)
        refine = 1
        mesh = make_centered_regular_rect_mesh((0,0), (10,1), n=(9,9),
                            periodicity=(True, False),
                            post_refine_factor=refine)
        mesh_data = rcon.distribute_mesh(mesh)
    else:
        mesh_data = rcon.receive_mesh()

    for order in [3]:
        discr = rcon.make_discretization(mesh_data, order=order,
			debug=["cuda_no_plan",
			#"print_op_code"
			],
			default_scalar_type=numpy.float64)

        from hedge.visualization import SiloVisualizer, VtkVisualizer
        #vis = VtkVisualizer(discr, rcon, "shearflow-%d" % order)
        vis = SiloVisualizer(discr, rcon)

        shearflow = SteadyShearFlow(gamma=gamma, mu=mu)
        fields = shearflow.volume_interpolant(0, discr)

        from hedge.pde import NavierStokesOperator
        op = NavierStokesOperator(dimensions=2, gamma=gamma, mu=mu, bc=shearflow)

        navierstokes_ex = op.bind(discr)

        max_eigval = [0]
        def rhs(t, q):
            ode_rhs, speed = navierstokes_ex(t, q)
            max_eigval[0] = speed
            return ode_rhs
        rhs(0, fields)

        dt = discr.dt_factor(max_eigval[0], order=2)
        final_time = 0.02
        nsteps = int(final_time/dt)+1
        dt = final_time/nsteps

        if rcon.is_head_rank:
            print "---------------------------------------------"
            print "order %d" % order
            print "---------------------------------------------"
            print "dt", dt
            print "nsteps", nsteps
            print "#elements=", len(mesh.elements)

        from hedge.timestep import RK4TimeStepper
        stepper = RK4TimeStepper()

        # diagnostics setup ---------------------------------------------------
        from pytools.log import LogManager, add_general_quantities, \
                add_simulation_quantities, add_run_info

        logmgr = LogManager("navierstokes-%d.dat" % order, "w", rcon.communicator)
        add_run_info(logmgr)
        add_general_quantities(logmgr)
        add_simulation_quantities(logmgr, dt)
        discr.add_instrumentation(logmgr)
        stepper.add_instrumentation(logmgr)

        logmgr.add_watches(["step.max", "t_sim.max", "t_step.max"])

        # timestep loop -------------------------------------------------------
        t = 0

        for step in range(nsteps):
            logmgr.tick()

            if step % 100 == 0:
            #if False:
                visf = vis.make_file("shearflow-%d-%04d" % (order, step))

                true_fields = shearflow.volume_interpolant(t, discr)

                rhs_fields = rhs(t, fields)

                from pylo import DB_VARTYPE_VECTOR
                vis.add_data(visf,
                        [
                            ("rho", discr.convert_volume(op.rho(fields), kind="numpy")),
                            ("e", discr.convert_volume(op.e(fields), kind="numpy")),
                            ("rho_u", discr.convert_volume(op.rho_u(fields), kind="numpy")),
                            ("u", discr.convert_volume(op.u(fields), kind="numpy")),

                            ("true_rho", discr.convert_volume(op.rho(true_fields), kind="numpy")),
                            ("true_e", discr.convert_volume(op.e(true_fields), kind="numpy")),
                            ("true_rho_u", discr.convert_volume(op.rho_u(true_fields), kind="numpy")),
                            ("true_u", discr.convert_volume(op.u(true_fields), kind="numpy")),

                            ("rhs_rho", discr.convert_volume(op.rho(rhs_fields), kind="numpy")),
                            ("rhs_e", discr.convert_volume(op.e(rhs_fields), kind="numpy")),
                            ("rhs_rho_u", discr.convert_volume(op.rho_u(rhs_fields), kind="numpy")),
                            ],
                        expressions=[
                            ("diff_rho", "rho-true_rho"),
                            ("diff_e", "e-true_e"),
                            ("diff_rho_u", "rho_u-true_rho_u", DB_VARTYPE_VECTOR),

                            ("p", "0.4*(e- 0.5*(rho_u*u))"),
                            ],
                        time=t, step=step
                        )
                visf.close()

            fields = stepper(fields, t, dt, rhs)
            t += dt

            dt = discr.dt_factor(max_eigval[0], order=2)

        logmgr.tick()
        logmgr.save()

        true_fields = shearflow.volume_interpolant(t, discr)
        eoc_rec.add_data_point(order, discr.norm(fields[1]-true_fields[1]))
        print
        print eoc_rec.pretty_print("P.Deg.", "L2 Error")

if __name__ == "__main__":
    main()
