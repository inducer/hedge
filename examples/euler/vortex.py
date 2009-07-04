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




class Vortex:
    def __init__(self, beta, gamma, center, velocity):
        self.beta = beta
        self.gamma = gamma
        self.center = numpy.asarray(center)
        self.velocity = numpy.asarray(velocity)

    def __call__(self, t, x_vec):
        vortex_loc = self.center + t*self.velocity

        # coordinates relative to vortex center
        x_rel = x_vec[0] - vortex_loc[0]
        y_rel = x_vec[1] - vortex_loc[1]

        # Y.C. Zhou, G.W. Wei / Journal of Computational Physics 189 (2003) 159
        # also JSH/TW Nodal DG Methods, p. 209

        from math import pi
        r = numpy.sqrt(x_rel**2+y_rel**2)
        expterm = self.beta*numpy.exp(1-r**2)
        u = self.velocity[0] - expterm*y_rel/(2*pi)
        v = self.velocity[1] + expterm*x_rel/(2*pi)
        rho = (1-(self.gamma-1)/(16*self.gamma*pi**2)*expterm**2)**(1/(self.gamma-1))
        p = rho**self.gamma

        e = p/(self.gamma-1) + rho/2*(u**2+v**2)

        from hedge.tools import join_fields
        return join_fields(rho, e, rho*u, rho*v)

    def volume_interpolant(self, t, discr):
        return self(t, discr.nodes.T)

    def boundary_interpolant(self, t, discr, tag):
        return self(t, discr.get_boundary(tag).nodes.T)




def main(write_output=True):
    from hedge.backends import guess_run_context
    rcon = guess_run_context()

    gamma = 1.4

    from hedge.tools import EOCRecorder, to_obj_array
    eoc_rec = EOCRecorder()

    if rcon.is_head_rank:
        from hedge.mesh import make_rect_mesh
        mesh = make_rect_mesh((0,-5), (10,5), max_area=0.15)
        mesh_data = rcon.distribute_mesh(mesh)
    else:
        mesh_data = rcon.receive_mesh()

    for order in [3, 4, 5]:
    #for order in [1,2,3,4,5,6]:
        discr = rcon.make_discretization(mesh_data, order=order)

        from hedge.visualization import SiloVisualizer
        #vis = VtkVisualizer(discr, rcon, "vortex-%d" % order)
        vis = SiloVisualizer(discr, rcon)

        vortex = Vortex(beta=5, gamma=gamma, 
                center=[5,0], 
                velocity=[1,0])
        fields = vortex.volume_interpolant(0, discr)

        from hedge.pde import EulerOperator
        op = EulerOperator(dimensions=2, gamma=1.4, bc=vortex)

        euler_ex = op.bind(discr)

        max_eigval = [0]
        def rhs(t, q):
            ode_rhs, speed = euler_ex(t, q)
            max_eigval[0] = speed
            return ode_rhs
        rhs(0, fields)

        dt = discr.dt_factor(max_eigval[0])
        final_time = 0.2
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

        if write_output:
            log_file_name = "euler-%d.dat" % order
        else:
            log_file_name = None

        logmgr = LogManager(log_file_name, "w", rcon.communicator)
        add_run_info(logmgr)
        add_general_quantities(logmgr)
        add_simulation_quantities(logmgr, dt)
        discr.add_instrumentation(logmgr)
        stepper.add_instrumentation(logmgr)

        logmgr.add_watches(["step.max", "t_sim.max", "t_step.max"])

        # timestep loop -------------------------------------------------------
        t = 0

        try:
            for step in range(nsteps):
                logmgr.tick()

                if step % 1 == 0 and write_output:
                #if False:
                    visf = vis.make_file("vortex-%d-%04d" % (order, step))

                    #true_fields = vortex.volume_interpolant(t, discr)

                    from pylo import DB_VARTYPE_VECTOR
                    vis.add_data(visf,
                            [
                                ("rho", op.rho(fields)),
                                ("e", op.e(fields)),
                                ("rho_u", op.rho_u(fields)),
                                ("u", op.u(fields)),

                                #("true_rho", op.rho(true_fields)),
                                #("true_e", op.e(true_fields)),
                                #("true_rho_u", op.rho_u(true_fields)),
                                #("true_u", op.u(true_fields)),

                                #("rhs_rho", op.rho(rhs_fields)),
                                #("rhs_e", op.e(rhs_fields)),
                                #("rhs_rho_u", op.rho_u(rhs_fields)),
                                ],
                            #expressions=[
                                #("diff_rho", "rho-true_rho"),
                                #("diff_e", "e-true_e"),
                                #("diff_rho_u", "rho_u-true_rho_u", DB_VARTYPE_VECTOR),

                                #("p", "0.4*(e- 0.5*(rho_u*u))"),
                                #],
                            time=t, step=step
                            )
                    visf.close()

                fields = stepper(fields, t, dt, rhs)
                t += dt

                dt = discr.dt_factor(max_eigval[0])

            true_fields = vortex.volume_interpolant(t, discr)
            eoc_rec.add_data_point(order, discr.norm(fields-true_fields))
            print
            print eoc_rec.pretty_print("P.Deg.", "L2 Error")

        finally:
            if write_output:
                vis.close()

            logmgr.tick()
            logmgr.save()

            discr.close()

    # after order loop
    assert eoc_rec.estimate_order_of_convergence()[0,1] > 6



if __name__ == "__main__":
    main()



# entry points for py.test ----------------------------------------------------
from pytools.test import mark_test
@mark_test(long=True)
def test_euler_vortex():
    main(write_output=True)
