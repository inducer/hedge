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

# modifies isentropic vortex solution so that rho->Arho, P->A^gamma rho^gamma
# this will be analytic solution if appropriate source terms are added
# to the RHS. coded for vel_x=1, vel_y=0


class Vortex:
    def __init__(self, beta, gamma, center, velocity, densityA):
        self.beta = beta
        self.gamma = gamma
        self.center = numpy.asarray(center)
        self.velocity = numpy.asarray(velocity)
        self.densityA = densityA

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
        rho = self.densityA*(1-(self.gamma-1)/(16*self.gamma*pi**2)*expterm**2)**(1/(self.gamma-1))
        p = rho**self.gamma

        e = p/(self.gamma-1) + rho/2*(u**2+v**2)

        from hedge.tools import join_fields
        return join_fields(rho, e, rho*u, rho*v)

    def volume_interpolant(self, t, discr):
        return discr.convert_volume(
                        self(t, discr.nodes.T
                            .astype(discr.default_scalar_type)),
                        kind=discr.compute_kind)

    def boundary_interpolant(self, t, discr, tag):
        return discr.convert_boundary(
                        self(t, discr.get_boundary(tag).nodes.T
                            .astype(discr.default_scalar_type)),
                         tag=tag, kind=discr.compute_kind)



class SourceTerms:
    def __init__(self, beta, gamma, center, velocity, densityA):
        self.beta = beta
        self.gamma = gamma
        self.center = numpy.asarray(center)
        self.velocity = numpy.asarray(velocity)
        self.densityA = densityA



    def __call__(self,t,x_vec,q):

        vortex_loc = self.center + t*self.velocity

        # coordinates relative to vortex center
        x_rel = x_vec[0] - vortex_loc[0]
        y_rel = x_vec[1] - vortex_loc[1]

        # sources written in terms of A=1.0 solution 
        # (standard isentropic vortex)

        from math import pi
        r = numpy.sqrt(x_rel**2+y_rel**2)
        expterm = self.beta*numpy.exp(1-r**2)
        u = self.velocity[0] - expterm*y_rel/(2*pi)
        v = self.velocity[1] + expterm*x_rel/(2*pi)
        rho = (1-(self.gamma-1)/(16*self.gamma*pi**2)*expterm**2)**(1/(self.gamma-1))
        p = rho**self.gamma

        #computed necessary derivatives
        expterm_t = 2*expterm*x_rel
        expterm_x = -2*expterm*x_rel
        expterm_y = -2*expterm*y_rel
        u_x = -expterm*y_rel/(2*pi)*(-2*x_rel)
        v_y = expterm*x_rel/(2*pi)*(-2*y_rel)

        #derivatives for rho (A=1)
        facG=self.gamma-1
        rho_t = (1/facG)*(1-(facG)/(16*self.gamma*pi**2)*expterm**2)**(1/facG-1)* \
                (-facG/(16*self.gamma*pi**2)*2*expterm*expterm_t)
        rho_x = (1/facG)*(1-(facG)/(16*self.gamma*pi**2)*expterm**2)**(1/facG-1)* \
                (-facG/(16*self.gamma*pi**2)*2*expterm*expterm_x)
        rho_y = (1/facG)*(1-(facG)/(16*self.gamma*pi**2)*expterm**2)**(1/facG-1)* \
                (-facG/(16*self.gamma*pi**2)*2*expterm*expterm_y)

        #derivatives for rho (A=1) to the power of gamma
        rho_gamma_t = self.gamma*rho**(self.gamma-1)*rho_t
        rho_gamma_x = self.gamma*rho**(self.gamma-1)*rho_x
        rho_gamma_y = self.gamma*rho**(self.gamma-1)*rho_y


        factorA=self.densityA**self.gamma-self.densityA
        #construct source terms
        source_rho = x_vec[0]-x_vec[0]
        source_e = (factorA/(self.gamma-1))*(rho_gamma_t + self.gamma*(u_x*rho**self.gamma+u*rho_gamma_x)+ \
                self.gamma*(v_y*rho**self.gamma+v*rho_gamma_y))
        source_rhou = factorA*rho_gamma_x
        source_rhov = factorA*rho_gamma_y

        from hedge.tools import join_fields
        return join_fields(source_rho, source_e, source_rhou, source_rhov, x_vec[0]-x_vec[0])

    def volume_interpolant(self,t,q,discr):
        return discr.convert_volume(
                self(t,discr.nodes.T,q),
                kind=discr.compute_kind)


def main(write_output=True):
    from hedge.backends import guess_run_context
    rcon = guess_run_context(
                    #["cuda"]
                    )

    gamma = 1.4

    # at A=1 we have case of isentropic vortex, source terms 
    # arise for other values
    densityA = 2.0

    from hedge.tools import EOCRecorder, to_obj_array
    eoc_rec = EOCRecorder()

    if rcon.is_head_rank:
        from hedge.mesh import \
                make_rect_mesh, \
                make_centered_regular_rect_mesh

        refine = 1
        mesh = make_centered_regular_rect_mesh((0,-5), (10,5), n=(9,9),
                post_refine_factor=refine)
        mesh_data = rcon.distribute_mesh(mesh)
    else:
        mesh_data = rcon.receive_mesh()

    for order in [4,5]:
        discr = rcon.make_discretization(mesh_data, order=order,
                        debug=[#"cuda_no_plan",
                        #"print_op_code"
                        ],
                        default_scalar_type=numpy.float64)

        from hedge.visualization import SiloVisualizer, VtkVisualizer
        #vis = VtkVisualizer(discr, rcon, "vortex-%d" % order)
        vis = SiloVisualizer(discr, rcon)

        vortex = Vortex(beta=5, gamma=gamma,
                center=[5,0],
                velocity=[1,0], densityA=densityA)
        fields = vortex.volume_interpolant(0, discr)
        sources=SourceTerms(beta=5, gamma=gamma,
                center=[5,0],
                velocity=[1,0], densityA=densityA)

        from hedge.models.gas_dynamics import (
                GasDynamicsOperator, GammaLawEOS)
        from hedge.mesh import TAG_ALL

        op = GasDynamicsOperator(dimensions=2,
                mu=0.0, prandtl=0.72, spec_gas_const=287.1, 
                equation_of_state=GammaLawEOS(vortex.gamma),
                bc_inflow=vortex, bc_outflow=vortex, bc_noslip=vortex,
                inflow_tag=TAG_ALL, source=sources)

        euler_ex = op.bind(discr)

        max_eigval = [0]
        def rhs(t, q):
            ode_rhs, speed = euler_ex(t, q)
            max_eigval[0] = speed
            return ode_rhs
        rhs(0, fields)

        if rcon.is_head_rank:
            print "---------------------------------------------"
            print "order %d" % order
            print "---------------------------------------------"
            print "#elements=", len(mesh.elements)

        # limiter setup -------------------------------------------------------
        from hedge.models.gas_dynamics import SlopeLimiter1NEuler
        limiter = SlopeLimiter1NEuler(discr, gamma, 2, op)

        # time stepper --------------------------------------------------------
        from hedge.timestep import SSPRK3TimeStepper, RK4TimeStepper
        #stepper = SSPRK3TimeStepper(limiter=limiter)
        #stepper = SSPRK3TimeStepper()
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
        add_simulation_quantities(logmgr)
        discr.add_instrumentation(logmgr)
        stepper.add_instrumentation(logmgr)

        logmgr.add_watches(["step.max", "t_sim.max", "t_step.max"])

        # timestep loop -------------------------------------------------------
        t = 0

        #fields = limiter(fields)

        try:
            from hedge.timestep import times_and_steps
            step_it = times_and_steps(
                    final_time=.1,
                    #max_steps=500,
                    logmgr=logmgr,
                    max_dt_getter=lambda t: 0.4*op.estimate_timestep(discr,
                        stepper=stepper, t=t, max_eigenvalue=max_eigval[0]))

            for step, t, dt in step_it:
                if step % 1 == 0 and write_output:
                #if False:
                    visf = vis.make_file("vortex-%d-%04d" % (order, step))

                    true_fields = vortex.volume_interpolant(t, discr)

                    #rhs_fields = rhs(t, fields)

                    from pyvisfile.silo import DB_VARTYPE_VECTOR
                    vis.add_data(visf,
                            [
                                ("rho", discr.convert_volume(op.rho(fields), kind="numpy")),
                                ("e", discr.convert_volume(op.e(fields), kind="numpy")),
                                ("rho_u", discr.convert_volume(op.rho_u(fields), kind="numpy")),
                                ("u", discr.convert_volume(op.u(fields), kind="numpy")),

                                #("true_rho", discr.convert_volume(op.rho(true_fields), kind="numpy")),
                                #("true_e", discr.convert_volume(op.e(true_fields), kind="numpy")),
                                #("true_rho_u", discr.convert_volume(op.rho_u(true_fields), kind="numpy")),
                                #("true_u", discr.convert_volume(op.u(true_fields), kind="numpy")),

                                #("rhs_rho", discr.convert_volume(op.rho(rhs_fields), kind="numpy")),
                                #("rhs_e", discr.convert_volume(op.e(rhs_fields), kind="numpy")),
                                #("rhs_rho_u", discr.convert_volume(op.rho_u(rhs_fields), kind="numpy")),
                                ],
                            expressions=[
                                #("diff_rho", "rho-true_rho"),
                                #("diff_e", "e-true_e"),
                                #("diff_rho_u", "rho_u-true_rho_u", DB_VARTYPE_VECTOR),

                                ("p", "0.4*(e- 0.5*(rho_u*u))"),
                                ],
                            time=t, step=step
                            )
                    visf.close()

                fields = stepper(fields, t, dt, rhs)

            true_fields = vortex.volume_interpolant(t, discr)
            l2_error = discr.norm(fields-true_fields)
            l2_error_rho = discr.norm(op.rho(fields)-op.rho(true_fields))
            l2_error_e = discr.norm(op.e(fields)-op.e(true_fields))
            l2_error_rhou = discr.norm(op.rho_u(fields)-op.rho_u(true_fields))
            l2_error_u = discr.norm(op.u(fields)-op.u(true_fields))

            eoc_rec.add_data_point(order, l2_error_rho)
            print
            print eoc_rec.pretty_print("P.Deg.", "L2 Error")

            logmgr.set_constant("l2_error", l2_error)
            logmgr.set_constant("l2_error_rho", l2_error_rho)
            logmgr.set_constant("l2_error_e", l2_error_e)
            logmgr.set_constant("l2_error_rhou", l2_error_rhou)
            logmgr.set_constant("l2_error_u", l2_error_u)
            logmgr.set_constant("refinement", refine)

        finally:
            if write_output:
                vis.close()

            logmgr.close()
            discr.close()

    # after order loop
    #assert eoc_rec.estimate_order_of_convergence()[0,1] > 6




if __name__ == "__main__":
    main()



# entry points for py.test ----------------------------------------------------
from pytools.test import mark_test
@mark_test.long
def test_euler_vortex():
    main(write_output=False)
