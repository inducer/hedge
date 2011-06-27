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




def main(write_output=True):
    from pytools import add_python_path_relative_to_script
    add_python_path_relative_to_script("..")

    from hedge.backends import guess_run_context
    rcon = guess_run_context()

    from hedge.tools import EOCRecorder
    eoc_rec = EOCRecorder()


    if rcon.is_head_rank:
        from hedge.mesh.generator import \
                make_rect_mesh, \
                make_centered_regular_rect_mesh

        refine = 4
        mesh = make_centered_regular_rect_mesh((0,-5), (10,5), n=(9,9),
                post_refine_factor=refine)
        mesh_data = rcon.distribute_mesh(mesh)
    else:
        mesh_data = rcon.receive_mesh()

    # a second mesh to regrid to
    if rcon.is_head_rank:
        from hedge.mesh.generator import \
                make_rect_mesh, \
                make_centered_regular_rect_mesh

        refine = 4
        mesh2 = make_centered_regular_rect_mesh((0,-5), (10,5), n=(8,8),
                post_refine_factor=refine)
        mesh_data2 = rcon.distribute_mesh(mesh2)
    else:
        mesh_data2 = rcon.receive_mesh()



    for order in [3,4]:
        discr = rcon.make_discretization(mesh_data, order=order,
                        default_scalar_type=numpy.float64,
                        quad_min_degrees={
                            "gasdyn_vol": 3*order,
                            "gasdyn_face": 3*order,
                            })

        discr2 = rcon.make_discretization(mesh_data2, order=order,
                        default_scalar_type=numpy.float64,
                        quad_min_degrees={
                            "gasdyn_vol": 3*order,
                            "gasdyn_face": 3*order,
                            })


        from hedge.visualization import SiloVisualizer, VtkVisualizer
        vis = VtkVisualizer(discr, rcon, "vortex-%d" % order)
        #vis = SiloVisualizer(discr, rcon)

        from gas_dynamics_initials import Vortex
        vortex = Vortex()
        fields = vortex.volume_interpolant(0, discr)

        from hedge.models.gas_dynamics import GasDynamicsOperator
        from hedge.mesh import TAG_ALL

        op = GasDynamicsOperator(dimensions=2, gamma=vortex.gamma, mu=vortex.mu,
                prandtl=vortex.prandtl, spec_gas_const=vortex.spec_gas_const,
                bc_inflow=vortex, bc_outflow=vortex, bc_noslip=vortex,
                inflow_tag=TAG_ALL, source=None)

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
            print "#elements for mesh 1 =", len(mesh.elements)
            print "#elements for mesh 2 =", len(mesh2.elements)


        # limiter ------------------------------------------------------------
        from hedge.models.gas_dynamics import SlopeLimiter1NEuler
        limiter = SlopeLimiter1NEuler(discr, vortex.gamma, 2, op)

        from hedge.timestep import SSPRK3TimeStepper
        #stepper = SSPRK3TimeStepper(limiter=limiter)
        stepper = SSPRK3TimeStepper()

        #from hedge.timestep import RK4TimeStepper
        #stepper = RK4TimeStepper()

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
        try:
            final_time = 0.2
            from hedge.timestep import times_and_steps
            step_it = times_and_steps(
                    final_time=final_time, logmgr=logmgr,
                    max_dt_getter=lambda t: op.estimate_timestep(discr,
                        stepper=stepper, t=t, max_eigenvalue=max_eigval[0]))

            for step, t, dt in step_it:
                if step % 10 == 0 and write_output:
                #if False:
                    visf = vis.make_file("vortex-%d-%04d" % (order, step))

                    #true_fields = vortex.volume_interpolant(t, discr)

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
                #fields = limiter(fields)

                #regrid to discr2 at some arbitrary time
                if step == 21:

                    #get interpolated fields
                    fields = discr.get_regrid_values(fields, discr2, dtype=None, use_btree=True, thresh=1e-8)
                    #get new stepper (old one has reference to discr
                    stepper = SSPRK3TimeStepper()
                    #new bind
                    euler_ex = op.bind(discr2)
                    #new rhs
                    max_eigval = [0]
                    def rhs(t, q):
                        ode_rhs, speed = euler_ex(t, q)
                        max_eigval[0] = speed
                        return ode_rhs
                    rhs(t+dt, fields)
                    #add logmanager
                    #discr2.add_instrumentation(logmgr)
                    #new step_it
                    step_it = times_and_steps(
                        final_time=final_time, logmgr=logmgr,
                        max_dt_getter=lambda t: op.estimate_timestep(discr2,
                            stepper=stepper, t=t, max_eigenvalue=max_eigval[0]))

                    #new visualization
                    vis.close()
                    vis = VtkVisualizer(discr2, rcon, "vortexNewGrid-%d" % order)
                    discr=discr2



                assert not numpy.isnan(numpy.sum(fields[0]))

            true_fields = vortex.volume_interpolant(final_time, discr)
            l2_error = discr.norm(fields-true_fields)
            l2_error_rho = discr.norm(op.rho(fields)-op.rho(true_fields))
            l2_error_e = discr.norm(op.e(fields)-op.e(true_fields))
            l2_error_rhou = discr.norm(op.rho_u(fields)-op.rho_u(true_fields))
            l2_error_u = discr.norm(op.u(fields)-op.u(true_fields))

            eoc_rec.add_data_point(order, l2_error)
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
    # assert eoc_rec.estimate_order_of_convergence()[0,1] > 6




if __name__ == "__main__":
    main()
