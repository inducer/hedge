from __future__ import division
import numpy
import numpy.linalg as la




class Sod:
    def __init__(self, gamma):
        self.gamma = gamma
        self.prandtl = 0.72

    def __call__(self, t, x_vec):

        from hedge.tools import heaviside
        from hedge.tools import heaviside_a

        x_rel = x_vec[0]
        y_rel = x_vec[1]

        from math import pi
        r = numpy.sqrt(x_rel**2+y_rel**2)
        r_shift=r-3.0
        u = 0.0
        v = 0.0
        from numpy import sign
        rho = heaviside(-r_shift)+.125*heaviside_a(r_shift,1.0)
        e = (1.0/(self.gamma-1.0))*(heaviside(-r_shift)+.1*heaviside_a(r_shift,1.0))
        p = (self.gamma-1.0)*e

        from hedge.tools import join_fields
        return join_fields(rho, e, rho*u, rho*v)


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
    rcon = guess_run_context()

    from hedge.tools import to_obj_array

    if rcon.is_head_rank:
        from hedge.mesh.generator import make_rect_mesh
        mesh = make_rect_mesh((-5,-5), (5,5), max_area=0.01)
        mesh_data = rcon.distribute_mesh(mesh)
    else:
        mesh_data = rcon.receive_mesh()

    for order in [1]:
        discr = rcon.make_discretization(mesh_data, order=order,
                        default_scalar_type=numpy.float64)

        from hedge.visualization import SiloVisualizer, VtkVisualizer
        vis = VtkVisualizer(discr, rcon, "Sod2D-%d" % order)
        #vis = SiloVisualizer(discr, rcon)

        sod_field = Sod(gamma=1.4)
        fields = sod_field.volume_interpolant(0, discr)

        from hedge.models.gas_dynamics import GasDynamicsOperator
        from hedge.mesh import TAG_ALL
        op = GasDynamicsOperator(dimensions=2, gamma=sod_field.gamma, mu=0.0,
                prandtl=sod_field.prandtl,
                bc_inflow=sod_field,
                bc_outflow=sod_field,
                bc_noslip=sod_field,
                inflow_tag=TAG_ALL,
                source=None)

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

        # limiter setup ------------------------------------------------------------
        from hedge.models.gas_dynamics import SlopeLimiter1NEuler
        limiter =  SlopeLimiter1NEuler(discr, sod_field.gamma, 2, op)

        # integrator setup---------------------------------------------------------
        from hedge.timestep import SSPRK3TimeStepper, RK4TimeStepper
        stepper = SSPRK3TimeStepper(limiter=limiter)
        #stepper = SSPRK3TimeStepper()
        #stepper = RK4TimeStepper()

        # diagnostics setup ---------------------------------------------------
        from pytools.log import LogManager, add_general_quantities, \
                add_simulation_quantities, add_run_info

        logmgr = LogManager("euler-%d.dat" % order, "w", rcon.communicator)
        add_run_info(logmgr)
        add_general_quantities(logmgr)
        add_simulation_quantities(logmgr)
        discr.add_instrumentation(logmgr)
        stepper.add_instrumentation(logmgr)

        logmgr.add_watches(["step.max", "t_sim.max", "t_step.max"])

        # filter setup-------------------------------------------------------------
        from hedge.discretization import Filter, ExponentialFilterResponseFunction
        mode_filter = Filter(discr,
                ExponentialFilterResponseFunction(min_amplification=0.9,order=4))

        # timestep loop -------------------------------------------------------
        try:
            from hedge.timestep import times_and_steps
            step_it = times_and_steps(
                    final_time=1.0, logmgr=logmgr,
                    max_dt_getter=lambda t: op.estimate_timestep(discr,
                        stepper=stepper, t=t, max_eigenvalue=max_eigval[0]))

            for step, t, dt in step_it:
                if step % 5 == 0:
                #if False:
                    visf = vis.make_file("vortex-%d-%04d" % (order, step))

                    #true_fields = vortex.volume_interpolant(t, discr)

                    #from pyvisfile.silo import DB_VARTYPE_VECTOR
                    vis.add_data(visf,
                            [
                                ("rho", discr.convert_volume(op.rho(fields), kind="numpy")),
                                ("e", discr.convert_volume(op.e(fields), kind="numpy")),
                                ("rho_u", discr.convert_volume(op.rho_u(fields), kind="numpy")),
                                ("u", discr.convert_volume(op.u(fields), kind="numpy")),

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
                # fields = limiter(fields)
                # fields = mode_filter(fields)

                assert not numpy.isnan(numpy.sum(fields[0]))
        finally:
            vis.close()
            logmgr.close()
            discr.close()

        # not solution, just to check against when making code changes
        true_fields = sod_field.volume_interpolant(t, discr)
        print discr.norm(fields-true_fields)

if __name__ == "__main__":
    main()
