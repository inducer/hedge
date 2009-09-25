from __future__ import division
import numpy
import numpy.linalg as la




class Sod:
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, t, x_vec):

        x_rel = x_vec[0]
        y_rel = x_vec[1]

        from math import pi
        r = numpy.sqrt(x_rel**2+y_rel**2)
        r_shift=r-3.0
        u = 0.0
        v = 0.0
        from numpy import sign
        rho = sign(-r_shift)*(1+sign(-r_shift))/2.0+.125*(1.0-sign(-r_shift)*(1+sign(-r_shift))/2.0)
        e = (1.0/(self.gamma-1.0))*(sign(-r_shift)*(1+sign(-r_shift))/2.0+.1*(1-sign(-r_shift)*(1+sign(-r_shift))/2.0))
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

    gamma = 1.4

    from hedge.tools import EOCRecorder, to_obj_array
    eoc_rec = EOCRecorder()
    
    if rcon.is_head_rank:
        from hedge.mesh import make_rect_mesh
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

        sodfield = Sod(gamma=gamma)
        fields = sodfield.volume_interpolant(0, discr)

        from hedge.models.gasdynamics import GasDynamicsOperator
        from hedge.mesh import TAG_ALL
        op = GasDynamicsOperator(dimensions=2, discr=discr, gamma=1.4, 
                prandtl=0.72, spec_gas_const=287.1, inflow_tag=TAG_ALL,
                bc_inflow=sodfield,
                bc_outflow=sodfield,bc_noslip=sodfield,
                euler=True,source=None)

        euler_ex = op.bind(discr)

        max_eigval = [0]
        def rhs(t, q):
            ode_rhs, speed = euler_ex(t, q)
            max_eigval[0] = speed
            return ode_rhs
        rhs(0, fields)

        dt = discr.dt_factor(max_eigval[0])
        final_time = 2.5
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

        logmgr = LogManager("euler-%d.dat" % order, "w", rcon.communicator)
        add_run_info(logmgr)
        add_general_quantities(logmgr)
        add_simulation_quantities(logmgr, dt)
        discr.add_instrumentation(logmgr)
        stepper.add_instrumentation(logmgr)

        logmgr.add_watches(["step.max", "t_sim.max", "t_step.max"])


        # limiter setup-------------------------------------------------------------
        from hedge.models.gasdynamics import SlopeLimiter1NEuler
        limiter =  SlopeLimiter1NEuler(discr,gamma, 2, op)


        # filter setup-------------------------------------------------------------
        from hedge.discretization import Filter, ExponentialFilterResponseFunction
        antialiasing = Filter(discr,
                ExponentialFilterResponseFunction(min_amplification=0.9,order=4))


        # timestep loop -------------------------------------------------------
        t = 0

        from numpy import shape

        #limit IC...appears to result in a problem (no idea why)
        #fields = limiter(fields)

        for step in range(nsteps):
            logmgr.tick()

            if step % 5 == 0:
            #if False:
                visf = vis.make_file("vortex-%d-%04d" % (order, step))

                #true_fields = vortex.volume_interpolant(t, discr)

                from pylo import DB_VARTYPE_VECTOR
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
            #apply slope limiter
            fields = limiter(fields)
            #fields = antialiasing(fields)
            t += dt

            dt = discr.dt_factor(max_eigval[0])
            if(numpy.isnan(numpy.sum(fields[0]))==True):
                print 'Solution is blowing up'

        logmgr.tick()
        logmgr.save()

        #not solution, just to check against when making code changes
        true_fields = sodfield.volume_interpolant(t, discr)
        eoc_rec.add_data_point(order, discr.norm(fields-true_fields))
        print
        print eoc_rec.pretty_print("P.Deg.", "L2 Error")

if __name__ == "__main__":
    main()
