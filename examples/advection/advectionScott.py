#Scott's version of advection...to test and modify

#goal: understand 1D portion of code well, try certain features.
#periodic vs inflow/outflow, different initial conditions


#could try to code/impliment things like filter and slope-limiter


from __future__ import division
import numpy
import numpy.linalg as la
from pylab import *
import pdb


def main() :
    from hedge.timestep import RK4TimeStepper
    from hedge.tools import mem_checkpoint
    from math import sin, cos, pi, sqrt
    from math import floor

    from hedge.backends import guess_run_context
    rcon = guess_run_context(disable=set(["cuda"]))

    #initial condition (and can provide BC at inflow)
    def f(x):
        return cos(pi*x)

    def u_analytic(x, el, t):
        return f((-numpy.dot(v, x)/norm_v+t*norm_v))

    #Scott: added to generate grid on which u is given
    def x_getter(x, el, t):
	return x

    def boundary_tagger(vertices, el, face_nr, all_v):
        if numpy.dot(el.face_normals[face_nr], v) < 0:
            return ["inflow"]
        else:
            return ["outflow"]

    dim = 1

    v = numpy.array([1])
    if rcon.is_head_rank:
        from hedge.mesh import make_uniform_1d_mesh
        #mesh = make_uniform_1d_mesh(0, 2, 10, periodic=True)
        mesh = make_uniform_1d_mesh(0, 2, 10, left_tag="inflow", right_tag="outflow")

    norm_v = la.norm(v)

    if rcon.is_head_rank:
        mesh_data = rcon.distribute_mesh(mesh)
    else:
        mesh_data = rcon.receive_mesh()

    discr = rcon.make_discretization(mesh_data, order=4)
    vis_discr = discr

    from hedge.visualization import VtkVisualizer, SiloVisualizer
    vis = VtkVisualizer(vis_discr, rcon, "fld")
    #vis = SiloVisualizer(vis_discr, rcon)

    # operator setup ----------------------------------------------------------
    from hedge.data import \
            ConstantGivenFunction, \
            TimeConstantGivenFunction, \
            TimeDependentGivenFunction
    from hedge.pde import StrongAdvectionOperator, WeakAdvectionOperator
    op = WeakAdvectionOperator(v, 
            inflow_u=TimeDependentGivenFunction(u_analytic),
            flux_type="upwind")

    u = discr.interpolate_volume_function(lambda x, el: u_analytic(x, el, 0))

    # timestep setup ----------------------------------------------------------
    stepper = RK4TimeStepper()

    dt = discr.dt_factor(op.max_eigenvalue())
    dt = dt/2.0
    nsteps = int(1/dt)

    if rcon.is_head_rank:
        print "%d elements, dt=%g, nsteps=%d" % (
                len(discr.mesh.elements),
                dt,
                nsteps)

    # diagnostics setup -------------------------------------------------------
    from pytools.log import LogManager, \
            add_general_quantities, \
            add_simulation_quantities, \
            add_run_info

    logmgr = LogManager("advection.dat", "w", rcon.communicator)
    add_run_info(logmgr)
    add_general_quantities(logmgr)
    add_simulation_quantities(logmgr, dt)
    discr.add_instrumentation(logmgr)

    stepper.add_instrumentation(logmgr)

    from hedge.log import Integral, LpNorm
    u_getter = lambda: u
    logmgr.add_quantity(Integral(u_getter, discr, name="int_u"))
    logmgr.add_quantity(LpNorm(u_getter, discr, p=1, name="l1_u"))
    logmgr.add_quantity(LpNorm(u_getter, discr, name="l2_u"))

    logmgr.add_watches(["step.max", "t_sim.max", "l2_u", "t_step.max"])

    x_temp = discr.interpolate_volume_function(lambda x, el: x_getter(x, el, 0))
    plot(x_temp,u)
    show()
    #savefig('output1')
    #pdb.set_trace()


    # timestep loop -----------------------------------------------------------
    rhs = op.bind(discr)
    for step in xrange(nsteps):
        logmgr.tick()

        t = step*dt

        if step % 5 == 0:
            visf = vis.make_file("fld-%04d" % step)
            vis.add_data(visf, [ ("u", u), ], 
                        time=t, 
                        step=step
                        )
            visf.close()
            plot(x_temp,u)
            hold(0)
            show()
            #scott added to test l2 error
	    #u_true = discr.interpolate_volume_function(lambda x, el: u_analytic(x, el, step*dt))    
            #error = u-u_true
            #error_l2 = discr.norm(error)
            #print error_l2

        #the second slot is time the solution is at so t+dt solution is returned
        u = stepper(u, t, dt, rhs)

    vis.close()

    logmgr.tick()
    logmgr.save()


if __name__ == "__main__":
    main()
