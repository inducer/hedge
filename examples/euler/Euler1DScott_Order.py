#based on vortex.py.  this impliments 1D Euler equations in (x,t) coordinates



from __future__ import division
import numpy
import numpy.linalg as la
from pylab import *

class SmoothFields:
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, t, x_vec):

	from math import pi
	from math import sin
	u = 1
        rho = 2 + .2*numpy.sin(x_vec-t)
	p=1
	e= p/(self.gamma-1) + rho*(u**2)/2.0

        from hedge.tools import join_fields
        return join_fields(rho, e, rho*u)

    def volume_interpolant(self, t, discr):
        return self(t, discr.nodes.T)

    def boundary_interpolant(self, t, discr, tag):
        return self(t, discr.get_boundary(tag).nodes.T)

def main():
    from hedge.backends import guess_run_context
    rcon = guess_run_context()

    gamma = 1.4

    from hedge.tools import EOCRecorder, to_obj_array
    eoc_rec = EOCRecorder()
    
    from math import pi
    if rcon.is_head_rank:
        from hedge.mesh import make_uniform_1d_mesh
        #mesh = make_uniform_1d_mesh(0, 2*pi, 10, periodic=True)
	mesh = make_uniform_1d_mesh(0, 2*pi, 10, left_tag="inflow", right_tag="outflow")
    
    if rcon.is_head_rank:
        mesh_data = rcon.distribute_mesh(mesh)
    else:
        mesh_data = rcon.receive_mesh()

    #Scott: added to generate grid on which u is given
    def x_getter(x, el, t):
	return x

    order=4
    discr = rcon.make_discretization(mesh_data, order=order)

    from hedge.visualization import SiloVisualizer
    #vis = VtkVisualizer(discr, rcon, "vortex-%d" % order)
    vis = SiloVisualizer(discr, rcon)

    smoothfields = SmoothFields(gamma=gamma)
    fields = smoothfields.volume_interpolant(0, discr)
    from hedge.pde import EulerOperator
    op = EulerOperator(dimensions=1, gamma=1.4, bc=smoothfields)

    #for i, oi in enumerate(op.op_template()):
        #print i, oi

    euler_ex = op.bind(discr)

    max_eigval = [0]
    def rhs(t, q):
        ode_rhs, speed = euler_ex(t, q)
        max_eigval[0] = speed
        return ode_rhs

    x_temp = discr.interpolate_volume_function(lambda x, el: x_getter(x, el, 0))

    rhs(0, fields)
    dt = discr.dt_factor(max_eigval[0])
    dt = dt/4.0
    final_time = 10.0
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

    # timestep loop -------------------------------------------------------
    t = 0
 
    for step in range(nsteps):
        logmgr.tick()
	
	if step % 5 == 0:
	    true_fields = smoothfields.volume_interpolant(t, discr)
            plot(x_temp,fields[0]-true_fields[0])
            hold(0)
            show()

        fields = stepper(fields, t, dt, rhs)
        t += dt

        dt = discr.dt_factor(max_eigval[0])
	dt = dt/4.0

    logmgr.tick()
    logmgr.save()

    true_fields = smoothfields.volume_interpolant(t, discr)
    eoc_rec.add_data_point(order, discr.norm(fields-true_fields))
    print
    print eoc_rec.pretty_print("P.Deg.", "L2 Error")

if __name__ == "__main__":
    main()
