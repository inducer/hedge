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




from __future__ import division
import numpy
import numpy.linalg as la


def main():
    from hedge.timestep import RK4TimeStepper
    from hedge.tools import mem_checkpoint
    from math import sin, cos, pi, sqrt, tanh
    from math import floor

    from hedge.backends import guess_run_context
    rcon = guess_run_context(disable=set(["cuda"]))

    dim = 2

    if rcon.is_head_rank:
        #from hedge.mesh import make_disk_mesh
        #mesh = make_disk_mesh()
	from hedge.mesh import make_rect_mesh
	mesh = make_rect_mesh(a=(-1,-1),b=(1,1),max_area=0.008)

    if rcon.is_head_rank:
        mesh_data = rcon.distribute_mesh(mesh)
    else:
        mesh_data = rcon.receive_mesh()
 
    discr = rcon.make_discretization(mesh_data, order=4)
    vis_discr = discr

    
    if False:
        from hedge.tools import join_fields
        v = join_fields([discr.interpolate_volume_function(lambda x,el: x[i])
                                           for i in range(discr.dimensions)])
    
    if True:
        class VField:
            shape = (2,)

            def __call__(self, pt, el):
                x, y = pt
                #fac = (1-x**2)*(1-y**2) # Correction-Factor to keep Divergence as small as possible
                fac = 1
                return (-y*fac, x*fac)

        v = discr.interpolate_volume_function(VField())

    if False:
	rho = 1./30.
	sigma = 0.05

        class VField:
            shape = (2,)

            def __call__(self, pt, el):
                if x[1] <= 0.5:
                    vx = tanh((x[1]-0.25)/rho)
                else:
                    vx = tanh((0.75-x[1])/rho)

        v = join_fields(discr.interpolate_volume_function(v_x),
                        discr.interpolate_volume_function(lambda x,el: 
                                                   sigma*sin(2*pi*x[0])))

    from hedge.visualization import VtkVisualizer, SiloVisualizer
    vis = VtkVisualizer(vis_discr, rcon, "fld")
    #vis = SiloVisualizer(vis_discr, rcon)

    # operator setup ----------------------------------------------------------
    from hedge.data import \
            ConstantGivenFunction, \
            TimeConstantGivenFunction, \
            TimeDependentGivenFunction
    from hedge.pde import SpaceDependentWeakAdvectionOperator
    op = SpaceDependentWeakAdvectionOperator(dim, v, flux_type="upwind")

    # Initialize the domain at t = 0:
    if False:
        def initial(pt, el):
            from math import exp
            x = (pt-numpy.array([0.3, 0.7]))*8
            return exp(-numpy.dot(x, x))
    if True:
        def initial(pt, el):
            #from math import abs
            x, y = pt
            if abs(x) < 0.5 and abs(y) < 0.2:
                return 2
            else:
                return 1

    u = discr.interpolate_volume_function(initial)
    
    # timestep setup ----------------------------------------------------------
    stepper = RK4TimeStepper()

    dt = discr.dt_factor(op.max_eigenvalue()) * 50
    nsteps = int(700/dt)

    if rcon.is_head_rank:
        print "%d elements, dt=%g, nsteps=%d" % (
                len(discr.mesh.elements),
                dt,
                nsteps)

    # filter setup-------------------------------------------------------------
    from hedge.discretization import ExponentialFilterResponseFunction, Filter
    if False:
        def mod_resp_f(mid, ldis):
            if sum(mid) == 2:
                return 0.9
            elif sum(mid) == 3:
                return 0.5
            elif sum(mid) == 4:
                return 0.1
            else:
                return 1

    if False:
        efrf = ExponentialFilterResponseFunction(min_amplification=0.9, order=4)

        def mod_resp_f(mid, ldis):
            return efrf(mid, ldis)
    
    antialiasing = Filter(discr,ExponentialFilterResponseFunction(min_amplification=0.9, order=4))
    #antialiasing = Filter(discr, mod_resp_f)


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

    # timestep loop -----------------------------------------------------------
    #import rpdb2; rpdb2.start_embedded_debugger_interactive_password()
    rhs = op.bind(discr)
    for step in xrange(nsteps):
        logmgr.tick()

        t = step*dt

        if step % 10 == 0:
            visf = vis.make_file("fld-%04d" % step)
            vis.add_data(visf, [ ("u", u), ("v", v)], 
                        time=t, 
                        step=step
                        )
            visf.close()


        u = stepper(u, t, dt, rhs)
        # Use Filter:
        u = antialiasing(u)

    vis.close()

    logmgr.tick()
    logmgr.save()


if __name__ == "__main__":
    main()
