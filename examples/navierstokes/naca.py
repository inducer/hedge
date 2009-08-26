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




class Naca:
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, t, x_vec):
        # JSH/TW Nodal DG Methods, p.326 

        rho = numpy.empty_like(x_vec[0])
        rho.fill(1)
        u = 100
        v = 0
        p = 1
        rho_u = rho * u
        rho_v = rho * v
        e = p / (self.gamma - 1) + rho / 2 *(u ** 2 + v ** 2)

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



def nacamesh():
    def airfoil_from_generator():
        import meshpy.naca as naca
        return naca.generate_naca(naca_digits="0012", number_of_points=50,
                sharp_trailing_edge=True, uniform_distribution=False,
                verbose=True)

    def round_trip_connect(seq):
        result = []
        for i in range(len(seq)):
            result.append((i, (i+1)%len(seq)))
        return result

    def needs_refinement(vertices, area):
        barycenter =  sum(numpy.array(v) for v in vertices)/3

        pt_back = numpy.array([1,0])

        max_area_front = 0.002*la.norm(barycenter) + 1e-5
        max_area_back = 0.02*la.norm(barycenter-pt_back) + 1e-3
        return bool(area > min(max_area_front, max_area_back))

    import sys
    points = airfoil_from_generator()

    from meshpy.geometry import GeometryBuilder
    from meshpy.triangle import write_gnuplot_mesh

    builder = GeometryBuilder()
    builder.add_geometry(points=points,
            facets=round_trip_connect(points))
    builder.wrap_in_box(2)

    from meshpy.triangle import MeshInfo, build
    mi = MeshInfo()
    builder.set(mi)
    mi.set_holes([builder.center()])
    mesh = build(mi, refinement_func=needs_refinement)

    print "%d elements" % len(mesh.elements)

    #print mesh.elements[0]
    #raw_input()

    write_gnuplot_mesh("mesh.dat", mesh)

    return mesh




def main():
    from hedge.backends import guess_run_context
    rcon = guess_run_context(
    ["cuda"]
    )

    gamma = 1.4
    prandtl = 0.72
    spec_gas_const = 287.1

    from hedge.tools import EOCRecorder, to_obj_array
    eoc_rec = EOCRecorder()
    
    if rcon.is_head_rank:
        mesh = nacamesh()
        mesh_data = rcon.distribute_mesh(mesh)
    else:
        mesh_data = rcon.receive_mesh()

    #print mesh.elements[0]
    #raw_input()

    for order in [3]:
        discr = rcon.make_discretization(mesh_data, order=order,
			debug=["cuda_no_plan",
			#"print_op_code"
			],
			default_scalar_type=numpy.float64)

        from hedge.visualization import SiloVisualizer, VtkVisualizer
        #vis = VtkVisualizer(discr, rcon, "shearflow-%d" % order)
        vis = SiloVisualizer(discr, rcon)

        naca = Naca(gamma=gamma)
        fields = naca.volume_interpolant(0, discr)

        from hedge.pde import NavierStokesWithHeatOperator
        op = NavierStokesWithHeatOperator(dimensions=2, gamma=gamma,
                prandtl=prandtl, spec_gas_const=spec_gas_const, bc=naca)

        navierstokes_ex = op.bind(discr)

        max_eigval = [0]
        def rhs(t, q):
            ode_rhs, speed = navierstokes_ex(t, q)
            max_eigval[0] = speed
            return ode_rhs
        rhs(0, fields)

        dt = discr.dt_factor(max_eigval[0], order=2)
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
                visf = vis.make_file("naca-%d-%04d" % (order, step))

                #true_fields = naca.volume_interpolant(t, discr)

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
            t += dt

            dt = discr.dt_factor(max_eigval[0], order=2)

        logmgr.tick()
        logmgr.save()

        true_fields = naca.volume_interpolant(t, discr)
        eoc_rec.add_data_point(order, discr.norm(fields-true_fields))
        print
        print eoc_rec.pretty_print("P.Deg.", "L2 Error")

if __name__ == "__main__":
    main()
