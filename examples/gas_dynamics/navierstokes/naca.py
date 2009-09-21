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




def make_nacamesh():
    def airfoil_from_generator():
        import meshpy.naca as naca
        # parameters for the airfoil generator
        naca_digits = "0012"
        number_of_points = 30
        sharp_trailing_edge = True
        uniform_distribution = False
        verbose = True
        return naca.generate_naca(naca_digits=naca_digits,
                number_of_points=number_of_points,
                sharp_trailing_edge=sharp_trailing_edge,
                uniform_distribution=uniform_distribution,
                verbose=verbose)

    def round_trip_connect(seq):
        result = []
        for i in range(len(seq)):
            result.append((i, (i+1)%len(seq)))
        return result

    def needs_refinement(vertices, area):
        barycenter =  sum(numpy.array(v) for v in vertices)/3

        pt_back = numpy.array([1,0])

        max_area_front = 0.02*la.norm(barycenter)**2 + 1e-4
        max_area_back = 0.06*la.norm(barycenter-pt_back)**2 + 1e-4
        return bool(area > min(max_area_front, max_area_back))

    import sys
    points = airfoil_from_generator()

    from meshpy.geometry import GeometryBuilder, Marker
    from meshpy.triangle import write_gnuplot_mesh

    profile_marker = Marker.FIRST_USER_MARKER
    builder = GeometryBuilder()
    builder.add_geometry(points=points,
            facets=round_trip_connect(points),
            facet_markers=profile_marker)
    builder.wrap_in_box(8, (10, 8))

    from meshpy.triangle import MeshInfo, build
    mi = MeshInfo()
    builder.set(mi)
    mi.set_holes([builder.center()])

    mesh = build(mi, refinement_func=needs_refinement,
            allow_boundary_steiner=False,
            generate_faces=True)

    write_gnuplot_mesh("mesh.dat", mesh)

    print "%d elements" % len(mesh.elements)

    fvi2fm = mesh.face_vertex_indices_to_face_marker

    face_marker_to_tag = {
            profile_marker: "noslip",
            Marker.MINUS_X: "inflow",
            Marker.PLUS_X: "outflow",
            Marker.MINUS_Y: "minus_y",
            Marker.PLUS_Y: "plus_y"
            }

    def bdry_tagger(fvi, el, fn, all_v):
        face_marker = fvi2fm[fvi]
        return [face_marker_to_tag[face_marker]]

    from hedge.mesh import make_conformal_mesh
    return make_conformal_mesh(
            mesh.points, mesh.elements, bdry_tagger,
            periodicity=[None, ("minus_y", "plus_y")])




def main():
    from hedge.backends import guess_run_context
    rcon = guess_run_context(
    ["cuda"]
    )

    if rcon.is_head_rank:
        mesh = make_nacamesh()
        mesh_data = rcon.distribute_mesh(mesh)
    else:
        mesh_data = rcon.receive_mesh()

    for order in [3]:
        discr = rcon.make_discretization(mesh_data, order=order,
			debug=["cuda_no_plan"],
			default_scalar_type=numpy.float64)

        from hedge.visualization import SiloVisualizer, VtkVisualizer
        #vis = VtkVisualizer(discr, rcon, "shearflow-%d" % order)
        vis = SiloVisualizer(discr, rcon)

        from hedge.models.gas_dynamics_initials import UniformMachFlow
        naca = UniformMachFlow()
        fields = naca.volume_interpolant(0, discr)
        gamma, mu, prandtl, spec_gas_const = naca.properties()

        from hedge.models.gasdynamics import GasDynamicsOperator
        op = GasDynamicsOperator(dimensions=2, discr=discr,
                gamma=gamma, prandtl=prandtl, spec_gas_const=spec_gas_const, mu=0,
                bc_inflow=naca, bc_outflow=naca, bc_noslip=naca,
                inflow_tag="inflow", outflow_tag="outflow", noslip_tag="noslip",
                euler=False)

        navierstokes_ex = op.bind(discr)

        max_eigval = [0]
        def rhs(t, q):
            ode_rhs, speed = navierstokes_ex(t, q)
            max_eigval[0] = speed
            return ode_rhs
        rhs(0, fields)

        dt = discr.dt_factor(max_eigval[0], order=2)
        final_time = 20
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

            if step % 1000 == 0:
            #if False:
                visf = vis.make_file("naca-%d-%04d" % (order, step))

                #true_fields = naca.volume_interpolant(t, discr)

                #rhs_fields = rhs(t, fields)

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

                            #("rhs_rho", discr.convert_volume(op.rho(rhs_fields), kind="numpy")),
                            #("rhs_e", discr.convert_volume(op.e(rhs_fields), kind="numpy")),
                            #("rhs_rho_u", discr.convert_volume(op.rho_u(rhs_fields), kind="numpy")),
                            ],
                        expressions=[
                            #("diff_rho", "rho-true_rho"),
                            #("diff_e", "e-true_e"),
                            #("diff_rho_u", "rho_u-true_rho_u", DB_VARTYPE_VECTOR),

                            ("p", "(0.4)*(e- 0.5*(rho_u*u))"),
                            ],
                        time=t, step=step
                        )
                visf.close()

            old_fields = fields

            fields = stepper(fields, t, dt, rhs)
            t += dt

            dt = discr.dt_factor(max_eigval[0], order=2)

        logmgr.tick()
        logmgr.save()

        true_fields = naca.volume_interpolant(t, discr)
        eoc_rec.add_data_point(order, discr.norm(fields-old_fields))
        print
        print eoc_rec.pretty_print("P.Deg.", "Residual")

if __name__ == "__main__":
    main()
