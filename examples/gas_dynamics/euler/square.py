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
import pycuda.autoinit




def make_squaremesh():

    def round_trip_connect(seq):
        result = []
        for i in range(len(seq)):
            result.append((i, (i+1)%len(seq)))
        return result

    def needs_refinement(vertices, area):
        x =  sum(numpy.array(v) for v in vertices)/3

        max_area_volume = 0.7e-2 + 0.03*(0.05*x[1]**2 + 0.3*min(x[0]+1,0)**2)

        max_area_corners = 1e-3 + 0.001*max(
                la.norm(x-corner)**4 for corner in obstacle_corners)

        return bool(area > min(max_area_volume, max_area_corners))

    import sys

    from meshpy.geometry import make_box
    points, facets, _ = make_box((-0.5,-0.5), (0.5,0.5))
    obstacle_corners = points[:]

    from meshpy.geometry import GeometryBuilder, Marker
    from meshpy.triangle import write_gnuplot_mesh

    profile_marker = Marker.FIRST_USER_MARKER
    builder = GeometryBuilder()
    builder.add_geometry(points=points, facets=facets,
            facet_markers=profile_marker)

    points, facets, facet_markers = make_box((-16, -22), (25, 22))
    builder.add_geometry(points=points, facets=facets,
            facet_markers=facet_markers)

    from meshpy.triangle import MeshInfo, build
    mi = MeshInfo()
    builder.set(mi)
    mi.set_holes([(0,0)])

    mesh = build(mi, refinement_func=needs_refinement,
            allow_boundary_steiner=True,
            generate_faces=True)

    print "%d elements" % len(mesh.elements)

    write_gnuplot_mesh("mesh.dat", mesh)

    fvi2fm = mesh.face_vertex_indices_to_face_marker

    face_marker_to_tag = {
            profile_marker: "noslip",
            Marker.MINUS_X: "inflow",
            Marker.PLUS_X: "outflow",
            Marker.MINUS_Y: "inflow",
            Marker.PLUS_Y: "inflow"
            }

    def bdry_tagger(fvi, el, fn, all_v):
        face_marker = fvi2fm[fvi]
        return [face_marker_to_tag[face_marker]]

    from hedge.mesh import make_conformal_mesh
    return make_conformal_mesh(
            mesh.points, mesh.elements, bdry_tagger,
            )




def main():
    from hedge.backends import guess_run_context
    is_cpu = False
    if is_cpu == False:
        rcon = guess_run_context(["cuda"])
    else:
        rcon = guess_run_context()

    if rcon.is_head_rank:
        if True:
            mesh = make_squaremesh()
        else:
            from hedge.mesh import make_rect_mesh
            mesh = make_rect_mesh(
                   boundary_tagger=lambda fvi, el, fn, all_v: ["inflow"],
                   max_area=0.1)

        mesh_data = rcon.distribute_mesh(mesh)
    else:
        mesh_data = rcon.receive_mesh()

    from pytools import add_python_path_relative_to_script
    add_python_path_relative_to_script("..")

    for order in [3]:
        from gas_dynamics_initials import UniformMachFlow
        square = UniformMachFlow(reynolds=float("inf"))

        from hedge.models.gas_dynamics import GasDynamicsOperator
        op = GasDynamicsOperator(dimensions=2,
                gamma=square.gamma, mu=square.mu,
                prandtl=square.prandtl, spec_gas_const=square.spec_gas_const,
                bc_inflow=square, bc_outflow=square, bc_noslip=square,
                inflow_tag="inflow", outflow_tag="outflow", noslip_tag="noslip",
                euler=True)

        if is_cpu == False:
            discr = rcon.make_discretization(mesh_data, order=order,
                        debug=["cuda_no_plan",
                            #"cuda_dump_kernels",
                            #"dump_dataflow_graph",
                            #"dump_optemplate_stages",
                            #"dump_dataflow_graph",
                            #"print_op_code"
                            #"cuda_no_plan_el_local"
                            ],
                        default_scalar_type=numpy.float64,
                        tune_for=op.op_template(),
                        init_cuda=False)
        else:
            discr = rcon.make_discretization(mesh_data, order=order,
                        default_scalar_type=numpy.float32,
                        tune_for=op.op_template()
                        )

        from hedge.visualization import SiloVisualizer, VtkVisualizer
        #vis = VtkVisualizer(discr, rcon, "shearflow-%d" % order)
        vis = SiloVisualizer(discr, rcon)

        if is_cpu == False:
            from hedge.backends.cuda.tools import RK4TimeStepper
        else:
            from hedge.timestep import RK4TimeStepper
        stepper = RK4TimeStepper()

        # diagnostics setup ---------------------------------------------------
        from pytools.log import LogManager, add_general_quantities, \
                add_simulation_quantities, add_run_info

        if is_cpu == False:
            logmgr = LogManager("cns-square-gpu-dp-%d.dat" % order, "w", rcon.communicator)
        else:
            logmgr = LogManager("cns-square-cpu-dp-%d.dat" % order, "w", rcon.communicator)
        add_run_info(logmgr)
        add_general_quantities(logmgr)
        discr.add_instrumentation(logmgr)
        stepper.add_instrumentation(logmgr)


        from pytools.log import LogQuantity
        class ChangeSinceLastStep(LogQuantity):
            """Records the change of a variable between a time step and the previous
               one"""

            def __init__(self, name="change"):
                LogQuantity.__init__(self, name, "1", "Change since last time step")

                self.old_fields = 0

            def __call__(self):
                result = discr.norm(fields - self.old_fields)
                self.old_fields = fields
                return result

        #logmgr.add_quantity(ChangeSinceLastStep())

        fields = square.volume_interpolant(0, discr)

        navierstokes_ex = op.bind(discr)

        max_eigval = [0]
        def rhs(t, q):
            ode_rhs, speed = navierstokes_ex(t, q)
            max_eigval[0] = speed
            return ode_rhs
        rhs(0, fields)

        dt = discr.dt_factor(max_eigval[0], order=2)
        final_time = 200
        nsteps = int(final_time/dt)+1
        #nsteps = 500
        #final_time = nsteps * dt
        #dt = final_time/nsteps

        if rcon.is_head_rank:
            print "---------------------------------------------"
            print "order %d" % order
            print "---------------------------------------------"
            print "dt", dt
            print "nsteps", nsteps
            print "#elements=", len(mesh.elements)

        add_simulation_quantities(logmgr, dt)
        logmgr.add_watches(["step.max", "t_sim.max", "t_step.max"])


        # timestep loop -------------------------------------------------------
        t = 0

        try:
            for step in range(nsteps):
                logmgr.tick()

                #if (step % 10000 == 0): #and step < 950000) or (step % 500 == 0 and step > 950000):
                #if False:
                if step % 1000 == 0:
                    visf = vis.make_file("square-%d-%06d" % (order, step))

                    #rhs_fields = rhs(t, fields)

                    from pylo import DB_VARTYPE_VECTOR
                    from hedge.discretization import ones_on_boundary
                    vis.add_data(visf,
                            [
                                ("rho", discr.convert_volume(op.rho(fields), kind="numpy")),
                                ("e", discr.convert_volume(op.e(fields), kind="numpy")),
                                ("rho_u", discr.convert_volume(op.rho_u(fields), kind="numpy")),
                                ("u", discr.convert_volume(op.u(fields), kind="numpy")),

                                #("rhs_rho", discr.convert_volume(op.rho(rhs_fields), kind="numpy")),
                                #("rhs_e", discr.convert_volume(op.e(rhs_fields), kind="numpy")),
                                #("rhs_rho_u", discr.convert_volume(op.rho_u(rhs_fields), kind="numpy")),
                                ],
                            expressions=[
                                ("p", "(0.4)*(e- 0.5*(rho_u*u))"),
                                ],
                            time=t, step=step
                            )
                    visf.close()

                fields = stepper(fields, t, dt, rhs)
                t += dt

                dt = discr.dt_factor(max_eigval[0], order=1)

            logmgr.tick()
            logmgr.set_constant("is_cpu", is_cpu)
        finally:
            logmgr.save()
            discr.close()

if __name__ == "__main__":
    main()
