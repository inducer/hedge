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


def make_boxmesh():
    from meshpy.tet import MeshInfo, build
    from meshpy.geometry import GeometryBuilder, Marker, make_box

    geob = GeometryBuilder()

    box_marker = Marker.FIRST_USER_MARKER
    extent_small = 0.1*numpy.ones(3, dtype=numpy.float64)

    geob.add_geometry(*make_box(-extent_small, extent_small))

    # make small "separator box" for region attribute

    geob.add_geometry(
            *make_box(
                -extent_small*4,
                numpy.array([4, 0.4, 0.4], dtype=numpy.float64)))

    geob.add_geometry(
            *make_box(
                numpy.array([-1, -1, -1], dtype=numpy.float64),
                numpy.array([5, 1, 1], dtype=numpy.float64)))

    mesh_info = MeshInfo()
    geob.set(mesh_info)
    mesh_info.set_holes([(0, 0, 0)])

    # region attributes
    mesh_info.regions.resize(1)
    mesh_info.regions[0] = (
            # point in region
            list(extent_small*2) + [
            # region number
            1,
            # max volume in region
            #0.0001
            0.005
            ])

    mesh = build(mesh_info, max_volume=0.02,
            volume_constraints=True, attributes=True)
    print "%d elements" % len(mesh.elements)
    #mesh.write_vtk("box-in-box.vtk")
    #print "done writing"

    fvi2fm = mesh.face_vertex_indices_to_face_marker

    face_marker_to_tag = {
            box_marker: "noslip",
            Marker.MINUS_X: "inflow",
            Marker.PLUS_X: "outflow",
            Marker.MINUS_Y: "inflow",
            Marker.PLUS_Y: "inflow",
            Marker.PLUS_Z: "inflow",
            Marker.MINUS_Z: "inflow"
            }

    def bdry_tagger(fvi, el, fn, all_v):
        face_marker = fvi2fm[fvi]
        return [face_marker_to_tag[face_marker]]

    from hedge.mesh import make_conformal_mesh
    return make_conformal_mesh(
            mesh.points, mesh.elements, bdry_tagger)


def main():
    from hedge.backends import guess_run_context
    rcon = guess_run_context(["cuda"])

    if rcon.is_head_rank:
        mesh = make_boxmesh()
        #from hedge.mesh import make_rect_mesh
        #mesh = make_rect_mesh(
        #       boundary_tagger=lambda fvi, el, fn, all_v: ["inflow"])
        mesh_data = rcon.distribute_mesh(mesh)
    else:
        mesh_data = rcon.receive_mesh()

    for order in [3]:
        from pytools import add_python_path_relative_to_script
        add_python_path_relative_to_script("..")

        from gas_dynamics_initials import UniformMachFlow
        box = UniformMachFlow(angle_of_attack=0)

        from hedge.models.gas_dynamics import GasDynamicsOperator
        op = GasDynamicsOperator(dimensions=3,
                gamma=box.gamma, mu=box.mu,
                prandtl=box.prandtl, spec_gas_const=box.spec_gas_const,
                bc_inflow=box, bc_outflow=box, bc_noslip=box,
                inflow_tag="inflow", outflow_tag="outflow", noslip_tag="noslip")

        discr = rcon.make_discretization(mesh_data, order=order,
                        debug=[
                            #"cuda_no_plan",
                            #"cuda_dump_kernels",
                            #"dump_dataflow_graph",
                            #"dump_optemplate_stages",
                            #"dump_dataflow_graph",
                            #"print_op_code",
                            "cuda_no_plan_el_local",
                            ],
                        default_scalar_type=numpy.float32,
                        tune_for=op.op_template())

        from hedge.visualization import SiloVisualizer, VtkVisualizer  # noqa
        #vis = VtkVisualizer(discr, rcon, "shearflow-%d" % order)
        vis = SiloVisualizer(discr, rcon)

        fields = box.volume_interpolant(0, discr)

        navierstokes_ex = op.bind(discr)

        max_eigval = [0]

        def rhs(t, q):
            ode_rhs, speed = navierstokes_ex(t, q)
            max_eigval[0] = speed
            return ode_rhs

        rhs(0, fields)

        if rcon.is_head_rank:
            print "---------------------------------------------"
            print "order %d" % order
            print "---------------------------------------------"
            print "#elements=", len(mesh.elements)

        from hedge.timestep import RK4TimeStepper
        stepper = RK4TimeStepper()

        # diagnostics setup ---------------------------------------------------
        from pytools.log import LogManager, add_general_quantities, \
                add_simulation_quantities, add_run_info

        logmgr = LogManager("navierstokes-%d.dat" % order, "w", rcon.communicator)
        add_run_info(logmgr)
        add_general_quantities(logmgr)
        add_simulation_quantities(logmgr)
        discr.add_instrumentation(logmgr)
        stepper.add_instrumentation(logmgr)

        logmgr.add_watches(["step.max", "t_sim.max", "t_step.max"])

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

        logmgr.add_quantity(ChangeSinceLastStep())

        # timestep loop -------------------------------------------------------
        try:
            from hedge.timestep import times_and_steps
            step_it = times_and_steps(
                    final_time=200,
                    #max_steps=500,
                    logmgr=logmgr,
                    max_dt_getter=lambda t: op.estimate_timestep(discr,
                        stepper=stepper, t=t, max_eigenvalue=max_eigval[0]))

            for step, t, dt in step_it:
                if step % 200 == 0:
                #if False:
                    visf = vis.make_file("box-%d-%06d" % (order, step))

                    #rhs_fields = rhs(t, fields)

                    vis.add_data(visf,
                            [
                                ("rho", discr.convert_volume(
                                    op.rho(fields), kind="numpy")),
                                ("e", discr.convert_volume(
                                    op.e(fields), kind="numpy")),
                                ("rho_u", discr.convert_volume(
                                    op.rho_u(fields), kind="numpy")),
                                ("u", discr.convert_volume(
                                    op.u(fields), kind="numpy")),

                                # ("rhs_rho", discr.convert_volume(
                                #     op.rho(rhs_fields), kind="numpy")),
                                # ("rhs_e", discr.convert_volume(
                                #     op.e(rhs_fields), kind="numpy")),
                                # ("rhs_rho_u", discr.convert_volume(
                                #     op.rho_u(rhs_fields), kind="numpy")),
                                ],
                            expressions=[
                                ("p", "(0.4)*(e- 0.5*(rho_u*u))"),
                                ],
                            time=t, step=step
                            )
                    visf.close()

                fields = stepper(fields, t, dt, rhs)

        finally:
            vis.close()
            logmgr.save()
            discr.close()

if __name__ == "__main__":
    main()
