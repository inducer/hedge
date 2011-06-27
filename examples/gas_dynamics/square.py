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

        return bool(area > 2.5*min(max_area_volume, max_area_corners))

    from meshpy.geometry import make_box
    points, facets, _, _ = make_box((-0.5,-0.5), (0.5,0.5))
    obstacle_corners = points[:]

    from meshpy.geometry import GeometryBuilder, Marker

    profile_marker = Marker.FIRST_USER_MARKER
    builder = GeometryBuilder()
    builder.add_geometry(points=points, facets=facets,
            facet_markers=profile_marker)

    points, facets, _, facet_markers = make_box((-16, -22), (25, 22))
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

    from meshpy.triangle import write_gnuplot_mesh
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

    from hedge.mesh import make_conformal_mesh_ext
    vertices = numpy.asarray(mesh.points, dtype=float, order="C")
    from hedge.mesh.element import Triangle
    return make_conformal_mesh_ext(
            vertices,
            [Triangle(i, el_idx, vertices)
                for i, el_idx in enumerate(mesh.elements)],
            bdry_tagger)




def main():
    import logging
    logging.basicConfig(level=logging.INFO)

    from hedge.backends import guess_run_context
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
    add_python_path_relative_to_script(".")

    for order in [3]:
        from gas_dynamics_initials import UniformMachFlow
        square = UniformMachFlow(gaussian_pulse_at=numpy.array([-2, 2]),
                pulse_magnitude=0.003)

        from hedge.models.gas_dynamics import (
                GasDynamicsOperator,
                GammaLawEOS)

        op = GasDynamicsOperator(dimensions=2,
                equation_of_state=GammaLawEOS(square.gamma), mu=square.mu,
                prandtl=square.prandtl, spec_gas_const=square.spec_gas_const,
                bc_inflow=square, bc_outflow=square, bc_noslip=square,
                inflow_tag="inflow", outflow_tag="outflow", noslip_tag="noslip")

        discr = rcon.make_discretization(mesh_data, order=order,
                        debug=["cuda_no_plan",
                            "cuda_dump_kernels",
                            #"dump_dataflow_graph",
                            #"dump_optemplate_stages",
                            #"dump_dataflow_graph",
                            #"dump_op_code"
                            #"cuda_no_plan_el_local"
                            ],
                        default_scalar_type=numpy.float64,
                        tune_for=op.op_template(),
                        quad_min_degrees={
                            "gasdyn_vol": 3*order,
                            "gasdyn_face": 3*order,
                            }
                        )

        from hedge.visualization import SiloVisualizer, VtkVisualizer
        #vis = VtkVisualizer(discr, rcon, "shearflow-%d" % order)
        vis = SiloVisualizer(discr, rcon)

        from hedge.timestep.runge_kutta import (
                LSRK4TimeStepper, ODE23TimeStepper, ODE45TimeStepper)
        from hedge.timestep.dumka3 import Dumka3TimeStepper
        #stepper = LSRK4TimeStepper(dtype=discr.default_scalar_type,
                #vector_primitive_factory=discr.get_vector_primitive_factory())

        stepper = ODE23TimeStepper(dtype=discr.default_scalar_type,
                rtol=1e-6,
                vector_primitive_factory=discr.get_vector_primitive_factory())
        # Dumka works kind of poorly
        #stepper = Dumka3TimeStepper(dtype=discr.default_scalar_type,
                #rtol=1e-7, pol_index=2,
                #vector_primitive_factory=discr.get_vector_primitive_factory())

        #from hedge.timestep.dumka3 import Dumka3TimeStepper
        #stepper = Dumka3TimeStepper(3, rtol=1e-7)

        # diagnostics setup ---------------------------------------------------
        from pytools.log import LogManager, add_general_quantities, \
                add_simulation_quantities, add_run_info

        logmgr = LogManager("cns-square-sp-%d.dat" % order, "w", rcon.communicator)

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

        add_simulation_quantities(logmgr)
        logmgr.add_watches(["step.max", "t_sim.max", "t_step.max"])

        # filter setup ------------------------------------------------------------
        from hedge.discretization import Filter, ExponentialFilterResponseFunction
        mode_filter = Filter(discr,
                ExponentialFilterResponseFunction(min_amplification=0.95, order=6))

        # timestep loop -------------------------------------------------------
        fields = square.volume_interpolant(0, discr)

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

        try:
            from hedge.timestep import times_and_steps
            step_it = times_and_steps(
                    final_time=1000,
                    #max_steps=500,
                    logmgr=logmgr,
                    max_dt_getter=lambda t: next_dt,
                    taken_dt_getter=lambda: taken_dt)

            model_stepper = LSRK4TimeStepper()
            next_dt = op.estimate_timestep(discr,
                    stepper=model_stepper, t=0, 
                    max_eigenvalue=max_eigval[0])

            for step, t, dt in step_it:
                #if (step % 10000 == 0): #and step < 950000) or (step % 500 == 0 and step > 950000):
                #if False:
                if step % 5 == 0:
                    visf = vis.make_file("square-%d-%06d" % (order, step))

                    #from pyvisfile.silo import DB_VARTYPE_VECTOR
                    vis.add_data(visf,
                            [
                                ("rho", discr.convert_volume(op.rho(fields), kind="numpy")),
                                ("e", discr.convert_volume(op.e(fields), kind="numpy")),
                                ("rho_u", discr.convert_volume(op.rho_u(fields), kind="numpy")),
                                ("u", discr.convert_volume(op.u(fields), kind="numpy")),
                            ],
                            expressions=[
                                ("p", "(0.4)*(e- 0.5*(rho_u*u))"),
                                ],
                            time=t, step=step
                            )
                    visf.close()

                if stepper.adaptive:
                    fields, t, taken_dt, next_dt = stepper(fields, t, dt, rhs)
                else:
                    taken_dt = dt
                    fields = stepper(fields, t, dt, rhs)
                    dt = op.estimate_timestep(discr,
                            stepper=model_stepper, t=0,
                            max_eigenvalue=max_eigval[0])

                #fields = mode_filter(fields)

        finally:
            vis.close()
            logmgr.save()
            discr.close()

if __name__ == "__main__":
    main()
