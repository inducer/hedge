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




def ramp(t):
    # http://www.acm.caltech.edu/%7Ebruno/hyde_bruno_3d_jcp.pdf
    if t <= 0:
        return 0
    elif t >= 0:
        return 1
    else:
        from math import exp
        return 1-exp(2*exp(-1/t)/(t-1))

class NodalGivenFunction:
    def volume_interpolant(self, t, discr):
        return discr.convert_volume(
                        self(t, discr.nodes.T
                            .astype(discr.default_scalar_type)),
                        kind=discr.compute_kind)

    def boundary_interpolant(self, t, discr, tag):
        return discr.convert_boundary(
                        self(t, discr.get_boundary(tag).nodes.T
                            .astype(discr.default_scalar_type)),
                         tag=tag, kind=discr.compute_kind)




class ConcentricCylindersProblem:
    class InitialCondition(NodalGivenFunction):
        def __call__(self, t, x_vec):
            ones = numpy.ones_like(x_vec[0])
            zeros = numpy.zeros_like(x_vec[0])

            from hedge.tools import join_fields
            return join_fields(5*ones,zeros,zeros,zeros,zeros,zeros)

    class BoundaryCondition(NodalGivenFunction):
        def __call__(self, t, x_vec):
            ones = numpy.ones_like(x_vec[0])
            zeros = numpy.zeros_like(x_vec[0])

            x = x_vec[0]
            y = x_vec[1]
            r = x**2 + y**2

            tfac = ramp(t)
            theta = (numpy.arctan2(y, x)-20*t) % (2*numpy.pi)
            rho = 5+(r<0.75)*numpy.exp(-20*numpy.sin(theta)**2)

            from hedge.tools import join_fields
            return join_fields(rho,zeros,zeros,zeros,zeros,zeros)

    @staticmethod
    def make_mesh(r=1, r_inner=0.5, faces=50, max_area=0.01,
            boundary_tagger=(lambda fvi, el, fn, all_v: [])):
        from math import cos, sin, pi

        def needs_refinement(vertices, area):
            return area > max_area

        from meshpy.geometry import GeometryBuilder
        geob = GeometryBuilder()

        def make_circle(r):
            return [(r*cos(angle), r*sin(angle))
                    for angle in numpy.linspace(0, 2*pi, faces, endpoint=False)]

        geob.add_cycle(make_circle(1))
        geob.add_cycle(make_circle(0.5))

        import meshpy.triangle as triangle

        mesh_info = triangle.MeshInfo()
        geob.set(mesh_info)
        mesh_info.set_holes([(0,0)])

        generated_mesh = triangle.build(mesh_info, refinement_func=needs_refinement)

        from hedge.mesh import make_conformal_mesh
        return make_conformal_mesh(
                generated_mesh.points,
                generated_mesh.elements,
                boundary_tagger)



class SolidBodyRotationProblem:
    class BoundaryCondition(NodalGivenFunction):
        def __call__(self, t, x_vec):
            ones = numpy.ones_like(x_vec[0])
            zeros = numpy.zeros_like(x_vec[0])

            x = x_vec[0]
            y = x_vec[1]

            from hedge.tools import join_fields

            rho = ones + 0.3*(x>0)
            scale = (1-x**2)*(1-y**2)
            u = join_fields(y*scale,-x*scale)

            return self.op.from_primitive(x_vec, rho, u)

    InitialCondition = BoundaryCondition

    @staticmethod
    def make_mesh():
        from hedge.mesh import make_rect_mesh
        return make_rect_mesh(a=[-1,-1], b=(1,1), max_area=0.003,
                periodicity=(True,True),subdivisions=(10,10))





class SolidBodyRotationProblem:
    class BoundaryCondition(NodalGivenFunction):
        def __call__(self, t, x_vec):
            ones = numpy.ones_like(x_vec[0])
            zeros = numpy.zeros_like(x_vec[0])

            x = x_vec[0]
            y = x_vec[1]

            from hedge.tools import join_fields

            rho = ones + 0.3*(x>0)
            scale = (1-x**2)*(1-y**2)
            u = join_fields(y*scale,-x*scale)

            return self.op.from_primitive(x_vec, rho, u)

    InitialCondition = BoundaryCondition

    @staticmethod
    def make_mesh():
        from hedge.mesh import make_rect_mesh
        return make_rect_mesh(a=[-1,-1], b=(1,1), max_area=0.003,
                periodicity=(True,True),subdivisions=(10,10))





class ShearLayerProblem:
    class BoundaryCondition(NodalGivenFunction):
        def __call__(self, t, x_vec):
            ones = numpy.ones_like(x_vec[0])
            zeros = numpy.zeros_like(x_vec[0])

            x = x_vec[0]
            y = x_vec[1]

            from hedge.tools import join_fields

            def ramp_at_zero(x):
                return ramp(x+1)

            rho = ones
            ux = zeros
            uy = numpy.sign(x)*(1-numpy.sin(numpy.pi*(x+0.5))**10)
            u = join_fields(ux, uy)

            return self.op.from_primitive(x_vec, rho, u)

    InitialCondition = BoundaryCondition

    @staticmethod
    def make_mesh():
        from hedge.mesh import make_rect_mesh
        return make_rect_mesh(a=[-1,-1], b=(1,1), max_area=0.003,
                periodicity=(True,True),subdivisions=(25,25))





def main(write_output=True):
    from hedge.backends import guess_run_context
    rcon = guess_run_context(
                    #["cuda"]
                    )

    for order in [3]:
        problem = SolidBodyRotationProblem()
        #problem = ShearLayerProblem()

        if rcon.is_head_rank:
            mesh = problem.make_mesh()
            mesh_data = rcon.distribute_mesh(mesh)
            from hedge.visualization import write_gnuplot_mesh
            write_gnuplot_mesh("mesh.dat", mesh)
        else:
            mesh_data = rcon.receive_mesh()

        discr = rcon.make_discretization(mesh_data, order=order,
                        default_scalar_type=numpy.float64)

        from hedge.visualization import SiloVisualizer, VtkVisualizer
        #vis = VtkVisualizer(discr, rcon, "bgk-%d" % order)
        vis = SiloVisualizer(discr, rcon)

        from hedge.models.gas_dynamics.bgk_flow import BGKFlowOperator

        from hedge.mesh import TAG_ALL
        bc = problem.BoundaryCondition()
        op = BGKFlowOperator(bc, dimensions=2, tau=numpy.infty)
        bc.op = op

        rhs = op.bind(discr)

        dt = discr.dt_factor(op.max_eigenvalue()) * 0.6
        final_time = 2.0
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

        if write_output:
            log_file_name = "euler-%d.dat" % order
        else:
            log_file_name = None

        logmgr = LogManager(log_file_name, "w", rcon.communicator)
        add_run_info(logmgr)
        add_general_quantities(logmgr)
        add_simulation_quantities(logmgr, dt)
        discr.add_instrumentation(logmgr)
        stepper.add_instrumentation(logmgr)

        logmgr.add_watches(["step.max", "t_sim.max", "t_step.max"])

        # timestep loop -------------------------------------------------------
        t = 0

        ic = problem.InitialCondition()
        ic.op = op
        fields = ic.volume_interpolant(0, discr)

        from hedge.discretization import Filter, ExponentialFilterResponseFunction
        antialiasing = Filter(discr,
                ExponentialFilterResponseFunction(min_amplification=0.9,order=4))

        try:
            for step in range(nsteps):
                logmgr.tick()

                prim_f = op.to_primitive(discr.nodes.T, fields)

                if step % 5 == 0 and write_output:
                    visf = vis.make_file("bgk-%d-%04d" % (order, step))

                    from pylo import DB_VARTYPE_VECTOR
                    vis.add_data(visf,
                            [
                                ("rho", discr.convert_volume(prim_f[0], kind="numpy")),
                                ("u", discr.convert_volume(prim_f[1:3], kind="numpy")),
                                ("sig11", discr.convert_volume(prim_f[3], kind="numpy")),
                                ("sig22", discr.convert_volume(prim_f[4], kind="numpy")),
                                ("sig12", discr.convert_volume(prim_f[5], kind="numpy")),
                                ] + [
                                ("a%d" % i, discr.convert_volume(fields[i], kind="numpy"))
                                for i in range(1, 2+2*discr.dimensions)
                                ],
                            time=t, step=step
                            )
                    visf.close()

                fields = stepper(fields, t, dt, rhs)
                t += dt

                #fields = antialiasing(fields)

            logmgr.tick()

        finally:
            if write_output:
                vis.close()

            logmgr.save()

            discr.close()




if __name__ == "__main__":
    main()



# entry points for py.test ----------------------------------------------------
from pytools.test import mark_test
@mark_test.long
def test_euler_vortex():
    main()
