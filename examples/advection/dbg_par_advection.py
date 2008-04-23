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




import pylinear.array as num
import pylinear.computation as comp




def dot(x, y): 
    from operator import add
    return reduce(add, (xi*yi for xi, yi in zip(x,y)))




class StrongAdvectionOperator:
    def __init__(self, discr, a, inflow_u=None):
        self.a = a
        self.discr = discr
        self.inflow_u = inflow_u

        from hedge.flux import zero, trace_sign, make_normal, local, neighbor, average
        from hedge.discretization import bind_flux, bind_nabla, bind_mass_matrix, \
                bind_inverse_mass_matrix

        normal = make_normal(self.discr.dimensions)
        flux_weak = dot(normal, a) * average# - 0.5 *(local-neighbor)
        flux_strong = dot(normal, a)*local - flux_weak
        self.flux = bind_flux(self.discr, flux_strong)

        self.nabla = bind_nabla(discr)
        self.mass = bind_mass_matrix(discr)
        self.m_inv = bind_inverse_mass_matrix(discr)

    def rhs(self, t, u):
        from pytools import argmax
        from hedge.discretization import pair_with_boundary

        if self.inflow_u is not None:
            bc_in = self.discr.interpolate_boundary_function(
                    lambda x: self.inflow_u(t, x),
                    "inflow")

            return dot(self.a, self.nabla*u) - self.m_inv*(
                    self.flux * u + 
                    self.flux * pair_with_boundary(u, bc_in, "inflow"))
        else:
            return dot(self.a, self.nabla*u) - m_inv*(self.flux * u)





def main() :
    from hedge.element import \
            TriangularElement, \
            TetrahedralElement
    from hedge.timestep import RK4TimeStepper
    from hedge.mesh import \
            make_disk_mesh, \
            make_square_mesh, \
            make_regular_square_mesh, \
            make_single_element_mesh, \
            make_ball_mesh, \
            make_box_mesh
    from hedge.discretization import \
            Discretization, \
            bind_flux, \
            bind_nabla, \
            bind_weak_nabla, \
            bind_mass_matrix, \
            bind_inverse_mass_matrix, \
            pair_with_boundary
    from hedge.visualization import SiloVisualizer, make_silo_file
    from hedge.tools import dot
    from pytools.arithmetic_container import ArithmeticList
    from pytools.stopwatch import Job
    from math import sin, cos, pi, sqrt
    from hedge.parallel import \
            guess_parallelization_context, \
            reassemble_volume_field

    def u_analytic(t, x):
        return sin(3*(a*x+t))

    def boundary_tagger(vertices, el, face_nr):
        if el.face_normals[face_nr] * a > 0:
            return ["inflow"]
        else:
            return ["outflow"]

    pcon = guess_parallelization_context()

    dim = 2
    order = 3
    periodic = False

    if dim == 2:
        a = num.array([1,0])
        if pcon.is_head_rank:
            #mesh = make_square_mesh(boundary_tagger=boundary_tagger, max_area=0.1)
            #mesh = make_square_mesh(boundary_tagger=boundary_tagger, max_area=0.2)
            #mesh = make_regular_square_mesh(a=-r, b=r, boundary_tagger=boundary_tagger, n=3)
            #mesh = make_single_element_mesh(boundary_tagger=boundary_tagger)
            #mesh = make_disk_mesh(r=pi, boundary_tagger=boundary_tagger, max_area=0.5)
            mesh = make_disk_mesh(boundary_tagger=boundary_tagger)
        el_class = TriangularElement
    elif dim == 3:
        a = num.array([0,0,0.5])
        if pcon.is_head_rank:
            if periodic:
                mesh = make_box_mesh(dimensions=(1,1,2*pi/3),
                        periodic=periodic, max_volume=0.01)
            else:
                mesh = make_box_mesh(max_volume=0.01, 
                        boundary_tagger=boundary_tagger)
                #mesh = make_ball_mesh(boundary_tagger=boundary_tagger)
        el_class = TetrahedralElement
    else:
        raise RuntimeError, "bad number of dimensions"

    if pcon.is_head_rank:
        mesh_data = pcon.distribute_mesh(mesh)
    else:
        mesh_data = pcon.receive_mesh()

    discr = pcon.make_discretization(mesh_data, el_class(order))
    vis = SiloVisualizer(discr)
    op = StrongAdvectionOperator(discr, a, u_analytic)
    u = discr.interpolate_volume_function(lambda x: u_analytic(0, x))

    if pcon.is_head_rank:
        gdiscr = Discretization(mesh, el_class(order))
        gop = StrongAdvectionOperator(gdiscr, a, u_analytic)
        gu = gdiscr.interpolate_volume_function(lambda x: u_analytic(0, x))
        gvis = SiloVisualizer(gdiscr)
    else:
        gdiscr = None

    print "%d elements" % len(discr.mesh.elements)

    dt = discr.dt_factor(comp.norm_2(a))
    stepfactor = 1
    nsteps = int(4/dt)

    stepper = RK4TimeStepper()
    gstepper = RK4TimeStepper()
    for step in range(nsteps):
        if step % stepfactor == 0:
            print "timestep %d, t=%f, l2=%f" % (step, dt*step, sqrt(u*(op.mass*u)))

        t = step*dt
        if True:
            rhs = op.rhs(t, u)
            g_ass_u = reassemble_volume_field(pcon, gdiscr, discr, u)
            g_ass_rhs = reassemble_volume_field(pcon, gdiscr, discr, rhs)
            g_ass_bdry = reassemble_volume_field(pcon, gdiscr, discr, 
                    discr.received_bdrys[1-pcon.rank])
            if pcon.is_head_rank:
                grhs = gop.rhs(t, gu)
                silo = make_silo_file("fld-%04d" % step)
                gvis.add_to_silo(silo, [
                            ("ser_u", gu), 
                            ("par_u", g_ass_u), 
                            ("par_rhs", g_ass_rhs), 
                            ("ser_rhs", grhs), 
                            ("bdry", g_ass_bdry), 
                            ], 
                            expressions=[("d_rhs", "par_rhs-ser_rhs")],
                            time=t, 
                            step=step
                            )
                silo.close()

        u = stepper(u, t, dt, op.rhs)
        if pcon.is_head_rank:
            gu = gstepper(gu, t, dt, gop.rhs)

        #u_true = discr.interpolate_volume_function(lambda x: u_analytic(t, x))


if __name__ == "__main__":
    import cProfile as profile
    main()


