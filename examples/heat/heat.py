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
from pytools.arithmetic_container import ArithmeticList
from hedge.tools import Rotation, dot




def coefficient_to_matrix(discr, coeff):
    return num.diagonal_matrix(
            discr.interpolate_volume_function(coeff),
            flavor=num.SparseExecuteMatrix)

    


class StrongHeatOperator:
    def __init__(self, discr, coeff=lambda x: 1):
        self.discr = discr

        from hedge.flux import zero, make_normal, local, neighbor, average

        normal = make_normal(discr.dimensions)
        flux_weak = average*normal
        flux_strong = local*normal - flux_weak

        self.nabla = discr.nabla
        self.stiff = discr.stiffness_operator
        self.mass = discr.mass_operator
        self.m_inv = discr.inverse_mass_operator

        self.q_flux = discr.get_flux_operator(flux_strong)
        self.u_flux = discr.get_flux_operator(flux_strong)
        
        from math import sqrt
        self.sqrt_coeff = coefficient_to_matrix(discr, lambda x: sqrt(coeff(x)))

    def q(self, u):
        from hedge.discretization import pair_with_boundary

        sqrt_coeff_u = self.sqrt_coeff * u
        bc_u = -self.discr.boundarize_volume_field(sqrt_coeff_u)

        q = self.m_inv * (
                self.sqrt_coeff*(self.stiff * u)
                - (self.u_flux*sqrt_coeff_u)
                - self.u_flux*pair_with_boundary(sqrt_coeff_u, bc_u)
                )
        return q

    def rhs(self, t, u):
        from hedge.discretization import pair_with_boundary

        q = self.q(u)

        sqrt_coeff_q = self.sqrt_coeff * q
        bc_q = self.discr.boundarize_volume_field(sqrt_coeff_q)

        rhs_u = self.m_inv * (
                self.sqrt_coeff*dot(self.stiff, q)
                - dot(self.q_flux, sqrt_coeff_q)
                - dot(self.q_flux, pair_with_boundary(sqrt_coeff_q, bc_q))
                )

        return rhs_u




def main() :
    from hedge.element import \
            TriangularElement, \
            TetrahedralElement
    from hedge.timestep import RK4TimeStepper
    from hedge.mesh import \
            make_disk_mesh, \
            make_regular_square_mesh, \
            make_square_mesh, \
            make_ball_mesh
    from hedge.visualization import SiloVisualizer, make_silo_file
    from math import sin, cos, pi, exp, sqrt
    from hedge.parallel import guess_parallelization_context

    pcon = guess_parallelization_context()

    dim = 2

    if dim == 2:
        if pcon.is_head_rank:
            mesh = make_disk_mesh(r=0.5)
            #mesh = make_regular_square_mesh(
                    #n=9, periodicity=(True,True))
            #mesh = make_regular_square_mesh(n=9)
            #mesh = make_square_mesh(max_area=0.008)
            #mesh.transform(Rotation(pi/8))
        el_class = TriangularElement
    elif dim == 3:
        if pcon.is_head_rank:
            mesh = make_ball_mesh(max_volume=0.001)
        el_class = TetrahedralElement
    else:
        raise RuntimeError, "bad number of dimensions"

    if pcon.is_head_rank:
        print "%d elements" % len(mesh.elements)
        mesh_data = pcon.distribute_mesh(mesh)
    else:
        mesh_data = pcon.receive_mesh()

    discr = pcon.make_discretization(mesh_data, el_class(3))
    stepper = RK4TimeStepper()
    vis = SiloVisualizer(discr)

    dt = discr.dt_factor(1)**2/2
    nsteps = int(1/dt)
    if pcon.is_head_rank:
        print "dt", dt
        print "nsteps", nsteps

    def u0(x):
        
        if comp.norm_2(x) < 0.2:
            #return exp(-100*x*x)
            return 1
        else:
            return 0

    op = StrongHeatOperator(discr)
    u = discr.interpolate_volume_function(u0)

    for step in range(nsteps):
        t = step*dt
        if step % 10 == 0:
            print "timestep %d, t=%f, l2=%g" % (
                    step, t, sqrt(u*(op.mass*u)))

        if step % 10 == 0:
            silo = make_silo_file("fld-%04d" % step, pcon)
            vis.add_to_silo(silo,
                    [("u", u), ], 
                    time=t,
                    step=step, write_coarse_mesh=True)
            silo.close()

        u = stepper(u, t, dt, op.rhs)

if __name__ == "__main__":
    #import cProfile as profile
    #profile.run("main()", "wave2d.prof")
    main()

