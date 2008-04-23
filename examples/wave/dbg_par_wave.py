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




class StrongWaveOperator:
    def __init__(self, discr, source_f=None):
        self.discr = discr
        self.source_f = source_f

        from hedge.flux import zero, trace_sign, make_normal, local, neighbor, average
        from hedge.discretization import bind_flux, bind_nabla, bind_mass_matrix, \
                bind_inverse_mass_matrix

        normal = make_normal(discr.dimensions)
        flux_weak = average*normal
        flux_strong = local*normal - flux_weak

        self.nabla = bind_nabla(discr)
        self.mass = bind_mass_matrix(discr)
        self.m_inv = bind_inverse_mass_matrix(discr)

        self.flux = bind_flux(discr, flux_strong)

    def rhs(self, t, y):
        from hedge.discretization import pair_with_boundary

        u = y[0]
        v = y[1:]

        #bc_v = self.discr.boundarize_volume_field(v)
        bc_u = -self.discr.boundarize_volume_field(u)

        rhs = ArithmeticList([])
        # rhs u
        rhs.append(dot(self.nabla, v) 
                - self.m_inv*(
                    dot(self.flux, v) 
                    #+ dot(bflux, pair_with_boundary(v, bc_v))
                    ))

        if self.source_f is not None:
            rhs[0] += self.source_f(t)

        # rhs v
        #rhs.extend(self.nabla*u 
                #-self.m_inv*(
                    #self.flux*u 
                    #+ self.flux*pair_with_boundary(u, bc_u)
                    #))
        #rhs.extend( -(self.m_inv*( self.flux*u )))
        #rhs.extend(self.nabla*u)
        rhs.extend(self.flux*pair_with_boundary(u, bc_u))
        return rhs




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
    from pylo import DB_VARTYPE_VECTOR
    from hedge.visualization import \
            SiloVisualizer, \
            make_silo_file, \
            get_rank_partition
    from hedge.discretization import Discretization
    from pytools.stopwatch import Job
    from math import sin, cos, pi, exp, sqrt
    from hedge.parallel import \
            guess_parallelization_context, \
            reassemble_volume_field

    pcon = guess_parallelization_context()

    dim = 2
    order = 3

    if dim == 2:
        if pcon.is_head_rank:
            mesh = make_disk_mesh()
        #mesh = make_regular_square_mesh(n=5)
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

    discr = pcon.make_discretization(mesh_data, el_class(order))
    stepper = RK4TimeStepper()
    vis = SiloVisualizer(discr)

    dt = discr.dt_factor(1)
    nsteps = int(1/dt)
    if pcon.is_head_rank:
        print "dt", dt
        print "nsteps", nsteps

    def ic_u(x):
        return exp(-x*x*128)

    op = StrongWaveOperator(discr)
    fields = ArithmeticList(
            [discr.interpolate_volume_function(ic_u)])
    fields.extend(op.nabla*fields[0])

    if pcon.is_head_rank:
        gdiscr = Discretization(mesh, el_class(order))
        gvis = SiloVisualizer(gdiscr)
        gstepper = RK4TimeStepper()
        gop = StrongWaveOperator(gdiscr)

        gfields = ArithmeticList(
                [gdiscr.interpolate_volume_function(ic_u)])
        gfields.extend(gop.nabla*gfields[0])
    else:
        gdiscr = None

    gpart = reassemble_volume_field(pcon, gdiscr, discr, 
            get_rank_partition(pcon, discr))

    for step in range(nsteps):
        t = step*dt
        if step % 1 == 0:
            print "timestep %d, t=%f, l2=%g" % (
                    step, t, sqrt(fields[0]*(op.mass*fields[0])))

        if True:
            rhs = op.rhs(t, fields)
            g_ass_rhs = reassemble_volume_field(pcon, gdiscr, discr, rhs)
            if pcon.is_head_rank:
                grhs = gop.rhs(t, gfields)
                silo = make_silo_file("rhs-%04d" % step)
                gvis.add_to_silo(silo,
                        scalars=[
                            ("u", gfields[0]), 
                            ("par_rhs_u", g_ass_rhs[0]), 
                            ("ser_rhs_u", grhs[0]), 
                            ("partition", gpart),
                            ],
                        vectors=[
                            ("v", gfields[1:]), 
                            ("par_rhs_v", g_ass_rhs[1:]), 
                            ("ser_rhs_v", grhs[1:]), 
                            ],
                        expressions=[
                            ("d_rhs_v", "par_rhs_v-ser_rhs_v", 
                                DB_VARTYPE_VECTOR),
                            ("d_rhs_u", "par_rhs_u-ser_rhs_u"),
                            ]
                            )
                silo.close()

        if False:
            silo = make_silo_file("fld-%04d" % step, pcon)
            vis.add_to_silo(silo,
                    [("u", fields[0]), ], 
                    [("v", fields[1:]), ],
                    time=t,
                    step=step)
            silo.close()

        fields = stepper(fields, t, dt, op.rhs)
        if pcon.is_head_rank:
            gfields = gstepper(gfields, t, dt, gop.rhs)

if __name__ == "__main__":
    #import cProfile as profile
    #profile.run("main()", "wave2d.prof")
    main()

