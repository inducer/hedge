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
import pylinear.array as num
import pylinear.computation as comp




def double_cross(tbl, field):
    from pytools.arithmetic_container import ArithmeticList
    return ArithmeticList([
          tbl[0][1]*field[1] + tbl[0][2]*field[2]
        - tbl[1][1]*field[0] - tbl[2][2]*field[0],

          tbl[1][0]*field[0] + tbl[1][2]*field[2] 
        - tbl[0][0]*field[1] - tbl[2][2]*field[1],
          
          tbl[2][1]*field[1] + tbl[2][0]*field[0] 
        - tbl[0][0]*field[2] - tbl[1][1]*field[2],
        ])




def main():
    from hedge.element import TriangularElement, TetrahedralElement
    from hedge.timestep import RK4TimeStepper
    from hedge.mesh import make_ball_mesh, make_cylinder_mesh, make_box_mesh
    from hedge.discretization import \
            Discretization, \
            bind_flux, \
            bind_nabla, \
            bind_mass_matrix, \
            bind_inverse_mass_matrix, \
            pair_with_boundary
    from hedge.visualization import SiloVisualizer
    from hedge.silo import DB_VARTYPE_VECTOR
    from hedge.flux import zero, make_normal, local, neighbor, average
    from hedge.tools import dot, cross, EOCRecorder
    from math import sqrt
    from analytic_solutions import \
            RealPartAdapter, \
            SplitComplexAdapter, \
            CartesianAdapter, \
            CylindricalCavityMode, \
            RectangularCavityMode
    from pytools.arithmetic_container import ArithmeticList

    # field order is [Ex Ey Ez Hx Hy Hz]

    R = 1
    d = 2
    epsilon = 1
    mu = 1

    eoc_rec = EOCRecorder()

    cylindrical = False

    if cylindrical:
        mode = CylindricalCavityMode(m=1, n=1, p=1,
                radius=R, height=d, 
                epsilon=epsilon, mu=mu)
        r_sol = CartesianAdapter(RealPartAdapter(mode))
        c_sol = SplitComplexAdapter(CartesianAdapter(mode))
        mesh = make_cylinder_mesh(radius=R, height=d, max_volume=0.01)
    else:
        mode = RectangularCavityMode(epsilon, mu, (1,1,1))
        r_sol = RealPartAdapter(mode)
        c_sol = SplitComplexAdapter(mode)
        mesh = make_box_mesh(max_volume=0.01)

    #for order in [3]:
    for order in [1,2,3,4,5,6]:
        print "---------------------------------------------"
        print "order %d" % order
        print "---------------------------------------------"
        discr = Discretization(mesh, TetrahedralElement(order))
        vis = SiloVisualizer(discr)

        print "%d elements" % len(discr.mesh.elements)

        dt = discr.dt_factor(1/sqrt(epsilon*mu))
        final_time = 0.3
        nsteps = int(final_time/dt)+1
        dt = final_time/nsteps

        print "dt", dt
        print "nsteps", nsteps

        nabla = bind_nabla(discr)
        mass = bind_mass_matrix(discr)
        m_inv = bind_inverse_mass_matrix(discr)

        def l2_norm(field):
            return sqrt(dot(field, mass*field))

        def curl(field):
            return cross(nabla, field)

        def vis_solution():
            dt = 0.1
            for step in range(10):
                t = step*dt
                print step, t
                mode.set_time(t)
                fields = discr.interpolate_volume_function(r_sol)

                vis("em-%04d.silo" % step,
                        vectors=[("e", fields[0:3]), ("h", fields[3:6]), ],
                        expressions=[
                        ("mag_e", "magnitude(e)"),
                        ("mag_h", "magnitude(h)"),
                        ],
                        write_coarse_mesh=True,
                        time=t, step=step
                        )

        def check_pde():
            dt = 0.1

            for step in range(10):
                t = step*dt
                mode.set_time(t)
                fields = discr.interpolate_volume_function(c_sol)

                er = fields[0:3]
                hr = fields[3:6]
                ei = fields[6:9]
                hi = fields[9:12]

                vis("em-%04d.silo" % step,
                        vectors=[
                            ("er", er), 
                            ("ei", ei), 
                            ("hr", hr), 
                            ("hi", hi), 
                            ("curl_er", curl(er)), 
                            ("om_hi", -mode.omega*hi), 
                            ("curl_hr", curl(hr)), 
                            ("om_ei", mu*epsilon*mode.omega*hi), 
                            ],
                        expressions=[
                        ("diff_er", "curl_er-om_hi", DB_VARTYPE_VECTOR),
                        ("diff_hr", "curl_hr-om_ei", DB_VARTYPE_VECTOR),
                        ],
                        write_coarse_mesh=True,
                        time=t, step=step
                        )
                er_res = curl(er) + mode.omega*hi
                ei_res = curl(ei) - mode.omega*hr
                hr_res = curl(hr) - mu*epsilon*mode.omega*ei
                hi_res = curl(hi) + mu*epsilon*mode.omega*er

                print "time=%f, rel l2 residual in Re[E]=%g\tIm[E]=%g\tRe[H]=%g\tIm[H]=%g" % (
                        t,
                        l2_norm(er_res),#/l2_norm(er),
                        l2_norm(ei_res),#/l2_norm(ei),
                        l2_norm(hr_res),#/l2_norm(hr),
                        l2_norm(hi_res),#/l2_norm(hi),
                        )

        #vis_solution()
        #check_pde()
        #continue

        normal = make_normal(discr.dimensions)

        n_jump = bind_flux(discr, 1/2*normal*(local-neighbor))
        n_n_jump_tbl = [[bind_flux(discr, 1/2*normal[i]*normal[j]*(local-neighbor))
                for i in range(discr.dimensions)]
                for j in range(discr.dimensions)]

        mode.set_time(0)
        fields = discr.interpolate_volume_function(r_sol)

        alpha = 1

        def rhs(t, y):
            e = fields[0:3]
            h = fields[3:6]

            bc_e = -discr.boundarize_volume_field(e)
            bc_h = discr.boundarize_volume_field(h)

            h_pair = pair_with_boundary(h, bc_h)
            e_pair = pair_with_boundary(e, bc_e)

            rhs = ArithmeticList([])

            # rhs e
            rhs.extend(curl(h)
                    - m_inv*(
                        cross(n_jump, h)
                        + cross(n_jump, h_pair)
                        - alpha*double_cross(n_n_jump_tbl, e)
                        - alpha*double_cross(n_n_jump_tbl, e_pair)
                        ) 
                    )
            # rhs h
            rhs.extend(-curl(e)
                    + m_inv*(
                        cross(n_jump, e)
                        + cross(n_jump, e_pair)
                        + alpha*double_cross(n_n_jump_tbl, h)
                        + alpha*double_cross(n_n_jump_tbl, h_pair)
                        )
            )
            return rhs

        stepper = RK4TimeStepper()
        from time import time
        last_tstep = time()
        t = 0
        for step in range(nsteps):
            print "timestep %d, t=%f l2[e]=%g l2[h]=%g secs=%f" % (
                    step, t, l2_norm(fields[0:3]), l2_norm(fields[3:6]),
                    time()-last_tstep)
            last_tstep = time()

            #vis("cylmode-%04d.silo" % step,
                    #vectors=[("e", fields[0:3]), 
                        #("h", fields[3:6]), ],
                    #expressions=[
                        #],
                    #write_coarse_mesh=True,
                    #time=t, step=step
                    #)

            fields = stepper(fields, t, dt, rhs)
            t += dt

        mode.set_time(t)
        true_fields = discr.interpolate_volume_function(r_sol)
        eoc_rec.add_data_point(order, l2_norm(fields-true_fields))

        vis("em-%04d.silo" % order,
                vectors=[
                    ("e", fields[0:3]), 
                    ("h", fields[3:6]), 
                    ("etrue", true_fields[0:3]), 
                    ("htrue", true_fields[3:6]), 
                    ],
                expressions=[
                    ("eerr", "e-etrue", DB_VARTYPE_VECTOR),
                    ("herr", "h-htrue", DB_VARTYPE_VECTOR),
                    ],
                write_coarse_mesh=True,
                time=t, step=step
                )

        print
        print eoc_rec.pretty_print("P.Deg.", "L2 Error")

if __name__ == "__main__":
    import cProfile as profile
    #profile.run("main()", "wave2d.prof")
    main()
