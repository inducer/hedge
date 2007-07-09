from __future__ import division
import pylinear.array as num
import pylinear.computation as comp




def main():
    from hedge.element import TriangularElement, TetrahedralElement
    from hedge.timestep import RK4TimeStepper
    from hedge.mesh import make_ball_mesh, make_cylinder_mesh
    from hedge.discretization import \
            Discretization, \
            bind_flux, \
            bind_nabla, \
            bind_mass_matrix, \
            bind_inverse_mass_matrix, \
            pair_with_boundary
    from hedge.visualization import SiloVisualizer
    from hedge.silo import DB_VARTYPE_VECTOR
    from hedge.flux import zero, normal, jump, local, neighbor, average
    from hedge.tools import dot, cross
    from analytic_solutions import \
            RealPartAdapter, \
            SplitComplexAdapter, \
            CartesianAdapter, \
            CylindricalCavityMode
    from pytools.arithmetic_container import ArithmeticList

    # field order is [Ex Ey Ez Hx Hy Hz]

    R = 1
    d = 2
    epsilon = 1
    mu = 1

    mesh = make_cylinder_mesh(radius=R, height=d, max_volume=0.01)
    discr = Discretization(mesh, TetrahedralElement(3))
    vis = SiloVisualizer(discr)

    print "%d elements" % len(discr.mesh.elements)

    mode = CylindricalCavityMode(m=1, n=1, p=1,
            radius=R, height=d, 
            epsilon=epsilon, mu=mu)
    r_sol = CartesianAdapter(RealPartAdapter(mode))
    fields = discr.interpolate_volume_function(r_sol)

    dt = discr.dt_factor(1)
    nsteps = int(1/dt)
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

            vis("cylmode-%04d.silo" % step,
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
        c_sol = SplitComplexAdapter(CartesianAdapter(mode))

        for step in range(10):
            t = step*dt
            mode.set_time(t)
            fields = discr.interpolate_volume_function(c_sol)

            er = fields[0:3]
            hr = fields[3:6]
            ei = fields[6:9]
            hi = fields[9:12]

            #vis("cylmode-%04d.silo" % step,
                    #vectors=[("curl_er", curl(er)), ("om_hi", -mode.omega*hi), ],
                    #expressions=[
                    #("diff", "curl_er-om_hi", DB_VARTYPE_VECTOR),
                    #],
                    #write_coarse_mesh=True,
                    #time=t, step=step
                    #)
            er_res = curl(er) + mode.omega*hi
            ei_res = curl(ei) - mode.omega*hr
            hr_res = curl(hr) - mu*epsilon*mode.omega*ei
            hi_res = curl(hi) + mu*epsilon*mode.omega*er

            print "time=%f, rel l2 error in Re[E]=%g\tIm[E]=%g\tRe[H]=%g\tIm[H]=%g" % (
                    t,
                    l2_norm(er_res)/l2_norm(er),
                    l2_norm(ei_res)/l2_norm(ei),
                    l2_norm(hr_res)/l2_norm(hr),
                    l2_norm(hi_res)/l2_norm(hi),
                    )

    #vis_solution()
    #check_pde()

    normal = normal(discr.dimensions)
    jump = jump(discr.dimensions)

    flux_e = -normal*(local-average)
    flux_h = normal*(local-average)

    bflux_e = bind_flux(discr, flux_e)
    bflux_h = bind_flux(discr, flux_h)

    def rhs(t, y):
        e = fields[0:3]
        h = fields[3:6]

        bc_e = -discr.boundarize_volume_field(e)
        bc_h = discr.boundarize_volume_field(h)

        rhs = ArithmeticList([])
        # rhs e
        rhs.extend(curl(h)
                + m_inv*(
                    cross(bflux_e, h)
                    + cross(bflux_e, pair_with_boundary(h, bc_h))
                    ) 
                )
        # rhs h
        rhs.extend(-curl(e)
                +m_inv*(
                    cross(bflux_h, e)
                    + cross(bflux_h, pair_with_boundary(e, bc_e))
                    )
        )
        return rhs

    stepper = RK4TimeStepper()
    from time import time
    last_tstep = time()
    for step in range(nsteps):
        t = step*dt
        print "timestep %d, t=%f l2[e]=%g l2[h]=%g secs=%f" % (
                step, t, l2_norm(fields[0:3]), l2_norm(fields[3:6]),
                time()-last_tstep)
        last_tstep = time()

        vis("cylmode-%04d.silo" % step,
                vectors=[("e", fields[0:3]), 
                    ("h", fields[3:6]), ],
                expressions=[
                    ],
                write_coarse_mesh=True,
                time=t, step=step
                )

        fields = stepper(fields, t, dt, rhs)

if __name__ == "__main__":
    import cProfile as profile
    #profile.run("main()", "wave2d.prof")
    main()


