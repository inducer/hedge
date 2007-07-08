from __future__ import division
import pylinear.array as num
import pylinear.computation as comp




def main():
    from hedge.element import \
            TriangularElement, \
            TetrahedralElement
    from hedge.timestep import RK4TimeStepper
    from hedge.mesh import \
            make_ball_mesh, \
            make_cylinder_mesh
    from hedge.discretization import \
            Discretization, \
            bind_flux, \
            bind_boundary_flux, \
            bind_nabla, \
            bind_mass_matrix, \
            bind_inverse_mass_matrix, \
            pair_with_boundary
    from hedge.visualization import \
            VtkVisualizer, \
            SiloVisualizer
    from hedge.silo import DB_VARTYPE_VECTOR
    from pytools.stopwatch import Job
    from math import sin, cos, pi, exp, sqrt, atan2
    from hedge.flux import zero, normal, jump, local, neighbor, average
    from hedge.tools import Rotation, dot, cross
    from analytic_solutions import \
            RealPartAdapter, \
            SplitComplexAdapter, \
            CartesianAdapter, \
            CylindricalCavityMode

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
    #fields = discr.interpolate_volume_function(r_sol)

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
    check_pde()
    return

    normal = normal(discr.dimensions)
    flux_weak = average*normal
    flux_strong = local*normal - flux_weak

    flux = bind_flux(discr, flux_strong)
    bflux = bind_boundary_flux(discr, flux_strong)

    def rhs(t, y):
        u = fields[0]
        v = fields[1:]

        #bc_v = discr.boundarize_volume_field(v)
        bc_u = -discr.boundarize_volume_field(u)

        rhs = ArithmeticList([])
        # rhs u
        rhs.append(dot(nabla, v) 
                - m_inv*(
                    dot(flux, v) 
                    #+ dot(bflux, pair_with_boundary(v, bc_v))
                    ) 
                + source_u_vec)
        # rhs v
        rhs.extend(nabla*u 
                -m_inv*(
                    flux*u 
                    + bflux*pair_with_boundary(u, bc_u)
                    ))
        return rhs

    stepper = RK4TimeStepper()
    #vis = VtkVisualizer(discr)
    for step in range(nsteps):
        t = step*dt
        if step % 10 == 0:
            print "timestep %d, t=%f, l2=%g" % (
                    step, t, sqrt(fields[0]*(mass*fields[0])))

        if t > 0.1:
            source_u_vec = discr.volume_zeros()

        if step % 10 == 0:
            vis("fld-%04d.silo" % step,
                    [("u", fields[0]), ], 
                    [("v", fields[1:]), ],
                    time=t,
                    step=step)
        fields = stepper(fields, t, dt, rhs)

if __name__ == "__main__":
    import cProfile as profile
    #profile.run("main()", "wave2d.prof")
    main()


