import pylinear.array as num
import pylinear.computation as comp




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
    from pytools.arithmetic_container import ArithmeticList
    from pytools.stopwatch import Job
    from math import sin, cos, pi, exp, sqrt
    from hedge.flux import zero, normal, jump, local, neighbor, average
    from hedge.tools import Rotation, dot

    dim = 3

    if dim == 2:
        mesh = make_disk_mesh()
        #mesh = make_regular_square_mesh(n=5)
        #mesh = make_square_mesh(max_area=0.008)
        #mesh.transform(Rotation(pi/8))
        el_class = TriangularElement
    elif dim == 3:
        mesh = make_ball_mesh(max_volume=0.001)
        el_class = TetrahedralElement
    else:
        raise RuntimeError, "bad number of dimensions"

    discr = Discretization(mesh, el_class(3))
    print "%d elements" % len(discr.mesh.elements)

    fields = ArithmeticList([discr.volume_zeros()]) # u
    fields.extend([discr.volume_zeros() for i in range(discr.dimensions)]) # v

    dt = discr.dt_factor(1)
    nsteps = int(1/dt)
    print "dt", dt
    print "nsteps", nsteps

    normal = normal(discr.dimensions)
    flux_weak = average*normal
    flux_strong = local*normal - flux_weak

    nabla = bind_nabla(discr)
    mass = bind_mass_matrix(discr)
    m_inv = bind_inverse_mass_matrix(discr)

    flux = bind_flux(discr, flux_strong)
    bflux = bind_boundary_flux(discr, flux_strong)

    def source_u(x):
        return exp(-x*x*256)

    source_u_vec = discr.interpolate_volume_function(source_u)

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
    vis = SiloVisualizer(discr)
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

