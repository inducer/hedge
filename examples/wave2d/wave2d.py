import pylinear.array as num
import pylinear.computation as comp




def main() :
    from hedge.element import TriangularElement
    from hedge.timestep import RK4TimeStepper
    from hedge.mesh import \
            make_disk_mesh, \
            make_regular_square_mesh, \
            make_square_mesh
    from hedge.discretization import \
            Discretization, \
            bind_flux, \
            bind_boundary_flux, \
            bind_nabla, \
            bind_inverse_mass_matrix
    from pytools.arithmetic_container import ArithmeticList
    from pytools.stopwatch import Job
    from math import sin, cos, pi, exp
    from hedge.flux import zero, normal_2d, jump_2d, \
            local, neighbor, average
    from hedge.tools import Rotation, dot

    mesh = make_disk_mesh()
    #mesh = make_regular_square_mesh(n=5)
    #mesh = make_square_mesh(max_area=0.008)
    #mesh.transform(Rotation(pi/8))

    order = 4
    discr = Discretization(mesh, TriangularElement(order))
    print "%d elements" % len(discr.mesh.elements)

    # u, v1, v2
    fields = ArithmeticList([
        discr.volume_zeros(), 
        discr.volume_zeros(), 
        discr.volume_zeros()])

    dt = 0.5*discr.dt_factor(1)
    nsteps = int(1/dt)
    print "dt", dt
    print "nsteps", nsteps

    bc = discr.boundary_zeros()
    flux_weak = average*normal_2d
    flux_strong = local*normal_2d - flux_weak

    nabla = bind_nabla(discr)
    m_inv = bind_inverse_mass_matrix(discr)
    flux = bind_flux(discr, flux_strong)
    bflux = bind_boundary_flux(discr, flux_strong)

    def rhs(t, y):
        u = fields[0]
        v = fields[1:]

        def source_u(x):
            return exp(-x*x*64)

        source_u_vec = discr.interpolate_volume_function(source_u)

        rhs = ArithmeticList([# rhs u
                dot(nabla, v) -m_inv * dot(flux, v) +source_u_vec
                ])
        rhs.extend(# rhs v
            nabla*u -m_inv * (flux * u + bflux * (u,bc))
            )
        return rhs

    stepper = RK4TimeStepper()
    for step in range(nsteps):
        t = step*dt
        print "timestep %d, t=%f" % (step, t)

        if step % 4 == 0:
            discr.visualize_vtk("fld-%04d.vtk" % step,
                    [("u", fields[0]), ], 
                    [("v", zip(*fields[1:])), ]
                    )
        fields = stepper(fields, t, dt, rhs)

if __name__ == "__main__":
    import cProfile as profile
    #profile.run("main()", "wave2d.prof")
    main()

