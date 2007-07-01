import pylinear.array as num
import pylinear.computation as comp




def dot(x, y): 
    return sum(xi*yi for xi, yi in zip(x,y))




def main() :
    from hedge.element import TriangularElement
    from hedge.timestep import RK4TimeStepper
    from hedge.mesh import \
            make_disk_mesh, \
            make_regular_square_mesh, \
            make_square_mesh
    from hedge.discretization import Discretization
    from pytools.arithmetic_container import ArithmeticList
    from pytools.stopwatch import Job
    from math import sin, cos, pi, exp
    from hedge.flux import zero, normal_2d, jump_2d, \
            local, neighbor, average
    from hedge.tools import Rotation

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

    dt = discr.dt_factor(1)
    nsteps = int(1/dt)
    print "dt", dt
    print "nsteps", nsteps

    bc = discr.boundary_zeros()
    flux_weak = average*normal_2d
    flux_strong = local*normal_2d - flux_weak

    def rhs(t, y):
        u = fields[0]
        v = fields[1:]

        def source_u(x):
            return exp(-x*x*64)

        source_u_vec = discr.interpolate_volume_function(source_u)

        flux = flux_strong

        return ArithmeticList([# rhs u
                 discr.differentiate(0, v[0])
                +discr.differentiate(1, v[1])
                -discr.apply_inverse_mass_matrix(
                    discr.lift_interior_flux(flux[0], v[0])
                    +discr.lift_interior_flux(flux[1], v[1])
                    )
                +source_u_vec
                ,
                # rhs v0
                 discr.differentiate(0, u)
                 -discr.apply_inverse_mass_matrix(
                     discr.lift_interior_flux(flux[0], u)
                     +discr.lift_boundary_flux(flux[0], u, bc)
                     )
                ,
                # rhs v1
                 discr.differentiate(1, u)
                 -discr.apply_inverse_mass_matrix(
                     discr.lift_interior_flux(flux[1], u)
                     +discr.lift_boundary_flux(flux[1], u, bc)
                     )
                ])

    stepper = RK4TimeStepper()
    for step in range(nsteps):
        t = step*dt
        print "timestep %d, t=%f" % (step, t)

        discr.visualize_vtk("fld-%04d.vtk" % step,
                [("u", fields[0]), ], 
                [("v", zip(*fields[1:])), ]
                )
        fields = stepper(fields, t, dt, rhs)

if __name__ == "__main__":
    import cProfile as profile
    profile.run("main()", "wave2d.prof")
    main()

