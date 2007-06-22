import pylinear.array as num
import pylinear.computation as comp




def dot(x, y): 
    from operator import add
    return reduce(add, (xi*yi for xi, yi in zip(x,y)))




def main() :
    from hedge.element import TriangularElement
    from hedge.timestep import RK4TimeStepper
    from hedge.mesh import \
            make_disk_mesh, \
            make_square_mesh, \
            make_regular_square_mesh, \
            make_single_element_mesh
    from hedge.discretization import Discretization, generate_ones_on_boundary
    from hedge.flux import zero, trace_sign, \
            if_bc_equals, normal_2d, jump_2d, \
            local, neighbor, average
    from pytools.arithmetic_container import ArithmeticList
    from pytools.stopwatch import Job
    from math import sin, cos,pi

    a = num.array([1,0])

    def u_analytic(t, x):
        return sin(a*x+t)

    def boundary_tagger_circle(vertices, (v1, v2)):
        center = (num.array(vertices[v1])+num.array(vertices[v2]))/2
        
        if center * a > 0:
            return "inflow"
        else:
            return "outflow"

    def boundary_tagger_square(vertices, (v1, v2)):
        p1 = num.array(vertices[v1])
        p2 = num.array(vertices[v2])
        
        if abs((p1-p2) * a) < 1e-4 and p1[0]>0.45:
            return "inflow"
        else:
            return "outflow"


    #mesh = make_square_mesh(boundary_tagger=boundary_tagger_square, max_area=0.1)
    #mesh = make_square_mesh(boundary_tagger=boundary_tagger_square, max_area=0.2)
    #mesh = make_regular_square_mesh(boundary_tagger=boundary_tagger_square, n=3)
    #mesh = make_single_element_mesh(boundary_tagger=boundary_tagger_square)
    mesh = make_disk_mesh(r=pi, boundary_tagger=boundary_tagger_circle, max_area=0.5)
    discr = Discretization(mesh, TriangularElement(2))

    print "%d elements" % len(discr.mesh.elements)

    #discr.visualize_vtk("bdry.vtk",
            #[("outflow", generate_ones_on_boundary(discr, "outflow")), 
                #("inflow", generate_ones_on_boundary(discr, "inflow"))])
    #return 

    u = discr.interpolate_volume_function(lambda x: u_analytic(0, x))

    dt = 1e-2
    stepfactor = 1
    nsteps = int(2/dt)

    flux_weak = dot(normal_2d, a) * average - 0.5 *(local-neighbor)
    flux_strong = dot(normal_2d, a)*local - flux_weak

    def rhs_strong(t, u):
        from pytools import argmax

        bc_in = discr.interpolate_boundary_function(
                lambda x: u_analytic(t, x),
                "inflow")
        rhsint =   a[0]*discr.differentiate(0, u)
                #+ a[1]*discr.differentiate(1, u)
        rhsflux = discr.lift_interior_flux(flux_strong, u)
        rhsbdry = discr.lift_boundary_flux(flux_strong, u, bc_in, "inflow")

        return rhsint-discr.apply_inverse_mass_matrix(rhsflux+rhsbdry)

    def rhs_weak(t, u):
        from pytools import argmax

        bc_in = discr.interpolate_boundary_function(
                lambda x: u_analytic(t, x),
                "inflow")

        bc_out = discr.boundarize_volume_field(u, "outflow")

        rhsint =   a[0]*discr.apply_stiffness_matrix_t(0, u)
                #+ a[1]*discr.apply_stiffness_matrix_t(1, u)
        rhsflux = discr.lift_interior_flux(flux_weak, u)
        rhsbdry = discr.lift_boundary_flux(flux_weak, u, bc_in, "inflow") + \
                discr.lift_boundary_flux(flux_weak, u, bc_out, "outflow")

        return -rhsint+discr.apply_inverse_mass_matrix(rhsflux+rhsbdry)

    stepper = RK4TimeStepper()
    for step in range(nsteps):
        if step % stepfactor == 0:
            print "timestep %d, t=%f" % (step, dt*step)
        u = stepper(u, step*dt, dt, rhs_weak)

        if step % stepfactor == 0:
            job = Job("visualization")
            t = (step+1)*dt
            u_true = discr.interpolate_volume_function(
                    lambda x: u_analytic(t, x))

            discr.visualize_vtk("fld-%04d.vtk" % step,
                    [("u", u), 
                        ("error", u_true-u), 
                        ], 
                    )
            job.done()

if __name__ == "__main__":
    import cProfile as profile
    main()


