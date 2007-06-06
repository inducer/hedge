import pylinear.array as num
import pylinear.computation as comp




def main() :
    from hedge.element import TriangularElement
    from hedge.timestep import RK4TimeStepper
    from hedge.mesh import make_disk_mesh
    from hedge.discretization import \
            Discretization, \
            generate_ones_on_boundary
    from pytools.arithmetic_container import ArithmeticList
    from pytools.stopwatch import Job
    from math import sin, cos

    a = num.array([1,0])
    def u_analytic(t, x):
        return sin(4*(a*x+t))

    def boundary_tagger(vertices, (v1, v2)):
        center = (num.array(vertices[v1])+num.array(vertices[v2]))/2
        
        if center * a > 0:
            return "inflow"
        else:
            return "outflow"

    mesh = make_disk_mesh(boundary_tagger=boundary_tagger)

    discr = Discretization(mesh, TriangularElement(4))
    print "%d elements" % len(discr.mesh.elements)

    #discr.visualize_vtk("bdry.vtk",
            #[("outflow", generate_ones_on_boundary(discr, "outflow")), 
                #("inflow", generate_ones_on_boundary(discr, "inflow"))])
    #return 

    u = discr.interpolate_volume_function(
            lambda x: u_analytic(0, x))

    dt = 1e-2
    nsteps = int(1/dt)

    class CentralNX:
        def local_coeff(self, normal):
            return 0.5*normal[0]
        def neighbor_coeff(self, normal):
            return -0.5*normal[0]

    class CentralNY:
        def local_coeff(self, normal):
            return 0.5*normal[1]
        def neighbor_coeff(self, normal):
            return -0.5*normal[1]

    central_nx = CentralNX()
    central_ny = CentralNY()

    rhscnt = [0]

    def rhs(t, u):
        from pytools import argmax

        bc = discr.interpolate_boundary_function("inflow",
                lambda x: u_analytic(t, x))

        rhsint = +a[0]*discr.differentiate(0, u) \
                + a[1]*discr.differentiate(1, u)
        rhsflux =- a[0]*discr.lift_interior_flux(central_nx, u) \
                -  a[1]*discr.lift_interior_flux(central_ny, u)
        rhsbdry = \
                -  a[0]*discr.lift_boundary_flux(central_nx, u, bc,
                        "inflow") \
                -  a[1]*discr.lift_boundary_flux(central_ny, u, bc,
                        "inflow")

        maxidx = argmax(rhsflux)
        print "MAXES", max(rhsflux), maxidx, discr.find_face(maxidx)
        raw_input()

        if False:
        #if rhscnt[0] % 1 == 0:
            discr.visualize_vtk("rhs-%04d.vtk" % rhscnt[0],
                    [("u", u),
                        ("int", rhsint), 
                        ("flux", rhsflux),
                        ("bdry", rhsbdry),
                        ])
        rhscnt[0] += 1

        return rhsint+rhsflux+rhsbdry

    stepper = RK4TimeStepper()
    for step in range(nsteps):
        job = Job("timestep %d" % step)
        u = stepper(u, step*dt, dt, rhs)
        job.done()

        if False:
            job = Job("visualization")
            t = (step+1)*dt
            u_true = discr.interpolate_volume_function(
                    lambda x: u_analytic(t, x))

            discr.visualize_vtk("fld-%04d.vtk" % step,
                    [("u", u), ("u_true", u_true), ("rhsu", rhs(t, u))], 
                    )
            job.done()

if __name__ == "__main__":
    import cProfile as profile
    #profile.run("main()", "wave2d.prof")
    main()


