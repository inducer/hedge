import pylinear.array as num
import pylinear.computation as comp




def main() :
    from hedge.element import TriangularElement
    from hedge.timestep import RK4TimeStepper
    from hedge.mesh import \
            make_disk_mesh, \
            make_regular_square_mesh, \
            make_single_element_mesh
    from hedge.discretization import \
            Discretization, \
            generate_ones_on_boundary
    from pytools.arithmetic_container import ArithmeticList
    from pytools.stopwatch import Job
    from math import sin, cos

    a = num.array([1,0])
    #def u_analytic(t, x):
        #return sin(4*(a*x+t))
    #def u_analytic(t, x):
        #return 0.1*(a*x+t)
    def f(x):
        if x < 0.5:
            return 0
        else:
            #return sin(x-0.5)**2
            return (x-0.5)

    def u_analytic(t, x):
        return f(a*x+t)

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

    #mesh = make_regular_square_mesh(boundary_tagger=boundary_tagger_square, n=2)
    mesh = make_single_element_mesh(boundary_tagger=boundary_tagger_square)

    discr = Discretization(mesh, TriangularElement(8))
    print "%d elements" % len(discr.mesh.elements)

    #discr.visualize_vtk("bdry.vtk",
            #[("outflow", generate_ones_on_boundary(discr, "outflow")), 
                #("inflow", generate_ones_on_boundary(discr, "inflow"))])
    #return 

    u = discr.interpolate_volume_function(lambda x: u_analytic(0, x))

    dt = 1e-2
    nsteps = int(1/dt)

    class CentralWeak:
        def __init__(self, coordinate):
            self.coordinate = coordinate
        def local_coeff(self, normal):
            return 0.5*normal[self.coordinate]
        def neighbor_coeff(self, normal):
            return 0.5*normal[self.coordinate]

    class CentralStrong:
        def __init__(self, coordinate):
            self.coordinate = coordinate
        def local_coeff(self, normal):
            return 0.5*normal[self.coordinate]
        def neighbor_coeff(self, normal):
            return -0.5*normal[self.coordinate]

    rhscnt = [0]

    def rhs_strong(t, u):
        from pytools import argmax

        central_nx = CentralStrong(0)
        central_ny = CentralStrong(1)

        bc = discr.interpolate_boundary_function("inflow",
                lambda x: u_analytic(t, x))

        rhsint =   a[0]*discr.differentiate(0, u)
                #+ a[1]*discr.differentiate(1, u)
        rhsflux =-  a[0]*discr.lift_interior_flux(central_nx, u)
                #-  a[1]*discr.lift_interior_flux(central_ny, u)
        rhsbdry = \
                - a[0]*discr.lift_boundary_flux("inflow",
                        central_nx, u, bc)
                #-  a[1]*discr.lift_boundary_flux(central_ny, u, bc,
                        #"inflow")

        if False:
            maxidx = argmax(rhsflux)
            print "MAXES", max(rhsflux), maxidx, discr.find_face(maxidx)
            raw_input()

        #if False:
        if rhscnt[0] % 1 == 0:
            discr.visualize_vtk("rhs-%04d.vtk" % rhscnt[0],
                    [("u", u),
                        ("int", rhsint), 
                        ("iflux", rhsflux),
                        ("bdry", rhsbdry),
                        ("rhs", rhsint+rhsflux+rhsbdry),
                        ("flux", rhsflux+rhsbdry),
                        ])
        rhscnt[0] += 1

        return rhsint+discr.apply_inverse_mass_matrix(rhsflux+rhsbdry)

    def rhs_weak(t, u):
        from pytools import argmax

        central_nx = CentralWeak(0)
        central_ny = CentralWeak(1)

        bc = discr.interpolate_boundary_function("inflow",
                lambda x: u_analytic(t, x))

        rhsint =   -a[0]*discr.apply_stiffness_matrix_t(0, u)
                #+ a[1]*discr.differentiate(1, u)
        rhsflux =  a[0]*discr.lift_interior_flux(central_nx, u)
                #-  a[1]*discr.lift_interior_flux(central_ny, u)
        rhsbdry = \
                   a[0]*discr.lift_boundary_flux("inflow", 
                           central_nx, u, bc)
                #-  a[1]*discr.lift_boundary_flux(central_ny, u, bc,
                        #"inflow")

        if False:
            maxidx = argmax(rhsflux)
            print "MAXES", max(rhsflux), maxidx, discr.find_face(maxidx)
            raw_input()

        #if False:
        if rhscnt[0] % 1 == 0:
            discr.visualize_vtk("rhs-%04d.vtk" % rhscnt[0],
                    [("u", u),
                        ("int", rhsint), 
                        ("iflux", rhsflux),
                        ("bdry", rhsbdry),
                        ("rhs", rhsint+rhsflux+rhsbdry),
                        ("flux", rhsflux+rhsbdry),
                        ])
        rhscnt[0] += 1

        return rhsint+discr.apply_inverse_mass_matrix(rhsflux+rhsbdry)

    stepper = RK4TimeStepper()
    for step in range(nsteps):
        job = Job("timestep %d" % step)
        u = stepper(u, step*dt, dt, rhs_strong)
        job.done()

        if False:
            job = Job("visualization")
            t = (step+1)*dt
            u_true = discr.interpolate_volume_function(
                    lambda x: u_analytic(t, x))

            discr.visualize_vtk("fld-%04d.vtk" % step,
                    [("u", u), 
                        ("u_true", u_true), 
                        ("rhsu", rhs(t, u)),
                        ], 
                    )
            job.done()

if __name__ == "__main__":
    import cProfile as profile
    #profile.run("main()", "wave2d.prof")
    main()


