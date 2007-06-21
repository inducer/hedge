import pylinear.array as num
import pylinear.computation as comp




def dot(x, y): 
    from operator import add
    return reduce(add, (xi*yi for xi, yi in zip(x,y)))




def make_my_mesh():
    from hedge.mesh import ConformalMesh
    array = num.array

    points = [
            array([-0.5, -0.5]), 
            array([-0.5, 0.5]), 
            array([0.5, 0.5]), 
            array([0.5, -0.5]), 
            array([0.0, 0.0]), 
            array([-0.5, 0.0]), 
            array([0.0, -0.5]), 
            array([0.5, 0.0]), 
            array([0.0, 0.5])]
    if False:
        elements = [
                [7, 2, 4],
                [5, 0, 4],
                [4, 8, 1],
                [4, 1, 5],
                [4, 0, 6],
                [6, 3, 4],
                [4, 3, 7],
                [4, 2, 8],
                ]
    else:
        elements = [
                [8,7,4],
                [8,7,2],
                [6,7,3],
                [7,4,6],
                [5,6,0],
                [5,6,4],
                [5,8,4],
                [1,5,8],
                ]

    boundary_tags = {
            frozenset([3,7]): "inflow",
            frozenset([2,7]): "inflow",
            frozenset([6,3]): "outflow",
            frozenset([2,8]): "outflow",
            frozenset([1,8]): "outflow",
            frozenset([1,5]): "outflow",
            frozenset([0,5]): "outflow",
            frozenset([0,6]): "outflow",
            frozenset([3,6]): "outflow",
            }
    return ConformalMesh(points, elements, boundary_tags)




def main() :
    from hedge.element import TriangularElement
    from hedge.timestep import RK4TimeStepper
    from hedge.mesh import \
            make_disk_mesh, \
            make_square_mesh, \
            make_regular_square_mesh, \
            make_single_element_mesh
    from hedge.discretization import \
            Discretization, \
            generate_ones_on_boundary, \
            SymmetryMap
    from hedge.flux import zero, trace_sign, \
            if_bc_equals, normal_2d, jump_2d, \
            local, neighbor, average
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


    #mesh = make_square_mesh(boundary_tagger=boundary_tagger_square, max_area=0.1)
    #mesh = make_square_mesh(boundary_tagger=boundary_tagger_square, max_area=0.2)
    #mesh = make_regular_square_mesh(boundary_tagger=boundary_tagger_square, n=3)
    #mesh = make_single_element_mesh(boundary_tagger=boundary_tagger_square)
    mesh = make_my_mesh()

    #print mesh.vertices
    #for el in mesh.elements:
        #print el.vertices

    discr = Discretization(mesh, TriangularElement(4))
    print "%d elements" % len(discr.mesh.elements)

    #discr.visualize_vtk("bdry.vtk",
            #[("outflow", generate_ones_on_boundary(discr, "outflow")), 
                #("inflow", generate_ones_on_boundary(discr, "inflow"))])
    #return 

    sym_map = SymmetryMap(discr, 
            lambda x: num.array([x[0], -x[1]]),
            {0:3, 2:1, 5:6, 7:4})
    discr.sym_map = sym_map

    u = discr.interpolate_volume_function(lambda x: u_analytic(0, x))

    dt = 1e-3
    stepfactor = 100
    nsteps = int(10/dt)

    rhscnt = [0]
    rhsstep = stepfactor

    flux = dot(normal_2d, a) * local \
            - dot(normal_2d, a) * average \
            + 0.5 *(local-neighbor)

    def rhs_strong(t, u):
        from pytools import argmax

        bc = discr.interpolate_boundary_function("inflow",
                lambda x: u_analytic(t, x))

        rhsint =   a[0]*discr.differentiate(0, u)
                #+ a[1]*discr.differentiate(1, u)
        rhsflux = discr.lift_interior_flux(flux, u, rhscnt[0] == 3)
        rhsbdry = discr.lift_boundary_flux("inflow", flux, u, bc)

        if False:
            maxidx = argmax(rhsflux)
            print "MAXES", max(rhsflux), maxidx, discr.find_face(maxidx)
            raw_input()

        mflux = discr.apply_inverse_mass_matrix(rhsflux+rhsbdry)
        #if False:
        if rhscnt[0] % rhsstep == 0 or rhscnt[0] < 10:
            discr.visualize_vtk("rhs-%04d.vtk" % rhscnt[0],
                    [
                        ("u", u),
                        ("se_u", u-sym_map(u)),

                        ("int", rhsint), 
                        ("se_int", rhsint-sym_map(rhsint)),

                        ("iflux", rhsflux),
                        ("se_iflux", rhsflux-sym_map(rhsflux)),

                        ("bdry", rhsbdry),
                        ("se_bdry", rhsbdry-sym_map(rhsbdry)),

                        ("flux", rhsflux+rhsbdry),
                        ("se_flux", rhsflux+rhsbdry-sym_map(rhsflux+rhsbdry)),

                        ("mflux", mflux),
                        ("se_mflux", mflux-sym_map(mflux)),
                        ])
        rhscnt[0] += 1

        return rhsint-discr.apply_inverse_mass_matrix(rhsflux+rhsbdry)

    stepper = RK4TimeStepper()
    for step in range(nsteps):
        if step % 10 == 0:
            print "timestep %d, t=%f" % (step, dt*step)
        u = stepper(u, step*dt, dt, rhs_strong)

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


