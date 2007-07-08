import pylinear.array as num
import pylinear.computation as comp




def dot(x, y): 
    from operator import add
    return reduce(add, (xi*yi for xi, yi in zip(x,y)))




def main() :
    from hedge.element import \
            TriangularElement, \
            TetrahedralElement
    from hedge.timestep import RK4TimeStepper
    from hedge.mesh import \
            make_disk_mesh, \
            make_square_mesh, \
            make_regular_square_mesh, \
            make_single_element_mesh, \
            make_ball_mesh
    from hedge.discretization import \
            Discretization, \
            generate_ones_on_boundary, \
            bind_flux, \
            bind_boundary_flux, \
            bind_nabla, \
            bind_weak_nabla, \
            bind_mass_matrix, \
            bind_inverse_mass_matrix, \
            pair_with_boundary
    from hedge.visualization import SiloVisualizer
    from hedge.flux import zero, trace_sign, normal, jump, local, neighbor, average
    from hedge.tools import dot
    from pytools.arithmetic_container import ArithmeticList
    from pytools.stopwatch import Job
    from math import sin, cos, pi, sqrt

    def u_analytic(t, x):
        return sin(3*(a*x+t))

    def boundary_tagger(vertices, el, face_nr):
        if el.face_normals[face_nr] * a > 0:
            return "inflow"
        else:
            return "outflow"

    dim = 3
    if dim == 2:
        a = num.array([1,0])
        #mesh = make_square_mesh(boundary_tagger=boundary_tagger, max_area=0.1)
        #mesh = make_square_mesh(boundary_tagger=boundary_tagger, max_area=0.2)
        #mesh = make_regular_square_mesh(a=-r, b=r, boundary_tagger=boundary_tagger, n=3)
        #mesh = make_single_element_mesh(boundary_tagger=boundary_tagger)
        #mesh = make_disk_mesh(r=pi, boundary_tagger=boundary_tagger, max_area=0.5)
        #mesh = make_disk_mesh(boundary_tagger=boundary_tagger)
        el_class = TriangularElement
    elif dim == 3:
        a = num.array([1,0,0])
        mesh = make_ball_mesh(boundary_tagger=boundary_tagger)
        el_class = TetrahedralElement
    else:
        raise RuntimeError, "bad number of dimensions"

    discr = Discretization(mesh, el_class(3))
    vis = SiloVisualizer(discr)

    print "%d elements" % len(discr.mesh.elements)

    #vis("bdry.vtk",
            #[("outflow", generate_ones_on_boundary(discr, "outflow")), 
                #("inflow", generate_ones_on_boundary(discr, "inflow"))])
    #return 

    u = discr.interpolate_volume_function(lambda x: u_analytic(0, x))

    dt = discr.dt_factor(comp.norm_2(a))
    stepfactor = 1
    nsteps = int(2/dt)

    normal = normal(discr.dimensions)
    jump = jump(discr.dimensions)

    flux_weak = dot(normal, a) * average# - 0.5 *(local-neighbor)
    flux_strong = dot(normal, a)*local - flux_weak

    nabla = bind_nabla(discr)
    weak_nabla = bind_weak_nabla(discr)
    mass = bind_mass_matrix(discr)
    m_inv = bind_inverse_mass_matrix(discr)

    def rhs_strong(t, u):
        from pytools import argmax

        bc_in = discr.interpolate_boundary_function(
                lambda x: u_analytic(t, x),
                "inflow")

        flux = bind_flux(discr, flux_strong)
        bflux = bind_boundary_flux(discr, flux_strong, "inflow")

        return dot(a, nabla*u) - m_inv*(flux*u + bflux * pair_with_boundary(u, bc_in))

    def rhs_weak(t, u):
        from pytools import argmax

        bc_in = discr.interpolate_boundary_function(
                lambda x: u_analytic(t, x),
                "inflow")

        bc_out = discr.boundarize_volume_field(u, "outflow")

        flux = bind_flux(discr, flux_weak)
        b_in_flux = bind_boundary_flux(discr, flux_weak, "inflow")
        b_out_flux = bind_boundary_flux(discr, flux_weak, "outflow")

        return -dot(a, weak_nabla*u) +m_inv*(
                flux*u
                + b_in_flux * pair_with_boundary(u, bc_in)
                + b_out_flux * pair_with_boundary(u, bc_out)
                )

    stepper = RK4TimeStepper()
    for step in range(nsteps):
        if step % stepfactor == 0:
            print "timestep %d, t=%f, l2=%f" % (step, dt*step, sqrt(u*(mass*u)))
        u = stepper(u, step*dt, dt, rhs_strong)

        t = (step+1)*dt
        #u_true = discr.interpolate_volume_function(
                #lambda x: u_analytic(t, x))

        vis("fld-%04d.silo" % step,
                [
                    ("u", u), 
                    #("u_true", u_true), 
                    ], 
                #expressions=[("error", "u-u_true")]
                time=t, 
                step=step
                )

if __name__ == "__main__":
    import cProfile as profile
    main()


