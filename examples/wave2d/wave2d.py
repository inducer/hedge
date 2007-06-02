import pylinear.array as num
import pylinear.computation as comp




def dot(x, y): 
    return sum(xi*yi for xi, yi in zip(x,y))




def main() :
    from hedge.element import TriangularElement
    from hedge.timestep import RK4TimeStepper
    from hedge.mesh import make_disk_mesh
    from hedge.discretization import Discretization
    from pytools.arithmetic_container import ArithmeticList
    from pytools.stopwatch import Job
    from math import sin, cos

    discr = Discretization(make_disk_mesh(), TriangularElement(4))
    print "%d elements" % len(discr.mesh.elements)

    # u, v1, v2
    fields = ArithmeticList([
        discr.volume_zeros(), 
        discr.volume_zeros(), 
        discr.volume_zeros()])

    dt = 4e-3
    nsteps = int(1/dt)

    class Avg:
        def local_coeff(self, normal):
            return -0.5
        def neighbor_coeff(self, normal):
            return 0.5

    class CentralNX:
        def local_coeff(self, normal):
            return 0.5*normal[0]
        def neighbor_coeff(self, normal):
            return 0.5*normal[0]

    class CentralNY:
        def local_coeff(self, normal):
            return 0.5*normal[1]
        def neighbor_coeff(self, normal):
            return 0.5*normal[1]

    central_nx = CentralNX()
    central_ny = CentralNY()

    bc = discr.boundary_zeros("dirichlet")

    #coe = discr.constant_on_elements()
    #discr.visualize_vtk("flux.vtk", [("flux", 
        #coe+discr.lift_interior_flux(Avg(), coe))])
    #return

    from math import exp

    def rhs(t, y):
        u = fields[0]
        v = fields[1:]

        def source_u(x):
            return exp(-x*x*64)

        source_u_vec = discr.interpolate_volume_function(source_u)

        return ArithmeticList([# rhs u
                -discr.apply_stiffness_matrix_t(0, v[0])
                -discr.apply_stiffness_matrix_t(1, v[1])
                +discr.lift_interior_flux(central_nx, v[0])
                +discr.lift_interior_flux(central_ny, v[1])
                +source_u_vec
                ,
                # rhs v1
                -discr.apply_stiffness_matrix_t(0, u)
                +discr.lift_interior_flux(central_nx, u)
                +discr.lift_boundary_flux(central_nx, u, bc, "dirichlet")
                ,
                # rhs v2
                -discr.apply_stiffness_matrix_t(1, u)
                +discr.lift_interior_flux(central_ny, u)
                +discr.lift_boundary_flux(central_ny, u, bc, "dirichlet")
                ])

    stepper = RK4TimeStepper()
    for step in range(nsteps):
        discr.visualize_vtk("fld-%04d.vtk" % step,
                [("u", fields[0])], 
                [("v", zip(*fields[1:]))]
                )
        job = Job("timestep")
        fields = stepper(fields, step*dt, dt, rhs)
        job.done()

if __name__ == "__main__":
    import cProfile as profile
    #profile.run("main()", "wave2d.prof")
    main()

