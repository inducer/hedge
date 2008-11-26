# Hedge - the Hybrid'n'Easy DG Environment
# Copyright (C) 2007 Andreas Kloeckner
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.




from __future__ import division
import numpy
import numpy.linalg as la




def make_pml_mesh(a=(0,0), b=(1,1), max_area=None, 
        boundary_tagger=(lambda fvi, el, fn: []),
        pml_width=0.1,
        periodicity=None, subdivisions=None,
        refine_func=None):
    """Create an unstructured rectangular mesh.

    @arg a: the lower left hand point of the rectangle
    @arg b: the upper right hand point of the rectangle
    @arg max_area: maximum area of each triangle.
    @arg periodicity: either None, or a tuple of bools specifying whether
      the mesh is to be periodic in x and y.
    @arg subdivisions: If not C{None}, this is a 2-tuple specifying
      the number of facet subdivisions in X and Y.
    @arg refine_func: A refinement function as taken by C{meshpy.triangle.build}.
    """
    a = numpy.array(a)
    b = numpy.array(b)

    pml_a = a + pml_width
    pml_b = b - pml_width


    def round_trip_connect(start, end):
        for i in range(start, end):
            yield i, i+1
        yield end, start

    if max_area is not None:
        if refine_func is not None:
            raise ValueError, "cannot specify both refine_func and max_area"
        def refine_func(vertices, area):
            return area > max_area

    marker2tag = {
            1: "minus_x", 
            2: "minus_y", 
            3: "plus_x", 
            4: "plus_y", 
            }

    points = [
            a, (b[0],a[1]), b, (a[0],b[1]),
            pml_a, (pml_b[0],pml_a[1]), pml_b, (pml_a[0],pml_b[1]),
            ]
    facets = list(round_trip_connect(0, 3)) + list(round_trip_connect(4, 7))
    facet_markers = [2,3,4,1]

    if subdivisions is not None:
        from meshpy.triangle import subdivide_facets
        points, facets, facet_markers = subdivide_facets(
                [subdivisions[0], subdivisions[1], 
                    subdivisions[0], subdivisions[1]],
                points, facets, facet_markers)
            
    from hedge.mesh import finish_2d_rect_mesh
    return finish_2d_rect_mesh(points, facets, facet_markers, marker2tag, 
            refine_func, periodicity, boundary_tagger)



def main():
    from hedge.element import TriangularElement
    from hedge.timestep import RK4TimeStepper
    from hedge.mesh import make_disk_mesh
    from hedge.visualization import \
            VtkVisualizer, \
            SiloVisualizer, \
            get_rank_partition
    from pylo import DB_VARTYPE_VECTOR
    from math import sqrt, pi, exp
    from hedge.pde import TEMaxwellOperator, TMMaxwellOperator, GedneyPMLMaxwellOperator
    from hedge.backends import guess_run_context
    from hedge.data import GivenFunction, TimeIntervalGivenFunction

    rcon = guess_run_context()

    epsilon0 = 8.8541878176e-12 # C**2 / (N m**2)
    mu0 = 4*pi*1e-7 # N/A**2.
    epsilon = 1*epsilon0
    mu = 1*mu0

    cylindrical = False
    periodic = False

    pml_width = 0.2
    mesh = make_pml_mesh(a=(-1,-1), b=(1,1), pml_width=pml_width, max_area=0.003)

    if rcon.is_head_rank:
        mesh_data = rcon.distribute_mesh(mesh)
    else:
        mesh_data = rcon.receive_mesh()

    class CurrentSource:
        shape = (3,)

        def __call__(self, x, el):
            sc = numpy.array([1,4,3])[:discr.dimensions]
            return numpy.array([1,1,1])*exp(-20*la.norm(sc*x))

    order = 3
    discr = rcon.make_discretization(mesh_data, order=order)

    vis = VtkVisualizer(discr, rcon, "em-%d" % order, compressor="zlib")
    #vis = SiloVisualizer(discr, rcon

    dt = discr.dt_factor(1/sqrt(mu*epsilon))
    final_time = 12e-9
    nsteps = int(final_time/dt)+1
    dt = final_time/nsteps

    if rcon.is_head_rank:
        print "order %d" % order
        print "dt", dt
        print "nsteps", nsteps
        print "#elements=", len(mesh.elements)

    class PMLTMMaxwellOperator(TMMaxwellOperator, GedneyPMLMaxwellOperator):
        pass
    class PMLTEMaxwellOperator(TEMaxwellOperator, GedneyPMLMaxwellOperator):
        pass

    from hedge.mesh import TAG_ALL, TAG_NONE
    op = PMLTMMaxwellOperator(epsilon, mu, flux_type=1,
            current=TimeIntervalGivenFunction(
                GivenFunction(CurrentSource()), off_time=final_time/10),
            #pec_tag=TAG_NONE,
            #absorb_tag=TAG_ALL,
            )
    fields = op.assemble_fields(discr=discr)

    stepper = RK4TimeStepper()

    # diagnostics setup ---------------------------------------------------
    from pytools.log import LogManager, add_general_quantities, \
            add_simulation_quantities, add_run_info

    logmgr = LogManager("maxwell-%d.dat" % order, "w", rcon.communicator)
    add_run_info(logmgr)
    add_general_quantities(logmgr)
    add_simulation_quantities(logmgr, dt)
    discr.add_instrumentation(logmgr)
    stepper.add_instrumentation(logmgr)

    from pytools.log import IntervalTimer
    vis_timer = IntervalTimer("t_vis", "Time spent visualizing")
    logmgr.add_quantity(vis_timer)

    from hedge.log import EMFieldGetter, add_em_quantities
    field_getter = EMFieldGetter(discr, op, lambda: fields)
    add_em_quantities(logmgr, op, field_getter)
    
    logmgr.add_watches(["step.max", "t_sim.max", "W_field", "t_step.max"])

    # timestep loop -------------------------------------------------------

    t = 0
    rhs = op.bind(discr, sigma=op.sigma_from_width(discr, pml_width))

    vis_step = [0]
    for step in range(nsteps):
        logmgr.tick()

        if step % 5 == 0:
            e, h, d, b = op.split_ehdb(fields)
            visf = vis.make_file("em-%d-%04d" % (order, step))
            vis.add_data(visf, [ 
                ("e", e), 
                ("h", h), 
                ("d", d), 
                ("b", b), 
                #("pml_rhs_e", pml_rhs_e),
                #("pml_rhs_h", pml_rhs_h),
                #("pml_rhs_d", pml_rhs_d),
                #("pml_rhs_b", pml_rhs_b),
                ], time=t, step=step)
            visf.close()

        fields = stepper(fields, t, dt, rhs)
        t += dt

if __name__ == "__main__":
    import cProfile as profile
    #profile.run("main()", "wave2d.prof")
    main()

