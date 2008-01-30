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
import pylinear.array as num
import pylinear.computation as comp




def min_vertex_distance(discr):
    def min_vertex_distance_for_el(el):
        vertices = [discr.mesh.points[vi] 
                for vi in el.vertex_indices]

        return min(min(comp.norm_2(vi-vj)
                for i, vi in enumerate(vertices)
                if i != j)
                for j, vj in enumerate(vertices))

    return min(min_vertex_distance_for_el(el) 
            for el in discr.mesh.elements)




def make_corrugated_rect_mesh(a=(0,0), b=(1,1), max_area=0.1, 
        boundary_tagger=(lambda fvi, el, fn: []),
        n=3):
    import meshpy.triangle as triangle

    def round_trip_connect(start, end):
        for i in range(start, end):
            yield i, i+1
        yield end, start

    def needs_refinement(vert_origin, vert_destination, vert_apex, area):
        return area > max_area

    a = num.asarray(a)
    b = num.asarray(b)
    size = b-a
    x = num.array([1,0])
    y = num.array([0,1])
    xsize = size[0]
    ysize = size[1]

    marker2tag = {
            1: "minus_x", 
            2: "plus_x", 
            3: "bdry", 
            }

    points = [a, num.array([a[0], b[1]])]
    facets = [(0,1)]

    last_lower_idx = 0
    last_upper_idx = 1

    def add_pt(pt):
        result = len(points)
        points.append(pt)
        return result

    for i in range(n):
        last_lower_pt = points[last_lower_idx]
        last_upper_pt = points[last_upper_idx]
        upper_tooth_idx = add_pt(last_upper_pt + x*xsize*0.5 + y*ysize)
        upper_tube_idx = add_pt(last_upper_pt + x*xsize)
        lower_tooth_idx = add_pt(last_lower_pt + x*xsize*0.5 - y*ysize)
        lower_tube_idx = add_pt(last_lower_pt + x*xsize)

        facets.append((last_upper_idx, upper_tooth_idx))
        facets.append((upper_tooth_idx, upper_tube_idx))
        facets.append((last_lower_idx, lower_tooth_idx))
        facets.append((lower_tooth_idx, lower_tube_idx))

        last_lower_idx = lower_tube_idx
        last_upper_idx = upper_tube_idx

    facets.append((last_lower_idx, last_upper_idx))
    facet_markers = [1] + [3] * (len(facets)-2) + [2]
    subdivisions = [5] + [3] * (len(facets)-2) + [5]

    assert len(facets) == len(facet_markers)
    assert len(facets) == len(subdivisions)

    points, facets, facet_markers = triangle.subdivide_facets(
            subdivisions, points, facets, facet_markers)
            
    mesh_info = triangle.MeshInfo()
    mesh_info.set_points(points)
    mesh_info.set_facets(facets, facet_markers)
    
    mesh_periodicity = [("minus_x", "plus_x"), None]
    periodic_tags = set(mesh_periodicity[0])

    generated_mesh = triangle.build(mesh_info, 
            refinement_func=needs_refinement,
            allow_boundary_steiner=False)

    fvi2fm = dict((frozenset(fvi), marker) for fvi, marker in
        zip(generated_mesh.facets, generated_mesh.facet_markers))

    def wrapped_boundary_tagger(fvi, el, fn):
        btag = marker2tag[fvi2fm[frozenset(fvi)]]
        if btag in periodic_tags:
            print el.face_normals[fn], btag, fvi2fm[frozenset(fvi)]
            return [btag]
        else:
            return [btag] + boundary_tagger(fvi, el, fn)

    from hedge.mesh import make_conformal_mesh
    return make_conformal_mesh(
            generated_mesh.points,
            generated_mesh.elements,
            #boundary_tagger,
            wrapped_boundary_tagger,
            periodicity=mesh_periodicity
            )





def main() :
    from hedge.element import \
            TriangularElement, \
            TetrahedralElement
    from hedge.timestep import RK4TimeStepper
    from hedge.discretization import Discretization, ones_on_boundary, integral
    from hedge.visualization import SiloVisualizer, VtkVisualizer
    from hedge.tools import dot, mem_checkpoint
    from pytools.arithmetic_container import ArithmeticList
    from pytools.stopwatch import Job
    from math import sin, cos, pi, sqrt
    from hedge.parallel import \
            guess_parallelization_context, \
            reassemble_volume_field
    from math import floor

    def f(x):
        return sin(pi*x)
        #if int(floor(x)) % 2 == 0:
            #return 1
        #else:
            #return 0

    def u_analytic(x, t):
        return f((a*x/norm_a+t*norm_a))

    def boundary_tagger(vertices, el, face_nr):
        if el.face_normals[face_nr] * a > 0:
            return ["inflow"]
        else:
            return ["outflow"]

    pcon = guess_parallelization_context()

    dim = 2

    job = Job("mesh")
    if dim == 2:
        a = num.array([-1,0])
        if pcon.is_head_rank:
            from hedge.mesh import \
                    make_disk_mesh, \
                    make_square_mesh, \
                    make_rect_mesh, \
                    make_regular_square_mesh, \
                    make_regular_rect_mesh, \
                    make_single_element_mesh
        
            #mesh = make_square_mesh(max_area=0.0003, boundary_tagger=boundary_tagger)
            #mesh = make_regular_square_mesh(a=-r, b=r, boundary_tagger=boundary_tagger, n=3)
            #mesh = make_single_element_mesh(boundary_tagger=boundary_tagger)
            #mesh = make_disk_mesh(r=pi, boundary_tagger=boundary_tagger, max_area=0.5)
            #mesh = make_disk_mesh(boundary_tagger=boundary_tagger)
            
            if False:
                mesh = make_regular_rect_mesh(
                        (-0.5, -1.5),
                        (5, 1.5),
                        n=(10,5),
                        boundary_tagger=boundary_tagger,
                        periodicity=(True, False),
                        )
            if False:
                mesh = make_rect_mesh(
                        (-0.5, -1.5),
                        (5, 1.5),
                        max_area=0.3,
                        boundary_tagger=boundary_tagger,
                        periodicity=(True, False),
                        subdivisions=(10,5),
                        )
            mesh = make_corrugated_rect_mesh(
                    (-0.5, -0.5),
                    (0.5, 0.5),
                    boundary_tagger=boundary_tagger,
                    max_area=0.08)

        el_class = TriangularElement
    elif dim == 3:
        a = num.array([0,0,0.3])
        if pcon.is_head_rank:
            from hedge.mesh import make_cylinder_mesh, make_ball_mesh, make_box_mesh

            mesh = make_cylinder_mesh(max_volume=0.004, boundary_tagger=boundary_tagger,
                    periodic=False, radial_subdivisions=32)
            #mesh = make_box_mesh(dimensions=(1,1,2*pi/3), max_volume=0.01,
                    #boundary_tagger=boundary_tagger)
            #mesh = make_box_mesh(max_volume=0.01, boundary_tagger=boundary_tagger)
            #mesh = make_ball_mesh(boundary_tagger=boundary_tagger)
            #mesh = make_cylinder_mesh(max_volume=0.01, boundary_tagger=boundary_tagger)
        el_class = TetrahedralElement
    else:
        raise RuntimeError, "bad number of dimensions"

    norm_a = comp.norm_2(a)

    if pcon.is_head_rank:
        mesh_data = pcon.distribute_mesh(mesh)
    else:
        mesh_data = pcon.receive_mesh()
    job.done()

    job = Job("discretization")
    mesh_data = mesh_data.reordered_by("cuthill")
    discr = pcon.make_discretization(mesh_data, el_class(7))
    vis_discr = discr
    job.done()

    dt = discr.dt_factor(norm_a)
    nsteps = int(700/dt)

    if pcon.is_head_rank:
        print "%d elements, dt=%g, nsteps=%d" % (
                len(discr.mesh.elements),
                dt,
                nsteps)

    vis = SiloVisualizer(vis_discr, pcon)
    #vis = VtkVisualizer(vis_discr, pcon, "fld")

    # operator setup ----------------------------------------------------------
    from hedge.data import \
            ConstantGivenFunction, \
            TimeConstantGivenFunction, \
            TimeDependentGivenFunction
    from hedge.operators import StrongAdvectionOperator, WeakAdvectionOperator
    op = StrongAdvectionOperator(discr, a, 
            inflow_u=TimeConstantGivenFunction(ConstantGivenFunction()),
            #inflow_u=TimeDependentGivenFunction(u_analytic)),
            flux_type="lf")

    #from sizer import scanner
    #objs = scanner.Objects()
    #import code
    #code.interact(local = {'objs': objs})

    from pyrticle._internal import ShapeFunction
    sf = ShapeFunction(0.5*min_vertex_distance(discr), 2)

    def gauss_hump(x):
        from math import exp
        rsquared = (x*x)/(0.1**2)
        return exp(-rsquared)
    def gauss2_hump(x):
        from math import exp
        rsquared = (x*x)/(0.1**2)
        return exp(-rsquared)-0.5*exp(-rsquared/2)

    #u = discr.interpolate_volume_function(lambda x: u_analytic(x, 0))
    u = discr.interpolate_volume_function(sf)
    u /= integral(discr, u)

    stepper = RK4TimeStepper()

    # diagnostics setup -------------------------------------------------------
    from pytools.log import LogManager, \
            add_general_quantities, \
            add_simulation_quantities, \
            add_run_info

    logmgr = LogManager("advection.dat", pcon.communicator)
    add_run_info(logmgr)
    add_general_quantities(logmgr)
    add_simulation_quantities(logmgr, dt)
    discr.add_instrumentation(logmgr)

    from pytools.log import IntervalTimer
    vis_timer = IntervalTimer("t_vis", "Time spent visualizing")
    logmgr.add_quantity(vis_timer)
    stepper.add_instrumentation(logmgr)

    from hedge.log import Integral, L1Norm, L2Norm, VariableGetter
    logmgr.add_quantity(Integral(VariableGetter(locals(), "u"), discr))
    logmgr.add_quantity(L1Norm(VariableGetter(locals(), "u"), discr))
    logmgr.add_quantity(L2Norm(VariableGetter(locals(), "u"), discr))

    logmgr.add_watches(["step.max", "t_sim.max", "l2_u", "t_step.max"])

    # timestep loop -----------------------------------------------------------
    for step in range(nsteps):
        logmgr.tick()

        t = step*dt

        if step % 5 == 0:
            vis_timer.start()
            visf = vis.make_file("fld-%04d" % step)
            vis.add_data(visf, [
                        ("u", u), 
                        #("u_true", u_true), 
                        ], 
                        #expressions=[("error", "u-u_true")]
                        time=t, 
                        step=step
                        )
            visf.close()
            vis_timer.stop()

        u = stepper(u, t, dt, op.rhs)

        #u_true = discr.interpolate_volume_function(
                #lambda x: u_analytic(t, x))

    vis.close()

    logmgr.tick()
    logmgr.save()


if __name__ == "__main__":
    #import cProfile as profile
    #profile.run("main()", "advec.prof")
    main()


