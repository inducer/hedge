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




def make_mesh(a, b, pml_width=0.25, **kwargs):
    from meshpy.geometry import GeometryBuilder, make_box
    geob = GeometryBuilder()
    
    box_points, box_facets, _ = make_box(a, b)
    geob.add_geometry(box_points, box_facets)
    geob.wrap_in_box(pml_width)

    mesh_mod = geob.mesher_module()
    mi = mesh_mod.MeshInfo()
    geob.set(mi)

    built_mi = mesh_mod.build(mi, **kwargs)

    print "%d elements" % len(built_mi.elements)

    def boundary_tagger(fvi, el, fn):
        return []


    from hedge.mesh import make_conformal_mesh
    return make_conformal_mesh(
            built_mi.points,
            built_mi.elements, 
            boundary_tagger)




def main():
    from hedge.element import TriangularElement
    from hedge.timestep import RK4TimeStepper
    from hedge.mesh import make_disk_mesh
    from pylo import DB_VARTYPE_VECTOR
    from math import sqrt, pi, exp
    from hedge.pde import \
            MaxwellOperator, \
            GedneyPMLTEMaxwellOperator, \
            GedneyPMLTMMaxwellOperator, \
            GedneyPMLMaxwellOperator, \
            TEMaxwellOperator
    from hedge.backends import guess_run_context, FEAT_CUDA

    rcon = guess_run_context(disable=set([FEAT_CUDA]))

    #epsilon0 = 8.8541878176e-12 # C**2 / (N m**2)
    #mu0 = 4*pi*1e-7 # N/A**2.
    epsilon0 = 1 
    mu0 = 1 
    epsilon = 1*epsilon0
    mu = 1*mu0

    c = 1/sqrt(mu*epsilon)

    cylindrical = False
    periodic = False

    pml_width = 0.5
    mesh = make_mesh(a=numpy.array((-1,-1,-1)), b=numpy.array((1,1,1)), 
    #mesh = make_mesh(a=numpy.array((-1,-1)), b=numpy.array((1,1)), 
            pml_width=pml_width, max_volume=0.03)

    if rcon.is_head_rank:
        mesh_data = rcon.distribute_mesh(mesh)
    else:
        mesh_data = rcon.receive_mesh()

    class CurrentSource:
        shape = (3,)

        def __call__(self, x, el):
            #sc = numpy.array([1,4,3])[:discr.dimensions]
            return numpy.array([1,1,1])*exp(-20*la.norm(x))

    final_time = 20/c
    order = 3
    discr = rcon.make_discretization(mesh_data, order=order)

    from hedge.visualization import VtkVisualizer, SiloVisualizer
    #vis = VtkVisualizer(discr, rcon, "em-%d" % order)
    vis = SiloVisualizer(discr, rcon)

    from hedge.mesh import TAG_ALL, TAG_NONE
    from hedge.data import GivenFunction, TimeHarmonicGivenFunction
    op = GedneyPMLMaxwellOperator(epsilon, mu, flux_type=1,
            current=TimeHarmonicGivenFunction(
                GivenFunction(CurrentSource()),
                omega=2*pi*0.3*sqrt(mu*epsilon)),
            pec_tag=TAG_NONE,
            absorb_tag=TAG_ALL,
            )

    #fields = max_op.assemble_eh(discr=discr)
    fields = op.assemble_ehdb(discr=discr)

    stepper = RK4TimeStepper()

    dt = discr.dt_factor(op.max_eigenvalue())
    nsteps = int(final_time/dt)+1
    dt = final_time/nsteps

    if rcon.is_head_rank:
        print "order %d" % order
        print "dt", dt
        print "nsteps", nsteps
        print "#elements=", len(mesh.elements)

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

    from hedge.log import LpNorm
    class FieldIdxGetter:
        def __init__(self, whole_getter, idx):
            self.whole_getter = whole_getter
            self.idx = idx

        def __call__(self):
            return self.whole_getter()[self.idx]

    #for i in range(len(fields)):
        #logmgr.add_quantity(
                #LpNorm(FieldIdxGetter(lambda: fields, i), discr, name="norm_f%d" % i))
    #for i in range(len(fields)):
        #logmgr.add_quantity(
                #LpNorm(FieldIdxGetter(lambda: rhs(t, fields), i), discr, name="rhs_f%d" % i))

    for i, fi in enumerate(op.op_template()):
        print i, fi
        print 

    # timestep loop -------------------------------------------------------

    t = 0
    rhs = op.bind(discr, sigma=op.sigma_from_width(discr, pml_width))

    vis_step = [0]
    for step in range(nsteps):
        logmgr.tick()
        logmgr.save()

        if step % 1 == 0:
            e, h, d, b = op.split_ehdb(fields)
            visf = vis.make_file("em-%d-%04d" % (order, step))
            #pml_rhs_e, pml_rhs_h, pml_rhs_d, pml_rhs_b = \
                    #op.split_ehdb(rhs(t, fields))
            vis.add_data(visf, [ 
                ("e", e), 
                ("h", h), 
                ("d", d), 
                ("b", b), 
                #("pml_rhs_e", pml_rhs_e),
                #("pml_rhs_h", pml_rhs_h),
                #("pml_rhs_d", pml_rhs_d),
                #("pml_rhs_b", pml_rhs_b),
                #("max_rhs_e", max_rhs_e),
                #("max_rhs_h", max_rhs_h),
                #("sigma", sigma),
                ], time=t, step=step)
            visf.close()

        fields = stepper(fields, t, dt, rhs)
        t += dt

if __name__ == "__main__":
    import cProfile as profile
    #profile.run("main()", "wave2d.prof")
    main()

