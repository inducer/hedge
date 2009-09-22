# -*- coding: utf8 -*-
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




def test_simp_orthogonality():
    """Test orthogonality of simplicial bases using Grundmann-Moeller cubature"""
    from hedge.quadrature import SimplexCubature
    from hedge.element import TriangularElement, TetrahedralElement

    from numpy import dot

    for order, ebound in [
            (1, 2e-15),
            (2, 5e-15),
            (3, 1e-14),
            #(4, 3e-14),
            #(7, 3e-14),
            #(9, 2e-13),
            ]:
        for ldis in [TriangularElement(order), TetrahedralElement(order)]:
            cub = SimplexCubature(order, ldis.dimensions)
            basis = ldis.basis_functions()

            maxerr = 0
            for i, f in enumerate(basis):
                for j, g in enumerate(basis):
                    if i == j:
                        true_result = 1
                    else:
                        true_result = 0
                    result = cub(lambda x: f(x)*g(x))
                    err = abs(result-true_result)
                    maxerr = max(maxerr, err)
                    if err > ebound:
                        print "bad", order,i,j, err
                    assert err < ebound
            #print order, maxerr




def test_ab_coefficients():
    """Check that our AB coefficient generator reproduces known values."""
    _ABCoefficients = [
            # from R. Verfuerth,
            # "Skript Numerische Behandlung von Differentialgleichungen"
            None,
            [1],
            [3/2, -1/2],
            [23/12, -16/12, 5/12],
            [55/24, -59/24, 37/24, -9/24],
            [1901/720, -2774/720, 2616/720, -1274/720, 251/720]
            ]

    from hedge.timestep.ab import make_ab_coefficients
    for order in range(1,len(_ABCoefficients)):
        assert la.norm(make_ab_coefficients(order)
                - numpy.array(_ABCoefficients[order])) < 5e-14




from math import sqrt, log, sin, cos, exp
from hedge.tools import EOCRecorder
class CheckMultirateTimesteperAccuracy:
    """Check that the multirate timestepper has the advertised accuracy
    """
    def __init__(self,
            method,
            order,
            step_ratio,
            outfile,
            ode):

        self.method      = method
        self.order       = order
        self.step_ratio  = step_ratio
        self.out = outfile
        self.ode         = ode()

    def get_error(self, stepper, dt, name=None):
        t = self.ode.t_start
        y = self.ode.initial_values
        final_t = self.ode.t_end

        nsteps = int((final_t-t)/dt)

        if name is not None:
            outf = open(name, "w")
            format = "%g\t%g\t" + "%g\t" * len(y)

        times = []
        hist = []
        for i in range(nsteps):
            y = stepper(y, t, (self.ode.f2f_rhs, self.ode.s2f_rhs, self.ode.f2s_rhs, self.ode.s2s_rhs))
            t += dt
            hist.append(y)

        if False:
            times.append(t)
            from matplotlib.pyplot import plot, show
            plot(times, [h[0] for h in hist], "o", hold=True)
            plot(times, [self.soln_0(t) for t in times], hold=True)
            show()

        from ode_systems import Basic, Tria

        if isinstance(self.ode, Basic) or isinstance(self.ode, Tria):
            return abs(y[0]-self.ode.soln_0(t))
        else:
            return abs(
                    sqrt(y[0]**2 + y[1]**2)
                    - sqrt(self.ode.soln_0(t)**2 + self.ode.soln_1(t)**2)
                    )

    def __call__(self):
        import fpformat as fpf
        eocrec = EOCRecorder()
        for n in range(4,9):
            dt = 2**(-n)

            from hedge.timestep.multirate_ab import \
                     TwoRateAdamsBashforthTimeStepper

            stepper = TwoRateAdamsBashforthTimeStepper(
                    self.method, dt, self.step_ratio, self.order)

            error = self.get_error(stepper, dt, "mrab-%d.dat" % self.order)
            self.out.write("& %s" % fpf.sci(error,2))
            eocrec.add_data_point(1/dt, error)

        print "------------------------------------------------------"
        print "ORDER %d" % self.order
        print "------------------------------------------------------"
        print eocrec.pretty_print()

        orderest = eocrec.estimate_order_of_convergence()[0,1]
        print orderest, self.order
        #assert orderest > order*0.80

        #self.out.write("& %s" % fpf.fix(orderest,2))



def test_multirate_timestep_accuracy():
    """Check that the multirate timestepper has the advertised accuracy"""

    from hedge.timestep.multirate_ab.methods import methods

    from ode_systems import Basic, \
             Full, \
             Real, \
             Comp, \
             CC,\
             Tria, \
             Inh, \
             Inh2, \
             StiffUncoupled, \
             NonStiffUncoupled, \
             WeakCoupled, \
             StrongCoupled, \
             ExtForceNonStiff, \
             ExtForceStiff, \
             StiffCoupled2, \
             StiffComp, \
             StiffComp2, \
             StiffOscil,\
             WeakCoupledInit

    min_order = 1
    max_order = 6
    step_ratio = 10
    ode_arg_set = [Basic,Full,Real,Comp,CC,
                Tria,Inh,Inh2]
    for ode_arg in ode_arg_set:
        order_list =  range(min_order, max_order)

        for method in methods:

        # outputfile setup: ---------------------------------------------
        #outfilename = "mrab-out/mrab-EOC-%s.tex" % str(method)
        #outfile = open(outfilename, "w")
        #outfile.write("\\begin{tabular}{l")
        #for i in order_list:
        #    outfile.write("c")
        #outfile.write("}" + "\n")
        #outfile.write("N & h & h/2 & h/4 & Rate")
        #outfile.write("\\""\\" + "\n")
        #outfile.write("\\hline" + "\n")

        for order in order_list:
            outfile.write("%s" %order)
            checkup = CheckMultirateTimesteperAccuracy(
                    method,
                    order,
                    step_ratio,
                    outfile,
                    ode = ode_arg)
            checkup()
            #outfile.write("\\""\\" + "\n")
            #outfile.write("\\hline" + "\n")

        #outfile.write("\\hline" + "\n")
        #outfile.write("\\end{tabular}" + "\n")




def test_timestep_accuracy():
    """Check that all timesteppers have the advertised accuracy"""
    from math import sqrt, log, sin, cos
    from hedge.tools import EOCRecorder

    def rhs(t, y):
        u = y[0]
        v = y[1]
        return numpy.array([v, -u/t**2], dtype=numpy.float64)

    def soln(t):
        inner = sqrt(3)/2*log(t)
        return sqrt(t)*(
                5*sqrt(3)/3*sin(inner)
                + cos(inner)
                )

    def get_error(stepper, dt):
        t = 1
        y = numpy.array([1, 3], dtype=numpy.float64)
        final_t = 10
        nsteps = int((final_t-t)/dt)

        hist = []
        for i in range(nsteps):
            y = stepper(y, t, dt, rhs)
            t += dt
            hist.append(y)

        return abs(y[0]-soln(t))

    def verify_timestep_order(stepper_getter, order):
        eocrec = EOCRecorder()
        for n in range(4,9):
            dt = 2**(-n)
            stepper = stepper_getter()
            error = get_error(stepper,dt)
            eocrec.add_data_point(1/dt, error)

        #print stepper
        #print "------------------------------------------------------"
        #print "ORDER %d" % o
        #print "------------------------------------------------------"
        #print eocrec.pretty_print()

        orderest = eocrec.estimate_order_of_convergence()[0,1]
        #print orderest, order
        assert orderest > order*0.95

    from hedge.timestep.rk4 import RK4TimeStepper
    from hedge.timestep.ab import AdamsBashforthTimeStepper

    for o in range(1,5):
        verify_timestep_order(lambda : AdamsBashforthTimeStepper(o), o)
    verify_timestep_order(RK4TimeStepper, 4)





def test_face_vertex_order():
    """Verify that face_indices() emits face vertex indices in the right order"""
    from hedge.element import \
            IntervalElement, \
            TriangularElement, \
            TetrahedralElement

    for el in [
            IntervalElement(5),
            TriangularElement(5),
            TetrahedralElement(5)]:
        vertex_indices = el.vertex_indices()
        for fn, (face_vertices, face_indices) in enumerate(zip(
                el.geometry.face_vertices(vertex_indices),
                el.face_indices())):
            face_vertices_i = 0
            for fi in face_indices:
                if fi == face_vertices[face_vertices_i]:
                    face_vertices_i += 1

            assert face_vertices_i == len(face_vertices)





def test_newton_interpolation():
    """Verify Newton interpolation"""
    from hedge.interpolation import newton_interpolation_function

    x = [-1.5, -0.75, 0, 0.75, 1.5]
    y = [-14.1014, -0.931596, 0, 0.931596, 14.1014]
    nf = newton_interpolation_function(x, y)

    errors = [abs(yi-nf(xi)) for xi, yi in zip(x, y)]
    #print errors
    assert sum(errors) < 1e-14





def test_orthonormality_jacobi_1d():
    """Verify that the Jacobi polymials are orthogonal in 1D"""
    from hedge.polynomial import JacobiFunction
    from hedge.quadrature import LegendreGaussQuadrature

    max_n = 10
    int = LegendreGaussQuadrature(4*max_n) # overkill...

    class WeightFunction:
        def __init__(self, alpha, beta):
            self.alpha = alpha
            self.beta = beta

        def __call__(self, x):
            return (1-x)**self.alpha * (1+x)**self.beta

    for alpha, beta, ebound in [
            (0, 0, 5e-14),
            (1, 0, 4e-14),
            (3, 2, 3e-14),
            (0, 2, 3e-13),
            (5, 0, 3e-13),
            (3, 4, 1e-14)
            ]:
        jac_f = [JacobiFunction(alpha, beta, n) for n in range(max_n)]
        wf = WeightFunction(alpha, beta)
        maxerr = 0

        for i, fi in enumerate(jac_f):
            for j, fj in enumerate(jac_f):
                result = int(lambda x: wf(x)*fi(x)*fj(x))

                if i == j:
                    true_result = 1
                else:
                    true_result = 0
                err = abs(result-true_result)
                maxerr = max(maxerr, err)
                if abs(result-true_result) > ebound:
                    print "bad", alpha, beta, i, j, abs(result-true_result)
                assert abs(result-true_result) < ebound
        #print alpha, beta, maxerr





def test_transformed_quadrature():
    """Test 1D quadrature on arbitrary intervals"""
    from math import exp, sqrt, pi

    def gaussian_density(x, mu, sigma):
        return 1/(sigma*sqrt(2*pi))*exp(-(x-mu)**2/(2*sigma**2))

    from hedge.quadrature import LegendreGaussQuadrature, TransformedQuadrature

    mu = 17
    sigma = 12
    tq = TransformedQuadrature(LegendreGaussQuadrature(20), mu-6*sigma, mu+6*sigma)

    result = tq(lambda x: gaussian_density(x, mu, sigma))
    assert abs(result - 1) < 1e-9




def test_warp():
    """Check some assumptions on the node warp factor calculator"""
    n = 17
    from hedge.element import WarpFactorCalculator
    wfc = WarpFactorCalculator(n)

    assert abs(wfc.int_f(-1)) < 1e-12
    assert abs(wfc.int_f(1)) < 1e-12

    from hedge.quadrature import LegendreGaussQuadrature

    lgq = LegendreGaussQuadrature(n)
    assert abs(lgq(wfc)) < 6e-14




def test_simp_nodes():
    """Verify basic assumptions on simplex interpolation nodes"""
    from hedge.element import \
            IntervalElement, \
            TriangularElement, \
            TetrahedralElement

    els = [
            IntervalElement(19),
            TriangularElement(8),
            TriangularElement(17),
            TetrahedralElement(13)]

    for el in els:
        eps = 1e-10

        unodes = list(el.unit_nodes())
        assert len(unodes) == el.node_count()
        for ux in unodes:
            for uc in ux:
                assert uc >= -1-eps
            assert sum(ux) <= 1+eps

        try:
            equnodes = list(el.equidistant_unit_nodes())
        except AttributeError:
            assert isinstance(el, IntervalElement)
        else:
            assert len(equnodes) == el.node_count()
            for ux in equnodes:
                for uc in ux:
                    assert uc >= -1-eps
                assert sum(ux) <= 1+eps

        for indices in el.node_tuples():
            for index in indices:
                assert index >= 0
            assert sum(indices) <= el.order




def test_tri_nodes_against_known_values():
    """Check triangle nodes against a previous implementation"""
    from hedge.element import TriangularElement, TetrahedralElement
    triorder = 8
    tri = TriangularElement(triorder)

    def tri_equilateral_nodes_reference(self):
        # This is the old, more explicit, less general way of computing
        # the triangle nodes. Below, we compare its results with that of the
        # new routine.

        alpha_opt = [0.0000, 0.0000, 1.4152, 0.1001, 0.2751, 0.9800, 1.0999,
                1.2832, 1.3648, 1.4773, 1.4959, 1.5743, 1.5770, 1.6223, 1.6258]

        try:
            alpha = alpha_opt[self.order-1]
        except IndexError:
            alpha = 5/3

        from hedge.element import WarpFactorCalculator
        from math import sin, cos, pi

        warp = WarpFactorCalculator(self.order)

        edge1dir = numpy.array([1,0])
        edge2dir = numpy.array([cos(2*pi/3), sin(2*pi/3)])
        edge3dir = numpy.array([cos(4*pi/3), sin(4*pi/3)])

        for bary in self.equidistant_barycentric_nodes():
            lambda1, lambda2, lambda3 = bary

            # find equidistant (x,y) coordinates in equilateral triangle
            point = self.barycentric_to_equilateral(bary)

            # compute blend factors
            blend1 = 4*lambda1*lambda2 # nonzero on AB
            blend2 = 4*lambda3*lambda2 # nonzero on BC
            blend3 = 4*lambda3*lambda1 # nonzero on AC

            # calculate amount of warp for each node, for each edge
            warp1 = blend1*warp(lambda2 - lambda1)*(1 + (alpha*lambda3)**2)
            warp2 = blend2*warp(lambda3 - lambda2)*(1 + (alpha*lambda1)**2)
            warp3 = blend3*warp(lambda1 - lambda3)*(1 + (alpha*lambda2)**2)

            # return warped point
            yield point + warp1*edge1dir + warp2*edge2dir + warp3*edge3dir

    if False:
        outf = open("trinodes1.dat", "w")
        for ux in tri.equilateral_nodes():
            outf.write("%g\t%g\n" % tuple(ux))
        outf = open("trinodes2.dat", "w")
        for ux in tri_equilateral_nodes_reference(tri):
            outf.write("%g\t%g\n" % tuple(ux))

    for n1, n2 in zip(tri.equilateral_nodes(),
            tri_equilateral_nodes_reference(tri)):
        assert la.norm(n1-n2) < 3e-15

    def node_indices_2(order):
        for n in range(0, order+1):
             for m in range(0, order+1-n):
                 yield m,n

    assert set(tri.node_tuples()) == set(node_indices_2(triorder))




def test_simp_basis_grad():
    """Do a simplistic FD-style check on the differentiation matrix"""
    from itertools import izip
    from hedge.element import \
            IntervalElement, \
            TriangularElement, \
            TetrahedralElement
    from random import uniform

    els = [
            (1, IntervalElement(5)),
            (1, TriangularElement(8)),
            (3,TetrahedralElement(7))]

    for err_factor, el in els:
        d = el.dimensions
        for i_bf, (bf, gradbf) in \
                enumerate(izip(el.basis_functions(), el.grad_basis_functions())):
            for i in range(10):
                base = -0.95
                remaining = 1.90
                r = numpy.zeros((d,))
                for i in range(d):
                    rn = uniform(0, remaining)
                    r[i] = base+rn
                    remaining -= rn

                from pytools import wandering_element
                h = 1e-4
                gradbf_v = numpy.array(gradbf(r))
                approx_gradbf_v = numpy.array([
                    (bf(r+h*dir) - bf(r-h*dir))/(2*h)
                    for dir in [numpy.array(dir) for dir in wandering_element(d)]
                    ])
                err = la.norm(approx_gradbf_v-gradbf_v, numpy.Inf)
                #print el.dimensions, el.order, i_bf, err
                assert err < err_factor*h





def test_tri_face_node_distribution():
    """Test whether the nodes on the faces of the triangle are distributed
    according to the same proportions on each face.

    If this is not the case, then reusing the same face mass matrix
    for each face would be invalid.
    """

    from hedge.element import TriangularElement

    tri = TriangularElement(8)
    unodes = tri.unit_nodes()
    projected_face_points = []
    for face_i in tri.face_indices():
        start = unodes[face_i[0]]
        end = unodes[face_i[-1]]
        dir = end-start
        dir /= numpy.dot(dir, dir)
        pfp = numpy.array([numpy.dot(dir, unodes[i]-start) for i in face_i])
        projected_face_points.append(pfp)

    first_points =  projected_face_points[0]
    for points in projected_face_points[1:]:
        error = la.norm(points-first_points, numpy.Inf)
        assert error < 1e-15




def test_simp_face_normals_and_jacobians():
    """Check computed face normals and face jacobians on simplicial elements
    """
    from hedge.element import \
            IntervalElement, \
            TriangularElement, \
            TetrahedralElement
    from hedge.mesh import Triangle
    from numpy import dot

    for el in [
            IntervalElement(3),
            TetrahedralElement(1),
            TriangularElement(4),
            ]:
        for i in range(50):
            geo = el.geometry

            vertices = [numpy.random.randn(el.dimensions)
                    for vi in range(el.dimensions+1)]
            #array = num.array
            #vertices = [array([-1, -1.0, -1.0]), array([1, -1.0, -1.0]), array([-1.0, 1, -1.0]), array([-1.0, -1.0, 1.0])]
            map = geo.get_map_unit_to_global(vertices)

            unodes = el.unit_nodes()
            nodes = [map(v) for v in unodes]

            all_vertex_indices = range(el.dimensions+1)

            for face_i, fvi, normal, jac in \
                    zip(el.face_indices(),
                            geo.face_vertices(all_vertex_indices),
                            *geo.face_normals_and_jacobians(vertices, map)):
                mapped_corners = [vertices[i] for i in fvi]
                mapped_face_basis = [mc-mapped_corners[0] for mc in mapped_corners[1:]]

                # face vertices must be among all face nodes
                close_nodes = 0
                for fi in face_i:
                    face_node = nodes[fi]
                    for mc in mapped_corners:
                        if la.norm(mc-face_node) < 1e-13:
                            close_nodes += 1

                assert close_nodes == len(mapped_corners)

                opp_node = (set(all_vertex_indices) - set(fvi)).__iter__().next()
                mapped_opposite = vertices[opp_node]

                if el.dimensions == 1:
                    true_jac = 1
                elif el.dimensions == 2:
                    true_jac = la.norm(mapped_corners[1]-mapped_corners[0])/2
                elif el.dimensions == 3:
                    from hedge.tools import orthonormalize
                    mapped_face_projection = numpy.array(
                            orthonormalize(mapped_face_basis))
                    projected_corners = (
                            [ numpy.zeros((2,))]
                            + [dot(mapped_face_projection, v) for v in mapped_face_basis])
                    true_jac = abs(Triangle
                            .get_map_unit_to_global(projected_corners)
                            .jacobian())
                else:
                    assert False, "this test does not support %d dimensions yet" % el.dimensions

                #print abs(true_jac-jac)/true_jac
                #print "aft, bef", la.norm(mapped_end-mapped_start),la.norm(end-start)

                assert abs(true_jac - jac)/true_jac < 1e-13
                assert abs(la.norm(normal) - 1) < 1e-13
                for mfbv in mapped_face_basis:
                    assert abs(dot(normal, mfbv)) < 1e-13

                for mc in mapped_corners:
                    assert dot(mapped_opposite-mc, normal) < 0




def test_tri_map():
    """Verify that the mapping and node-building operations maintain triangle vertices"""
    from hedge.element import TriangularElement

    n = 8
    tri = TriangularElement(n)

    node_dict = dict((ituple, idx) for idx, ituple in enumerate(tri.node_tuples()))
    corner_indices = [node_dict[0,0], node_dict[n,0], node_dict[0,n]]
    unodes = tri.unit_nodes()
    corners = [unodes[i] for i in corner_indices]

    for i in range(10):
        vertices = [numpy.random.randn(2) for vi in range(3)]
        map = tri.geometry.get_map_unit_to_global(vertices)
        global_corners = [map(pt) for pt in corners]
        for gc, v in zip(global_corners, vertices):
            assert la.norm(gc-v) < 1e-12




def test_tri_map_jacobian_and_mass_matrix():
    """Verify whether tri map jacobians recover known values of triangle area"""
    from hedge.element import TriangularElement
    from math import sqrt, exp, pi

    for i in range(1,10):
        edata = TriangularElement(i)
        ones = numpy.ones((edata.node_count(),))
        unit_tri_area = 2
        error = la.norm(
            numpy.dot(ones,numpy.dot(edata.mass_matrix(), ones))-unit_tri_area)
        assert error < 1e-14

    for i in range(10):
        vertices = [numpy.random.randn(2) for vi in range(3)]
        map = edata.geometry.get_map_unit_to_global(vertices)
        mat = numpy.zeros((2,2))
        mat[:,0] = (vertices[1] - vertices[0])
        mat[:,1] = (vertices[2] - vertices[0])
        tri_area = abs(la.det(mat)/2)
        tri_area_2 = abs(unit_tri_area*map.jacobian())

        assert abs(tri_area - tri_area_2)/tri_area < 5e-15




def test_affine_map():
    """Check that our cheapo geometry-targeted linear algebra actually works."""
    from hedge.tools import AffineMap
    for d in range(1, 5):
    #for d in [3]:
        for i in range(100):
            a = numpy.random.randn(d, d)+10*numpy.eye(d)
            b = numpy.random.randn(d)

            m = AffineMap(a, b)

            assert abs(m.jacobian() - la.det(a)) < 1e-10
            assert la.norm(m.inverted().matrix - la.inv(a)) < 1e-10*la.norm(a)

            x = numpy.random.randn(d)

            m_inv = m.inverted()

            assert la.norm(x-m_inv(m(x))) < 1e-10




def test_all_periodic_no_boundary():
    """Test that an all-periodic brick has no boundary."""
    from hedge.mesh import make_box_mesh, TAG_ALL

    mesh = make_box_mesh(periodicity=(True,True,True))

    def count(iterable):
        result = 0
        for i in iterable:
            result += 1
        return result

    assert count(mesh.tag_to_boundary[TAG_ALL]) == 0




if __name__ == "__main__":
    #from py.test.cmdline import main
    #main([__file__])
    test_multirate_timestep_accuracy()
