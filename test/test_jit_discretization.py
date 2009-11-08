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
from hedge.backends.jit import Discretization as discr_class
from hedge_test_util import Monomial
import pytools.test




def test_1d_mass_mat_trig():
    """Check the integral of some trig functions on an interval using the mass matrix"""
    from hedge.mesh import make_uniform_1d_mesh
    from hedge.element import IntervalElement
    from math import sqrt, pi, cos, sin
    from numpy import dot

    mesh = make_uniform_1d_mesh(-4*pi, 9*pi, 17, periodic=True)
    discr = discr_class(mesh, IntervalElement(8),
            debug=discr_class.noninteractive_debug_flags())

    f = discr.interpolate_volume_function(
            lambda x, el: cos(x[0])**2)
    ones = discr.interpolate_volume_function(
            lambda x, el: 1)

    from hedge.optemplate import MassOperator
    mass_op = MassOperator()

    num_integral_1 = dot(ones, mass_op.apply(discr, f))
    num_integral_2 = dot(f, mass_op.apply(discr, ones))
    num_integral_3 = discr.integral(f)

    true_integral = 13*pi/2
    err_1 = abs(num_integral_1-true_integral)
    err_2 = abs(num_integral_2-true_integral)
    err_3 = abs(num_integral_3-true_integral)

    assert err_1 < 1e-10
    assert err_2 < 1e-10
    assert err_3 < 1e-10





def test_tri_mass_mat_trig():
    """Check the integral of some trig functions on a square using the mass matrix"""

    from hedge.mesh import make_square_mesh
    from hedge.element import TriangularElement
    from math import sqrt, pi, cos, sin

    mesh = make_square_mesh(a=-pi, b=pi, max_area=(2*pi/10)**2/2)
    discr = discr_class(mesh, TriangularElement(8),
            debug=discr_class.noninteractive_debug_flags())

    f = discr.interpolate_volume_function(
            lambda x, el: cos(x[0])**2*sin(x[1])**2)
    ones = discr.interpolate_volume_function(
            lambda x, el: 1)

    from hedge.optemplate import MassOperator
    mass_op = MassOperator()

    num_integral_1 = numpy.dot(ones, mass_op.apply(discr, f))
    num_integral_2 = numpy.dot(f, mass_op.apply(discr, ones))
    true_integral = pi**2
    err_1 = abs(num_integral_1-true_integral)
    err_2 = abs(num_integral_2-true_integral)
    #print err_1, err_2
    assert err_1 < 1e-10
    assert err_2 < 1e-10




def test_tri_diff_mat():
    """Check differentiation matrix along the coordinate axes on a disk

    Uses sines as the function to differentiate.
    """
    from hedge.mesh import make_disk_mesh
    from hedge.element import TriangularElement
    from math import sin, cos, sqrt

    from hedge.optemplate import make_nabla
    nabla = make_nabla(2)

    for coord in [0, 1]:
        mesh = make_disk_mesh()
        discr = discr_class(make_disk_mesh(), TriangularElement(4),
            debug=discr_class.noninteractive_debug_flags())
        f = discr.interpolate_volume_function(
                lambda x, el: sin(3*x[coord]))
        df = discr.interpolate_volume_function(
                lambda x, el: 3*cos(3*x[coord]))

        df_num = nabla[coord].apply(discr, f)
        #discr.visualize_vtk("diff-err.vtk",
                #[("f", f), ("df", df), ("df_num", df_num), ("error", error)])

        linf_error = la.norm(df_num-df, numpy.Inf)
        #print linf_error
        assert linf_error < 3e-5




def test_2d_gauss_theorem():
    """Verify Gauss's theorem explicitly on a mesh"""

    from hedge.element import TriangularElement
    from hedge.mesh import make_disk_mesh
    from math import sin, cos, sqrt, exp, pi
    from numpy import dot

    mesh = make_disk_mesh()
    order = 2

    discr = discr_class(mesh, order=order,
            debug=discr_class.noninteractive_debug_flags())
    ref_discr = discr_class(mesh, order=order)

    from hedge.flux import make_normal, FluxScalarPlaceholder
    from pymbolic.primitives import IfPositive

    normal = make_normal(discr.dimensions)
    flux_f_ph = FluxScalarPlaceholder(0)
    one_sided_x = flux_f_ph.int*normal[0]
    one_sided_y = flux_f_ph.int*normal[1]

    def f1(x, el):
        return sin(3*x[0])+cos(3*x[1])
    def f2(x, el):
        return sin(2*x[0])+cos(x[1])

    from hedge.discretization import ones_on_volume
    ones = ones_on_volume(discr)
    f1_v = discr.interpolate_volume_function(f1)
    f2_v = discr.interpolate_volume_function(f2)

    from hedge.optemplate import BoundaryPair, Field, make_nabla, \
            get_flux_operator
    nabla = make_nabla(discr.dimensions)
    diff_optp = nabla[0] * Field("f1") + nabla[1] * Field("f2")

    divergence = nabla[0].apply(discr, f1_v) + nabla[1].apply(discr, f2_v)
    int_div = discr.integral(divergence)

    flux_optp = (
            get_flux_operator(one_sided_x)
            *BoundaryPair(Field("f1"), Field("fz")) +
            get_flux_operator(one_sided_y)
            *BoundaryPair(Field("f2"), Field("fz")))

    from hedge.mesh import TAG_ALL
    bdry_val = discr.compile(flux_optp)(f1=f1_v, f2=f2_v,
            fz=discr.boundary_zeros(TAG_ALL))
    ref_bdry_val = ref_discr.compile(flux_optp)(f1=f1_v, f2=f2_v,
            fz=discr.boundary_zeros(TAG_ALL))

    boundary_int = dot(bdry_val, ones)

    if False:
        from hedge.visualization import SiloVisualizer
        vis = SiloVisualizer(discr)
        visf = vis.make_file("test")

        from hedge.tools import make_obj_array
        from hedge.mesh import TAG_ALL
        vis.add_data(visf, [
            ("bdry", bdry_val),
            ("ref_bdry", ref_bdry_val),
            ("div", divergence),
            ("f", make_obj_array([f1_v, f2_v])),
            ("n", discr.volumize_boundary_field(
                discr.boundary_normals(TAG_ALL), TAG_ALL)),
            ],
            expressions=[("bdiff", "bdry-ref_bdry")])

        #print abs(boundary_int-int_div)

    assert abs(boundary_int-int_div) < 5e-15




def test_simp_cubature():
    """Check that Grundmann-Moeller cubature works as advertised"""
    from pytools import generate_nonnegative_integer_tuples_summing_to_at_most
    from hedge.quadrature import SimplexCubature

    for dim in range(2,3+1):
        for s in range(3+1):
            cub = SimplexCubature(s, dim)
            for comb in generate_nonnegative_integer_tuples_summing_to_at_most(
                    2*s+1, dim):
                f = Monomial(comb)
                i_f = cub(f)
                err = abs(i_f - f.theoretical_integral())
                assert err < 2e-15




def test_simp_mass_and_diff_matrices_by_monomial():
    """Verify simplicial mass and differentiation matrices using monomials"""

    from hedge.element import \
            IntervalElement, \
            TriangularElement, \
            TetrahedralElement
    from pytools import generate_nonnegative_integer_tuples_summing_to_at_most

    from operator import add, mul

    thresh = 1e-13

    from numpy import dot
    for el in [
            IntervalElement(5),
            TriangularElement(3),
            TetrahedralElement(5),
            ]:
        for comb in generate_nonnegative_integer_tuples_summing_to_at_most(
                el.order, el.dimensions):
            ones = numpy.ones((el.node_count(),))
            unodes = el.unit_nodes()
            f = Monomial(comb)
            f_n = numpy.array([f(x) for x in unodes])
            int_f_n = dot(ones, dot(el.mass_matrix(), f_n))
            int_f = f.theoretical_integral()
            err = la.norm(int_f - int_f_n)
            if err > thresh:
                print "bad", el, comb, int_f, int_f_n, err
            assert err < thresh

            dmats = el.differentiation_matrices()
            for i in range(el.dimensions):
                df = f.diff(i)
                df = numpy.array([df(x) for x in unodes])/2
                df_n = dot(dmats[i], f_n)
                err = la.norm(df - df_n, numpy.Inf)
                if err > thresh:
                    print "bad-diff", comb, i, err
                assert err < thresh




def test_simp_gauss_theorem():
    """Verify Gauss's theorem explicitly on simplicial elements"""

    from hedge.element import \
            IntervalElement, \
            TriangularElement, \
            TetrahedralElement
    from operator import add
    from math import sin, cos, sqrt, exp, pi

    from numpy import dot

    def f1_1d(x):
        return sin(3*x[0])

    def f1_2d(x):
        return sin(3*x[0])+cos(3*x[1])
    def f2_2d(x):
        return sin(2*x[0])+cos(x[1])

    def f1_3d(x):
        return sin(3*x[0])+cos(3*x[1])+sin(1.9*x[2])
    def f2_3d(x):
        return sin(1.2*x[0])+cos(2.1*x[1])-cos(1.5*x[2])
    def f3_3d(x):
        return 5*sin(-0.2*x[0])-3*cos(x[1])+cos(x[2])

    #def f1_3d(x):
        #return 1
    #def f2_3d(x):
        #return 0
    #def f3_3d(x):
        #return 0

    def d(imap, coordinate, field):
        col = imap.matrix[:, coordinate]
        matrices = el.differentiation_matrices()
        return reduce(add, (coeff*dot(dmat, field)
                    for dmat, coeff in zip(matrices, col)))

    array = numpy.array

    intervals = [
            [array([-0.5]), array([17.])]
            ]

    triangles = [
            [array([-7.1687642250744492, 0.63058995062684642]), array([9.9744219044921199, 6.6530989283689781]), array([12.269380138171147, -17.529689194536481])],
            [array([-3.1285787297852634, -16.579403405465403]), array([-5.2882160938912515, -6.2209234150214137]), array([11.251223490342774, 4.6571427341871727])],
            [array([4.7407472917152553, -18.406868078408063]), array([1.8224524488556271, 11.551374404003361]), array([2.523148394963088, 1.632574414790982])],
            [array([-11.523714017493292, -14.2820557378961]), array([-0.44311816855771136, 19.572194735728861]), array([5.2855990566779445, -9.8743423935894388])],
            [array([1.113949150102217, -3.2255502625302639]), array([-13.028732972681315, 2.1525752429773379]), array([-2.3929000970202705, 6.2884649052982198])],
            [array([-8.0878061368549741, -14.604092423350167]), array([4.5339922477199996, 8.3770287646932022]), array([-5.2180549365480156, -1.9930760171433717])],
            [array([-1.9047012017294165, -3.6517882549544485]), array([3.1461902282192784, 5.7397849191668229]), array([-11.072761256907262, -8.3758508470287509])],
            [array([8.6609581113102934, 9.1121629958018566]), array([3.8230948675835497, -14.004679313330751]), array([10.975623610855521, 1.6267418698764553])],
            [array([13.959685276941629, -12.201892555481464]), array([-7.8057604576925499, -3.5283871457281757]), array([-0.41961743047735317, -3.2615635891671872])],
            [array([-9.8469907360335078, 6.0635407355366242]), array([7.8727080309703439, 7.634505157189091]), array([-2.7723038834027118, 8.5441656500931789])],
            ]
    tets = [
            #[make_random_vector(3, num.Float) for i in range(4)]
            #for j in range(10)
            [array([-0.087835976182812386, 8.4241880323369127, 2.6557808710807933]), array([5.4875560966799677, -7.5530368326122499, 8.4703868377747877]), array([-8.4888098806626751, 1.8369058453192324, -6.9041468708803713]), array([17.327527449499168, -9.0586108433594319, 5.7459746913914636])],
            [array([16.993689961344593, -12.116305360441197, -12.711045554409088]), array([-2.0324332275643817, -5.0524187595904335, 5.9257028535230383]), array([6.4221144729287687, -7.2496949199427245, -1.1590391996379827]), array([-5.7529432953399171, -6.9587987820990262, 3.7223773892240426])],
            [array([-0.4423263927732628, -1.6306971591009138, -1.2554069824001064]), array([-9.1171749892163785, 14.232868970928301, 4.6548620163014505]), array([16.554360867165595, -2.1451702825571202, -1.9050837421951314]), array([-8.7455417971698139, 19.016251630886945, -15.137691422305545])],
            [array([-1.9251811954429843, -4.5369007736338665, 9.2675942450331963]), array([-13.586778017089083, -3.6666239130220553, -14.095112617514117]), array([-15.014799506040006, -3.4363888726140681, -0.85237426213437206]), array([6.3854978041452597, 13.293981904633554, -7.8432774486183146])],
            [array([-6.761839340374304, 14.864784704377955, 1.574274771089831]), array([-0.1823468063801317, -21.892423945260102, 11.565172070570537]), array([-0.14658389181168049, 13.07241603902848, 7.2652184007323042]), array([-20.35574011769496, 14.816503793175773, -7.2800214849607254])],
            [array([23.294362873156878, 13.644282203469114, 10.383738204469243]), array([-19.792088993452555, 0.4209925297886693, -7.3933945447515388]), array([-2.832898385995708, -1.6480401382241885, -6.2689214950820924]), array([-0.081772347748623617, -3.3803599922239673, -19.614368674546114])],
            [array([0.43913744703796659, -16.473036116412242, -0.8653853759721295]), array([-7.3270752283484297, -0.97723025169973787, 2.1330514627504464]), array([3.8730334021748307, -9.0983850278388143, 3.3578300089831501]), array([18.639504439820936, 20.594835769217696, -10.666261239487298])],
            [array([-12.786230591302058, -9.2931510923111169, -2.1598642469378935]), array([-4.0458439207057459, -9.0298998997705144, -0.11666215074316685]), array([7.5023999981398424, 4.8603369473110583, 2.1813627427875013]), array([2.9579841500551272, -22.563123335973565, 10.335559822513606])],
            [array([-7.7732699602949893, 15.816977096296963, -6.8826683632918728]), array([7.6233333630240256, -9.3309869383569026, 0.50189282953625991]), array([-11.272342858699034, 1.089016041114454, -6.0359393299451476]), array([-6.4746449930954348, -0.026130504314747997, -2.2786267101817677])],
            [array([-18.243993907118757, 5.0646875774948974, -9.2110046334596856]), array([-8.1550804560957264, -3.1021806460634913, 7.5622831439916105]), array([19.460768761970783, 17.494565076685859, 16.295621155355697]), array([4.6186236213250131, -1.3869183721072562, -0.2159066724152843])],
            ]

    for el_geoms, el, f in [
            (intervals, IntervalElement(9), [f1_1d]),
            (triangles, TriangularElement(9), [f1_2d, f2_2d]),
            (tets, TetrahedralElement(1), [f1_3d, f2_3d, f3_3d]),
            ]:
        for vertices in el_geoms:
            ones = numpy.ones((el.node_count(),))
            face_ones = numpy.ones((len(el.face_indices()[0]),))

            map = el.geometry.get_map_unit_to_global(vertices)
            imap = map.inverted()

            mapped_points = [map(node) for node in el.unit_nodes()]

            f_n = [numpy.array([fi(x) for x in mapped_points])
                    for fi in f]
            df_n = [d(imap, i, f_n[i]) for i, fi_n in enumerate(f_n)]

            int_div_f = abs(map.jacobian())*sum(
                    dot(ones, dot(el.mass_matrix(), dfi_n)) for dfi_n in df_n)

            if False:
                boundary_comp = [
                        array([
                            fjac
                            * dot(face_ones,
                                dot(el.face_mass_matrix(),
                                    num.take(fi_n, face_indices)) )
                            * n_coord
                            for fi_n, n_coord in zip(f_n, n)])
                        for face_indices, n, fjac
                        in zip(el.face_indices(), *el.face_normals_and_jacobians(vertices, map))
                        ]

            boundary_sum = sum(
                    sum(
                        fjac
                        * dot(face_ones,
                            dot(el.face_mass_matrix(),
                                numpy.take(fi_n, face_indices)))
                        * n_coord
                        for fi_n, n_coord in zip(f_n, n))
                    for face_indices, n, fjac
                    in zip(el.face_indices(),
                        *el.geometry.face_normals_and_jacobians(vertices, map))
                    )

            #print el.face_normals_and_jacobians(map)[1]
            #print 'mp', [mapped_points[fi] for fi in el.face_indices()[2]]
            #print num.take(f_n[0], el.face_indices()[2])
            #print 'bc', boundary_comp
            #print 'bs', boundary_sum
            #print 'idiv', int_div_f
            #print abs(boundary_sum-int_div_f)
            assert abs(boundary_sum-int_div_f) < 1e-12




def test_1d_mass_matrix_vs_quadrature():
    """Check that a 1D mass matrix for Legendre-Gauss points gives the right weights"""
    from hedge.quadrature import LegendreGaussQuadrature
    from hedge.polynomial import legendre_vandermonde

    for n in range(13):
        lgq = LegendreGaussQuadrature(n)
        vdm = legendre_vandermonde(lgq.points, n)
        mass_mat = la.inv(numpy.dot(vdm, vdm.T))
        ones = numpy.ones((mass_mat.shape[0],))

        assert la.norm(
            la.solve(numpy.dot(vdm, vdm.T), ones) -
                numpy.array(lgq.weights), numpy.Inf) < 2e-14




def test_mapping_differences_tri():
    """Check that triangle interpolation is independent of mapping to reference
    """
    from hedge.element import TriangularElement
    from random import random
    from pytools import generate_permutations

    def shift(list):
        return list[1:] + [list[0]]

    class LinearCombinationOfFunctions:
        def __init__(self, coefficients, functions, premap):
            self.coefficients = coefficients
            self.functions = functions
            self.premap = premap

        def __call__(self, x):
            return sum(coeff*f(self.premap(x)) for coeff, f in
                    zip(self.coefficients, self.functions))

    def random_barycentric_coordinates(dim):
        remain = 1
        coords = []
        for i in range(dim):
            coords.append(random() * remain)
            remain -= coords[-1]
        coords.append(remain)
        return coords

    tri = TriangularElement(5)

    for trial_number in range(10):
        vertices = [numpy.random.randn(2) for vi in range(3)]
        map = tri.geometry.get_map_unit_to_global(vertices)
        nodes = [map(node) for node in tri.unit_nodes()]
        node_values = numpy.array([random() for node in nodes])

        functions = []
        for pvertices in generate_permutations(vertices):
            pmap = tri.geometry.get_map_unit_to_global(pvertices)
            pnodes = [pmap(node) for node in tri.unit_nodes()]

            # map from pnode# to node#
            nodematch = {}
            for pi, pn in enumerate(pnodes):
                for i, n in enumerate(nodes):
                    if la.norm(n - pn) < 1e-13:
                        nodematch[pi] = i
                        break

            pnode_values = numpy.array([node_values[nodematch[pi]]
                    for pi in range(len(nodes))])

            interp_f = LinearCombinationOfFunctions(
                    la.solve(tri.vandermonde(), pnode_values),
                    tri.basis_functions(),
                    pmap.inverted())

            # verify interpolation property
            #for n, nv in zip(pnodes, pnode_values):
                #assert abs(interp_f(n) - nv) < 1e-13

            functions.append(interp_f)

        for subtrial_number in range(15):
            pt_in_element = sum(
                    coeff*vertex
                    for coeff, vertex in zip(
                        random_barycentric_coordinates(2),
                        vertices))
            f_values = [f(pt_in_element) for f in functions]
            avg = sum(f_values) / len(f_values)
            err = [abs(fv-avg) for fv in f_values]
            assert max(err) < 5e-13




def test_interior_fluxes_tri():
    """Check triangle surface integrals computed using interior fluxes
    against their known values.
    """

    from math import pi, sin, cos

    def round_trip_connect(start, end):
        for i in range(start, end):
            yield i, i+1
        yield end, start

    a = -pi
    b = pi
    points = [
            (a,0), (b,0),
            (a,-1), (b,-1),
            (a,1), (b,1)
            ]

    import meshpy.triangle as triangle

    mesh_info = triangle.MeshInfo()
    mesh_info.set_points(points)
    mesh_info.set_facets(
            [(0,1),(1,3),(3,2),(2,0),(0,4),(4,5),(1,5)]
            )

    mesh_info.regions.resize(2)
    mesh_info.regions[0] = [
            0,-0.5, # coordinate
            1, # lower element tag
            0.1, # max area
            ]
    mesh_info.regions[1] = [
            0,0.5, # coordinate
            2, # upper element tag
            0.01, # max area
            ]

    generated_mesh = triangle.build(mesh_info,
            attributes=True,
            volume_constraints=True)
    #triangle.write_gnuplot_mesh("mesh.dat", generated_mesh)

    def element_tagger(el):
        if generated_mesh.element_attributes[el.id] == 1:
            return ["upper"]
        else:
            return ["lower"]

    from hedge.mesh import make_conformal_mesh
    mesh = make_conformal_mesh(
            generated_mesh.points,
            generated_mesh.elements)

    from hedge.element import TriangularElement
    from hedge.discretization import ones_on_volume
    discr = discr_class(mesh, TriangularElement(4),
            debug=discr_class.noninteractive_debug_flags())

    def f_u(x, el):
        if generated_mesh.element_attributes[el.id] == 1:
            return cos(x[0]-x[1])
        else:
            return 0

    def f_l(x, el):
        if generated_mesh.element_attributes[el.id] == 0:
            return sin(x[0]-x[1])
        else:
            return 0

    u_l = discr.interpolate_volume_function(f_l)
    u_u = discr.interpolate_volume_function(f_u)
    u = u_u + u_u

    #discr.visualize_vtk("dual.vtk", [("u", u)])

    from hedge.flux import make_normal, FluxScalarPlaceholder
    from hedge.optemplate import Field, get_flux_operator
    fluxu = FluxScalarPlaceholder()
    res = discr.compile(
            get_flux_operator(
                (fluxu.int - fluxu.ext) * make_normal(discr.dimensions)[1])
            * Field("u"))(u=u)

    from hedge.discretization import ones_on_volume
    ones = ones_on_volume(discr)
    err = abs(numpy.dot(res, ones))
    #print err
    assert err < 5e-14




@pytools.test.mark_test.long
def test_interior_fluxes_tet():
    """Check tetrahedron surface integrals computed using interior fluxes
    against their known values.
    """

    import meshpy.tet as tet
    from math import pi, sin, cos

    mesh_info = tet.MeshInfo()

    # construct a two-box extrusion of this base
    base = [(-pi,-pi,0), (pi,-pi,0), (pi,pi,0), (-pi,pi,0)]

    # first, the nodes
    mesh_info.set_points(
            base
            +[(x,y,z+pi) for x,y,z in base]
            +[(x,y,z+pi+1) for x,y,z in base]
            )

    # next, the facets

    # vertex indices for a box missing the -z face
    box_without_minus_z = [
        [4,5,6,7],
        [0,4,5,1],
        [1,5,6,2],
        [2,6,7,3],
        [3,7,4,0],
        ]

    def add_to_all_vertex_indices(facets, increment):
        return [[pt+increment for pt in facet] for facet in facets]

    mesh_info.set_facets(
        [[0,1,2,3]] # base
        +box_without_minus_z # first box
        +add_to_all_vertex_indices(box_without_minus_z, 4) # second box
        )

    # set the volume properties -- this is where the tet size constraints are
    mesh_info.regions.resize(2)
    mesh_info.regions[0] = [0,0,pi/2, # point in volume -> first box
            0, # region tag (user-defined number)
            0.5, # max tet volume in region
            ]
    mesh_info.regions[1] = [0,0,pi+0.5, # point in volume -> second box
            1, # region tag (user-defined number, arbitrary)
            0.1, # max tet volume in region
            ]

    generated_mesh = tet.build(mesh_info, attributes=True, volume_constraints=True)

    from hedge.mesh import make_conformal_mesh
    mesh = make_conformal_mesh(
            generated_mesh.points,
            generated_mesh.elements)

    from hedge.element import TetrahedralElement
    from hedge.discretization import ones_on_volume
    discr = discr_class(mesh, TetrahedralElement(4),
            debug=discr_class.noninteractive_debug_flags())

    def f_u(x, el):
        if generated_mesh.element_attributes[el.id] == 1:
            return cos(x[0]-x[1]+x[2])
        else:
            return 0

    def f_l(x, el):
        if generated_mesh.element_attributes[el.id] == 0:
            return sin(x[0]-x[1]+x[2])
        else:
            return 0

    u_l = discr.interpolate_volume_function(f_l)
    u_u = discr.interpolate_volume_function(f_u)
    u = u_l + u_u

    # visualize the produced field
    #from hedge.visualization import SiloVisualizer
    #vis = SiloVisualizer(discr)
    #visf = vis.make_file("sandwich")
    #vis.add_data(visf,
            #[("u_l", u_l), ("u_u", u_u)],
            #expressions=[("u", "u_l+u_u")])

    # make sure the surface integral of the difference
    # between top and bottom is zero
    from hedge.flux import make_normal, FluxScalarPlaceholder
    from hedge.optemplate import Field, get_flux_operator

    fluxu = FluxScalarPlaceholder()
    res = discr.compile(get_flux_operator(
            (fluxu.int - fluxu.ext)
            *make_normal(discr.dimensions)[1]) * Field("u"))(u=u)

    from hedge.discretization import ones_on_volume
    ones = ones_on_volume(discr)
    assert abs(numpy.dot(res,ones)) < 5e-14




@pytools.test.mark_test.long
def test_symmetry_preservation_2d():
    """Test whether we preserve symmetry in a symmetric 2D advection problem"""
    from numpy import dot

    def make_mesh():
        array = numpy.array

        #
        #    1---8---2
        #    |7 /|\ 1|
        #    | / | \ |
        #    |/ 6|0 \|
        #    5---4---7
        #    |\ 5|3 /|
        #    | \ | / |
        #    |4 \|/ 2|
        #    0---6---3
        #
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

        def boundary_tagger(vertices, el, face_nr, all_v):
            if dot(el.face_normals[face_nr], v) < 0:
                return ["inflow"]
            else:
                return ["outflow"]

        from hedge.mesh import make_conformal_mesh
        return make_conformal_mesh(points, elements, boundary_tagger)

    from hedge.discretization import SymmetryMap
    from hedge.element import TriangularElement
    from hedge.timestep import RK4TimeStepper
    from math import sqrt, sin
    from hedge.models.advection import StrongAdvectionOperator
    from hedge.data import TimeDependentGivenFunction

    v = numpy.array([-1,0])

    mesh = make_mesh()
    discr = discr_class(mesh, order=4,
            debug=discr_class.noninteractive_debug_flags())
    #ref_discr = DynamicDiscretization(mesh, order=4)

    def f(x):
        if x < 0.5: return 0
        else: return (x-0.5)

    #def f(x):
        #return sin(5*x)

    def u_analytic(x, el, t):
        return f(-dot(v, x)+t)

    u = discr.interpolate_volume_function(
            lambda x, el: u_analytic(x, el, 0))

    sym_map = SymmetryMap(discr,
            lambda x: numpy.array([x[0], -x[1]]),
            {0:3, 2:1, 5:6, 7:4})

    for flux_type in StrongAdvectionOperator.flux_types:
        stepper = RK4TimeStepper()
        op = StrongAdvectionOperator(v,
                inflow_u=TimeDependentGivenFunction(u_analytic),
                flux_type=flux_type)

        dt = discr.dt_factor(op.max_eigenvalue())
        nsteps = int(1/dt)
        rhs = op.bind(discr)
        #test_comp = [ "bflux"]
        #test_rhs = op.bind(discr, test_comp)
        #ref_rhs = op.bind(ref_discr, test_comp)
        for step in xrange(nsteps):
            u = stepper(u, step*dt, dt, rhs)
            sym_error_u = u-sym_map(u)
            sym_error_u_l2 = discr.norm(sym_error_u)

            if False:
                from hedge.visualization import SiloVisualizer
                vis = SiloVisualizer(discr)
                visf = vis.make_file("test-%s-%04d" % (flux_type, step))
                vis.add_data(visf,[
                    ("u", u),
                    ("sym_u", sym_map(u)),
                    ("sym_diff", u-sym_map(u)),
                    ("rhs", rhs(step*dt, u)),
                    ("rhs_test", test_rhs(step*dt, u)),
                    ("rhs_ref", ref_rhs(step*dt, u)),
                    ("rhs_diff", test_rhs(step*dt, u)-ref_rhs(step*dt, u)),
                    ])

                print sym_error_u_l2
            assert sym_error_u_l2 < 4e-15





@pytools.test.mark_test.long
def test_convergence_advec_2d():
    """Test whether 2D advection actually converges"""

    import pyublas
    from hedge.mesh import make_disk_mesh, make_regular_rect_mesh
    from hedge.element import TriangularElement
    from hedge.timestep import RK4TimeStepper
    from hedge.tools import EOCRecorder
    from math import sin, pi, sqrt
    from hedge.models.advection import StrongAdvectionOperator
    from hedge.data import TimeDependentGivenFunction

    v = numpy.array([0.27,0])
    norm_a = la.norm(v)

    from numpy import dot

    def f(x):
        return sin(x)

    def u_analytic(x, el, t):
        return f((-dot(v, x)/norm_a+t*norm_a))

    def boundary_tagger(vertices, el, face_nr, all_v):
        if dot(el.face_normals[face_nr], v) < 0:
            return ["inflow"]
        else:
            return ["outflow"]

    for mesh in [
            # non-periodic
            make_disk_mesh(r=pi, boundary_tagger=boundary_tagger, max_area=0.5),
            # periodic
            make_regular_rect_mesh(a=(0,0), b=(2*pi, 1), n=(8,4),
                periodicity=(True, False),
                boundary_tagger=boundary_tagger,
                )
            ]:
        for flux_type in StrongAdvectionOperator.flux_types:
            eoc_rec = EOCRecorder()

            for order in [1,2,3,4,5,6]:
                discr = discr_class(mesh, TriangularElement(order),
                        debug=discr_class.noninteractive_debug_flags())
                op = StrongAdvectionOperator(v,
                        inflow_u=TimeDependentGivenFunction(u_analytic),
                        flux_type=flux_type)

                u = discr.interpolate_volume_function(
                        lambda x, el: u_analytic(x, el, 0))
                dt = discr.dt_factor(norm_a)
                nsteps = int(1/dt)

                stepper = RK4TimeStepper()
                rhs = op.bind(discr)
                for step in range(nsteps):
                    u = stepper(u, step*dt, dt, rhs)

                u_true = discr.interpolate_volume_function(
                        lambda x, el: u_analytic(x, el, nsteps*dt))
                error = u-u_true
                error_l2 = discr.norm(error)
                eoc_rec.add_data_point(order, error_l2)

            if False:
                print "%s\n%s\n" % (flux_type.upper(), "-" * len(flux_type))
                print eoc_rec.pretty_print(abscissa_label="Poly. Order",
                        error_label="L2 Error")

            assert eoc_rec.estimate_order_of_convergence()[0,1] > 4
            assert eoc_rec.estimate_order_of_convergence(2)[-1,1] > 10




@pytools.test.mark_test.long
def test_elliptic():
    """Test various properties of elliptic operators."""

    from hedge.tools import unit_vector
    def matrix_rep(op):
        h,w = op.shape
        mat = numpy.zeros(op.shape)
        for j in range(w):
            mat[:,j] = op(unit_vector(w, j))
        return mat

    def check_grad_mat():
        import pyublas
        if not pyublas.has_sparse_wrappers():
            return

        grad_mat = op.grad_matrix()

        #print len(discr), grad_mat.nnz, type(grad_mat)
        for i in range(10):
            u = numpy.random.randn(len(discr))

            mat_result = grad_mat * u
            op_result = numpy.hstack(op.grad(u))

            err = la.norm(mat_result-op_result)*la.norm(op_result)
            assert la.norm(mat_result-op_result)*la.norm(op_result) \
                    < 1e-5

    def check_matrix_tgt():
        big = num.zeros((20, 20), flavor=num.SparseBuildMatrix)
        small = num.array([[1,2,3],[4,5,6],[7,8,9]])
        print small
        from hedge._internal import MatrixTarget
        tgt = MatrixTarget(big, 4, 4)
        tgt.begin(small.shape[0], small.shape[1])
        print "YO"
        tgt.add_coefficients(4, 4, small)
        print "DUDE"
        tgt.finalize()
        print big

    import pymbolic
    v_x = pymbolic.var("x")
    truesol = pymbolic.parse("math.sin(x[0]**2*x[1]**2)")
    truesol_c = pymbolic.compile(truesol, variables=["x"])
    rhs = pymbolic.simplify(pymbolic.laplace(truesol, [v_x[0], v_x[1]]))
    rhs_c = pymbolic.compile(rhs, variables=["x", "el"])

    from hedge.mesh import make_disk_mesh, TAG_ALL, TAG_NONE
    mesh = make_disk_mesh(r=0.5, max_area=0.1, faces=20)
    mesh = mesh.reordered_by("cuthill")

    from hedge.backends import SerialRunContext
    rcon = SerialRunContext(discr_class)

    from hedge.tools import EOCRecorder
    eocrec = EOCRecorder()
    for order in [1,2,3,4,5]:
        for flux in ["ldg", "ip"]:
            from hedge.element import TriangularElement
            discr = discr_class(mesh, TriangularElement(order),
                    debug=discr_class.noninteractive_debug_flags())

            from hedge.data import GivenFunction
            from hedge.models.poisson import WeakPoissonOperator
            op = WeakPoissonOperator(discr.dimensions,
                    dirichlet_tag=TAG_ALL,
                    dirichlet_bc=GivenFunction(
                        lambda x, el: truesol_c(x)),
                    neumann_tag=TAG_NONE,
                    flux=flux)

            bound_op = op.bind(discr)

            if order <= 3:
                mat = matrix_rep(bound_op)
                sym_err = la.norm(mat-mat.T)
                #print sym_err
                assert sym_err<1e-12
                #check_grad_mat()

            from hedge.iterative import parallel_cg
            truesol_v = discr.interpolate_volume_function(
                    lambda x, el: truesol_c(x))
            sol_v = -parallel_cg(
                    rcon, -bound_op,
                    bound_op.prepare_rhs(GivenFunction(rhs_c)),
                    tol=1e-10, max_iterations=40000)

            eocrec.add_data_point(order, discr.norm(sol_v-truesol_v))

    #print eocrec.pretty_print()
    assert eocrec.estimate_order_of_convergence()[0,1] > 8




def test_projection():
    """Test whether projection between different orders works"""

    from hedge.mesh import make_disk_mesh
    from hedge.discretization import Projector
    from hedge.element import TriangularElement
    from hedge.tools import EOCRecorder
    from math import sin, pi, sqrt

    from numpy import dot

    a = numpy.array([1,3])

    def u_analytic(x, el):
        return sin(dot(a, x))

    mesh = make_disk_mesh(r=pi, max_area=0.5)

    discr2 = discr_class(mesh, TriangularElement(2),
            debug=discr_class.noninteractive_debug_flags())
    discr5 = discr_class(mesh, TriangularElement(5),
            debug=discr_class.noninteractive_debug_flags())
    p2to5 = Projector(discr2, discr5)
    p5to2 = Projector(discr5, discr2)

    u2 = discr2.interpolate_volume_function(u_analytic)
    u2_i = p5to2(p2to5(u2))
    assert discr2.norm(u2-u2_i) < 3e-15




def test_filter():
    """Exercise mode-based filtering."""

    from hedge.mesh import make_disk_mesh
    from hedge.element import TriangularElement
    from math import sin

    mesh = make_disk_mesh(r=3.4, max_area=0.5)
    discr = discr_class(mesh, TriangularElement(5),
            debug=discr_class.noninteractive_debug_flags())

    from hedge.discretization import Filter, ExponentialFilterResponseFunction
    half_filter = Filter(discr, lambda mid, ldis: 0.5)
    for eg in discr.element_groups:
        fmat = half_filter.get_filter_matrix(eg)
        n,m = fmat.shape
        assert la.norm(fmat - 0.5*numpy.eye(n, m)) < 2e-15

    from numpy import dot

    def test_freq(n):
        a = numpy.array([1,n])

        def u_analytic(x, el):
            return sin(dot(a, x))

        exp_filter = Filter(discr, ExponentialFilterResponseFunction(0.9, 3))

        u = discr.interpolate_volume_function(u_analytic)
        filt_u = exp_filter(u)

        int_error = abs(discr.integral(u) - discr.integral(filt_u))
        l2_ratio = discr.norm(filt_u) / discr.norm(u)
        assert int_error < 5e-15
        assert 0.96 < l2_ratio < 0.99999

    test_freq(3)
    test_freq(5)
    test_freq(9)
    test_freq(17)




def no_test_tri_mass_mat_gauss(self):
    """Check the integral of a Gaussian on a disk using the mass matrix"""

    # This is a bad test, since it's never exact. The Gaussian has infinite support,
    # and this *does* matter numerically.

    from hedge.mesh import make_disk_mesh
    from hedge.element import TriangularElement
    from math import sqrt, exp, pi

    sigma_squared = 1/219.3

    mesh = make_disk_mesh()
    discr = self.discr_class(make_disk_mesh(), TriangularElement(4), 
            debug=self.discr_class.noninteractive_debug_flags())
    f = discr.interpolate_volume_function(
            lambda x, el: exp(-x*x/(2*sigma_squared)))
    ones = discr.interpolate_volume_function(
            lambda x, el: 1)

    #discr.visualize_vtk("gaussian.vtk", [("f", f)])
    num_integral_1 = ones * (discr.mass_operator * f)
    num_integral_2 = f * (discr.mass_operator * ones)
    dim = 2
    true_integral = (2*pi)**(dim/2)*sqrt(sigma_squared)**dim
    err_1 = abs(num_integral_1-true_integral)
    err_2 = abs(num_integral_2-true_integral)
    self.assert_(err_1 < 1e-11)
    self.assert_(err_2 < 1e-11)




if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec sys.argv[1]
    else:
        from py.test.cmdline import main
        main([__file__])
