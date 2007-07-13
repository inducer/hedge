def plot_order_3_tri_poly_components():
    from hedge.polynomial import jacobi_function
    from hedge.tools import plot_1d

    i = 3
    j = 3
    f = jacobi_function(0, 0, i)
    g = jacobi_function(2*i+1, 0, j)
    plot_1d(f, -1, 1)
    plot_1d(g, -1, 1)

def plot_tri_polynomials_hi_res():
    import numpy
    import pylinear.array as num
    from hedge.element import TriangularElement
    from hedge.polynomial import generic_vandermonde
    from pyvtk import PolyData, PointData, VtkData, Scalars, Vectors

    tri_low = TriangularElement(3)
    tri_high = TriangularElement(14)

    unodes = tri_high.unit_nodes()

    low_vdm = tri_low.vandermonde()
    point_values = generic_vandermonde(unodes, tri_low.basis_functions()).T
    lagrange_values = (1/low_vdm.T) * point_values

    def three_vector(x):
        if len(x) == 3:
            return x
        elif len(x) == 2:
            return x[0], x[1], 0.
        elif len(x) == 1:
            return x[0], 0, 0.

    structure = PolyData(points=[three_vector(x) for x in unodes], 
            polygons=tri_high.generate_submesh_indices())
    pdatalist = []
    for i, row in enumerate(lagrange_values):
        pdatalist.append(Scalars(numpy.array(row), 
            name="basis%d" % i, lookup_table="default"))
    vtk = VtkData(structure, "Hedge visualization", PointData(*pdatalist))
    vtk.tofile("basis.vtk")

def plot_tri_quad_weights():
    import numpy
    import pylinear.array as num
    from hedge.element import TriangularElement
    from hedge.polynomial import generic_vandermonde
    from pyvtk import PolyData, PointData, VtkData, Scalars, Vectors

    edata = TriangularElement(3)
    unodes = edata.unit_nodes()
    mass_mat = edata.mass_matrix()
    ones = num.ones((mass_mat.shape[0],))

    point_values = mass_mat<<num.solve>>ones
    #point_values = mass_mat*ones

    def three_vector(x):
        if len(x) == 3:
            return x
        elif len(x) == 2:
            return x[0], x[1], 0.
        elif len(x) == 1:
            return x[0], 0, 0.

    structure = PolyData(points=[three_vector(x) for x in unodes], 
            polygons=edata.generate_submesh_indices())
    pdatalist = [
           Scalars(numpy.array(point_values), 
            name="qweights", lookup_table="default") ]
    vtk = VtkData(structure, "Hedge visualization", PointData(*pdatalist))
    vtk.tofile("quadweights.vtk")

def plot_real_tri_quad_weights():
    from hedge.cubature import TetrahedronCubatureData
    outf = file("quad.dat", "w")
    for x, y, w in TetrahedronCubatureData[15]:
        outf.write("%g\t%g\t%g\n" % (x,y,w))

def visualize_single_tet():
    from hedge.mesh import ConformalMesh
    from hedge.element import TetrahedralElement
    from hedge.discretization import Discretization
    from hedge.visualization import SiloVisualizer

    points = [
            (-1,-1,-1),
            (+1,-1,-1),
            (-1,+1,-1),
            (-1,-1,+1),
            ]
    elements = [(0,1,2,3)]
    mesh = ConformalMesh(points, elements)
    discr = Discretization(mesh, TetrahedralElement(2))
    vis = SiloVisualizer(discr)
    vis("single-tet.silo", [("zip", discr.volume_zeros())])



if __name__ == "__main__":
    #plot_tri_order_1_polynomials_hi_res()
    #plot_tri_quad_weights()
    #plot_real_tri_quad_weights()
    visualize_single_tet()
