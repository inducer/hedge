"""Mesh topology/geometry representation."""

from __future__ import division

__copyright__ = "Copyright (C) 2009 Xueyu Zhu, Andreas Kloeckner"

__license__ = """
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see U{http://www.gnu.org/licenses/}.
"""

import numpy
import numpy.linalg as la
from pytools import memoize, memoize_method, Record, single_valued
from hedge.discretization.local import \
        IntervalDiscretization, \
        TriangleDiscretization, \
        TetrahedronDiscretization



class Point:
    dimensions = 0

    def node_count(self):
        return 1



# tools -----------------------------------------------------------------------
def generate_triangle_vertex_tuples(order):
    yield (0, 0)
    yield (order, 0)
    yield (0, order)

def generate_triangle_edge_tuples(order):
    for i in range(1, order):
        yield (i, 0)
    for i in range(1, order):
        yield (order-i, i)
    for i in range(1, order):
        yield (0, order-i)

def generate_triangle_volume_tuples(order):
    for i in range(1, order):
        for j in range(1, order-i):
            yield (j, i)





# element info ----------------------------------------------------------------
class GmshElementBase(object):
    @memoize_method
    def hedge_to_gmsh_index_map(self):
        gmsh_tup_to_index = dict(
                (tup, i)
                for i, tup in enumerate(self.gmsh_node_tuples()))

        return [gmsh_tup_to_index[tup]
                for tup in self.node_tuples()]





class GmshIntervalElement(IntervalDiscretization, GmshElementBase):
    vertex_count = 2

    @memoize_method
    def gmsh_node_tuples(self):
        yield (0,)
        yield (self.order,)

        for i in range(1, self.order):
            yield (i,)




class GmshTriangularElement(TriangleDiscretization, GmshElementBase):
    vertex_count = 3

    @memoize_method
    def gmsh_node_tuples(self):
        for tup in generate_triangle_vertex_tuples(self.order):
            yield tup
        for tup in generate_triangle_edge_tuples(self.order):
            yield tup
        for tup in generate_triangle_volume_tuples(self.order):
            yield tup




class GmshTetrahedralElement(TetrahedronDiscretization, GmshElementBase):
    vertex_count = 4

    @memoize_method
    def gmsh_node_tuples(self):
        result = []
        result_set = set()

        def add_without_duplicating(tup):
            if tup not in result_set:
                result.append(tup)
                result_set.add(tup)

        o = self.order

        for f in [
                generate_triangle_vertex_tuples,
                generate_triangle_edge_tuples,
                generate_triangle_volume_tuples]:
            for i, j in f(o):
                add_without_duplicating((i, j, 0)) # u-v
            for i, j in f(o):
                add_without_duplicating((0, i, j)) # v-w
            for i, j in f(o):
                add_without_duplicating((j, 0, i)) # w-u
            for i, j in f(o):
                add_without_duplicating((o-i-j, i, j))

        # volume
        for i in range(1, o):
            for j in range(1, o-i):
                for k in range(1, o-j-i):
                    result.append((k, j, i))

        return result




GMSH_ELEMENT_TYPE_TO_INFO_MAP = {
        1:  GmshIntervalElement(1),
        2:  GmshTriangularElement(1),
        4:  GmshTetrahedralElement(1),
        8:  GmshIntervalElement(2),
        9:  GmshTriangularElement(2),
        11: GmshTetrahedralElement(2),
        15: Point(),
        21: GmshTriangularElement(3),
        23: GmshTriangularElement(4),
        25: GmshTriangularElement(5),
        26: GmshIntervalElement(4),
        27: GmshIntervalElement(5),
        28: GmshIntervalElement(6),
        29: GmshTetrahedralElement(3),
        30: GmshTetrahedralElement(4),
        31: GmshTetrahedralElement(5)
        }




# file reader -----------------------------------------------------------------
class GmshFileFormatError(RuntimeError):
    pass




class LineFeeder:
    def __init__(self, line_list):
        self.line_list = line_list
        self.i = 0

    def has_next_line(self):
        return self.i < len(self.line_list)

    def get_next_line(self):
        if self.i >= len(self.line_list):
            raise GmshFileFormatError("unexpected end of file")

        result = self.line_list[self.i].strip()
        self.i += 1
        return result




class LocalToGlobalMap(object):
    def __init__(self, nodes, ldis):
        self.nodes = nodes
        self.ldis  = ldis

        node_src_indices = numpy.array(
                ldis.hedge_to_gmsh_index_map(),
                dtype=numpy.intp)

        nodes = numpy.array(nodes, dtype=numpy.float64)
        reordered_nodes = nodes[node_src_indices, :]

        self.modal_coeff = la.solve(ldis.vandermonde(), reordered_nodes)
        # axis 0: node number, axis 1: xyz axis

    def __call__(self, r):
        """Given a point *r* on the reference element, return the
        corresponding point *x* in global coordinates.
        """
        mc = self.modal_coeff

        return numpy.array([sum([
            mc[i, axis] * mbf(r)
            for i, mbf in enumerate(self.ldis.basis_functions())])
            for axis in range(self.ldis.dimensions)])

    def is_affine(self):
        from pytools import any
        return any(
                max(mid) >= 2 and abs(mc) >= 1e-13
                for mid, mc in zip(
                    self.ldis.generate_mode_identifiers(),
                    self.modal_coeff)



def read_gmsh(filename, force_dimension=None, periodicity=None):
    """
    :param force_dimension: if not None, truncate point coordinates to this many dimensions.
    """
    import string
    # open target file
    mesh_file = open(filename, 'r')
    feeder = LineFeeder(mesh_file.readlines())
    mesh_file.close()

    element_type_map = GMSH_ELEMENT_TYPE_TO_INFO_MAP

    # collect the mesh information
    nodes = []
    elements = []

    # maps (tag_number, dimension) -> tag_name
    tag_name_map = {}

    gmsh_vertex_nrs_to_element = {}

    class ElementInfo(Record):
        pass

    while feeder.has_next_line():
        next_line = feeder.get_next_line()
        if not next_line.startswith("$"):
            raise GmshFileFormatError("expected start of section, '%s' found instead" % l)

        section_name = next_line[1:]

        if section_name == "MeshFormat":
            line_count = 0
            while True:
                next_line = feeder.get_next_line()
                if next_line == "$End"+section_name:
                    break

                if line_count == 0:
                    version_number, file_type, data_size = next_line.split()

                if line_count > 0:
                    raise GmshFileFormatError("more than one line found in MeshFormat section")

                if version_number != "2.1":
                    from warnings import warn
                    warn("mesh version 2.1 expected, '%s' found instead" % version_number)

                if file_type != "0":
                    raise GmshFileFormatError("only ASCII gmsh file type is supported")

                line_count += 1

        elif section_name == "Nodes":
            node_count = int(feeder.get_next_line())
            node_idx = 1

            while True:
                next_line = feeder.get_next_line()
                if next_line == "$End"+section_name:
                    break

                parts = next_line.split()
                if len(parts) != 4:
                    raise GmshFileFormatError("expected four-component line in $Nodes section")

                read_node_idx = int(parts[0])
                if read_node_idx != node_idx:
                    raise GmshFileFormatError("out-of-order node index found")

                nodes.append(numpy.array(
                        [float(x) for x in parts[1:force_dimension+1]],
                        dtype=numpy.float64))

                node_idx += 1

            if node_count+1 != node_idx:
                raise GmshFileFormatError("unexpected number of nodes found")

        elif section_name == "Elements":
            element_count = int(feeder.get_next_line())
            element_idx = 1
            while True:
                next_line = feeder.get_next_line()
                if next_line == "$End"+section_name:
                    break

                parts = [int(x) for x in next_line.split()]

                if len(parts) < 4:
                    raise GmshFileFormatError("too few entries in element line")

                read_element_idx = parts[0]
                if read_element_idx != element_idx:
                    raise GmshFileFormatError("out-of-order node index found")

                el_type_num = parts[1]
                try:
                    element_type = element_type_map[el_type_num]
                except KeyError:
                    raise GmshFileFormatError("unexpected element type %d"
                            % el_type_num)

                tag_count = parts[2]
                tags = parts[3:3+tag_count]

                # convert to zero-based
                node_indices = [x-1 for x in parts[3+tag_count:]]

                if element_type.node_count()!= len(node_indices):
                    raise GmshFileFormatError("unexpected number of nodes in element")

                gmsh_vertex_nrs = el.node_indices[:el.element_type.vertex_count]
                zero_based_idx = element_idx - 1
                el_info = ElementInfo(
                    index=zero_based_idx,
                    el_type=element_type,
                    node_indices=node_indices,
                    gmsh_vertex_indices=gmsh_vertex_nrs,
                    tag_numbers=tags)

                gmsh_vertex_nrs_to_element[set(gmsh_vertex_nrs] = el_info

                element_idx +=1
            if element_count+1 != element_idx:
                raise GmshFileFormatError("unexpected number of elements found")

        elif section_name == "PhysicalNames":
            name_count = int(feeder.get_next_line())
            name_idx = 1

            while True:
                next_line = feeder.get_next_line()
                if next_line == "$End"+section_name:
                    break

                dimension, number, name = next_line.split()
                dimension = int(dimension)
                number = int(number)

                tag_name_map.setdefault((number, dimension)).append(name)

                name_idx +=1

            if name_count+1 != name_idx:
                raise GmshFileFormatError("unexpected number of physical names found")
        else:
            # unrecognized section, skip
            while True:
                next_line = feeder.get_next_line()
                if next_line == "$End"+section_name:
                    break

    # check all tags refer to elements of same dimension
    tag_number_to_dim = {}

    # figure out dimensionalities
    node_dim = single_valued(len(node) for node in nodes)
    vol_dim = max(el.el_type.dimensions for el in elements)
    bdry_dim = vol_dim - 1

    vol_elements = [el for el in elements
            if el.el_type.dimensions == vol_dim]
    bdry_elements = [el for el in elements
            if el.el_type.dimensions == bdry_dim]

    # build hedge-compatible elements
    from hedge.mesh.element import TO_CURVED_CLASS
    hedge_vertices = []
    hedge_elements = []

    gmsh_node_nr_to_hedge_vertex_nr = {}
    hedge_el_to_gmsh_element = {}

    def get_vertex_nr(gmsh_node_nr):
        try:
            return gmsh_node_nr_to_hedge_vertex_nr[gmsh_node_nr]
        except KeyError:
            hedge_vertex_nr = len(hedge_vertices)
            hedge_vertices.append(nodes[gmsh_node_nr])
            gmsh_node_nr_to_hedge_vertex_nr[gmsh_node_nr] = hedge_vertex_nr
            return hedge_vertex_nr

    for el_nr, gmsh_el in enumerate(vol_elements):
        el_map = LocalToGlobalMap(
                [nodes[ni] for ni gmsh_el.node_indices],
                gmsh_el.element_type)
        is_affine = gmsh_el.map.is_affine()

        el_class = gmsh_el.element_type.geometry
        if not is_affine:
            try:
                el_class = TO_CURVED_CLASS[el_class]
            except KeyError:
                pass
            else:
                raise GmshFileFormatError("unsupported curved element type")

        vertex_indices = [get_vertex_nr(gmsh_node_nr)
                for gmsh_node_nr in gmsh_el.gmsh_vertex_indices]

        if is_affine:
            hedge_el = el_class(el_nr, hedge_vertices, vertex_indices)
        else:
            hedge_el = el_class(el_nr, vertex_indices, el_map)

        hedge_elements.append(hedge_el)
        hedge_el_to_gmsh_el[hedge_el] = gmsh_el

    from pytools import reverse_dictionary
    hedge_vertex_nr_to_gmsh_node_nr = reverse_dictionary(
            gmsh_node_nr_to_hedge_vertex_nr)

    del gmsh_node_nr_to_hedge_vertex_nr
    del nodes
    del vol_elements

    def volume_tagger(el, all_v):
        return [tag_name_map[tag_nr, el.dimensions]
                for tag_nr in hedge_el_to_gmsh_element[el].tags
                if (tag_nr, el.dimensions) in tag_name_map]

    def boundary_tagger(fvi, el, fn, all_v):
        gmsh_vertex_nrs = set(
                hedge_vertex_nr_to_gmsh_node_nr[face_vertex_index]
                for face_vertex_index in fvi)
        gmsh_element = gmsh_vertex_nrs_to_element[gmsh_vertex_nrs]

        return [tag_name_map[tag_nr, el.dimensions]
                for tag_nr in gmsh_element.tags
                if (tag_nr, el.dimensions) in tag_name_map]

    from hedge.mesh import make_conformal_mesh_ext
    return make_conformal_mesh_ext(
            hedge_vertices,
            hege_elements,
            volume_tagger,
            boundary_tagger
            periodicity)






if __name__ == "__main__":
    if False:
        z = GmshTriangularElement(5)
        print list(z.gmsh_node_tuples())
        print z.node_tuples()

        print

    z = GmshTriangularElement(2)
    #print list(z.gmsh_node_tuples())
    #print z.gmsh_node_tuples().next()
    #print
    #print set(z.gmsh_node_tuples())
    #print
    #print set(z.node_tuples()) - set(z.gmsh_node_tuples())
    #print "gmsh",list(z.gmsh_node_tuples())
    #print        list(z.gmsh_node_tuples())
    #print "hedge",[ x for x in z.node_tuples()]
    #map = z.hedge_to_gmsh_index_map()
    #print map
    print '------------------------------triangular------------------------------------------------'
    print z.unit_nodes()
    nodes =[numpy.array([0,0]),numpy.array([1,0]),numpy.array([1.0/2,numpy.sqrt(3.0)/2.0]),numpy.array([1.0/2,0]),numpy.array([3.0/4.0,numpy.sqrt(3)/4]),numpy.array([1.0/4,numpy.sqrt(3)/4.0])]
    print nodes
    p = numpy.array([0.5,0])
    print "f:", [ f(p)  for i,f in enumerate(list(z.basis_functions()))]
    print p,f
    print f(p),"----------------------here-------------------"
    print
    l_to_g = LocalToGlobalMap(nodes,z)
    print l_to_g(numpy.array([0.0,0.0]))
    print "-------------------tetrahedral------------------"
    z = GmshTetrahedralElement(1)
    node = [[-1,-1/numpy.sqrt(3),-1/numpy.sqrt(6)],[1,-1/numpy.sqrt(3),-1/numpy.sqrt(6)],[0,-2/numpy.sqrt(3),-1/numpy.sqrt(6)],[0,0,3/numpy.sqrt(6)]]
    nodes =[numpy.array(x) for x in node]
    print nodes
    p = numpy.array([-1,-1.0,-1.0])
    print "f:", [ f(p)  for i,f in enumerate(list(z.basis_functions()))]
    l_to_g = LocalToGlobalMap(nodes,z)
    print l_to_g(p)



