"""Reader for the GMSH file format."""

from __future__ import division

__copyright__ = "Copyright (C) 2009 Xueyu Zhu, Andreas Kloeckner"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import numpy as np
import numpy.linalg as la
from pytools import memoize_method, Record, single_valued
from hedge.discretization.local import \
        IntervalDiscretization, \
        TriangleDiscretization, \
        TetrahedronDiscretization
from meshpy.gmsh_reader import GmshMeshReceiverBase, GmshPoint



# {{{ tools

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

# }}}





# {{{ gmsh element info

class HedgeGmshElementBase(object):
    @memoize_method
    def get_lexicographic_gmsh_node_indices(self):
        gmsh_tup_to_index = dict(
                (tup, i)
                for i, tup in enumerate(self.gmsh_node_tuples()))

        return [gmsh_tup_to_index[tup]
                for tup in self.node_tuples()]

    @memoize_method
    def equidistant_vandermonde(self):
        from hedge.polynomial import generic_vandermonde

        return generic_vandermonde(
                list(self.equidistant_unit_nodes()),
                list(self.basis_functions()))




class HedgeGmshIntervalElement(IntervalDiscretization, HedgeGmshElementBase):
    @memoize_method
    def gmsh_node_tuples(self):
        return [(0,), (self.order,),] + [
                (i,) for i in range(1, self.order)]




class HedgeGmshIncompleteTriangularElement(HedgeGmshElementBase):
    dimensions = 2

    def __init__(self, order):
        self.order = order

    def node_count(self):
        return len(self.gmsh_node_tuples())

    @memoize_method
    def gmsh_node_tuples(self):
        result = []
        for tup in generate_triangle_vertex_tuples(self.order):
            result.append(tup)
        for tup in generate_triangle_edge_tuples(self.order):
            result.append(tup)
        return result





class HedgeGmshTriangularElement(TriangleDiscretization, HedgeGmshElementBase):
    @memoize_method
    def gmsh_node_tuples(self):
        result = []
        for tup in generate_triangle_vertex_tuples(self.order):
            result.append(tup)
        for tup in generate_triangle_edge_tuples(self.order):
            result.append(tup)
        for tup in generate_triangle_volume_tuples(self.order):
            result.append(tup)
        return result




class HedgeGmshTetrahedralElement(TetrahedronDiscretization, HedgeGmshElementBase):
    @memoize_method
    def gmsh_node_tuples(self):
        # gmsh's node ordering is on crack
        return {
                1: [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)],
                2: [
                    (0, 0, 0), (2, 0, 0), (0, 2, 0), (0, 0, 2), (1, 0, 0), (1, 1, 0),
                    (0, 1, 0), (0, 0, 1), (0, 1, 1), (1, 0, 1)],
                3: [
                    (0, 0, 0), (3, 0, 0), (0, 3, 0), (0, 0, 3), (1, 0, 0), (2, 0, 0),
                    (2, 1, 0), (1, 2, 0), (0, 2, 0), (0, 1, 0), (0, 0, 2), (0, 0, 1),
                    (0, 1, 2), (0, 2, 1), (1, 0, 2), (2, 0, 1), (1, 1, 0), (1, 0, 1),
                    (0, 1, 1), (1, 1, 1)],
                4: [
                    (0, 0, 0), (4, 0, 0), (0, 4, 0), (0, 0, 4), (1, 0, 0), (2, 0, 0),
                    (3, 0, 0), (3, 1, 0), (2, 2, 0), (1, 3, 0), (0, 3, 0), (0, 2, 0),
                    (0, 1, 0), (0, 0, 3), (0, 0, 2), (0, 0, 1), (0, 1, 3), (0, 2, 2),
                    (0, 3, 1), (1, 0, 3), (2, 0, 2), (3, 0, 1), (1, 1, 0), (1, 2, 0),
                    (2, 1, 0), (1, 0, 1), (2, 0, 1), (1, 0, 2), (0, 1, 1), (0, 1, 2),
                    (0, 2, 1), (1, 1, 2), (2, 1, 1), (1, 2, 1), (1, 1, 1)],
                5: [
                    (0, 0, 0), (5, 0, 0), (0, 5, 0), (0, 0, 5), (1, 0, 0), (2, 0, 0),
                    (3, 0, 0), (4, 0, 0), (4, 1, 0), (3, 2, 0), (2, 3, 0), (1, 4, 0),
                    (0, 4, 0), (0, 3, 0), (0, 2, 0), (0, 1, 0), (0, 0, 4), (0, 0, 3),
                    (0, 0, 2), (0, 0, 1), (0, 1, 4), (0, 2, 3), (0, 3, 2), (0, 4, 1),
                    (1, 0, 4), (2, 0, 3), (3, 0, 2), (4, 0, 1), (1, 1, 0), (1, 3, 0),
                    (3, 1, 0), (1, 2, 0), (2, 2, 0), (2, 1, 0), (1, 0, 1), (3, 0, 1),
                    (1, 0, 3), (2, 0, 1), (2, 0, 2), (1, 0, 2), (0, 1, 1), (0, 1, 3),
                    (0, 3, 1), (0, 1, 2), (0, 2, 2), (0, 2, 1), (1, 1, 3), (3, 1, 1),
                    (1, 3, 1), (2, 1, 2), (2, 2, 1), (1, 2, 2), (1, 1, 1), (2, 1, 1),
                    (1, 2, 1), (1, 1, 2)],
                }[self.order]

# }}}

# {{{ local-to-global map

class LocalToGlobalMap(object):
    def __init__(self, nodes, ldis):
        self.nodes = nodes
        self.ldis  = ldis

        node_src_indices = np.array(
                ldis.get_lexicographic_gmsh_node_indices(),
                dtype=np.intp)

        nodes = np.array(nodes, dtype=np.float64)
        reordered_nodes = nodes[node_src_indices, :]

        self.modal_coeff = la.solve(
                ldis.equidistant_vandermonde(), reordered_nodes)
        # axis 0: node number, axis 1: xyz axis

        if False:
            for i, c in zip(ldis.generate_mode_identifiers(), self.modal_coeff):
                print i, c

    def __call__(self, r):
        """Given a point *r* on the reference element, return the
        corresponding point *x* in global coordinates.
        """
        mc = self.modal_coeff

        return np.array([sum([
            mc[i, axis] * mbf(r)
            for i, mbf in enumerate(self.ldis.basis_functions())])
            for axis in range(self.ldis.dimensions)])

    def is_affine(self):
        from pytools import any

        has_high_order_geometry = any(
                sum(mid) >= 2 and abs(mc) >= 1e-13
                for mc_along_axis in self.modal_coeff.T
                for mid, mc in zip(
                    self.ldis.generate_mode_identifiers(),
                    mc_along_axis)
                )

        return not has_high_order_geometry

# }}}

# {{{ gmsh mesh receiver

class ElementInfo(Record):
    pass

class HedgeGmshMeshReceiver(GmshMeshReceiverBase):
    gmsh_element_type_to_info_map = {
            1:  HedgeGmshIntervalElement(1),
            2:  HedgeGmshTriangularElement(1),
            4:  HedgeGmshTetrahedralElement(1),
            8:  HedgeGmshIntervalElement(2),
            9:  HedgeGmshTriangularElement(2),
            11: HedgeGmshTetrahedralElement(2),
            15: GmshPoint(0),
            20: HedgeGmshIncompleteTriangularElement(3),
            21: HedgeGmshTriangularElement(3),
            22: HedgeGmshIncompleteTriangularElement(4),
            23: HedgeGmshTriangularElement(4),
            24: HedgeGmshIncompleteTriangularElement(5),
            25: HedgeGmshTriangularElement(5),
            26: HedgeGmshIntervalElement(3),
            27: HedgeGmshIntervalElement(4),
            28: HedgeGmshIntervalElement(5),
            29: HedgeGmshTetrahedralElement(3),
            30: HedgeGmshTetrahedralElement(4),
            31: HedgeGmshTetrahedralElement(5)
            }

    # {{{ intake

    def __init__(self, dimensions, tag_mapper):
        if dimensions is None:
            dimensions = 3
        self.dimensions = dimensions
        self.tag_mapper = tag_mapper

        # maps (tag_number, dimension) -> tag_name
        self.tag_name_map = {}
        self.gmsh_vertex_nrs_to_element = {}

    def set_up_nodes(self, count):
        self.nodes = np.empty((count, self.dimensions))

    def add_node(self, node_nr, point):
        self.nodes[node_nr] = point

    def add_element(self, element_nr, element_type, vertex_nrs,
            lexicographic_nodes, tag_numbers):
        self.gmsh_vertex_nrs_to_element[frozenset(vertex_nrs)] = ElementInfo(
                    index=element_nr,
                    el_type=element_type,
                    node_indices=lexicographic_nodes,
                    gmsh_vertex_indices=vertex_nrs,
                    tag_numbers=tag_numbers)

    def add_tag(self, name, index, dimension):
        self.tag_name_map[index, dimension] = self.tag_mapper(name)

    # }}}

    # {{{ mesh construction

    def build_mesh(self, periodicity, allow_internal_boundaries, tag_mapper):
        # figure out dimensionalities
        vol_dim = max(el.el_type.dimensions for key, el in
                self.gmsh_vertex_nrs_to_element.iteritems() )

        vol_elements = [el for key, el in self.gmsh_vertex_nrs_to_element.iteritems()
                if el.el_type.dimensions == vol_dim]

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
                hedge_vertices.append(self.nodes[gmsh_node_nr])
                gmsh_node_nr_to_hedge_vertex_nr[gmsh_node_nr] = hedge_vertex_nr
                return hedge_vertex_nr

        for el_nr, gmsh_el in enumerate(vol_elements):
            el_map = LocalToGlobalMap(
                    [self.nodes[ni] for ni in  gmsh_el.node_indices],
                    gmsh_el.el_type)
            is_affine = el_map.is_affine()

            el_class = gmsh_el.el_type.geometry
            if not is_affine:
                try:
                    el_class = TO_CURVED_CLASS[el_class]
                except KeyError:
                    raise NotImplementedError("unsupported curved gmsh element type %s" % el_class)

            vertex_indices = [get_vertex_nr(gmsh_node_nr)
                    for gmsh_node_nr in gmsh_el.gmsh_vertex_indices]

            if is_affine:
                hedge_el = el_class(el_nr, vertex_indices, hedge_vertices)
            else:
                hedge_el = el_class(el_nr, vertex_indices, el_map)

            hedge_elements.append(hedge_el)
            hedge_el_to_gmsh_element[hedge_el] = gmsh_el

        from pytools import reverse_dictionary
        hedge_vertex_nr_to_gmsh_node_nr = reverse_dictionary(
                gmsh_node_nr_to_hedge_vertex_nr)

        del vol_elements

        def volume_tagger(el, all_v):
            return [self.tag_name_map[tag_nr, el.dimensions]
                    for tag_nr in hedge_el_to_gmsh_element[el].tag_numbers
                    if (tag_nr, el.dimensions) in self.tag_name_map]

        def boundary_tagger(fvi, el, fn, all_v):
            gmsh_vertex_nrs = frozenset(
                    hedge_vertex_nr_to_gmsh_node_nr[face_vertex_index]
                    for face_vertex_index in fvi)

            try:
                gmsh_element = self.gmsh_vertex_nrs_to_element[gmsh_vertex_nrs]
            except KeyError:
                return []
            else:
                x = [self.tag_name_map[tag_nr, el.dimensions-1]
                        for tag_nr in gmsh_element.tag_numbers
                        if (tag_nr, el.dimensions-1) in self.tag_name_map]
                if len(x) > 1:
                    from pudb import set_trace; set_trace()
                return x

        vertex_array = np.array(hedge_vertices, dtype=np.float64)
        pt_dim = vertex_array.shape[-1]
        if pt_dim != vol_dim:
            from warnings import warn
            warn("Found %d-dimensional mesh embedded in %d-dimensional space. "
                    "Hedge only supports meshes of zero codimension (for now). "
                    "Maybe you want to set force_dimension=%d?"
                    % (vol_dim, pt_dim, vol_dim))

        from hedge.mesh import make_conformal_mesh_ext
        return make_conformal_mesh_ext(
                vertex_array,
                hedge_elements,
                boundary_tagger=boundary_tagger,
                volume_tagger=volume_tagger,
                periodicity=periodicity,
                allow_internal_boundaries=allow_internal_boundaries)

    # }}}

# }}}



# {{{ front-end functions

def read_gmsh(filename, force_dimension=None, periodicity=None,
        allow_internal_boundaries=False,
        tag_mapper=lambda tag: tag):
    """
    :param force_dimension: if not None, truncate point coordinates to this many dimensions.
    """

    mr = HedgeGmshMeshReceiver(force_dimension, tag_mapper)
    from meshpy.gmsh_reader import read_gmsh
    read_gmsh(mr, filename, force_dimension=force_dimension)

    return mr.build_mesh(periodicity=periodicity,
            allow_internal_boundaries=allow_internal_boundaries,
            tag_mapper=tag_mapper)




def generate_gmsh(source, dimensions, order=None, other_options=[],
            extension="geo", gmsh_executable="gmsh",
            force_dimension=None, periodicity=None,
            allow_internal_boundaries=False,
            tag_mapper=lambda tag: tag):

    mr = HedgeGmshMeshReceiver(force_dimension, tag_mapper)
    from meshpy.gmsh_reader import generate_gmsh
    generate_gmsh(mr, source, dimensions, order, other_options, extension,
            gmsh_executable, force_dimension=force_dimension)

    return mr.build_mesh(periodicity=periodicity,
            allow_internal_boundaries=allow_internal_boundaries,
            tag_mapper=tag_mapper)

# }}}




# vim: fdm=marker
