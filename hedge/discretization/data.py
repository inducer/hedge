# -*- coding: utf8 -*-

"""Data structures for hedge.discretization."""

from __future__ import division

__copyright__ = "Copyright (C) 2007 Andreas Kloeckner"

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
import hedge._internal
from pytools import memoize_method




# {{{ discretization-level quadrature info ------------------------------------
class QuadratureInfo(object):
    """Discretization-level quadrature info.

    Once fully filled, this structure has the following data members:

    :ivar node_count: number of volume quadrature nodes.
    :ivar int_faces_node_count: number of interior-face quadrature nodes.
    """

# }}}
# {{{ element groups ----------------------------------------------------------
class ElementGroupBase(object):
    pass




class StraightElementGroup(ElementGroupBase):
    """Once fully filled, this structure has the following data members:

    :ivar members: a list of :class:`hedge.mesh.Element` instances in this group.
    :ivar member_nrs: a list of the element ID numbers in this group.
    :ivar local_discretization: an instance of 
      :class:`hedge.discretization.local.LocalDiscretization`.
    :ivar ranges: a list of :class:`slice` objects indicating the DOF numbers for
      each element. Note: This is actually a C++ ElementRanges object.
    :ivar mass_matrix: The element-local mass matrix :math:`M`.
    :ivar inverse_mass_matrix: the element-local inverse mass matrix :math:`M^{-1}`.
    :ivar differentiation_matrices: local differentiation matrices :math:`D_r, D_s, D_t`, 
      i.e.  differentiation by :math:`r, s, t, \dots`.
    :ivar stiffness_matrices: the element-local stiffness matrices :math:`MD_r, MD_s,\dots`.
    :ivar jacobians: list of jacobians over all elements
    :ivar inverse_jacobians: inverses of L{jacobians}.
    :ivar diff_coefficients: a :math:`d\\times d`-matrix of coefficient vectors to turn
      :math:`(r,s,t)`-differentiation into :math:`(x,y,z)`.
    :ivar quadrature_info: a map from quadrature tag to QuadratureInfo instance.
    """

    # {{{ quadrature info
    class QuadratureInfo:
        """
        :ivar ldis_quad_info: an instance of 
          :class:`hedge.discretization.local.Element.QuadratureInfo`.
        :ivar ranges: a list of :class:`slice` objects indicating the DOF numbers for
          each element. Note: This is actually a C++ ElementRanges object.
        :ivar mass_matrix: The element-local mass matrix :math:`M`.
        :ivar el_faces_ranges: a list of :class:`slice` objects
          indicating the DOF ranges for each element in my
          segment of the facial quadrature vector.  Notice that
          each element's range includes all its faces.
          Note: This is actually a C++ ElementRanges object.
        """
        def __init__(self, el_group, min_degree, 
                start_vol_node, start_int_faces_node):
            ldis = el_group.local_discretization

            ldis_quad_info = self.ldis_quad_info = \
                    ldis.get_quadrature_info(min_degree)

            from hedge._internal import UniformElementRanges
            self.ranges = UniformElementRanges(
                    start_vol_node,
                    ldis_quad_info.node_count(),
                    len(el_group.members))
            self.el_faces_ranges = UniformElementRanges(
                    start_int_faces_node,
                    ldis.face_count()*ldis_quad_info.face_node_count(),
                    len(el_group.members))

    # }}}






class CurvedElementGroup(ElementGroupBase):
    pass




# }}}
# {{{ face groups -------------------------------------------------------------
class StraightFaceGroup(hedge._internal.StraightFaceGroup):
    """
    Each face group has its own element numbering.

    :ivar ldis_loc: An instance of 
        :hedge.discretization.local.LocalDiscretization`,
        used for the interior side of each face.
    :ivar ldis_opp: An instance of 
        :hedge.discretization.local.LocalDiscretization`,
        used for the exterior side of each face.
    :ivar local_el_inverse_jacobians: A list of inverse
        Jacobians for each element.

    The following attributes are inherited from the C++ level:

    :ivar face_pairs: A list of face pair instances.
    :ivar double_sided: A :class:`bool` indicating whether this 
        face group is double-sided, i.e. represents both sides
        of each face-pair, or only the interior side.
    :ivar index_lists: A numpy array of shape 
        *(index_list_count, index_list_length)*.
    :ivar face_count: The number of faces of each element
        in :attr:`ldis_loc`.
    :ivar local_el_write_base: a list of global volume 
        element base indices, indexed in local element numbers.
    :ivar quadrature_info: a map from quadrature tag to QuadratureInfo instance.

    Further, the following methods are inherited from the C++ level:

    .. method:: element_count()
    .. method:: face_length()
    """

    def __init__(self, double_sided, debug):
        hedge._internal.StraightFaceGroup.__init__(self, double_sided)
        from hedge.tools import IndexListRegistry
        self.fil_registry = IndexListRegistry(debug)
        self.quadrature_info = {}

    def register_face_index_list(self, identifier, generator):
        return self.fil_registry.register(identifier, generator)

    def commit(self, discr, ldis_loc, ldis_opp, get_write_el_base=None):
        """
        :param get_write_el_base: a function of *(read_el_base, element_id)* 
          returning the DOF index to which data should be written post-lift.
          This is needed since on a quadrature grid, element base indices in a
          face pair refer to interior boundary vectors and are hence only
          usable for reading.
        """
        if self.fil_registry.index_lists:
            self.index_lists = numpy.array(
                    self.fil_registry.index_lists,
                    dtype=numpy.uint32, order="C")
            del self.fil_registry

        if ldis_loc is None:
            self.face_count = 0
        else:
            self.face_count = ldis_loc.face_count()

        # number elements locally
        used_bases_and_els = list(set(
                (side.el_base_index, side.element_id)
                for fp in self.face_pairs
                for side in [fp.int_side, fp.ext_side]
                if side.element_id != hedge._internal.INVALID_ELEMENT))

        if get_write_el_base is None:
            def get_write_el_base(read_base, el_id):
                return read_base

        used_bases_and_els.sort()
        el_id_to_local_number = dict(
                (bae[1], i) for i, bae in enumerate(used_bases_and_els))
        self.local_el_write_base = numpy.fromiter(
                (get_write_el_base(*bae) 
                    for bae in used_bases_and_els), dtype=numpy.uint32)

        for fp in self.face_pairs:
            for side in [fp.int_side, fp.ext_side]:
                if side.element_id != hedge._internal.INVALID_ELEMENT:
                    side.local_el_number = el_id_to_local_number[side.element_id]

        # transfer inverse jacobians
        self.local_el_inverse_jacobians = numpy.fromiter(
                (abs(discr.mesh.elements[bae[1]].inverse_map.jacobian()) 
                    for bae in used_bases_and_els),
                dtype=float)

        self.ldis_loc = ldis_loc
        self.ldis_opp = ldis_opp

    class QuadratureInfo:
        """
        """
        def __init__(self):
            pass






class CurvedFaceGroup(hedge._internal.CurvedFaceGroup):
    def __init__(self, double_sided, debug):
        hedge._internal.CurvedFaceGroup.__init__(self, double_sided)
        from hedge.tools import IndexListRegistry
        self.fil_registry = IndexListRegistry(debug)

    def register_face_index_list(self, identifier, generator):
        return self.fil_registry.register(identifier, generator)

    def commit(self, discr, ldis_loc, ldis_opp):
        if self.fil_registry.index_lists:
            self.index_lists = numpy.array(
                    self.fil_registry.index_lists,
                    dtype=numpy.uint32, order="C")
            del self.fil_registry

        if ldis_loc is None:
            self.face_count = 0
        else:
            self.face_count = ldis_loc.face_count()

        # number elements locally
        used_bases_and_els = list(set(
                (side.el_base_index, side.element_id)
                for fp in self.face_pairs
                for side in [fp.int_side, fp.ext_side]
                if side.element_id != hedge._internal.INVALID_ELEMENT))

        used_bases_and_els.sort()
        el_id_to_local_number = dict(
                (bae[1], i) for i, bae in enumerate(used_bases_and_els))
        self.local_el_write_base = numpy.fromiter(
                (bae[0] for bae in used_bases_and_els), dtype=numpy.uint32)

        for fp in self.face_pairs:
            for side in [fp.int_side, fp.ext_side]:
                if side.element_id != hedge._internal.INVALID_ELEMENT:
                    side.local_el_number = el_id_to_local_number[side.element_id]

        # transfer inverse jacobians
        self.local_el_inverse_jacobians = numpy.fromiter(
                (abs(discr.mesh.elements[bae[1]].inverse_map.jacobian()) 
                    for bae in used_bases_and_els),
                dtype=float)

        self.ldis_loc = ldis_loc
        self.ldis_opp = ldis_opp




class StraightCurvedFaceGroup(hedge._internal.StraightCurvedFaceGroup):
    def __init__(self, double_sided, debug):
        hedge._internal.StraightCurvedFaceGroup.__init__(self, double_sided)
        from hedge.tools import IndexListRegistry
        self.fil_registry = IndexListRegistry(debug)

    def register_face_index_list(self, identifier, generator):
        return self.fil_registry.register(identifier, generator)

    def commit(self, discr, ldis_loc, ldis_opp):
        if self.fil_registry.index_lists:
            self.index_lists = numpy.array(
                    self.fil_registry.index_lists,
                    dtype=numpy.uint32, order="C")
            del self.fil_registry

        if ldis_loc is None:
            self.face_count = 0
        else:
            self.face_count = ldis_loc.face_count()

        # number elements locally
        used_bases_and_els = list(set(
                (side.el_base_index, side.element_id)
                for fp in self.face_pairs
                for side in [fp.int_side, fp.ext_side]
                if side.element_id != hedge._internal.INVALID_ELEMENT))

        used_bases_and_els.sort()
        el_id_to_local_number = dict(
                (bae[1], i) for i, bae in enumerate(used_bases_and_els))

        self.local_el_write_base = numpy.fromiter(
                (bae[0] for bae in used_bases_and_els), dtype=numpy.uint32)

        for fp in self.face_pairs:
            for side in [fp.int_side, fp.ext_side]:
                if side.element_id != hedge._internal.INVALID_ELEMENT:
                    side.local_el_number = el_id_to_local_number[side.element_id]

        # transfer inverse jacobians
        self.local_el_inverse_jacobians = numpy.fromiter(
                (abs(discr.mesh.elements[bae[1]].inverse_map.jacobian()) 
                    for bae in used_bases_and_els),
                dtype=float)

        self.ldis_loc = ldis_loc
        self.ldis_opp = ldis_opp




# }}}
# {{{ boundary ----------------------------------------------------------------
class Boundary(object):
    """
    :ivar nodes: an array of node coordinates.
    :ivar vol_indices: a numpy intp of volume indices of all nodes,
        for quick data extraction from volume data.
    :ivar face_groups: a list of :class:`FaceGroup` instances.
    :ivar fg_ranges: a list of lists of :class:`slice` objects indicating the 
      DOF numbers in the boundary vector for each face. Note: The entries of 
      this list are actually C++ ElementRanges objects. There is one list per face
      group object, in the same order.
    :ivar el_face_to_face_group_and_face_pair:
    """
    def __init__(self, discr, nodes, vol_indices, face_groups, fg_ranges,
            el_face_to_face_group_and_face_pair={}):
        self.discr = discr
        self.nodes = nodes
        self.vol_indices = numpy.asarray(vol_indices, dtype=numpy.intp)
        self.face_groups = face_groups
        self.fg_ranges = fg_ranges
        self.el_face_to_face_group_and_face_pair = \
                el_face_to_face_group_and_face_pair

    def find_facepair(self, el_face):
        fg, fp_idx = self.el_face_to_face_group_and_face_pair[el_face]

        return fg.face_pairs[fp_idx]

    def find_facepair_side(self, el_face):
        fp = self.find_facepair(el_face)
        el, face_nbr = el_face

        for flux_face in [fp.int_side, fp.ext_side]:
            if flux_face.element_id == el.id and flux_face.face_id == face_nbr:
                return flux_face
        raise KeyError, "flux face not found in boundary"

    def is_empty(self):
        return len(self.nodes) == 0

    class QuadratureInfo:
        """
        Unless otherwise noted, attributes have the same meaning as above, but for 
        the quadrature grid.

        :ivar ldis_quad_info: 
        :ivar face_groups:
        :ivar fg_ranges:
        :ivar fg_ldis_quad_infos: An array of :class:`QuadratureInfo` instances
            belonging to a :class:`hedge.discretization.local.LocalDiscretization`.
            There is one instance for each face group object.
        :ivar node_count:
        """
        def __init__(self, face_groups, fg_ranges, fg_ldis_quad_infos, node_count):
            self.face_groups = face_groups
            self.fg_ranges = fg_ranges
            self.fg_ldis_quad_infos = fg_ldis_quad_infos
            self.node_count = node_count

    @memoize_method
    def get_quadrature_info(self, quadrature_tag):
        from hedge._internal import UniformElementRanges

        q_face_groups = []
        q_fg_ranges = []
        fg_ldis_quad_infos = []

        fg_start = 0
        for fg, fg_range in zip(self.face_groups, self.fg_ranges):
            ldis = fg.ldis_loc
            ldis_q_info = ldis.get_quadrature_info(
                    self.discr.quad_min_degrees[quadrature_tag])
            fg_ldis_quad_infos.append(ldis_q_info)

            q_fg_range = UniformElementRanges(
                fg_start, # FIXME: need to vary element starts
                ldis_q_info.face_node_count(), len(fg_range))
            q_fg_ranges.append(q_fg_range)

            fg_start += q_fg_range.total_size

        return self.QuadratureInfo(
                face_groups=q_face_groups,
                fg_ranges=q_fg_ranges,
                fg_ldis_quad_infos=fg_ldis_quad_infos,
                node_count=fg_start)

        return q_info





# }}}
# vim: foldmethod=marker
