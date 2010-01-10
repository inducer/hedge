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




# {{{ quadrature info ---------------------------------------------------------
class QuadratureInfo(object):
    """Once fully filled, this structure has the following data members:

    :ivar node_count: number of quadrature nodes across the whole mesh.
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
        """
        def __init__(self, start_qnode, el_group, min_degree):
            ldis = el_group.local_discretization

            ldis_quad_info = self.ldis_quad_info = \
                    ldis.quadrature_info(min_degree)

            from hedge._internal import UniformElementRanges
            self.ranges = UniformElementRanges(
                    start_qnode,
                    ldis_quad_info.node_count(),
                    len(el_group.members))

            self.mass_matrix = numpy.asarray(
                    la.solve(
                        ldis.vandermonde().T,
                        numpy.dot(
                            ldis_quad_info.vandermonde().T,
                            numpy.diag(ldis_quad_info.volume_weights))),
                    order="C")

            self.stiffness_t_matrices = [numpy.asarray(
                    la.solve(
                        ldis.vandermonde().T,
                        numpy.dot(
                            diff_vdm.T,
                            numpy.diag(ldis_quad_info.volume_weights))),
                    order="C")
                    for diff_vdm in ldis_quad_info.diff_vandermonde_matrices()]

    # }}}






class CurvedElementGroup(ElementGroupBase):
    pass




# }}}
# {{{ face groups -------------------------------------------------------------
class StraightFaceGroup(hedge._internal.StraightFaceGroup):
    def __init__(self, double_sided, debug):
        hedge._internal.StraightFaceGroup.__init__(self, double_sided)
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
        self.local_el_to_global_el_base = numpy.fromiter(
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
        self.local_el_to_global_el_base = numpy.fromiter(
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
        self.local_el_to_global_el_base = numpy.fromiter(
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
    def __init__(self, nodes, ranges, vol_indices, face_groups,
            el_face_to_face_group_and_face_pair={}):
        self.nodes = nodes
        self.ranges = ranges
        self.vol_indices = vol_indices
        self.face_groups = face_groups
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




# }}}
# vim: foldmethod=marker
