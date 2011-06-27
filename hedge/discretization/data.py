# -*- coding: utf8 -*-

"""Data structures for hedge.discretization."""

from __future__ import division

__copyright__ = "Copyright (C) 2007 Andreas Kloeckner"

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
    :ivar volume_jacobians: Full-volume vector of jacobians on this
      quadrature grid.

    :ivar inverse_metric_derivatives: A list of lists of full-volume 
        vectors, such that the vector
        *inverse_metric_derivatives[xyz_axis][rst_axis]* gives the metric
        derivatives on the entire volume for this quadrature grid

        .. math::
            \frac{d r_{\mathtt{rst\_axis}} }{d x_{\mathtt{xyz\_axis}} }
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
    :ivar quadrature_info: a map from quadrature tag to QuadratureInfo instance.
    """

    def el_array_from_volume(self, vol_array):
        """Return a 2-dimensional view of *vol_array* in which the first
        dimension numbers elements within this element group and the second
        dimension numbers nodes within each of those elements.
        """
        return (vol_array[self.ranges.start:self.ranges.start+self.ranges.total_size]
                .reshape(len(self.ranges), -1))

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

        def el_array_from_volume(self, vol_array):
            """Return a 2-dimensional view of *vol_array* in which the first
            dimension numbers elements within this element group and the second
            dimension numbers nodes within each of those elements.
            """
            return (vol_array[
                self.ranges.start:self.ranges.start+self.ranges.total_size]
                .reshape(len(self.ranges), -1))
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

    Face groups on quadrature grids additionally have these
    properties:

    :ivar ldis_loc_quad_info: refer to 
      :class:`hedge.discretization.local.LocalDiscretization.QuadratureInfo`
      instance relevant for this face group's :attr:`ldis_loc`.
    :ivar ldis_opp_quad_info: refer to 
      :class:`hedge.discretization.local.LocalDiscretization.QuadratureInfo`
      instance relevant for this face group's :attr:`ldis_opp`.

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
        discr = self.discr

        from hedge._internal import UniformElementRanges

        quad_face_groups = []
        quad_fg_ranges = []
        fg_ldis_quad_infos = []

        fg_start = 0
        f_start = 0

        for fg, fg_range in zip(self.face_groups, self.fg_ranges):
            ldis = fg.ldis_loc

            # get quadrature info from local discretization
            ldis_quad_info = ldis.get_quadrature_info(
                    self.discr.quad_min_degrees[quadrature_tag])
            fg_ldis_quad_infos.append(ldis_quad_info)
            quad_fnc = ldis_quad_info.face_node_count()

            # create element ranges
            quad_fg_range = UniformElementRanges(
                fg_start,
                quad_fnc, len(fg_range))
            quad_fg_ranges.append(quad_fg_range)

            fg_start += quad_fg_range.total_size

            # create the quadrature face group
            fg_type = StraightFaceGroup
            quad_fg = type(fg)(double_sided=False,
                    debug="ilist_generation" in discr.debug)
            quad_face_groups.append(quad_fg)

            # create quadrature face pairs
            for fp in fg.face_pairs:
                el = discr.mesh.elements[fp.int_side.element_id]
                face_nr = fp.int_side.face_id
                ef = el, face_nr

                quad_fp = quad_fg.FacePair()

                def find_el_base_index(el):
                    group, idx = discr.group_map[el.id]
                    return group.quadrature_info[quadrature_tag].el_faces_ranges[idx].start

                face_indices = tuple(range(quad_fnc*face_nr, quad_fnc*(face_nr+1)))

                quad_fp.int_side.el_base_index = find_el_base_index(el)
                quad_fp.ext_side.el_base_index = f_start
                quad_fp.int_side.face_index_list_number = quad_fg.register_face_index_list(
                        identifier=face_nr,
                        generator=lambda: face_indices)
                quad_fp.ext_side.face_index_list_number = quad_fg.register_face_index_list(
                        identifier=(),
                        generator=lambda: tuple(xrange(quad_fnc)))
                self.discr._set_flux_face_data(quad_fp.int_side, ldis, ef)

                # check that all property assigns found their C++-side slots
                assert len(fp.__dict__) == 0
                assert len(fp.int_side.__dict__) == 0
                assert len(fp.ext_side.__dict__) == 0

                quad_fg.face_pairs.append(quad_fp)

                f_start += quad_fnc

            assert f_start == fg_start

            if len(quad_fg.face_pairs):
                def get_write_el_base(read_base, el_id):
                    return discr.find_el_range(el_id).start

                quad_fg.commit(discr, ldis, ldis, get_write_el_base)

                quad_fg.ldis_loc_quad_info = ldis_quad_info
                quad_fg.ldis_opp_quad_info = ldis_quad_info

        return self.QuadratureInfo(
                face_groups=quad_face_groups,
                fg_ranges=quad_fg_ranges,
                fg_ldis_quad_infos=fg_ldis_quad_infos,
                node_count=fg_start)

        return q_info





# }}}
# vim: foldmethod=marker
