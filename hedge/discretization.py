# -*- coding: utf8 -*-

"""Global function space discretization."""

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
import hedge.tools
import hedge.mesh
import hedge.optemplate
import hedge._internal
from pytools import memoize_method




class _FaceGroup(hedge._internal.FaceGroup):
    def __init__(self, double_sided, debug):
        hedge._internal.FaceGroup.__init__(self, double_sided)
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
                for side in [fp.loc, fp.opp]
                if side.element_id != hedge._internal.INVALID_ELEMENT))

        used_bases_and_els.sort()
        el_id_to_local_number = dict(
                (bae[1], i) for i, bae in enumerate(used_bases_and_els))
        self.local_el_to_global_el_base = numpy.fromiter(
                (bae[0] for bae in used_bases_and_els), dtype=numpy.uint32)

        for fp in self.face_pairs:
            for side in [fp.loc, fp.opp]:
                if side.element_id != hedge._internal.INVALID_ELEMENT:
                    side.local_el_number = el_id_to_local_number[side.element_id]

        # transfer inverse jacobians
        self.local_el_inverse_jacobians = numpy.fromiter(
                (abs(discr.mesh.elements[bae[1]].inverse_map.jacobian()) 
                    for bae in used_bases_and_els),
                dtype=float)

        self.ldis_loc = ldis_loc
        self.ldis_opp = ldis_opp




class _ElementGroup(object):
    """Once fully filled, this structure has the following data members:

    @ivar members: a list of hedge.mesh.Element instances in this group.
    @ivar member_nrs: a list of the element ID numbers in this group.
    @ivar local_discretization: an instance of hedge.element.Element.
    @ivar ranges: a list of C{slice} objects indicating the DOF numbers for
      each element. Note: This is actually a C++ ElementRanges object.
    @ivar mass_matrix: The element-local mass matrix M{M}.
    @ivar inverse_mass_matrix: the element-local inverese mass matrix M{M^{-1}}.
    @ivar differentiation_matrices: local differentiation matrices M{D_r, D_s, D_t}, 
      i.e.  differentiation by M{r, s, t, ....}.
    @ivar stiffness_matrices: the element-local stiffness matrices M{M*D_r, M*D_s,...}.
    @ivar jacobians: list of jacobians over all elements
    @ivar inverse_jacobians: inverses of L{jacobians}.
    @ivar diff_coefficients: a M{(d,d)}-matrix of coefficient vectors to turn
      M{(r,s,t)}-differentiation into M{(x,y,z)}.
    """
    pass




class _Boundary(object):
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

        for flux_face in [fp.loc, fp.opp]:
            if flux_face.element_id == el.id and flux_face.face_id == face_nbr:
                return flux_face
        raise KeyError, "flux face not found in boundary"




class OpTemplateFunction:
    def __init__(self, discr, pp_optemplate):
        self.discr = discr
        self.pp_optemplate = pp_optemplate

    def __call__(self, **vars):
        return self.discr.run_preprocessed_optemplate(self.pp_optemplate, vars)




class _PointEvaluator(object):
    def __init__(self, discr, el_range, interp_coeff):
        self.discr = discr
        self.el_range = el_range
        self.interp_coeff = interp_coeff

    def __call__(self, field):
        from hedge.tools import log_shape
        ls = log_shape(field)
        if ls != ():
            result = numpy.zeros(ls, dtype=self.discr.default_scalar_type)
            from pytools import indices_in_shape
            for i in indices_in_shape(ls):
                result[i] = numpy.dot(
                        self.interp_coeff, field[i][self.el_range])
            return result
        else:
            return numpy.dot(self.interp_coeff, field[self.el_range])


class Discretization(object):
    """The global approximation space.

    Instances of this class tie together a local discretization (i.e. polynomials
    on an elemnent) into a function space on a mesh. They provide creation
    functions such as interpolating given functions, differential operators and
    flux lifting operators.
    """

    @classmethod
    def all_debug_flags(cls):
        return set([
            "ilist_generation", 
            "node_permutation",
            "print_op_code",
            "dump_dataflow_graph",
            ])

    @classmethod
    def noninteractive_debug_flags(cls):
        return set([
            "ilist_generation", 
            "node_permutation",
            ])

    @staticmethod
    def get_local_discretization(mesh, local_discretization=None, order=None):
        if local_discretization is None and order is None:
            raise ValueError, "must supply either local_discretization or order"
        if local_discretization is not None and order is not None:
            raise ValueError, "must supply only one of local_discretization and order"
        if local_discretization is None:
            from hedge.element import ELEMENTS
            from pytools import one
            ldis_class = one(
                    ldis_class for ldis_class in ELEMENTS
                    if isinstance(mesh.elements[0], ldis_class.geometry))
            return ldis_class(order)
        else:
            return local_discretization

    def __init__(self, mesh, local_discretization=None, 
            order=None, debug=set(), default_scalar_type=numpy.float64,
            run_context=None):
        """

        @arg debug: A set of strings indicating which debug checks should
          be activated. See validity check below for the currently defined
          set of debug flags.
        """
          
        self.mesh = mesh

        local_discretization = self.get_local_discretization(
                mesh, local_discretization, order)

        self.dimensions = local_discretization.dimensions

        debug = set(debug)
        assert not debug.difference(self.all_debug_flags()), "Invalid debug flag specified"
        self.debug = debug

        self._build_element_groups_and_nodes(local_discretization)
        self._calculate_local_matrices()
        self._build_interior_face_groups()

        self.instrumented = False

        self.default_scalar_type = default_scalar_type

        self.exec_functions = {}

    def close(self):
        pass

    # instrumentation ---------------------------------------------------------
    def create_op_timers(self):
        from pytools.log import IntervalTimer

        self.gather_timer = IntervalTimer("t_gather", 
                "Time spent gathering fluxes")
        self.lift_timer = IntervalTimer("t_lift", 
                "Time spent lifting fluxes")
        self.mass_timer = IntervalTimer("t_mass", 
                "Time spent applying mass operators")
        self.diff_timer = IntervalTimer("t_diff",
                "Time spent applying applying differentiation operators")
        self.vector_math_timer = IntervalTimer("t_vector_math",
                "Time spent doing vector math")

        return [self.gather_timer, 
                self.lift_timer,
                self.mass_timer,
                self.diff_timer,
                self.vector_math_timer]

    def add_instrumentation(self, mgr):
        from pytools.log import IntervalTimer, EventCounter

        self.gather_counter = EventCounter("n_gather", 
                "Number of flux gather invocations")
        self.lift_counter = EventCounter("n_lift", 
                "Number of flux lift invocations")
        self.mass_counter = EventCounter("n_mass_op", 
                "Number of mass operator applications")
        self.diff_counter = EventCounter("n_diff",
                "Number of differentiation operator applications")

        self.gather_flop_counter = EventCounter("n_flops_gather",
                "Number of floating point operations in gather")
        self.lift_flop_counter = EventCounter("n_flops_lift",
                "Number of floating point operations in lift")
        self.mass_flop_counter = EventCounter("n_flops_mass",
                "Number of floating point operations in mass operator")
        self.diff_flop_counter = EventCounter("n_flops_diff",
                "Number of floating point operations in diff operator")
        self.vector_math_flop_counter = EventCounter("n_flops_vector_math",
                "Number of floating point operations in vector math")

        self.interpolant_counter = EventCounter("n_interp", 
                "Number of interpolant evaluations")

        self.interpolant_timer = IntervalTimer("t_interp", 
                "Time spent evaluating interpolants")

        for op in self.create_op_timers():
            mgr.add_quantity(op)

        mgr.add_quantity(self.gather_counter)
        mgr.add_quantity(self.lift_counter)
        mgr.add_quantity(self.mass_counter)
        mgr.add_quantity(self.diff_counter)

        mgr.add_quantity(self.gather_flop_counter)
        mgr.add_quantity(self.lift_flop_counter)
        mgr.add_quantity(self.mass_flop_counter)
        mgr.add_quantity(self.diff_flop_counter)
        mgr.add_quantity(self.vector_math_flop_counter)

        mgr.add_quantity(self.interpolant_counter)
        mgr.add_quantity(self.interpolant_timer)

        from pytools.log import time_and_count_function
        self.interpolate_volume_function = \
                time_and_count_function(
                        self.interpolate_volume_function,
                        self.interpolant_timer,
                        self.interpolant_counter)

        self.interpolate_boundary_function = \
                time_and_count_function(
                        self.interpolate_boundary_function,
                        self.interpolant_timer,
                        self.interpolant_counter)

        from pytools import single_valued
        try:
            order = single_valued(eg.local_discretization.order
                    for eg in self.element_groups)
        except ValueError:
            pass
        else:
            mgr.set_constant("dg_order", order)

        mgr.set_constant("default_type", self.default_scalar_type.__name__)
        mgr.set_constant("element_count", len(self.mesh.elements))
        mgr.set_constant("node_count", len(self.nodes))

        for f in self.all_debug_flags():
            mgr.set_constant("debug_%s" % f, f in self.debug)

        self.instrumented = True

    # initialization ----------------------------------------------------------
    def _build_element_groups_and_nodes(self, local_discretization):
        from hedge._internal import UniformElementRanges

        eg = _ElementGroup()
        eg.members = self.mesh.elements
        eg.member_nrs = numpy.fromiter((el.id for el in eg.members), dtype=numpy.uint32)
        eg.local_discretization = ldis = local_discretization
        eg.ranges = UniformElementRanges(
                0, 
                len(ldis.unit_nodes()), 
                len(self.mesh.elements))

        nodes_per_el = ldis.node_count()
        # mem layout:
        # [....element....][...element...]
        #  |    \
        #  [node.]
        #   | | |
        #   x y z

        # nodes should not be a multi-d array: this would break once
        # p-adaptivity is implemented
        self.nodes = numpy.empty(
                (len(self.mesh.elements)*nodes_per_el, self.dimensions),
                dtype=float, order="C")

        unit_nodes = numpy.empty( (nodes_per_el, self.dimensions),
                dtype=float, order="C")

        for i_node, node in enumerate(ldis.unit_nodes()):
            unit_nodes[i_node] = node

        from hedge._internal import map_element_nodes

        for el in self.mesh.elements:
            map_element_nodes(
                    self.nodes,
                    el.id*nodes_per_el*self.dimensions,
                    el.map,
                    unit_nodes,
                    self.dimensions)

        self.group_map = [(eg, i) for i in range(len(self.mesh.elements))]
        self.element_groups = [eg]

    def _calculate_local_matrices(self):
        for eg in self.element_groups:
            ldis = eg.local_discretization

            mmat = eg.mass_matrix = ldis.mass_matrix()
            immat = eg.inverse_mass_matrix = ldis.inverse_mass_matrix()
            dmats = eg.differentiation_matrices = \
                    ldis.differentiation_matrices()
            smats = eg.stiffness_matrices = \
                    [numpy.dot(mmat, d) for d in dmats]
            smats = eg.stiffness_t_matrices = \
                    [numpy.dot(d.T, mmat.T) for d in dmats]
            eg.minv_st = \
                    [numpy.dot(numpy.dot(immat,d.T), mmat) for d in dmats]

            eg.jacobians = numpy.array([
                abs(el.map.jacobian()) 
                for el in eg.members])
            eg.inverse_jacobians = numpy.array([
                abs(el.inverse_map.jacobian()) 
                for el in eg.members])

            eg.diff_coefficients = numpy.array([
                    [
                        [
                            el.inverse_map
                            .matrix[loc_coord, glob_coord]
                            for el in eg.members
                            ]
                        for loc_coord in range(ldis.dimensions)
                        ]
                    for glob_coord in range(ldis.dimensions)
                    ])

            eg.stiffness_coefficients = numpy.array([
                    [
                        [
                            abs(el.map.jacobian())*el.inverse_map
                            .matrix[loc_coord, glob_coord]
                            for el in eg.members
                            ]
                        for loc_coord in range(ldis.dimensions)
                        ]
                    for glob_coord in range(ldis.dimensions)
                    ])

    def _set_flux_face_data(self, f, ldis, (el, fi)):
        f.face_jacobian = el.face_jacobians[fi]
        f.element_id = el.id
        f.face_id = fi
        f.order = ldis.order
        f.normal = el.face_normals[fi]

        # This crude approximation is shamelessly stolen from sledge.
        # There's an important caveat, however (which took me the better
        # part of a week to figure out):
        # h on both sides of an interface must be the same, otherwise
        # the penalty term will behave very oddly.
        # This unification happens below.
        f.h = abs(el.map.jacobian()/f.face_jacobian)

    def _build_interior_face_groups(self):
        from hedge._internal import FacePair
        from hedge.element import FaceVertexMismatch

        fg = _FaceGroup(double_sided=True,
                debug="ilist_generation" in self.debug)

        all_ldis_l = []
        all_ldis_n = []

        debug_node_perm = "node_permutation" in self.debug

        # find and match node indices along faces
        for i, (local_face, neigh_face) in enumerate(self.mesh.interfaces):
            e_l, fi_l = local_face
            e_n, fi_n = neigh_face

            eslice_l, ldis_l = self.find_el_data(e_l.id)
            eslice_n, ldis_n = self.find_el_data(e_n.id)

            all_ldis_l.append(ldis_l)
            all_ldis_n.append(ldis_n)

            vertices_l = e_l.faces[fi_l]
            vertices_n = e_n.faces[fi_n]

            findices_l = ldis_l.face_indices()[fi_l]
            findices_n = ldis_n.face_indices()[fi_n]

            try:
                findices_shuffle_op_n = \
                        ldis_l.get_face_index_shuffle_to_match(
                        vertices_l, vertices_n)

                if debug_node_perm and ldis_l.has_facial_nodes and ldis_n.has_facial_nodes:
                    findices_shuffled_n = findices_shuffle_op_n(findices_n)

                    for i, j in zip(findices_l, findices_shuffled_n):
                        dist = self.nodes[eslice_l.start+i]-self.nodes[eslice_n.start+j]
                        assert la.norm(dist) < 1e-14

            except FaceVertexMismatch:
                # this happens if vertices_l is not a permutation of vertices_n.
                # periodicity is the only reason why that would be so.

                vertices_n, axis = self.mesh.periodic_opposite_faces[vertices_n]

                findices_shuffle_op_n = \
                        ldis_l.get_face_index_shuffle_to_match(vertices_l, vertices_n)

                if debug_node_perm and ldis_l.has_facial_nodes and ldis_n.has_facial_nodes:
                    findices_shuffled_n = findices_shuffle_op_n(findices_n)

                    for i, j in zip(findices_l, findices_shuffled_n):
                        dist = self.nodes[eslice_l.start+i]-self.nodes[eslice_n.start+j]
                        dist[axis] = 0 
                        assert la.norm(dist) < 1e-14

            # create and fill the face pair
            fp = FacePair()

            fp.loc.el_base_index = eslice_l.start
            fp.opp.el_base_index = eslice_n.start

            fp.loc.face_index_list_number = fg.register_face_index_list(
                    identifier=fi_l, 
                    generator=lambda: findices_l)
            fp.opp.face_index_list_number = fg.register_face_index_list(
                    identifier=(fi_n, findices_shuffle_op_n),
                    generator=lambda : findices_shuffle_op_n(findices_n))
            from pytools import get_write_to_map_from_permutation
            fp.opp_native_write_map = fg.register_face_index_list(
                    identifier=(fi_n, findices_shuffle_op_n, "wtm"),
                    generator=lambda : 
                    get_write_to_map_from_permutation(
                    findices_shuffle_op_n(findices_n), findices_n))

            self._set_flux_face_data(fp.loc, ldis_l, local_face)
            self._set_flux_face_data(fp.opp, ldis_n, neigh_face)

            # unify h across the faces
            fp.loc.h = fp.opp.h = max(fp.loc.h, fp.opp.h)

            assert len(fp.__dict__) == 0
            assert len(fp.loc.__dict__) == 0
            assert len(fp.opp.__dict__) == 0

            fg.face_pairs.append(fp)

        if len(fg.face_pairs):
            from pytools import single_valued
            ldis_l = single_valued(all_ldis_l)
            ldis_n = single_valued(all_ldis_n)

            fg.commit(self, ldis_l, ldis_n)

            self.face_groups = [fg]
        else:
            self.face_groups = []
        
    def boundary_nonempty(self, tag):
        return bool(self.mesh.tag_to_boundary.get(tag, []))

    @memoize_method
    def get_boundary(self, tag):
        """Get a _Boundary instance for a given `tag'.

        If there is no boundary tagged with `tag', an empty _Boundary instance
        is returned. Asking for a nonexistant boundary is not an error. 
        (Otherwise get_boundary would unnecessarily become non-local when run 
        in parallel.)
        """
        from hedge._internal import FacePair

        nodes = []
        face_ranges = {}
        vol_indices = []
        face_group = _FaceGroup(double_sided=False,
                debug="ilist_generation" in self.debug)
        ldis = None # if this boundary is empty, we might as well have no ldis
        el_face_to_face_group_and_face_pair = {}

        for ef in self.mesh.tag_to_boundary.get(tag, []):
            el, face_nr = ef

            el_slice, ldis = self.find_el_data(el.id)
            face_indices = ldis.face_indices()[face_nr]

            f_start = len(nodes)
            nodes += [self.nodes[el_slice.start+i] for i in face_indices]
            face_ranges[ef] = (f_start, len(nodes))
            vol_indices.extend(el_slice.start+i for i in face_indices)

            # create the face pair
            fp = FacePair()
            fp.loc.el_base_index = el_slice.start
            fp.opp.el_base_index = f_start
            fp.loc.face_index_list_number = face_group.register_face_index_list(
                    identifier=face_nr,
                    generator=lambda: face_indices)
            fp.opp.face_index_list_number = face_group.register_face_index_list(
                    identifier=(),
                    generator=lambda: tuple(xrange(len(face_indices))))
            self._set_flux_face_data(fp.loc, ldis, ef)
            assert len(fp.__dict__) == 0
            assert len(fp.loc.__dict__) == 0
            assert len(fp.opp.__dict__) == 0

            face_group.face_pairs.append(fp)

            # and make it possible to find it later
            el_face_to_face_group_and_face_pair[ef] = \
                    face_group, len(face_group.face_pairs)-1

        face_group.commit(self, ldis, ldis)

        bdry = _Boundary(
                nodes=numpy.array(nodes),
                ranges=face_ranges,
                vol_indices=numpy.asarray(vol_indices, dtype=numpy.intp),
                face_groups=[face_group],
                el_face_to_face_group_and_face_pair=
                el_face_to_face_group_and_face_pair)

        return bdry

    # vector construction -----------------------------------------------------
    def __len__(self):
        """Return the number of nodes in this discretization."""
        return len(self.nodes)

    def len_boundary(self, tag):
        return len(self.get_boundary(tag).nodes)

    def get_kind(self, field):
        return "numpy"

    compute_kind = "numpy"

    def convert_dtype(self, field, dtype):
        from hedge.tools import with_object_array_or_scalar
        if dtype is not None:
            return with_object_array_or_scalar(lambda f: f.astype(dtype), field)
        else:
            return field

    def convert_volume(self, field, kind):
        orig_kind = self.get_kind(field)

        if orig_kind != "numpy":
            raise ValueError, "unable to perform kind conversion: %s -> %s" % (
                    orig_kind, kind)

        return field

    def convert_boundary(self, field, tag, kind):
        orig_kind = self.get_kind(field)

        if orig_kind != "numpy":
            raise ValueError, "unable to perform kind conversion: %s -> %s" % (
                    orig_kind, kind)

        return field

    def convert_boundary_async(self, field, tag, kind, read_map=None):
        from hedge.tools import ImmediateFuture

        if read_map is not None:
            from hedge.tools import log_shape
            ls = log_shape(field)
            if field.dtype == object or ls == ():
                from hedge.tools import with_object_array_or_scalar
                field = with_object_array_or_scalar(
                        lambda f: f[read_map], field)
            else:
                field = numpy.asarray(
                        numpy.take(field, read_map, axis=len(ls)),
                        order="C")

        return ImmediateFuture(
                self.convert_boundary(field, tag, kind))

    def volume_empty(self, shape=(), dtype=None, kind="numpy"):
        if kind != "numpy":
            raise ValueError, "invalid vector kind requested"

        if dtype is None:
            dtype = self.default_scalar_type
        return numpy.empty(shape+(len(self.nodes),), dtype)

    def volume_zeros(self, shape=(), dtype=None, kind="numpy"):
        if kind != "numpy":
            raise ValueError, "invalid vector kind requested"

        if dtype is None:
            dtype = self.default_scalar_type
        return numpy.zeros(shape+(len(self.nodes),), dtype)

    def interpolate_volume_function(self, f, dtype=None, kind=None):
        if kind is None:
            kind = self.compute_kind

        try:
            # are we interpolating many fields at once?
            shape = f.shape
        except AttributeError:
            # no, just one
            shape = ()

        slice_pfx = (slice(None),)*len(shape)
        out = self.volume_empty(shape, dtype, kind="numpy")
        for eg in self.element_groups:
            for el, el_slice in zip(eg.members, eg.ranges):
                for point_nr in xrange(el_slice.start, el_slice.stop):
                    out[slice_pfx + (point_nr,)] = \
                                f(self.nodes[point_nr], el)
        return self.convert_volume(out, kind=kind)

    def boundary_empty(self, tag, shape=(), dtype=None, kind="numpy"):
        if kind not in ["numpy", "numpy-mpi-recv"]:
            raise ValueError, "invalid vector kind requested"

        if dtype is None:
            dtype = self.default_scalar_type
        return numpy.empty(shape+(len(self.get_boundary(tag).nodes),), dtype)

    def boundary_zeros(self, tag, shape=(), dtype=None, kind="numpy"):
        if kind not in ["numpy", "numpy-mpi-recv"]:
            raise ValueError, "invalid vector kind requested"
        if dtype is None:
            dtype = self.default_scalar_type

        return numpy.zeros(shape+(len(self.get_boundary(tag).nodes),), dtype)

    def interpolate_boundary_function(self, f, tag, dtype=None, kind=None):
        if kind is None:
            kind = self.compute_kind

        try:
            # are we interpolating many fields at once?
            shape = f.shape
        except AttributeError:
            # no, just one
            shape = ()

        out = self.boundary_zeros(tag, shape, dtype, kind="numpy")
        slice_pfx = (slice(None),)*len(shape)
        for point_nr, x in enumerate(self.get_boundary(tag).nodes):
            out[slice_pfx + (point_nr,)] = f(x, None) # FIXME

        return self.convert_boundary(out, tag, kind)

    @memoize_method
    def boundary_normals(self, tag, dtype=None, kind=None):
        if kind is None:
            kind = self.compute_kind

        result = self.boundary_zeros(shape=(self.dimensions,), tag=tag, dtype=dtype,
                kind="numpy")
        for fg in self.get_boundary(tag).face_groups:
            for face_pair in fg.face_pairs:
                oeb = face_pair.opp.el_base_index
                opp_index_list = fg.index_lists[face_pair.opp.face_index_list_number]
                for i in opp_index_list:
                    result[:,oeb+i] = face_pair.loc.normal

        return self.convert_boundary(result, tag, kind)

    def volumize_boundary_field(self, bfield, tag, kind=None):
        if kind is None:
            kind = self.compute_kind

        if kind != "numpy":
            raise ValueError("invalid target vector kind in volumize_boundary_field")

        bdry = self.get_boundary(tag)

        def f(subfld):
            result = self.volume_zeros(dtype=bfield.dtype, kind="numpy")
            result[bdry.vol_indices] = subfld
            return result

        from hedge.tools import with_object_array_or_scalar
        return with_object_array_or_scalar(f, bfield)

    def boundarize_volume_field(self, field, tag, kind=None):
        if kind is None:
            kind = self.compute_kind

        if kind != "numpy":
            raise ValueError("invalid target vector kind in boundarize_volume_field")

        bdry = self.get_boundary(tag)

        from hedge.tools import log_shape, is_obj_array
        ls = log_shape(field)

        if is_obj_array(field):
            if len(field) == 0:
                return numpy.zeros(())

            result = self.boundary_empty(tag, shape=ls, dtype=field[0].dtype)
            from pytools import indices_in_shape
            for i in indices_in_shape(ls):
                result[i] = field[i][bdry.vol_indices]

            return result
        else:
            return field[tuple(slice(None) for i in range(len(ls))) + (bdry.vol_indices,)]

    def boundarize_volume_field_async(self, field, tag, kind=None):
        from hedge.tools import ImmediateFuture
        return ImmediateFuture(
                self.boundarize_volume_field(field, tag, kind))

    def prepare_from_neighbor_map(self, indices):
        return numpy.array(indices, dtype=numpy.intp)

    # scalar reduction --------------------------------------------------------
    def nodewise_dot_product(self, a, b):
        return numpy.dot(a, b)

    @memoize_method
    def _mass_ones(self):
        from hedge.optemplate import MassOperator
        return MassOperator().apply(self, ones_on_volume(self))

    @memoize_method
    def mesh_volume(self):
        return self.integral(ones_on_volume(self))

    def integral(self, volume_vector):
        from hedge.tools import log_shape

        ls = log_shape(volume_vector)
        if ls == ():
            if isinstance(volume_vector, (int, float, complex)):
                # accept scalars as volume_vector
                empty = self.volume_empty(dtype=type(volume_vector))
                empty.fill(volume_vector)
                volume_vector = empty

            return self.nodewise_dot_product(
                    self._mass_ones(), volume_vector)
        else:
            result = numpy.zeros(shape=ls, dtype=float)
            
            from pytools import indices_in_shape
            for i in indices_in_shape(ls):
                vvi = volume_vector[i]
                if isinstance(vvi, (int, float)) and vvi == 0:
                    result[i] = 0
                else:
                    result[i] = self.nodewise_dot_product(
                            self._mass_ones(), volume_vector[i])

            return result

    @memoize_method
    def _compiled_mass_operator(self):
        from hedge.optemplate import MassOperator, Field
        mass_op_func = self.compile(MassOperator() * Field("f"))
        return lambda f: mass_op_func(f=f)

    def norm(self, volume_vector, p=2):
        if p == numpy.Inf:
            return numpy.abs(volume_vector).max()
        else:
            from hedge.tools import log_shape

            if p != 2:
                volume_vector = numpy.abs(volume_vector)**(p/2)

            return self.inner_product(
                    volume_vector,
                    volume_vector)**(1/p)

    def inner_product(self, a, b):
        mass_op = self._compiled_mass_operator()

        from hedge.tools import log_shape
        ls = log_shape(a)
        assert log_shape(b) == ls
        if ls == ():
            return float(self.nodewise_dot_product(
                    a, mass_op(b)))
        else:
            assert len(ls) == 1
            return float(sum(
                    self.nodewise_dot_product(
                        sub_a, mass_op(sub_b))
                    for sub_a, sub_b in zip(a,b)))

    # element data retrieval --------------------------------------------------
    def find_el_range(self, el_id):
        group, idx = self.group_map[el_id]
        return group.ranges[idx]

    def find_el_discretization(self, el_id):
        return self.group_map[el_id][0].local_discretization

    def find_el_data(self, el_id):
        group, idx = self.group_map[el_id]
        return group.ranges[idx], group.local_discretization

    def find_element(self, idx):
        for i, (start, stop) in enumerate(self.element_group):
            if start <= idx < stop:
                return i
        raise ValueError, "not a valid dof index"
        
    # misc stuff --------------------------------------------------------------
    @memoize_method
    def dt_non_geometric_factor(self):
        distinct_ldis = set(eg.local_discretization for eg in self.element_groups)
        return min(ldis.dt_non_geometric_factor() 
                for ldis in distinct_ldis)

    @memoize_method
    def dt_geometric_factor(self):
        return min(min(eg.local_discretization.dt_geometric_factor(
            [self.mesh.points[i] for i in el.vertex_indices], el)
            for el in eg.members)
            for eg in self.element_groups)

    def dt_factor(self, max_system_ev, stepper_class=None, *stepper_args):
        u"""Calculate the largest stable timestep, given a time stepper
        `stepper_class`. If none is given, RK4 is assumed.
        """

        # Calculating the correct timestep Δt for a DG scheme using the RK4
        # method is described in: "Nodal DG Methods, Algorithm, Analysis and
        # Applications" by J.S. Hesthaven & T. Warburton, p. 93, "Discrete
        # stability and timestep choise".  The implementation of timestep
        # calculation here is based upon this chapter.
        #
        # For a spatially continuous problem, the timestep can be calculated by
        # the following relation: 
        #
        #           max|λop| * Δt =  C_TimeStepper,
        #
        # where max|λop| is the maximum eigenvalue of the operator and
        # C_TimeStepper represents the maximum size of the stability region of
        # the timestepper along the imaginary axis. 
        #
        # For a DG-discretized problem another factor has to be added:
        #
        #            fDG = fNG * fG,
        #
        # fNG: non geometric factor fG:  geometric factor
        #
        # The discrete relation is: max|λop| * Δt = fDG * C_Timestepper
        #
        # Since the LocalDiscretization.dt_non_geometric_factor() and
        # LocalDiscretization.dt_geometric_factor() implicitly scale their
        # results for an RK4 time stepper, fDG includes already C_RK4 such as
        # fDG becomes fDG_RK4 and the relation is:
        #
        #           max|λop| * Δt = fDG_RK4
        #
        # As this is only sufficient for the use of RK4 timestepper but not for
        # any other implemented approache (e.g. Adams-Bashforth) additional
        # information about the size of the stability region is required to be
        # added into the relation.
        #
        # Unifying the relation with the size of the RK4 stability region and
        # multiplying it with the size of the specific timestepper stability
        # region brings out the correct relation:
        #
        #           max|λop| * Δt = fDG_RK4 / C_RK4 * C_TimeStepper
        #
        # C_TimeStepper gets calculated by a bisection method for every kind of
        # timestepper.


        rk4_dt = 1/max_system_ev \
                * self.dt_non_geometric_factor() \
                * self.dt_geometric_factor()

        from hedge.timestep import RK4TimeStepper
        if stepper_class is None or stepper_class == RK4TimeStepper:
            return rk4_dt
        else:
            assert isinstance(stepper_class, type)

            from hedge.timestep.stability import \
                    calculate_fudged_stability_region

            return rk4_dt \
                    * calculate_fudged_stability_region(
                            stepper_class, *stepper_args) \
                    / calculate_fudged_stability_region(RK4TimeStepper)

    def get_point_evaluator(self, point):
        for eg in self.element_groups:
            for el, rng in zip(eg.members, eg.ranges):
                if el.contains_point(point):
                    ldis = eg.local_discretization
                    basis_values = numpy.array([
                            phi(el.inverse_map(point)) 
                            for phi in ldis.basis_functions()])
                    vdm_t = ldis.vandermonde().T
                    return _PointEvaluator(
                            discr=self,
                            el_range=rng,
                            interp_coeff=la.solve(vdm_t, basis_values))

        raise RuntimeError, "point %s not found" % point

    # op template execution ---------------------------------------------------
    def compile(self, optemplate, post_bind_mapper=lambda x: x,
            is_vector_pred=lambda expr: True):
        ex = self.executor_class(self, optemplate, post_bind_mapper,
                is_vector_pred=is_vector_pred)

        if "dump_dataflow_graph" in self.debug:
            ex.code.dump_dataflow_graph()

        if self.instrumented:
            ex.instrument()
        return ex

    def add_function(self, name, func):
        self.exec_functions[name] = func




# random utilities ------------------------------------------------------------
class SymmetryMap(object):
    """A symmetry map on global DG functions.

    Suppose that the L{Mesh} on which a L{Discretization} is defined has
    is mapped onto itself by a nontrivial symmetry map M{f(.)}. Then
    this class allows you to carry out this map on vectors representing
    functions on this L{Discretization}.
    """
    def __init__(self, discr, sym_map, element_map, threshold=1e-13):
        self.discretization = discr

        complete_el_map = {}
        for i, j in element_map.iteritems():
            complete_el_map[i] = j
            complete_el_map[j] = i

        self.map = {}

        for eg in discr.element_groups:
            for el, el_slice in zip(eg.members, eg.ranges):
                mapped_i_el = complete_el_map[el.id]
                mapped_slice = discr.find_el_range(mapped_i_el)
                for i_pt in range(el_slice.start, el_slice.stop):
                    pt = discr.nodes[i_pt]
                    mapped_pt = sym_map(pt)
                    for m_i_pt in range(mapped_slice.start, mapped_slice.stop):
                        if la.norm(discr.nodes[m_i_pt] - mapped_pt) < threshold:
                            self.map[i_pt] = m_i_pt
                            break

                    if i_pt not in self.map:
                        for m_i_pt in range(mapped_slice.start, mapped_slice.stop):
                            print la.norm_2(discr.nodes[m_i_pt] - mapped_pt)
                        raise RuntimeError, "no symmetry match found"

    def __call__(self, vec):
        result = self.discretization.volume_zeros()
        for i, mapped_i in self.map.iteritems():
            result[mapped_i] = vec[i]
        return result




def generate_random_constant_on_elements(discr):
    result = discr.volume_zeros()
    import random
    for eg in discr.element_groups:
        for e_start, e_end in eg.ranges:
            result[e_start:e_end] = random.random()
    return result




def ones_on_boundary(discr, tag):
    result = discr.volume_zeros()

    try:
        faces = discr.mesh.tag_to_boundary[tag]
    except KeyError:
        pass
    else:
        for face in faces:
            el, fl = face

            (el_start, el_end), ldis = discr.find_el_data(el.id)
            fl_indices = ldis.face_indices()[fl]

            for i in fl_indices:
                result[el_start+i] = 1

    return result




def ones_on_volume(discr):
    result = discr.volume_empty()
    result.fill(1)
    return result




# projection between different discretizations --------------------------------
class Projector:
    def __init__(self, from_discr, to_discr):
        self.from_discr = from_discr
        self.to_discr = to_discr

        self.interp_matrices = []
        for from_eg, to_eg in zip(
                from_discr.element_groups, to_discr.element_groups):
            from_ldis = from_eg.local_discretization
            to_ldis = to_eg.local_discretization

            from_count = from_ldis.node_count()
            to_count = to_ldis.node_count()

            # check that the two element groups have the same members
            for from_el, to_el in zip(from_eg.members, to_eg.members):
                assert from_el is to_el

            from hedge.tools import permutation_matrix

            # assemble the from->to mode permutation matrix, guided by 
            # mode identifiers
            if to_count > from_count:
                to_node_ids_to_idx = dict(
                        (nid, i) for i, nid in 
                        enumerate(to_ldis.generate_mode_identifiers()))

                to_indices = [
                    to_node_ids_to_idx[from_nid]
                    for from_nid in from_ldis.generate_mode_identifiers()
                    ]

                pmat = permutation_matrix(
                    to_indices=to_indices, 
                    h=to_count, w=from_count)
            else:
                from_node_ids_to_idx = dict(
                        (nid, i) for i, nid in 
                        enumerate(from_ldis.generate_mode_identifiers()))

                from_indices = [
                    from_node_ids_to_idx[to_nid]
                    for to_nid in to_ldis.generate_mode_identifiers()
                    ]

                pmat = permutation_matrix(
                    from_indices=from_indices, 
                    h=to_count, w=from_count)

            # build interpolation matrix
            from_matrix = from_ldis.vandermonde()
            to_matrix = to_ldis.vandermonde()

            from hedge.tools import leftsolve
            from numpy import dot
            self.interp_matrices.append(
                    numpy.asarray(
                        leftsolve(from_matrix, dot(to_matrix, pmat)),
                        order="C"))

    def __call__(self, from_vec):
        from hedge._internal import perform_elwise_operator
        from hedge.tools import log_shape

        ls = log_shape(from_vec)
        result = self.to_discr.volume_zeros(ls, kind="numpy")

        from pytools import indices_in_shape
        for i in indices_in_shape(ls):
            for from_eg, to_eg, imat in zip(
                    self.from_discr.element_groups, 
                    self.to_discr.element_groups, 
                    self.interp_matrices):
                perform_elwise_operator(
                        from_eg.ranges, to_eg.ranges, 
                        imat, from_vec[i], result[i])

        return result




# filter ----------------------------------------------------------------------
class ExponentialFilterResponseFunction:
    """A typical exponential-falloff mode response filter function.

    See description in Section 5.6.1 of Hesthaven/Warburton.
    """
    def __init__(self, min_amplification=0.1, order=6):
        """Construct the filter function.

        @arg min_amplification: The amplification factor applied to the highest mode.
        @arg order: The order of the filter. This controls how fast (or slowly) the
          C{min_amplification} is reached.

        The amplification factor of the lowest-order (constant) mode is always 1.
        """
        from math import log
        self.alpha = -log(min_amplification)
        self.order = order

    def __call__(self, mode_idx, ldis):
        eta = sum(mode_idx)/ldis.order

        from math import exp
        return exp(-self.alpha * eta**self.order)




class Filter:
    def __init__(self, discr, mode_response_func):
        """Construct a filter.

        @arg discr: The L{Discretization} for which the filter is to be
          constructed.
        @mode_response_func: A function mapping 
          C{(mode_tuple, local_discretization)} to a float indicating the
          factor by which this mode is to be multiplied after filtering.
        """
        self.discr = discr

        self.filter_map = {}

        for eg in discr.element_groups:
            ldis = eg.local_discretization

            node_count = ldis.node_count()

            filter_coeffs = [mode_response_func(mid, ldis)
                for mid in ldis.generate_mode_identifiers()] 

            # build filter matrix
            vdm = ldis.vandermonde()
            from hedge.tools import leftsolve
            from numpy import dot
            mat = numpy.asarray(
                leftsolve(vdm,
                    dot(vdm, numpy.diag(filter_coeffs))),
                order="C")
            self.filter_map[eg] = mat

    def __call__(self, vec):
        from hedge.tools import log_shape

        ls = log_shape(vec)
        result = self.discr.volume_zeros(ls)

        from pytools import indices_in_shape
        for i in indices_in_shape(ls):
            from hedge._internal import perform_elwise_operator
            for eg in self.discr.element_groups:
                perform_elwise_operator(eg.ranges, eg.ranges, 
                        self.filter_map[eg], vec[i], result[i])

        return result

    def get_filter_matrix(self, el_group):
        return self.filter_map[el_group]
