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
import pyublas
import hedge.tools
import hedge.mesh
import hedge._internal




class _FaceGroup(hedge._internal.FaceGroup):
    def __init__(self, double_sided):
        hedge._internal.FaceGroup.__init__(self, double_sided)
        self.face_index_lists = []
        self.face_index_list_register = {}

    def register_face_indices(self, identifier, generator):

        try:
            return self.face_index_list_register[identifier]
        except KeyError:
            new_idx = len(self.face_index_lists)
            fil = generator()
            self.face_index_lists.append(fil)
            self.face_index_list_register[identifier] = new_idx
            return new_idx

    def commit_face_index_lists(self):
        from hedge._internal import IntVector

        for fil in self.face_index_lists:
            intvec = IntVector(fil)

            self.index_lists.append(intvec)

        del self.face_index_lists
        del self.face_index_list_register





class _ElementGroup(object):
    """Once fully filled, this structure has the following data members:

    @ivar members: a list of hedge.mesh.Element instances in this group.
    @ivar local_discretization: an instance of hedge.element.Element.
    @ivar ranges: a list of C{(start, end)} tuples indicating the DOF numbers for
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
    def __init__(self, nodes, ranges, index_map, face_groups_and_ldis,
            el_face_to_face_group_and_flux_face_index={}):
        self.nodes = nodes
        self.ranges = ranges
        self.index_map = index_map
        self.face_groups_and_ldis = face_groups_and_ldis
        self.el_face_to_face_group_and_flux_face_index = \
                el_face_to_face_group_and_flux_face_index




class Discretization(object):
    """The global approximation space.

    Instances of this class tie together a local discretization (i.e. polynomials
    on an elemnent) into a function space on a mesh. They provide creation
    functions such as interpolating given functions, differential operators and
    flux lifting operators.
    """

    def __init__(self, mesh, local_discretization, debug=False):
        self.mesh = mesh

        self.dimensions = local_discretization.dimensions
        self.debug = debug

        self._build_element_groups_and_nodes(local_discretization)
        self._calculate_local_matrices()
        self._build_interior_face_groups()
        self.boundaries = {}

        # instrumentation -----------------------------------------------------
        from pytools.log import IntervalTimer, EventCounter

        self.inner_flux_counter = EventCounter("n_inner_flux", 
                "Number of inner flux computations")
        self.inner_flux_timer = IntervalTimer("t_inner_flux", 
                "Time spent computing inner fluxes")
        self.bdry_flux_counter = EventCounter("n_bdry_flux", 
                "Number of boundary flux computations")
        self.bdry_flux_timer = IntervalTimer("t_bdry_flux", 
                "Time spent computing boundary fluxes")

        self.mass_op_counter = EventCounter("n_mass_op", 
                "Number of mass operator applications")
        self.mass_op_timer = IntervalTimer("t_mass_op", 
                "Time spent applying mass operators")
        self.diff_op_counter = EventCounter("n_diff_op",
                "Number of differentiation operator applications")
        self.diff_op_timer = IntervalTimer("t_diff_op",
                "Time spent applying applying differentiation operators")

        self.interpolant_counter = EventCounter("n_interp", 
                "Number of interpolant evaluations")
        self.interpolant_timer = IntervalTimer("t_interp", 
                "Time spent evaluating interpolants")

        self.instrumented = False

    # instrumentation ---------------------------------------------------------
    def add_instrumentation(self, mgr):
        mgr.add_quantity(self.inner_flux_counter)
        mgr.add_quantity(self.inner_flux_timer)
        mgr.add_quantity(self.bdry_flux_counter)
        mgr.add_quantity(self.bdry_flux_timer)
        mgr.add_quantity(self.mass_op_counter)
        mgr.add_quantity(self.mass_op_timer)
        mgr.add_quantity(self.diff_op_counter)
        mgr.add_quantity(self.diff_op_timer)
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

        self.perform_inner_flux = \
                time_and_count_function(
                        self.perform_inner_flux,
                        self.inner_flux_timer,
                        self.inner_flux_counter)

        self._perform_boundary_flux = \
                time_and_count_function(
                        self._perform_boundary_flux,
                        self.bdry_flux_timer,
                        self.bdry_flux_counter)

        self.instrumented = True

    # initialization ----------------------------------------------------------
    def _build_element_groups_and_nodes(self, local_discretization):
        from hedge._internal import UniformElementRanges

        eg = _ElementGroup()
        eg.members = self.mesh.elements
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
            smats = eg.stiffness_matrices = numpy.array(
                    [numpy.dot(mmat, d) for d in dmats])
            smats = eg.stiffness_t_matrices = numpy.array(
                    [numpy.dot(d.T, mmat.T) for d in dmats])
            eg.minv_st = numpy.array(
                    [numpy.dot(numpy.dot(immat,d.T), mmat) for d in dmats])

            eg.jacobians = numpy.array([
                abs(el.map.jacobian) 
                for el in eg.members])
            eg.inverse_jacobians = numpy.array([
                abs(el.inverse_map.jacobian) 
                for el in eg.members])

            eg.diff_coefficients = [ # over global diff. coordinate
                    [ # local diff. coordinate
                        numpy.array([
                            el.inverse_map
                            .matrix[loc_coord, glob_coord]
                            for el in eg.members
                            ])
                        for loc_coord in range(ldis.dimensions)
                        ]
                    for glob_coord in range(ldis.dimensions)
                    ]

            eg.stiffness_coefficients = [ # over global diff. coordinate
                    [ # local diff. coordinate
                        numpy.array([
                            abs(el.map.jacobian)*el.inverse_map
                            .matrix[loc_coord, glob_coord]
                            for el in eg.members
                            ])
                        for loc_coord in range(ldis.dimensions)
                        ]
                    for glob_coord in range(ldis.dimensions)
                    ]

    def _make_flux_face(self, ldis, (el, fi)):
        from hedge.flux import FluxFace

        f = FluxFace()

        f.face_jacobian = el.face_jacobians[fi]
        f.element_id = el.id
        f.face_id = fi
        f.order = ldis.order
        f.normal = pyublas.why_not(el.face_normals[fi])

        # This crude approximation is shamelessly stolen from sledge.
        # There's an important caveat, however (which took me the better
        # part of a week to figure out):
        # h on both sides of an interface must be the same, otherwise
        # the penalty term will behave very oddly.
        f.h = abs(el.map.jacobian/f.face_jacobian)

        return f

    def _build_interior_face_groups(self):
        from hedge._internal import FaceGroup, FacePair
        from hedge.element import FaceVertexMismatch

        fg = _FaceGroup(double_sided=True)

        # find and match node indices along faces
        for i, (local_face, neigh_face) in enumerate(self.mesh.interfaces):
            e_l, fi_l = local_face
            e_n, fi_n = neigh_face

            (estart_l, eend_l), ldis_l = self.find_el_data(e_l.id)
            (estart_n, eend_n), ldis_n = self.find_el_data(e_n.id)

            vertices_l = e_l.faces[fi_l]
            vertices_n = e_n.faces[fi_n]

            findices_l = ldis_l.face_indices()[fi_l]
            findices_n = ldis_n.face_indices()[fi_n]

            try:
                findices_shuffle_op_n = \
                        ldis_l.get_face_index_shuffle_to_match(
                        vertices_l, vertices_n)

                if self.debug:
                    findices_shuffled_n = findices_shuffle_op_n(findices_n)

                    for i, j in zip(findices_l, findices_shuffled_n):
                        dist = self.nodes[estart_l+i]-self.nodes[estart_n+j]
                        assert la.norm(dist) < 1e-14

            except FaceVertexMismatch:
                # this happens if vertices_l is not a permutation of vertices_n.
                # periodicity is the only reason why that would be so.

                vertices_n, axis = self.mesh.periodic_opposite_faces[vertices_n]

                findices_shuffle_op_n = \
                        ldis_l.get_face_index_shuffle_to_match(vertices_l, vertices_n)

                if self.debug:
                    findices_shuffled_n = findices_shuffle_op_n(findices_n)

                    for i, j in zip(findices_l, findices_shuffled_n):
                        dist = self.nodes[estart_l+i]-self.nodes[estart_n+j]
                        dist[axis] = 0 
                        assert la.norm(dist) < 1e-14

            # create and fill the face pair
            fp = FacePair()

            fp.el_base_index = estart_l
            fp.opp_el_base_index = estart_n

            fp.face_index_list_number = fg.register_face_indices(
                    identifier=fi_l, 
                    generator=lambda: findices_l)
            fp.opp_face_index_list_number = fg.register_face_indices(
                    identifier=(fi_n, findices_shuffle_op_n),
                    generator=lambda : findices_shuffle_op_n(findices_n))

            fp.flux_face_index = len(fg.flux_faces)
            fp.opp_flux_face_index = len(fg.flux_faces)+1

            fg.face_pairs.append(fp)

            # create the flux faces
            flux_face_l = self._make_flux_face(ldis_l, local_face)
            flux_face_n = self._make_flux_face(ldis_n, neigh_face)

            # unify h across the faces
            flux_face_l.h = flux_face_n.h = max(flux_face_l.h, flux_face_n.h)

            fg.flux_faces.append(flux_face_l)
            fg.flux_faces.append(flux_face_n)

        fg.commit_face_index_lists()

        if len(fg.face_pairs):
            self.face_groups = [(fg, ldis_l.face_mass_matrix())]
        else:
            self.face_groups = []
        
    def _get_boundary(self, tag):
        """Get a _Boundary instance for a given `tag'.

        If there is no boundary tagged with `tag', an empty _Boundary instance
        is returned. Asking for a nonexistant boundary is not an error. 
        (Otherwise get_boundary would unnecessarily become non-local when run 
        in parallel.)
        """
        try:
            return self.boundaries[tag]
        except KeyError:
            pass

        from hedge._internal import IndexMap, FacePair

        nodes = []
        face_ranges = {}
        index_map = []
        face_group = _FaceGroup(double_sided=False)
        ldis = None # if this boundary is empty, we might as well have no ldis
        el_face_to_face_group_and_flux_face_index = {}

        for ef in self.mesh.tag_to_boundary.get(tag, []):
            el, face_nr = ef

            (el_start, el_end), ldis = self.find_el_data(el.id)
            face_indices = ldis.face_indices()[face_nr]

            f_start = len(nodes)
            nodes += [self.nodes[el_start+i] for i in face_indices]
            face_ranges[ef] = (f_start, len(nodes))
            index_map.extend(el_start+i for i in face_indices)

            # create the face pair
            fp = FacePair()
            fp.el_base_index = el_start
            fp.opp_el_base_index = f_start
            fp.face_index_list_number = face_group.register_face_indices(
                    identifier=face_nr,
                    generator=lambda: face_indices)
            fp.opp_face_index_list_number = face_group.register_face_indices(
                    identifier=(),
                    generator=lambda: tuple(xrange(len(face_indices))))
            fp.flux_face_index = len(face_group.flux_faces)
            fp.opp_flux_face_index = FacePair.INVALID_INDEX
            face_group.face_pairs.append(fp)

            # create the flux face
            face_group.flux_faces.append(self._make_flux_face(ldis, ef))
            
            # and make it possible to find it later
            el_face_to_face_group_and_flux_face_index[ef] = \
                    face_group, len(face_group.flux_faces)-1

        face_group.commit_face_index_lists()

        bdry = _Boundary(
                nodes=nodes,
                ranges=face_ranges,
                index_map=IndexMap(len(self.nodes), len(index_map), index_map),
                face_groups_and_ldis=[(face_group, ldis)],
                el_face_to_face_group_and_flux_face_index=
                el_face_to_face_group_and_flux_face_index)

        self.boundaries[tag] = bdry
        return bdry

    # vector construction -----------------------------------------------------
    def __len__(self):
        """Return the number of nodes in this discretization."""
        return len(self.nodes)

    def len_boundary(self, tag):
        return len(self._get_boundary(tag).nodes)

    def volume_empty(self, shape=()):
        return numpy.empty(shape+(len(self.nodes),), dtype=float)

    def volume_zeros(self, shape=()):
        return numpy.zeros(shape+(len(self.nodes),), dtype=float)

    def interpolate_volume_function(self, f):
        try:
            # are we interpolating many fields at once?
            shape = f.shape
        except AttributeError:
            # no, just one
            result = self.volume_zeros()
            for i, x in enumerate(self.nodes):
                result[i] = f(x)
            return result
        else:
            result = self.volume_zeros(shape)
            for point_nr, x in enumerate(self.nodes):
                result[(slice(None),)*len(shape) + (point_nr,)] = f(x)
            return result

    def boundary_zeros(self, tag=hedge.mesh.TAG_ALL, shape=()):
        return numpy.zeros(shape+(len(self._get_boundary(tag).nodes),),
                dtype=float)

    def interpolate_boundary_function(self, f, tag=hedge.mesh.TAG_ALL):
        try:
            # are we interpolating many fields at once?
            shape = f.shape
        except AttributeError:
            # no, just one
            result = self.boundary_zeros(tag=tag)
            for i, x in enumerate(self._get_boundary(tag).nodes):
                result[i] = f(x)
            return result
        else:
            result = self.boundary_zeros(tag=tag, shape=shape)
            for point_nr, x in enumerate(self._get_boundary(tag).nodes):
                result[(slice(None),)*len(shape) + (point_nr,)] = f(x)
            return result

    def boundary_normals(self, tag=hedge.mesh.TAG_ALL):
        result = self.boundary_zeros(tag=tag, shape=(self.dimensions,))
        for fg, ldis in self._get_boundary(tag).face_groups_and_ldis:
            for face_pair in fg.face_pairs:
                flux_face = fg.flux_faces[face_pair.flux_face_index]
                oeb = face_pair.opp_el_base_index
                opp_index_list = fg.index_lists[face_pair.opp_face_index_list_number]
                for i in opp_index_list:
                    result[:,oeb+i] = flux_face.normal

        return result

    # element data retrieval --------------------------------------------------
    def find_el_range(self, el_id):
        group, idx = self.group_map[el_id]
        return group.ranges[idx]

    def find_el_discretization(self, el_id):
        return self.group_map[el_id][0].local_discretization

    def find_el_data(self, el_id):
        group, idx = self.group_map[el_id]
        return group.ranges[idx], group.local_discretization

    # inner flux computation --------------------------------------------------
    def perform_inner_flux(self, int_flux, ext_flux, target):
        """Perform fluxes in the interior of the domain on the 
        given operator target.  This performs the contribution::

          M_{i,j} := \sum_{interior faces f} \int_f
            (  
               int_flux(f) * \phi_j
               + 
               ext_flux(f) * \phi_{opp(j)}
             )
             \phi_i
        
        on the given target. opp(j) denotes the dof with its node
        opposite from node j on the face opposite f.

        Thus the matrix product M*u, where u is the full volume field
        results in::

          v_i := M_{i,j} u_j
            = \sum_{interior faces f} \int_f
            (  
               int_flux(f) * u^-
               + 
               ext_flux(f) * u^+
             )
             \phi_i

        For more on operator targets, see src/cpp/op_target.hpp.

        Both local_flux and neighbor_flux must be instances of
        hedge.flux.Flux, i.e. compiled fluxes. Typically, you will
        not call this routine, it will be called for you by flux
        operators obtained by get_flux_operator().
        """
        from hedge._internal import perform_flux_on_one_target, ChainedFlux, NullTarget

        if isinstance(target, NullTarget):
            return

        ch_int = ChainedFlux(int_flux)
        ch_ext = ChainedFlux(ext_flux)

        target.begin(len(self.nodes), len(self.nodes))
        for fg, fmm in self.face_groups:
            perform_flux_on_one_target(fg, fmm, ch_int, ch_ext, target)
        target.finalize()

    # boundary flux computation -----------------------------------------------
    def _perform_boundary_flux(self, 
            int_flux, int_target, 
            ext_flux, ext_target, 
            bdry):
        from hedge._internal import perform_flux, ChainedFlux

        ch_int = ChainedFlux(int_flux)
        ch_ext = ChainedFlux(ext_flux)

        int_target.begin(len(self.nodes), len(self.nodes))
        ext_target.begin(len(self.nodes), len(bdry.nodes))
        if bdry.nodes:
            for fg, ldis in bdry.face_groups_and_ldis:
                perform_flux(fg, ldis.face_mass_matrix(), 
                        ch_int, int_target, 
                        ch_ext, ext_target)
        int_target.finalize()
        ext_target.finalize()

    # misc stuff --------------------------------------------------------------
    def dt_non_geometric_factor(self):
        distinct_ldis = set(eg.local_discretization for eg in self.element_groups)
        return min(ldis.dt_non_geometric_factor() 
                for ldis in distinct_ldis)

    def dt_geometric_factor(self):
        return min(min(eg.local_discretization.dt_geometric_factor(
            [self.mesh.points[i] for i in el.vertex_indices], el)
            for el in eg.members)
            for eg in self.element_groups)

    def dt_factor(self, max_system_ev):
        return 1/max_system_ev \
                * self.dt_non_geometric_factor() \
                * self.dt_geometric_factor()

    def volumize_boundary_field(self, bfield, tag=hedge.mesh.TAG_ALL):
        from hedge._internal import perform_inverse_index_map
        from hedge.tools import make_vector_target, log_shape

        ls = log_shape(bfield)
        result = self.volume_zeros(ls)
        bdry = self._get_boundary(tag)

        if ls != ():
            from pytools import indices_in_shape
            for i in indices_in_shape(ls):
                target = make_vector_target(bfield[i], result[i])
                target.begin(len(self.nodes), len(bdry.nodes))
                perform_inverse_index_map(bdry.index_map, target)
                target.finalize()
        else:
            target = make_vector_target(bfield, result)
            target.begin(len(self.nodes), len(bdry.nodes))
            perform_inverse_index_map(bdry.index_map, target)
            target.finalize()

        return result

    def boundarize_volume_field(self, field, tag=hedge.mesh.TAG_ALL):
        from hedge._internal import perform_index_map
        from hedge.tools import make_vector_target, log_shape

        ls = log_shape(field)
        result = self.boundary_zeros(tag, ls)
        bdry = self._get_boundary(tag)

        if ls != ():
            from pytools import indices_in_shape
            for i in indices_in_shape(ls):
                target = make_vector_target(field[i], result[i])
                target.begin(len(bdry.nodes), len(self.nodes))
                perform_index_map(bdry.index_map, target)
                target.finalize()
        else:
            target = make_vector_target(field, result)
            target.begin(len(bdry.nodes), len(self.nodes))
            perform_index_map(bdry.index_map, target)
            target.finalize()

        return result

    def find_element(self, idx):
        for i, (start, stop) in enumerate(self.element_group):
            if start <= idx < stop:
                return i
        raise ValueError, "not a valid dof index"
        
    def find_face(self, idx):
        el_id = self.find_element(idx)
        el_start, el_stop = self.element_group[el_id]
        for f_id, face_indices in enumerate(self.element_map[el_id].face_indices()):
            if idx-el_start in face_indices:
                return el_id, f_id, idx-el_start
        raise ValueError, "not a valid face dof index"

    # operator binding functions --------------------------------------------------
    @property
    def nabla(self):
        return DiffOperatorVector(self,
                [DifferentiationOperator(self, i) for i in range(self.dimensions)])

    @property
    def minv_stiffness_t(self):
        return DiffOperatorVector(self,
                [MInvSTOperator(self, i) for i in range(self.dimensions)])

    @property
    def stiffness_operator(self):
        return DiffOperatorVector(self,
                [StiffnessOperator(self, i) for i in range(self.dimensions)])

    @property
    def stiffness_t_operator(self):
        return DiffOperatorVector(self,
                [StiffnessTOperator(self, i) for i in range(self.dimensions)])

    @property
    def mass_operator(self):
        return MassOperator(self)

    @property
    def inverse_mass_operator(self):
        return InverseMassOperator(self)

    def get_flux_operator(self, flux, direct=True):
        """Return a flux operator that can be multiplied with
        a volume field to obtain the lifted interior fluxes
        or with a boundary pair to obtain the lifted boundary
        flux.

        `direct' determines whether the operator is applied in a
        matrix-free fashion or uses precomputed matrices.
        """

        from hedge.flux import compile_flux
        from hedge.tools import is_obj_array

        def get_scalar_flux_operator(flux):
            if direct:
                return _DirectFluxOperator(self, flux)
            else:
                return _FluxMatrixOperator(self, flux)

        if isinstance(flux, tuple):
            # a tuple of int/ext fluxes
            return get_scalar_flux_operator([(0,) + flux])
        elif is_obj_array(flux):
            return _VectorFluxOperator(self, 
                    [get_scalar_flux_operator(compile_flux(flux_component)) 
                        for flux_component in flux])
        else:
            return get_scalar_flux_operator(compile_flux(flux))




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
            for el, (el_start, el_stop) in zip(eg.members, eg.ranges):
                mapped_i_el = complete_el_map[el.id]
                mapped_start, mapped_stop = discr.find_el_range(mapped_i_el)
                for i_pt in range(el_start, el_stop):
                    pt = discr.nodes[i_pt]
                    mapped_pt = sym_map(pt)
                    for m_i_pt in range(mapped_start, mapped_stop):
                        if la.norm(discr.nodes[m_i_pt] - mapped_pt) < threshold:
                            self.map[i_pt] = m_i_pt
                            break

                    if i_pt not in self.map:
                        for m_i_pt in range(mapped_start, mapped_stop):
                            print comp.norm_2(discr.nodes[m_i_pt] - mapped_pt)
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




def ones_on_boundary(discr, tag=hedge.mesh.TAG_ALL):
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




def ones_on_volume(discr, tag=hedge.mesh.TAG_ALL):
    result = discr.volume_zeros()

    from hedge._internal import UniformElementRanges

    for eg in discr.element_groups:
        if isinstance(eg.ranges, UniformElementRanges):
            result[eg.ranges.start:
                    eg.ranges.start+len(eg.ranges)*eg.ranges.el_size] = 1
        else:
            for e_start, e_end in eg.ranges:
                result[e_start:e_end] = 1

    return result




def integral(discr, volume_vector, tag=hedge.mesh.TAG_ALL):
    try:
        mass_ones = discr._mass_ones
    except AttributeError:
        discr._mass_ones = mass_ones = discr.mass_operator * ones_on_volume(discr, tag)
    
    from hedge.tools import log_shape

    ls = log_shape(volume_vector)
    if ls == ():
        return numpy.dot(mass_ones, volume_vector)
    else:
        result = numpy.zeros(shape=ls, dtype=float)
        
        from pytools import indices_in_shape
        for i in indices_in_shape(ls):
            result[i] = numpy.dot(mass_ones, volume_vector[i])
            #result[i] = (discr.mass_operator*volume_vector[i]).sum()

        return result




def norm(discr, volume_vector, p=2):
    if p == numpy.Inf:
        return numpy.abs(volume_vector).max()
    else:
        from hedge.tools import log_shape

        if p != 2:
            volume_vector = numpy.abs(volume_vector)**(p/2)

        ls = log_shape(volume_vector)
        if ls == ():
            return float(numpy.dot(
                    volume_vector,
                    discr.mass_operator * volume_vector)**(1/p))
        else:
            assert len(ls) == 1
            return float(sum(
                    numpy.dot(
                        subv,
                        discr.mass_operator * subv)
                    for subv in volume_vector)**(1/p))

    

# operator wrapper ------------------------------------------------------------
class OpWrapper(hedge.tools.OperatorBase):
    def __init__(self,  discr_op):
        operator.Operator(num.Float64).__init__(self)
        self.discr_op = discr_op

    def size1(self):
        return len(self.discr_op.discr)

    def size2(self):
        return len(self.discr_op.discr)

    def apply(self, before, after):
        after[:] = self.discr_op.__mul__(before)





# operator algebra ------------------------------------------------------------
class DiscretizationVectorOperator(object):
    def __init__(self, discr):
        self.discr = discr

    def __add__(self, op2):
        return _OperatorSum(self, op2)

    def __sub__(self, op2):
        return _OperatorSum(self, -op2)

    def __neg__(self):
        return _ScalarMultipleOperator(-1, self)

    def __rmul__(self, scalar):
        value = float(scalar)
        return _ScalarMultipleOperator(scalar, self)

class _ScalarMultipleOperator(DiscretizationVectorOperator):
    def __init__(self, scalar, op):
        self.scalar = scalar
        self.op = op

    def __mul__(self, field):
        return self.scalar*(self.op*field)

class _OperatorSum(DiscretizationVectorOperator):
    def __init__(self, op1, op2):
        self.op1 = op1
        self.op2 = op2

    def __mul__(self, field):
        return self.op1*field + self.op2*field




# diff operators --------------------------------------------------------------
class _DiffResultsCache(object):
    def __init__(self, vector):
        self.vector = vector
        self.cache = {}

    def lookup(self, diff_op):
        return self.cache.get(diff_op, None)

    def set(self, diff_op, value):
        self.cache[diff_op] = value




def cache_diff_results(vec):
    from hedge.tools import log_shape
    ls = log_shape(vec)

    if ls == ():
        return _DiffResultsCache(vec)
    else:
        result = numpy.empty(ls, dtype=object)
        from pytools import indices_in_shape
        for i in indices_in_shape(ls):
            result[i] = _DiffResultsCache(vec[i])
        return result




class DiffOperatorBase(DiscretizationVectorOperator):
    def __init__(self, discr, xyz_axis):
        DiscretizationVectorOperator.__init__(self, discr)

        self.xyz_axis = xyz_axis

        self.do_warn = True

        if self.discr.instrumented:
            from pytools.log import time_and_count_function
            self.__mul__ = \
                    time_and_count_function(
                            self.__mul__,
                            self.discr.diff_op_timer,
                            self.discr.diff_op_counter)

    def do_not_warn(self):
        self.do_warn = False

    def diff_rst(self, rst_axis, field):
        result = self.discr.volume_zeros()

        from hedge.tools import make_vector_target
        target = make_vector_target(field, result)

        target.begin(len(self.discr), len(self.discr))

        from hedge._internal import perform_elwise_operator
        for eg in self.discr.element_groups:
            perform_elwise_operator(eg.ranges, eg.ranges, 
                    self.matrices(eg)[rst_axis], target)

        target.finalize()

        return result

    def diff_xyz(self, field, result):
        if isinstance(field, _DiffResultsCache):
            cache = field
            field = cache.vector
            rst_derivatives = cache.lookup(self.__class__)
        else:
            cache = None
            rst_derivatives = None
            if self.do_warn:
                from warnings import warn
                warn("wrap operand of differential operator in cache_diff_results() for speed")

        if rst_derivatives is None:
            rst_derivatives = [self.diff_rst(i, field) 
                    for i in range(self.discr.dimensions)]
            if cache:
                cache.set(self.__class__, rst_derivatives)

        from hedge.tools import make_vector_target
        from hedge._internal import perform_elwise_scale

        for rst_axis in range(self.discr.dimensions):
            target = make_vector_target(rst_derivatives[rst_axis], result)

            target.begin(len(self.discr), len(self.discr))
            for eg in self.discr.element_groups:
                perform_elwise_scale(eg.ranges,
                        self.coefficients(eg)[self.xyz_axis][rst_axis],
                        target)
            target.finalize()
        return result

    def __mul__(self, field):
        from hedge.tools import log_shape
        lshape = log_shape(field)

        result = self.discr.volume_zeros(lshape)

        if lshape == ():
            from pytools import indices_in_shape
            self.diff_xyz(field, result)
        else:
            from pytools import indices_in_shape
            for i in indices_in_shape(lshape):
                self.diff_xyz(field[i], result[i])

        return result
            
    def perform_on(self, target):
        from hedge._internal import perform_elwise_scaled_operator

        target.begin(len(self.discr), len(self.discr))

        for eg in self.discr.element_groups:
            for coeff, mat in zip(self.coefficients(eg)[self.xyz_axis], 
                    self.matrices(eg)):
                perform_elwise_scaled_operator(
                        eg.ranges, eg.ranges, coeff, mat, target)

        target.finalize()

class DifferentiationOperator(DiffOperatorBase):
    @staticmethod
    def matrices(element_group): return element_group.differentiation_matrices

    @staticmethod
    def coefficients(element_group): return element_group.diff_coefficients

class MInvSTOperator(DiffOperatorBase):
    @staticmethod
    def matrices(element_group): return element_group.minv_st

    @staticmethod
    def coefficients(element_group): return element_group.diff_coefficients

class StiffnessOperator(DiffOperatorBase):
    @staticmethod
    def matrices(element_group): return element_group.stiffness_matrices

    @staticmethod
    def coefficients(element_group): return element_group.stiffness_coefficients

class StiffnessTOperator(DiffOperatorBase):
    @staticmethod
    def matrices(element_group): return element_group.stiffness_t_matrices

    @staticmethod
    def coefficients(element_group): return element_group.stiffness_coefficients

class DiffOperatorVector(object):
    def __init__(self, discr, operators):
        self.discr = discr
        self.operators = operators

    def __len__(self):
        return len(self.operators)

    def __getitem__(self, i):
        return self.operators[i]

    def __mul__(self, field):
        result = self.discr.volume_zeros(shape=(len(self),))
        for i, op in enumerate(self):
            op.diff_xyz(field, result[i])
        return result




# mass operators --------------------------------------------------------------
class MassOperatorBase(DiscretizationVectorOperator):
    def __init__(self, discr):
        DiscretizationVectorOperator.__init__(self, discr)

        if self.discr.instrumented:
            from pytools.log import time_and_count_function
            self.perform_on = \
                    time_and_count_function(
                            self.perform_on,
                            self.discr.mass_op_timer,
                            self.discr.mass_op_counter)


    def __mul__(self, field):
        from hedge.tools import log_shape, make_vector_target
        lshape = log_shape(field)
        result = self.discr.volume_zeros(lshape)

        from pytools import indices_in_shape
        for i in indices_in_shape(lshape):
            self.perform_on(make_vector_target(field[i], result[i]))

        return result

    def perform_on(self, target):
        from hedge._internal import perform_elwise_scaled_operator
        target.begin(len(self.discr), len(self.discr))
        for eg in self.discr.element_groups:
            perform_elwise_scaled_operator(eg.ranges, eg.ranges,
                   self.coefficients(eg), self.matrix(eg), 
                   target)
        target.finalize()

class MassOperator(MassOperatorBase):
    @staticmethod
    def matrix(element_group): return element_group.mass_matrix

    @staticmethod
    def coefficients(element_group): return element_group.jacobians

class InverseMassOperator(MassOperatorBase):
    @staticmethod
    def matrix(element_group): return element_group.inverse_mass_matrix

    @staticmethod
    def coefficients(element_group): return element_group.inverse_jacobians




# flux operators --------------------------------------------------------------
class _DirectFluxOperator(DiscretizationVectorOperator):
    def __init__(self, discr, flux):
        DiscretizationVectorOperator.__init__(self, discr)
        self.flux = flux

    def mul_to(self, field, out):
        from hedge.tools import make_vector_target, log_shape

        if isinstance(field, BoundaryPair):
            # boundary flux
            bpair = field
            bdry = self.discr._get_boundary(bpair.tag)

            if not bdry.nodes:
                return 0

            field = bpair.field
            bfield = bpair.bfield

            class ZeroVector:
                dtype = 0
                def __getitem__(self, idx):
                    return 0

            if log_shape(field) != () or log_shape(field) != ():
                if isinstance(bfield, int) and bfield == 0:
                    bfield = ZeroVector()
                if isinstance(field, int) and field == 0:
                    field = ZeroVector()

                for idx, int_flux, ext_flux in self.flux:
                    if not (0 <= idx < len(field)):
                        raise RuntimeError, "flux depends on out-of-bounds field index"
                    self.discr._perform_boundary_flux(
                            int_flux, make_vector_target(field[idx], out),
                            ext_flux, make_vector_target(bfield[idx], out), 
                            bdry)
            else:
                if len(self.flux) == 1:
                    idx, int_flux, ext_flux = self.flux[0]
                    if idx != 0:
                        raise RuntimeError, "flux depends on out-of-bounds field index"
                    self.discr._perform_boundary_flux(
                            int_flux, make_vector_target(field, out),
                            ext_flux, make_vector_target(bfield, out), 
                            bdry)
                elif len(self.flux) > 1:
                    raise RuntimeError, "only found one field to process, but flux has multiple field dependencies"

            return out

        else:
            # inner flux
            from hedge.tools import log_shape
            ls = log_shape(field)
            if ls != ():
                assert len(ls) == 1
                for idx, int_flux, ext_flux in self.flux:
                    if not (0 <= idx < len(field)):
                        raise RuntimeError, "flux depends on out-of-bounds field index"
                    self.discr.perform_inner_flux(
                            int_flux, ext_flux, make_vector_target(field[idx], out))
            else:
                if len(self.flux) > 1:
                    raise RuntimeError, "only found one field to process, but flux has multiple field dependencies"

                idx, int_flux, ext_flux = self.flux[0]
                if idx != 0:
                    raise RuntimeError, "flux depends on out-of-bounds field index"
                self.discr.perform_inner_flux(
                        int_flux, ext_flux, make_vector_target(field, out))

            return out

    def __mul__(self, field):
        result = self.discr.volume_zeros()
        return self.mul_to(field, result)

    def perform_inner(self, tgt):
        dof = len(self.discr)

        for idx, int_flux, ext_flux in self.flux:
            self.discr.perform_inner_flux(int_flux, ext_flux, 
                    tgt.rebased_target(0, dof*idx))

    def perform_int_bdry(self, tag, tgt):
        from hedge._internal import NullTarget
        dof = len(self.discr)

        for idx, int_flux, ext_flux in self.flux:
            self.discr._perform_boundary_flux(
                    int_flux, tgt.rebased_target(0, dof*idx),
                    ext_flux, NullTarget(), 
                    self.discr._get_boundary(tag))




class _FluxMatrixOperator(DiscretizationVectorOperator):
    def __init__(self, discr, flux):
        DiscretizationVectorOperator.__init__(self, discr)

        from hedge._internal import MatrixTarget
        self.flux = flux

        dof = len(self.discr)

        self.inner_matrices = {}
        for idx, int_flux, ext_flux in self.flux:
            inner_bmatrix = pyublas.zeros(shape=(dof, dof), flavor=pyublas.SparseBuildMatrix)
            discr.perform_inner_flux(int_flux, ext_flux, MatrixTarget(inner_bmatrix))
            self.inner_matrices[idx] = \
                    pyublas.asarray(inner_bmatrix, flavor=pyublas.SparseExecuteMatrix)

        self.bdry_ops = {}

    def __mul__(self, field):

        def get_bdry_op(idx, bdry, int_flux, ext_flux):
            try:
                return self.bdry_ops[idx, bdry]
            except KeyError:
                dof = len(self.discr)
                bdry_dof = len(bdry.nodes)

                from hedge._internal import MatrixTarget

                int_bmatrix = pyublas.zeros(shape=(dof,dof), flavor=pyublas.SparseBuildMatrix)
                ext_bmatrix = pyublas.zeros(shape=(dof,bdry_dof), flavor=pyublas.SparseBuildMatrix)
                self.discr._perform_boundary_flux(
                        int_flux, MatrixTarget(int_bmatrix), 
                        ext_flux, MatrixTarget(ext_bmatrix), 
                        bdry)
                int_matrix = pyublas.asarray(int_bmatrix, flavor=pyublas.SparseExecuteMatrix)
                ext_matrix = pyublas.asarray(ext_bmatrix, flavor=pyublas.SparseExecuteMatrix)

                self.bdry_ops[idx, bdry] = int_matrix, ext_matrix

                return int_matrix, ext_matrix

        if isinstance(field, BoundaryPair):
            # boundary flux
            bpair = field
            bdry = self.discr._get_boundary(bpair.tag)

            if not bdry.nodes:
                return 0

            class ZeroVector:
                def __getitem__(self, idx):
                    return 0

            field = bpair.field
            bfield = bpair.bfield

            from hedge.tools import log_shape
            ls_f = log_shape(field)
            ls_bf = log_shape(bfield)

            if ls_f != () or ls_bf != ():
                if isinstance(bfield, int) and bfield == 0:
                    bfield = ZeroVector()
                elif isinstance(field, int) and field == 0:
                    field = ZeroVector()

                assert len(ls) == 1
                result = self.discr.volume_zeros()
                for idx, int_flux, ext_flux in self.flux:
                    int_matrix, ext_matrix = get_bdry_op(idx, bdry, int_flux, ext_flux)
                    result += int_matrix*bpair.field + ext_matrix*bpair.bfield
                return result
            else:
                assert len(self.flux) == 1
                idx, int_flux, ext_flux = self.flux[0]
                assert idx == 0

                int_matrix, ext_matrix = get_bdry_op(0, bdry, int_flux, ext_flux)
                return int_matrix*bpair.field + ext_matrix*bpair.bfield
        else:
            from hedge.tools import log_shape

            ls = log_shape(field)
            if ls != ():
                assert len(ls) == 1
                result = self.discr.volume_zeros()
                for idx, int_flux, ext_flux in self.flux:
                    result += self.inner_matrices[idx] * field[idx]
                return result
            else:
                assert len(self.flux) == 1
                idx, int_flux, ext_flux = self.flux[0]
                assert idx == 0
                return self.inner_matrices[0] * field

    def matrix_inner(self):
        """Returns a BlockMatrix to compute the lifting of the domain-interior fluxes.

        This matrix can only capture the interior part of the flux, the exterior-facing
        part is not taken into account.

        The different components are assumed to be run together in one vector,
        in order.
        """
        from hedge.tools import BlockMatrix
        flen = len(self.discr.volume_zeros())
        return BlockMatrix(
                (0, flen*idx, self.inner_matrices[idx])
                for idx, int_flux, ext_flux in self.flux)




class _VectorFluxOperator(object):
    def __init__(self, discr, flux_operators):
        self.discr = discr
        self.flux_operators = flux_operators

        if discr.instrumented:
            from pytools.log import time_and_count_function
            self._do_inner_flux = \
                    time_and_count_function(
                            self._do_inner_flux,
                            self.discr.inner_flux_timer,
                            self.discr.inner_flux_counter,
                            increment=sum(len(fo) for fo in self.flux_operators))

    def __mul__(self, field):
        if isinstance(field, BoundaryPair):
            result = self.discr.volume_zeros(
                    shape=(len(self.flux_operators),))

            for i, fop_i in enumerate(self.flux_operators):
                fop_i.mul_to(field, result[i])

            return result
        else:
            return self._do_inner_flux(field)

    def _do_inner_flux(self, field):
        # this is for performance -- it is faster to apply several fluxes
        # to a single operand at once
        result = self.discr.volume_zeros(
                shape=(len(self.flux_operators),))

        def find_field_flux(flux_op, i_field):
            for idx, int_flux, ext_flux in flux_op.flux:
                if idx == i_field:
                    return int_flux, ext_flux
            return None

        from hedge.tools import log_shape
        if log_shape(field) == ():
            field = field[numpy.newaxis,:]

        from hedge._internal import \
                perform_multiple_double_sided_fluxes_on_single_operand, \
                ChainedFlux
        for i_field, f_i in enumerate(field):
            fluxes_and_results = []
            for i_result, fo in enumerate(self.flux_operators):
                scalar_flux = find_field_flux(fo, i_field)
                if scalar_flux is not None:
                    int_flux, ext_flux = scalar_flux
                    fluxes_and_results.append(
                            (ChainedFlux(int_flux), 
                                ChainedFlux(ext_flux), 
                                result[i_result]))
            for fg, fmm in self.discr.face_groups:
                perform_multiple_double_sided_fluxes_on_single_operand(
                        fg, fmm, fluxes_and_results, f_i)

        return result






    def perform_inner(self, tgt):
        dof = len(self.discr)
        for i, fo in enumerate(self.flux_operators):
            fo.perform_inner(tgt.rebased_target(dof*i, 0))

    def perform_int_bdry(self, tag, tgt):
        dof = len(self.discr)
        for i, fo in enumerate(self.flux_operators):
            fo.perform_int_bdry(tag, tgt.rebased_target(dof*i, 0))




# boundary treatment ----------------------------------------------------------
class BoundaryPair(object):
    """Represents a pairing of a volume and a boundary field, used for the
    application of boundary fluxes.

    Do not use class constructor. Use L{pair_with_boundary}() instead.
    """
    def __init__(self, field, bfield, tag):
        self.field = field
        self.bfield = bfield
        self.tag = tag




def pair_with_boundary(field, bfield, tag=hedge.mesh.TAG_ALL):
    """Create a L{BoundaryPair} out of the volume field C{field}
    and the boundary field C{field}.

    Also accepts ArithmeticList or 0 for either argument.
    """
    return BoundaryPair(field, bfield, tag)




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
        from hedge._internal import perform_elwise_operator, VectorTarget
        from hedge.tools import log_shape

        ls = log_shape(from_vec)
        result = self.to_discr.volume_zeros(ls)

        from pytools import indices_in_shape
        for i in indices_in_shape(ls):
            target = VectorTarget(from_vec[i], result[i])

            target.begin(len(self.to_discr), len(self.from_discr))
            for from_eg, to_eg, imat in zip(
                    self.from_discr.element_groups, 
                    self.to_discr.element_groups, 
                    self.interp_matrices):
                perform_elwise_operator(
                        from_eg.ranges, to_eg.ranges, 
                        imat, target)

            target.finalize()

        return result




# filter ----------------------------------------------------------------------
class ExponentialFilterResponseFunction:
    """A typical exponential-falloff mode response filter function.

    See description in Section 5.6.1 of Jacobs/Hesthaven.
    """
    def __init__(self, min_amplification=0.1, order=6):
        """Construct the filter function.

        @arg min_amplification: The amplification factor applied to the highest mode.
        @arg order: The order of the filter. This controls how fast (or slow) the
          C{min_amplification} is reached.

        The amplification factor of the lowest-order (constant) mode is always 1.
        """
        from math import log
        self.alpha = -log(min_amplification)
        self.order = order

    def __call__(self, mode_idx, ldis):
        eta = max(mode_idx)/ldis.order

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

        self.filter_matrices = []
        for eg in discr.element_groups:
            ldis = eg.local_discretization

            node_count = ldis.node_count()

            filter_coeffs = [mode_response_func(mid, ldis)
                for mid in ldis.generate_mode_identifiers()] 

            # build filter matrix
            vdm = ldis.vandermonde()
            from hedge.tools import leftsolve
            from numpy import dot
            self.filter_matrices.append(numpy.asarray(
                leftsolve(vdm,
                    dot(vdm, numpy.diag(filter_coeffs))),
                order="C"))

    def __call__(self, vec):
        from hedge.tools import log_shape

        ls = log_shape(vec)
        result = self.discr.volume_zeros(ls)

        from pytools import indices_in_shape
        for i in indices_in_shape(ls):
            from hedge._internal import perform_elwise_operator, VectorTarget

            target = VectorTarget(vec[i], result[i])

            target.begin(len(self.discr), len(self.discr))
            for eg, fmat in zip(self.discr.element_groups, self.filter_matrices):
                perform_elwise_operator(eg.ranges, eg.ranges, fmat, target)

            target.finalize()

        return result
