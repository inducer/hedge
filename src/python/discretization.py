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




import pylinear.array as num
import pylinear.computation as comp
import pylinear.operator as operator
from pytools.arithmetic_container import \
        work_with_arithmetic_containers, \
        ArithmeticList
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

            # allow prefetching a few entries past the end
            for i in range(4):
                intvec.append(intvec[-1])

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

    # initialization ----------------------------------------------------------
    def _build_element_groups_and_nodes(self, local_discretization):
        from hedge._internal import UniformElementRanges
        from hedge.tools import FixedSizeSliceAdapter

        eg = _ElementGroup()
        eg.members = self.mesh.elements
        eg.local_discretization = ldis = local_discretization
        eg.ranges = UniformElementRanges(
                0, 
                len(ldis.unit_nodes()), 
                len(self.mesh.elements))

        nodes_per_el = ldis.node_count()
        self.nodes = FixedSizeSliceAdapter(
                num.empty((self.dimensions * nodes_per_el * len(self.mesh.elements),)),
                self.dimensions)

        unit_nodes = FixedSizeSliceAdapter(
                num.empty((self.dimensions * nodes_per_el,)),
                self.dimensions)
        for i_node, node in enumerate(ldis.unit_nodes()):
            unit_nodes[i_node] = node

        from hedge._internal import map_element_nodes

        for el in self.mesh.elements:
            map_element_nodes(
                    self.nodes.adaptee,
                    el.id*nodes_per_el*self.dimensions,
                    el.map,
                    unit_nodes.adaptee,
                    self.dimensions)

        self.group_map = [(eg, i) for i in range(len(self.mesh.elements))]
        self.element_groups = [eg]

    def _calculate_local_matrices(self):
        from pytools.arithmetic_container import ArithmeticList
        AL = ArithmeticList

        for eg in self.element_groups:
            ldis = eg.local_discretization

            mmat = eg.mass_matrix = ldis.mass_matrix()
            immat = eg.inverse_mass_matrix = ldis.inverse_mass_matrix()
            dmats = eg.differentiation_matrices = \
                    ldis.differentiation_matrices()
            smats = eg.stiffness_matrices = AL(mmat*d for d in dmats)
            smats = eg.stiffness_t_matrices = AL(d.T*mmat.T for d in dmats)
            eg.minv_st = AL(immat*d.T*mmat for d in dmats)

            eg.jacobians = num.array([
                abs(el.map.jacobian) 
                for el in eg.members])
            eg.inverse_jacobians = num.array([
                abs(el.inverse_map.jacobian) 
                for el in eg.members])

            eg.diff_coefficients = [ # over global diff. coordinate
                    [ # local diff. coordinate
                        num.array([
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
                        num.array([
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
        f.normal = el.face_normals[fi]

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
                        assert comp.norm_2(dist) < 1e-14

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
                        assert comp.norm_2(dist) < 1e-14

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

    def volume_zeros(self):
        return num.zeros((len(self.nodes),))

    def interpolate_volume_function(self, f):
        self.interpolant_counter.add()

        try:
            # are we interpolating many fields at once?
            shape = f.shape
        except AttributeError:
            # no, just one
            result = self.volume_zeros()
            self.interpolant_timer.start()
            result[:] = (f(x) for x in self.nodes)
            self.interpolant_timer.stop()
            return result
        else:
            if len(f.shape) == 1:
                (count,) = f.shape
                result = ArithmeticList([self.volume_zeros() for i in range(count)])

                self.interpolant_timer.start()
                for point_nr, x in enumerate(self.nodes):
                    for field_nr, value in enumerate(f(x)):
                        result[field_nr][point_nr] = value
                self.interpolant_timer.stop()

                return result
            elif len(f.shape) == 2:
                h, w = f.shape
                result = [[self.volume_zeros() for j in range(w)] for i in range(h)]

                self.interpolant_timer.start()
                for point_nr, x in enumerate(self.nodes):
                    for i, row in enumerate(f(x)):
                        for j, entry in enumerate(row):
                            result[i][j][point_nr] = entry
                self.interpolant_timer.stop()

                from pytools.arithmetic_container import ArithmeticListMatrix
                return ArithmeticListMatrix(result)
            else:
                raise NotImplementedError, "only scalars, vectors and matrices are "\
                        "supported for volume interpolation"

    def boundary_zeros(self, tag=hedge.mesh.TAG_ALL):
        return num.zeros((len(self._get_boundary(tag).nodes),))

    def interpolate_boundary_function(self, f, tag=hedge.mesh.TAG_ALL):
        self.interpolant_counter.add()

        try:
            # are we interpolating many fields at once?
            shape = f.shape
        except AttributeError:
            # no, just one
            self.interpolant_timer.start()
            result = num.array([f(x) for x in self._get_boundary(tag).nodes])
            self.interpolant_timer.stop()
            return result
        else:
            if len(f.shape) == 1:
                (count,) = f.shape

                self.interpolant_timer.start()
                result = ArithmeticList([self.boundary_zeros(tag) for i in range(count)])
                for i, pt in enumerate(self._get_boundary(tag).nodes):
                    for field_nr, value in enumerate(f(pt)):
                        result[field_nr][i] = value
                self.interpolant_timer.stop()

                return result
            elif len(f.shape) == 2:
                h, w = f.shape

                self.interpolant_timer.start()
                result = [[self.boundary_zeros(tag) for j in range(w)] for i in range(h)]

                for point_nr, x in enumerate(self._get_boundary(tag).nodes):
                    for i, row in enumerate(f(x)):
                        for j, entry in enumerate(row):
                            result[i][j][point_nr] = entry
                self.interpolant_timer.stop()

                from pytools.arithmetic_container import ArithmeticListMatrix
                return ArithmeticListMatrix(result)
            else:
                raise NotImplementedError, "only scalars, vectors and matrices vectors are supported "\
                        "for boundary interpolation"

    def boundary_normals(self, tag=hedge.mesh.TAG_ALL):
        result = ArithmeticList([self.boundary_zeros(tag) for i in range(self.dimensions)])
        for fg, ldis in self._get_boundary(tag).face_groups_and_ldis:
            for face_pair in fg.face_pairs:
                flux_face = fg.flux_faces[face_pair.flux_face_index]
                normal = flux_face.normal
                oeb = face_pair.opp_el_base_index
                opp_index_list = fg.index_lists[face_pair.opp_face_index_list_number]
                for i in opp_index_list:
                    for j in range(self.dimensions):
                        result[j][oeb+i] = normal[j]

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
        self.inner_flux_counter.add()
        self.inner_flux_timer.start()

        from hedge._internal import perform_flux_on_one_target, ChainedFlux, NullTarget

        if isinstance(target, NullTarget):
            return

        ch_int = ChainedFlux(int_flux)
        ch_ext = ChainedFlux(ext_flux)

        target.begin(len(self.nodes), len(self.nodes))
        for fg, fmm in self.face_groups:
            perform_flux_on_one_target(fg, fmm, ch_int, ch_ext, target)
        target.finalize()

        self.inner_flux_timer.stop()

    # boundary flux computation -----------------------------------------------
    def perform_boundary_flux(self, 
            int_flux, int_target, 
            ext_flux, ext_target, 
            tag=hedge.mesh.TAG_ALL):
        self.bdry_flux_counter.add()
        self.bdry_flux_timer.start()

        from hedge._internal import perform_flux, ChainedFlux

        ch_int = ChainedFlux(int_flux)
        ch_ext = ChainedFlux(ext_flux)

        bdry = self._get_boundary(tag)

        int_target.begin(len(self.nodes), len(self.nodes))
        ext_target.begin(len(self.nodes), len(bdry.nodes))
        if bdry.nodes:
            for fg, ldis in bdry.face_groups_and_ldis:
                perform_flux(fg, ldis.face_mass_matrix(), 
                        ch_int, int_target, 
                        ch_ext, ext_target)
        int_target.finalize()
        ext_target.finalize()

        self.bdry_flux_timer.stop()

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

    @work_with_arithmetic_containers
    def volumize_boundary_field(self, bfield, tag=hedge.mesh.TAG_ALL):
        from hedge._internal import perform_inverse_index_map
        from hedge.tools import make_vector_target

        result = self.volume_zeros()
        bdry = self._get_boundary(tag)

        target = make_vector_target(bfield, result)
        target.begin(len(self.nodes), len(bdry.nodes))
        perform_inverse_index_map(bdry.index_map, target)
        target.finalize()

        return result

    @work_with_arithmetic_containers
    def boundarize_volume_field(self, field, tag=hedge.mesh.TAG_ALL):
        from hedge._internal import perform_index_map
        from hedge.tools import make_vector_target

        result = self.boundary_zeros(tag)

        bdry = self._get_boundary(tag)

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
        return ArithmeticList(
                DifferentiationOperator(self, i) for i in range(self.dimensions))

    @property
    def minv_stiffness_t(self):
        return ArithmeticList(
                MInvSTOperator(self, i) for i in range(self.dimensions))

    @property
    def stiffness_operator(self):
        return ArithmeticList(
                StiffnessOperator(self, i) for i in range(self.dimensions))

    @property
    def stiffness_t_operator(self):
        return ArithmeticList(
                StiffnessTOperator(self, i) for i in range(self.dimensions))
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

        def get_scalar_flux_operator(flux):
            if direct:
                return _DirectFluxOperator(self, flux)
            else:
                return _FluxMatrixOperator(self, flux)

        if isinstance(flux, tuple):
            # a tuple of int/ext fluxes
            return get_scalar_flux_operator([(0,) + flux])
        elif isinstance(flux, ArithmeticList):
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
                        if comp.norm_2(discr.nodes[m_i_pt] - mapped_pt) < threshold:
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




@work_with_arithmetic_containers
def integral(discr, volume_vector, tag=hedge.mesh.TAG_ALL):
    try:
        mass_ones = discr._mass_ones
    except AttributeError:
        discr._mass_ones = mass_ones = discr.mass_operator * ones_on_volume(discr, tag)
    
    return mass_ones * volume_vector




# pylinear operator wrapper ---------------------------------------------------
class PylinearOpWrapper(hedge.tools.PylinearOperator):
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
class _DiffResultCache(object):
    def __init__(self, vector):
        self.vector = vector
        self.cache = {}

def cache_diff_results(vec):
    from pytools.arithmetic_container import ArithmeticList
    if isinstance(vec, list):
        return [_DiffResultCache(subvec) for subvec in vec]
    else:
        return _DiffResultCache(vec)




class DiffOperatorBase(DiscretizationVectorOperator):
    def __init__(self, discr, xyz_axis):
        DiscretizationVectorOperator.__init__(self, discr)

        self.xyz_axis = xyz_axis

        self.do_warn = True

    def do_not_warn(self):
        self.do_warn = False

    def __mul__(self, field):
        # this emulates work_with_arithmetic_containers, but has to be more
        # general, because 
        # a) _DiffResultCache is not an object supporting arithmetic
        #    (and for good reason)
        # b) cache_diff_results returns a plain list (because of a)

        if isinstance(field, list):
            return ArithmeticList(self * subfield for subfield in field)

        self.discr.diff_op_counter.add()
        self.discr.diff_op_timer.start()

        if not isinstance(field, _DiffResultCache):
            if self.do_warn:
                from warnings import warn
                warn("wrap operand of diff.operator in cache_diff_results() for speed")

            result = self.discr.volume_zeros()
            from hedge.tools import make_vector_target
            self.perform_on(make_vector_target(field, result))
        else:
            rst_derivatives = self.get_rst_derivatives(field)

            result = self.discr.volume_zeros()

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

        self.discr.diff_op_timer.stop()

        return result

    def get_rst_derivatives(self, diff_result_cache):
        def diff(rst_axis):
            result = self.discr.volume_zeros()

            from hedge.tools import make_vector_target
            target = make_vector_target(diff_result_cache.vector, result)

            target.begin(len(self.discr), len(self.discr))

            from hedge._internal import perform_elwise_operator
            for eg in self.discr.element_groups:
                perform_elwise_operator(eg.ranges, eg.ranges, 
                        self.matrices(eg)[rst_axis], target)

            target.finalize()

            return result

        try:
            return diff_result_cache.cache[self.__class__]
        except KeyError:
            result = [diff(i) for i in range(self.discr.dimensions)]
            diff_result_cache.cache[self.__class__] = result
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





# mass operators --------------------------------------------------------------
class MassOperatorBase(DiscretizationVectorOperator):
    def __init__(self, discr):
        DiscretizationVectorOperator.__init__(self, discr)

    @work_with_arithmetic_containers
    def __mul__(self, field):
        from hedge.tools import make_vector_target
        result = self.discr.volume_zeros()
        self.perform_on(make_vector_target(field, result))
        return result

    def perform_on(self, target):
        self.discr.mass_op_counter.add()

        self.discr.mass_op_timer.start()
        from hedge._internal import perform_elwise_scaled_operator
        target.begin(len(self.discr), len(self.discr))
        for eg in self.discr.element_groups:
            perform_elwise_scaled_operator(eg.ranges, eg.ranges,
                   self.coefficients(eg), self.matrix(eg), 
                   target)
        target.finalize()
        self.discr.mass_op_timer.stop()

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

    def __mul__(self, field):
        from hedge.tools import make_vector_target

        def mul_single_dep(int_flux, ext_flux, result, field):
            if isinstance(field, BoundaryPair):
                bpair = field
                self.discr.perform_boundary_flux(
                        int_flux, make_vector_target(bpair.field, result),
                        ext_flux, make_vector_target(bpair.bfield, result), 
                        bpair.tag)
            else:
                self.discr.perform_inner_flux(
                        int_flux, ext_flux, make_vector_target(field, result))

        result = self.discr.volume_zeros()
        if isinstance(field, ArithmeticList):
            for idx, int_flux, ext_flux in self.flux:
                if not (0 <= idx < len(field)):
                    raise RuntimeError, "flux depends on out-of-bounds field index"
                mul_single_dep(int_flux, ext_flux, result, field[idx]) 
        else:
            if len(self.flux) > 1:
                raise RuntimeError, "only found one field to process, but flux has multiple field dependencies"
            if len(self.flux) == 1:
                idx, int_flux, ext_flux = self.flux[0]
                if idx != 0:
                    raise RuntimeError, "flux depends on out-of-bounds field index"
                mul_single_dep(int_flux, ext_flux, result, field) 
        return result

    def perform_inner(self, tgt):
        dof = len(self.discr)

        for idx, int_flux, ext_flux in self.flux:
            self.discr.perform_inner_flux(int_flux, ext_flux, 
                    tgt.rebased_target(0, dof*idx))

    def perform_int_bdry(self, tag, tgt):
        from hedge._internal import NullTarget
        dof = len(self.discr)

        for idx, int_flux, ext_flux in self.flux:
            self.discr.perform_boundary_flux(
                    int_flux, tgt.rebased_target(0, dof*idx),
                    ext_flux, NullTarget(), 
                    tag)




class _FluxMatrixOperator(DiscretizationVectorOperator):
    def __init__(self, discr, flux):
        DiscretizationVectorOperator.__init__(self, discr)

        from hedge._internal import MatrixTarget
        self.flux = flux

        dof = len(self.discr)

        self.inner_matrices = {}
        for idx, int_flux, ext_flux in self.flux:
            inner_bmatrix = num.zeros(shape=(dof, dof), flavor=num.SparseBuildMatrix)
            discr.perform_inner_flux(int_flux, ext_flux, MatrixTarget(inner_bmatrix))
            self.inner_matrices[idx] = \
                    num.asarray(inner_bmatrix, flavor=num.SparseExecuteMatrix)

        self.bdry_ops = {}

    def __mul__(self, field):
        dof = len(self.discr)

        def mul_single_dep(idx, int_flux, ext_flux, field):
            if isinstance(field, BoundaryPair):
                from hedge._internal import MatrixTarget

                bpair = field
                bdry_dof = self.discr.len_boundary(bpair.tag)

                try:
                    int_matrix, ext_matrix = self.bdry_ops[idx, bpair.tag]
                except KeyError:
                    int_bmatrix = num.zeros(shape=(dof,dof), flavor=num.SparseBuildMatrix)
                    ext_bmatrix = num.zeros(shape=(dof,bdry_dof), flavor=num.SparseBuildMatrix)
                    self.discr.perform_boundary_flux(
                            int_flux, MatrixTarget(int_bmatrix), 
                            ext_flux, MatrixTarget(ext_bmatrix), 
                            bpair.tag)
                    int_matrix = num.asarray(int_bmatrix, flavor=num.SparseExecuteMatrix)
                    ext_matrix = num.asarray(ext_bmatrix, flavor=num.SparseExecuteMatrix)

                    self.bdry_ops[idx, bpair.tag] = int_matrix, ext_matrix

                return int_matrix*bpair.field + ext_matrix*bpair.bfield
            else:
                return self.inner_matrices[idx] * field

        if isinstance(field, ArithmeticList):
            result = self.discr.volume_zeros()
            for idx, int_flux, ext_flux in self.flux:
                result += mul_single_dep(idx, int_flux, ext_flux, field[idx]) 
            return result
        else:
            assert len(self.flux) == 1
            idx, int_flux, ext_flux = self.flux[0]
            assert idx == 0
            return mul_single_dep(0, int_flux, ext_flux, field) 

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




    def __mul__(self, field):
        if isinstance(field, BoundaryPair) or (
                isinstance(field, list) and isinstance(field[0], BoundaryPair)):
            return ArithmeticList(fo * field for fo in self.flux_operators)
        else:
            # this is for performance -- it is faster to apply several fluxes
            # to a single operand at once
            result = ArithmeticList(
                    self.discr.volume_zeros() for f in self.flux_operators)

            if not isinstance(field, list):
                field = [field]

            def find_field_flux(flux_op, i_field):
                for idx, int_flux, ext_flux in flux_op.flux:
                    if idx == i_field:
                        return int_flux, ext_flux
                return None

            self.discr.inner_flux_timer.start()
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
                self.discr.inner_flux_counter.add(len(fluxes_and_results))
                for fg, fmm in self.discr.face_groups:
                    perform_multiple_double_sided_fluxes_on_single_operand(
                            fg, fmm, fluxes_and_results, f_i)
            self.discr.inner_flux_timer.stop()

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

    Accepts ArithmeticList or 0 for either argument.
    """
    if isinstance(field, ArithmeticList):
        assert isinstance(bfield, ArithmeticList)
        return ArithmeticList([
            BoundaryPair(sub_f, sub_bf, tag) for sub_f, sub_bf in zip(field, bfield)
            ])
    else:
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

                pmat = num.permutation_matrix(
                    to_indices=to_indices, 
                    h=to_count, w=from_count,
                    flavor=num.DenseMatrix)
            else:
                from_node_ids_to_idx = dict(
                        (nid, i) for i, nid in 
                        enumerate(from_ldis.generate_mode_identifiers()))

                from_indices = [
                    from_node_ids_to_idx[to_nid]
                    for to_nid in to_ldis.generate_mode_identifiers()
                    ]

                pmat = num.permutation_matrix(
                    from_indices=from_indices, 
                    h=to_count, w=from_count,
                    flavor=num.DenseMatrix)

            # build interpolation matrix
            from_matrix = from_ldis.vandermonde()
            to_matrix = to_ldis.vandermonde()
            #self.interp_matrices.append(from_matrix <<num.leftsolve>> (to_matrix*pmat))
            self.interp_matrices.append(to_matrix*pmat*(1/from_matrix))

    @work_with_arithmetic_containers
    def __call__(self, from_vec):
        from hedge._internal import perform_elwise_operator, VectorTarget
        result = self.to_discr.volume_zeros()

        target = VectorTarget(from_vec, result)

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

