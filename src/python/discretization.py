"""Global function space discretization."""

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
import hedge.mesh




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
    def __init__(self, nodes, ranges, index_map, face_groups_and_ldis):
        self.nodes = nodes
        self.ranges = ranges
        self.index_map = index_map
        self.face_groups_and_ldis = face_groups_and_ldis




class Discretization(object):
    def __init__(self, mesh, local_discretization, 
            reorder=hedge.mesh.REORDER_CMK):
        self.mesh = mesh
        self.dimensions = local_discretization.dimensions

        self.mesh.reorder(reorder)

        self._build_element_groups_and_nodes(local_discretization)
        self._calculate_local_matrices()
        self._find_face_data()
        self._build_interior_face_groups()
        self.boundaries = {}

    # initialization ----------------------------------------------------------
    def _build_element_groups_and_nodes(self, local_discretization):
        self.nodes = []
        from hedge._internal import UniformElementRanges

        eg = _ElementGroup()
        eg.members = self.mesh.elements
        eg.local_discretization = ldis = local_discretization
        eg.ranges = UniformElementRanges(
                0, 
                len(ldis.unit_nodes()), 
                len(self.mesh.elements))

        for el in self.mesh.elements:
            e_start = len(self.nodes)
            self.nodes += [el.map(node) for node in ldis.unit_nodes()]

        self.group_map = [(eg, i) for i in range(len(self.mesh.elements))]
        self.element_groups = [eg]

    def _calculate_local_matrices(self):
        for eg in self.element_groups:
            ldis = eg.local_discretization

            mmat = eg.mass_matrix = ldis.mass_matrix()
            immat = eg.inverse_mass_matrix = ldis.inverse_mass_matrix()
            dmats = eg.differentiation_matrices = \
                    ldis.differentiation_matrices()
            smats = eg.stiffness_matrices = [mmat*d for d in dmats]
            smats = eg.stiffness_t_matrices = [d.T*mmat.T for d in dmats]
            eg.minv_st = [immat*d.T*mmat for d in dmats]

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

    def _find_face_data(self):
        from hedge.flux import Face
        self.faces = []
        for eg in self.element_groups:
            ldis = eg.local_discretization
            for el in eg.members:
                el_faces = []
                for fi, (n, fj) in enumerate(
                        zip(el.face_normals, el.face_jacobians)):
                    f = Face()

                    # This crude approximation is shamelessly stolen from sledge.
                    # There's an important caveat, however (which took me the better
                    # part of a week to figure out):
                    # h on both sides of an interface must be the same, otherwise
                    # the penalty term will behave very oddly.
                    # In hedge, this unification is performed in connect_faces in the C++ core.
                    f.h = abs(el.map.jacobian/fj)

                    f.face_jacobian = fj
                    f.element_id = el.id
                    f.face_id = fi
                    f.order = ldis.order
                    f.normal = n
                    el_faces.append(f)

                self.faces.append(el_faces)

    def _build_interior_face_groups(self):
        from hedge._internal import FaceGroup, FacePair, UnsignedList
        from hedge.element import FaceVertexMismatch

        fg = FaceGroup()

        # map (el, face) tuples to their numbers within this face group
        face_number_map = {}

        # find and match node indices along faces
        for i, (local_face, neigh_face) in enumerate(self.mesh.both_interfaces()):
            face_number_map[local_face] = i

            e_l, fi_l = local_face
            e_n, fi_n = neigh_face

            (estart_l, eend_l), ldis_l = self.find_el_data(e_l.id)
            (estart_n, eend_n), ldis_n = self.find_el_data(e_n.id)

            vertices_l = e_l.faces[fi_l]
            vertices_n = e_n.faces[fi_n]

            findices_l = ldis_l.face_indices()[fi_l]
            findices_n = ldis_n.face_indices()[fi_n]

            try:
                findices_shuffled_n = \
                        ldis_l.shuffle_face_indices_to_match(
                        vertices_l, vertices_n, findices_n)

                for i, j in zip(findices_l, findices_shuffled_n):
                    dist = self.nodes[estart_l+i]-self.nodes[estart_n+j]
                    assert comp.norm_2(dist) < 1e-14

            except FaceVertexMismatch:
                # this happens if vertices_l is not a permutation of vertices_n.
                # periodicity is the only reason why that would be so.

                vertices_n, axis = self.mesh.periodic_opposite_map[vertices_n]

                findices_shuffled_n = \
                        ldis_l.shuffle_face_indices_to_match(
                        vertices_l, vertices_n, findices_n)

                for i, j in zip(findices_l, findices_shuffled_n):
                    dist = self.nodes[estart_l+i]-self.nodes[estart_n+j]
                    dist[axis] = 0 
                    assert comp.norm_2(dist) < 1e-14


            fp = FacePair()
            fp.face_indices = UnsignedList(estart_l+i for i in findices_l)
            fp.opposite_indices = UnsignedList(estart_n+i for i in findices_shuffled_n)
            fp.flux_face = self.faces[e_l.id][fi_l]
            fg.append(fp)

        # communicate face neighbor relationships to C++ core
        if len(fg):
            self.face_groups = [(fg, ldis_l.face_mass_matrix())]

            fg.connect_faces([
                    (face_number_map[local_face], face_number_map[neigh_face])
                    for local_face, neigh_face in self.mesh.interfaces
                    ])
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

        from hedge._internal import IndexMap, FaceGroup, FacePair, UnsignedList

        nodes = []
        face_ranges = {}
        index_map = []
        face_group = FaceGroup()
        ldis = None # if this boundary is empty, we might as well have no ldis

        for ef in self.mesh.tag_to_boundary.get(tag, []):
            el, face_nr = ef

            (el_start, el_end), ldis = self.find_el_data(el.id)
            face_indices = ldis.face_indices()[face_nr]

            f_start = len(nodes)
            nodes += [self.nodes[el_start+i] for i in face_indices]
            face_range = face_ranges[ef] = (f_start, len(nodes))
            index_map.extend(el_start+i for i in face_indices)

            fp = FacePair()
            fp.face_indices = UnsignedList(el_start+i for i in face_indices)
            fp.opposite_indices = UnsignedList(xrange(*face_range))
            fp.flux_face = self.faces[el.id][face_nr]
            face_group.append(fp)

        bdry = _Boundary(
                nodes=nodes,
                ranges=face_ranges,
                index_map=IndexMap(len(self.nodes), len(index_map), index_map),
                face_groups_and_ldis=[(face_group, ldis)])

        self.boundaries[tag] = bdry
        return bdry

    # vector construction -----------------------------------------------------
    def __len__(self):
        """Return the number of nodes in this discretization."""
        return len(self.nodes)

    def volume_zeros(self):
        return num.zeros((len(self.nodes),))

    def interpolate_volume_function(self, f):
        try:
            # are we interpolating many fields at once?
            count = f.target_dimensions
        except AttributeError:
            # no, just one
            count = 1

        if count > 1:
            result = ArithmeticList([self.volume_zeros() for i in range(count)])

            for point_nr, x in enumerate(self.nodes):
                for field_nr, value in enumerate(f(x)):
                    result[field_nr][point_nr] = value
            return result
        else:
            return num.array([f(x) for x in self.nodes])

    def interpolate_tag_volume_function(self, f, tag=hedge.mesh.TAG_ALL):
        try:
            # are we interpolating many fields at once?
            count = len(f)
        except:
            # no, just one
            count = 1

        if count > 1:
            result = ArithmeticList([self.volume_zeros() for i in range(count)])

            for el in self.mesh.tag_to_elements[tag]:
                e_start, e_end = self.find_el_range(el.id)
                for i, pt in enumerate(self.nodes[e_start:e_end]):
                    for field_nr, value in enumerate(f(pt)):
                        result[field_nr][e_start+i] = value
        else:
            result = self.volume_zeros()
            for el in self.mesh.tag_to_elements[tag]:
                e_start, e_end = self.find_el_range(el.id)
                for i, pt in enumerate(self.nodes[e_start:e_end]):
                    result[e_start+i] = f(pt)

        return result

    def boundary_zeros(self, tag=hedge.mesh.TAG_ALL):
        return num.zeros((len(self._get_boundary(tag).nodes),))

    def interpolate_boundary_function(self, f, tag=hedge.mesh.TAG_ALL):
        try:
            # are we interpolating many fields at once?
            count = f.target_dimensions
        except AttributeError:
            # no, just one
            count = 1

        if count > 1:
            result = ArithmeticList([self.boundary_zeros(tag) for i in range(count)])
            for i, pt in enumerate(self._get_boundary(tag).nodes):
                for field_nr, value in enumerate(f(pt)):
                    result[field_nr][i] = value
            return result
        else:
            return num.array([f(x) for x in self._get_boundary(tag).nodes])

    def boundary_normals(self, tag=hedge.mesh.TAG_ALL):
        result = ArithmeticList([self.boundary_zeros(tag) for i in range(self.dimensions)])
        for fg, ldis in self._get_boundary(tag).face_groups_and_ldis:
            for face_pair in fg:
                normal = face_pair.flux_face.normal
                for i in face_pair.opposite_indices:
                    for j in range(self.dimensions):
                        result[j][i] = normal[j]

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

    # local operators ---------------------------------------------------------
    def perform_mass_operator(self, target):
        from hedge._internal import perform_elwise_scaled_operator
        target.begin(len(self.nodes), len(self.nodes))
        for eg in self.element_groups:
            perform_elwise_scaled_operator(
                    eg.ranges, eg.jacobians, eg.mass_matrix, target)
        target.finalize()

    def perform_inverse_mass_operator(self, target):
        from hedge._internal import perform_elwise_scaled_operator
        target.begin(len(self.nodes), len(self.nodes))
        for eg in self.element_groups:
            perform_elwise_scaled_operator(eg.ranges, 
                   eg.inverse_jacobians, eg.inverse_mass_matrix, 
                   target)
        target.finalize()

    def perform_differentiation_operator(self, coordinate, target):
        from hedge._internal import perform_elwise_scaled_operator

        target.begin(len(self.nodes), len(self.nodes))

        for eg in self.element_groups:
            for coeff, mat in zip(eg.diff_coefficients[coordinate], 
                    eg.differentiation_matrices):
                perform_elwise_scaled_operator(eg.ranges, coeff, mat, target)

        target.finalize()

    def perform_stiffness_operator(self, coordinate, target):
        from hedge._internal import perform_elwise_scaled_operator

        target.begin(len(self.nodes), len(self.nodes))

        for eg in self.element_groups:
            for coeff, mat in zip(eg.stiffness_coefficients[coordinate], 
                    eg.stiffness_matrices):
                perform_elwise_scaled_operator(eg.ranges, coeff, mat, target)

        target.finalize()

    def perform_stiffness_t_operator(self, coordinate, target):
        from hedge._internal import perform_elwise_scaled_operator

        target.begin(len(self.nodes), len(self.nodes))

        for eg in self.element_groups:
            for coeff, mat in zip(eg.stiffness_coefficients[coordinate], 
                    eg.stiffness_t_matrices):
                perform_elwise_scaled_operator(eg.ranges, coeff, mat, target)

        target.finalize()

    def perform_minv_st_operator(self, coordinate, target):
        from hedge._internal import perform_elwise_scaled_operator

        target.begin(len(self.nodes), len(self.nodes))

        for eg in self.element_groups:
            for coeff, mat in zip(eg.diff_coefficients[coordinate], eg.minv_st):
                perform_elwise_scaled_operator(eg.ranges, coeff, mat, target)

        target.finalize()

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
        from hedge._internal import perform_flux, ChainedFlux

        ch_int = ChainedFlux(int_flux)
        ch_ext = ChainedFlux(ext_flux)

        target.begin(len(self.nodes), len(self.nodes))
        for fg, fmm in self.face_groups:
            perform_flux(fg, fmm, ch_int, target, ch_ext, target)
        target.finalize()

    # boundary flux computation -----------------------------------------------
    def perform_boundary_flux(self, 
            int_flux, int_target, 
            ext_flux, ext_target, 
            tag=hedge.mesh.TAG_ALL):
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
                [_DifferentiationOperator(self, i, self.perform_differentiation_operator) 
                    for i in range(self.dimensions)]
                )

    @property
    def stiffness_operator(self):
        return ArithmeticList(
                [_DifferentiationOperator(self, i, 
                    self.perform_stiffness_operator) 
                    for i in range(self.dimensions)]
                )

    @property
    def stiffness_t_operator(self):
        return ArithmeticList(
                [_DifferentiationOperator(self, i, 
                    self.perform_stiffness_t_operator) 
                    for i in range(self.dimensions)]
                )

    @property
    def minv_stiffness_t(self):
        return ArithmeticList(
                [_DifferentiationOperator(self, i, 
                    self.perform_minv_st_operator) 
                    for i in range(self.dimensions)]
                )

    @property
    def mass_operator(self):
        return _DiscretizationMethodOperator(
                self, self.perform_mass_operator)

    @property
    def inverse_mass_operator(self):
        return _DiscretizationMethodOperator(
                self, self.perform_inverse_mass_operator)

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




def generate_ones_on_boundary(discr, tag):
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




# pylinear operator wrapper ---------------------------------------------------
class PylinearOpWrapper(operator.Operator(num.Float64)):
    def __init__(self,  discr_op):
        operator.Operator(num.Float64).__init__(self)
        self.discr_op = discr_op

    def size1(self):
        return len(self.discr_op.discr)

    def size2(self):
        return len(self.discr_op.discr)

    def apply(self, before, after):
        after[:] = self.discr_op.__mul__(before)





# local operators -------------------------------------------------------------
class _DiscretizationVectorOperator(object):
    def __init__(self, discr):
        self.discr = discr

class _DifferentiationOperator(_DiscretizationVectorOperator):
    def __init__(self, discr, coordinate, perform_func):
        _DiscretizationVectorOperator.__init__(self, discr)

        self.coordinate = coordinate
        self.perform_func = perform_func

    @work_with_arithmetic_containers
    def __mul__(self, field):
        from hedge.tools import make_vector_target

        result = self.discr.volume_zeros()
        self.perform_func(self.coordinate, make_vector_target(field, result))
        return result

    def matrix(self):
        from hedge._internal import MatrixTarget

        bmatrix = num.zeros((0,0), flavor=num.SparseBuildMatrix)
        self.perform_func(self.coordinate, MatrixTarget(bmatrix))
        return num.asarray(bmatrix, flavor=num.SparseExecuteMatrix)

class _DiscretizationMethodOperator(_DiscretizationVectorOperator):
    def __init__(self, discr, perform_func):
        _DiscretizationVectorOperator.__init__(self, discr)
        self.perform_func = perform_func

    @work_with_arithmetic_containers
    def __mul__(self, field):
        from hedge.tools import make_vector_target
        result = self.discr.volume_zeros()
        self.perform_func(make_vector_target(field, result))
        return result

    def matrix(self):
        from hedge._internal import MatrixTarget

        bmatrix = num.zeros((0,0), flavor=num.SparseBuildMatrix)
        self.perform_func(MatrixTarget(bmatrix))
        return num.asarray(bmatrix, flavor=num.SparseExecuteMatrix)




# flux operators --------------------------------------------------------------
class _DirectFluxOperator(_DiscretizationVectorOperator):
    def __init__(self, discr, flux):
        _DiscretizationVectorOperator.__init__(self, discr)
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




class _FluxMatrixOperator(_DiscretizationVectorOperator):
    def __init__(self, discr, flux):
        _DiscretizationVectorOperator.__init__(self, discr)

        from hedge._internal import MatrixTarget

        self.flux = flux

        self.inner_matrices = {}
        for idx, int_flux, ext_flux in self.flux:
            inner_bmatrix = num.zeros(shape=(0,0), flavor=num.SparseBuildMatrix)
            discr.perform_inner_flux(int_flux, ext_flux, MatrixTarget(inner_bmatrix))
            self.inner_matrices[idx] = \
                    num.asarray(inner_bmatrix, flavor=num.SparseExecuteMatrix)

        self.bdry_ops = {}

    def __mul__(self, field):
        def mul_single_dep(idx, int_flux, ext_flux, field):
            if isinstance(field, BoundaryPair):
                from hedge._internal import MatrixTarget

                bpair = field

                try:
                    int_matrix, ext_matrix = self.bdry_ops[idx, bpair.tag]
                except KeyError:
                    int_bmatrix = num.zeros(shape=(0,0), flavor=num.SparseBuildMatrix)
                    ext_bmatrix = num.zeros(shape=(0,0), flavor=num.SparseBuildMatrix)
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
        """Returns a BlockMatrix to compute the lifting of the interior fluxes.

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
        return ArithmeticList(fo * field for fo in self.flux_operators)

    def matrix_inner(self):
        """Returns a BlockMatrix to compute the lifting of the interior fluxes.

        The different components are assumed to be run together in one vector,
        in order.
        """
        from hedge.tools import BlockMatrix
        flen = len(self.discr.volume_zeros())
        return BlockMatrix(
            (flen*i, 0, fo.matrix_inner()) 
            for i, fo in enumerate(self.flux_operators))





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


