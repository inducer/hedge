# Hedge - the Hybrid'n'Easy DG Environment
# Copyright (C) 2007 Andreas Kloeckner
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.




import pylinear.array as num
import pylinear.computation as comp
from pytools.arithmetic_container import work_with_arithmetic_containers




class _ElementGroup(object):
    """Once fully filled, this structure has the following data members:

    - members: a list of hedge.mesh.Element instances in this group.-----------
    - local_discretization: an instance of hedge.element.Element.
    - ranges: a list of (start, end) tuples indicating the DOF numbers for
      each element. Note: This is actually a C++ ElementRanges object.

    - mass_matrix
    - inverse_mass_matrix
    - differentiation_matrices: local differentiation matrices, i.e.
      differentiation by r, s, t, ....
    - jacobians
    - inverse_jacobians
    - diff_coefficients: a (d,d)-matrix of coefficient vectors to turn
      (r,s,t)-differentiation into (x,y,z).
    """
    pass




class _Boundary:
    def __init__(self, nodes, ranges, index_map, face_groups_and_ldis):
        self.nodes = nodes
        self.ranges = ranges
        self.index_map = index_map
        self.face_groups_and_ldis = face_groups_and_ldis




class Discretization:
    def __init__(self, mesh, local_discretization):
        self.mesh = mesh
        self.dimensions = local_discretization.dimensions

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
                    f.h = el.map.jacobian/fj # same as sledge
                    f.face_jacobian = fj
                    f.element_id = el.id
                    f.face_id = fi
                    f.order = ldis.order
                    f.normal = n
                    el_faces.append(f)

                self.faces.append(el_faces)

    def _build_interior_face_groups(self):
        from hedge._internal import FaceGroup
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

            except ValueError:
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

            fg.add_face(
                    [estart_l+i for i in findices_l],
                    [estart_n+i for i in findices_shuffled_n],
                    self.faces[e_l.id][fi_l])

        # communicate face neighbor relationships to C++ core
        if len(fg):
            self.face_groups = [(fg, ldis_l.face_mass_matrix())]

            fg.connect_faces([
                    (face_number_map[local_face], face_number_map[neigh_face])
                    for local_face, neigh_face in self.mesh.both_interfaces()
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

        from hedge._internal import IndexMap, FaceGroup

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

            face_group.add_face(
                    [el_start+i for i in face_indices],
                    range(*face_range),
                    self.faces[el.id][face_nr])

        bdry = _Boundary(
                nodes=nodes,
                ranges=face_ranges,
                index_map=IndexMap(len(self.nodes), len(index_map), index_map),
                face_groups_and_ldis=[(face_group, ldis)])

        self.boundaries[tag] = bdry
        return bdry

    # vector construction -----------------------------------------------------
    def volume_zeros(self):
        return num.zeros((len(self.nodes),))

    def interpolate_volume_function(self, f):
        try:
            # are we interpolating many fields at once?
            count = len(f)
        except:
            # no, just one
            count = 1

        if count > 1:
            from pytools.arithmetic_container import ArithmeticList
            result = ArithmeticList([self.volume_zeros() for i in range(count)])

            for point_nr, x in enumerate(self.nodes):
                for field_nr, value in enumerate(f(x)):
                    result[field_nr][point_nr] = value
            return result
        else:
            return num.array([f(x) for x in self.nodes])

    def interpolate_tag_volume_function(self, f, tag=None):
        try:
            # are we interpolating many fields at once?
            count = len(f)
        except:
            # no, just one
            count = 1

        if count > 1:
            from pytools.arithmetic_container import ArithmeticList
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

    def boundary_zeros(self, tag=None):
        return num.zeros((len(self._get_boundary(tag).nodes),))

    def interpolate_boundary_function(self, f, tag=None):
        return num.array([f(x) for x in self._get_boundary(tag).nodes])

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

    @work_with_arithmetic_containers
    def apply_mass_matrix(self, field):
        from hedge._internal import VectorTarget
        result = self.volume_zeros()
        self.perform_mass_operator(VectorTarget(field, result))
        return result

    def perform_inverse_mass_operator(self, target):
        from hedge._internal import perform_elwise_scaled_operator
        target.begin(len(self.nodes), len(self.nodes))
        for eg in self.element_groups:
            perform_elwise_scaled_operator(eg.ranges, 
                   eg.inverse_jacobians, eg.inverse_mass_matrix, 
                   target)
        target.finalize()

    @work_with_arithmetic_containers
    def apply_inverse_mass_matrix(self, field):
        from hedge._internal import VectorTarget
        result = self.volume_zeros()
        self.perform_inverse_mass_operator(VectorTarget(field, result))
        return result

    def perform_differentiation_operator(self, coordinate, target):
        from hedge._internal import perform_elwise_scaled_operator

        target.begin(len(self.nodes), len(self.nodes))

        for eg in self.element_groups:
            for coeff, mat in zip(eg.diff_coefficients[coordinate], 
                    eg.differentiation_matrices):
                perform_elwise_scaled_operator(eg.ranges, coeff, mat, target)

        target.finalize()

    def perform_minv_st_operator(self, coordinate, target):
        from hedge._internal import perform_elwise_scaled_operator
        target.begin(len(self.nodes), len(self.nodes))
        for eg in self.element_groups:
            for coeff, mat in zip(eg.diff_coefficients[coordinate], eg.minv_st):
                perform_elwise_scaled_operator(eg.ranges, coeff, mat, target)
        target.finalize()

    def differentiate(self, coordinate, field):
        from hedge._internal import VectorTarget
        result = self.volume_zeros()
        self.perform_differentiation_operator(coordinate,
                VectorTarget(field, result))
        return result

    def apply_minv_st(self, coordinate, field):
        from hedge._internal import VectorTarget
        result = self.volume_zeros()
        self.perform_minv_st_operator(coordinate,
                VectorTarget(field, result))
        return result

    # flux computations -------------------------------------------------------
    def perform_interior_flux(self, flux, target):
        """Perform interior fluxes on the given operator target.
        This adds the contribution

          M_{i,j} := \sum_{interior faces f} \int_f
            (  
               flux.local_coeff(f) * \phi_j
               + 
               flux.neighbor_coeff(f) * \phi_{opp(j)}
             )
             \phi_i
        
        to the given target. opp(j) denotes the dof with its node
        opposite from node j on the face opposite f.

        Thus the matrix product M*u, where u is the full volume field
        results in 

          v_i := M_{i,j} u_j
            = \sum_{interior faces f} \int_f
            (  
               flux.local_coeff(f) * u^-
               + 
               flux.neighbor_coeff(f) * u^+
             )
             \phi_i

        For more on operator targets, see src/cpp/op_target.hpp.
        """
        from hedge.flux import which_faces, compile_flux

        target.begin(len(self.nodes), len(self.nodes))
        for fg, fmm in self.face_groups:
            compile_flux(flux).perform(fg, which_faces.BOTH, fmm, target)
        target.finalize()

    def lift_interior_flux(self, flux, field):
        """Use perform_interior_flux() to directly compute the vector
        v mentioned in its definition. No matrix of the operator is 
        ever constructed.
        """
        from hedge._internal import VectorTarget

        result = num.zeros_like(field)
        self.perform_interior_flux(flux, VectorTarget(field, result))
        return result

    def prepare_interior_flux_op(self, flux):
        """Use perform_interior_flux() to compute the matrix M mentioned
        in its definition.
The return value of this function is meant to be passed directly to
        apply_interior_flux_op. For the serial Discretization class, the
        value returned from this function is simply the sparse matrix M in 
        CSR form. However, for parallel Discretization instances, the return
        value will be more complicated. Hence, user programs should not rely 
        on the structure of the return value of this function.
        """
        from hedge._internal import MatrixTarget

        matrix = num.zeros(shape=(0,0), flavor=num.SparseBuildMatrix)
        print "pif_m"
        self.perform_interior_flux(flux, MatrixTarget(matrix))
        print "end_pif_m"
        print matrix.shape, matrix.nnz
        conv = num.asarray(matrix, flavor=num.SparseExecuteMatrix)
        print "conv_pif_m"
        return conv

    def apply_interior_flux_op(self, data, field):
        """Use the result of prepare_interior_flux_op(), passed in as `data',
        to apply the operator described in perform_interior_flux on the 
        field passed in as `field'.
        """
        # I love programming when it's this easy. :)
        return data*field

    def lift_boundary_flux(self, flux, field, bfield, tag=None):
        from hedge._internal import VectorTarget
        from hedge.flux import which_faces, compile_flux

        result = num.zeros_like(field)

        bdry = self._get_boundary(tag)
        if not bdry.nodes:
            return result

        compiled_flux = compile_flux(flux)
        
        target_local = VectorTarget(field, result)
        target_local.begin(len(self.nodes), len(self.nodes))
        for fg, ldis in bdry.face_groups_and_ldis:
            compiled_flux.perform(fg, which_faces.LOCAL, 
                    ldis.face_mass_matrix(), target_local)
        target_local.finalize()

        target_bdry = VectorTarget(bfield, result)
        target_bdry.begin(len(self.nodes), len(bdry.nodes))
        for fg, ldis in bdry.face_groups_and_ldis:
            compiled_flux.perform(fg, which_faces.NEIGHBOR, 
                    ldis.face_mass_matrix(), target_bdry)
        target_bdry.finalize()

        return result
    
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
    def volumize_boundary_field(self, bfield, tag=None):
        from hedge._internal import \
                VectorTarget, \
                perform_inverse_index_map

        result = self.volume_zeros()
        bdry = self._get_boundary(tag)

        target = VectorTarget(bfield, result)
        target.begin(len(self.nodes), len(bdry.nodes))
        perform_inverse_index_map(bdry.index_map, target)
        target.finalize()

        return result

    @work_with_arithmetic_containers
    def boundarize_volume_field(self, field, tag=None):
        from hedge._internal import \
                VectorTarget, \
                perform_index_map

        result = self.boundary_zeros(tag)

        bdry = self._get_boundary(tag)

        target = VectorTarget(field, result)
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




class SymmetryMap:
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
                            self.map[m_i_pt] = i_pt
                            break

        for i in range(len(discr.nodes)):
            assert i in self.map

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
    for face in discr.mesh.tag_to_boundary[tag]:
        el, fl = face

        (el_start, el_end), ldis = discr.find_el_data(el.id)
        fl_indices = ldis.face_indices()[fl]

        for i in fl_indices:
            result[el_start+i] = 1
    return result




# bound operators -------------------------------------------------------------
class _DifferentiationOperator(object):
    def __init__(self, discr, coordinate):
        self.discr = discr
        self.coordinate = coordinate

    def __mul__(self, field):
       return self.discr.differentiate(self.coordinate, field)

class _WeakDifferentiationOperator(object):
    def __init__(self, discr, coordinate):
        self.discr = discr
        self.coordinate = coordinate

    def __mul__(self, field):
       return self.discr.apply_minv_st(self.coordinate, field)

class _MassMatrixOperator(object):
    def __init__(self, discr):
        self.discr = discr

    def __mul__(self, field):
       return self.discr.apply_mass_matrix(field)

class _InverseMassMatrixOperator(object):
    def __init__(self, discr):
        self.discr = discr

    def __mul__(self, field):
       return self.discr.apply_inverse_mass_matrix(field)

class _BoundaryPair:
    def __init__(self, field, bfield, tag):
        self.field = field
        self.bfield = bfield
        self.tag = tag

class _DirectFluxOperator(object):
    def __init__(self, discr, flux):
        self.discr = discr
        self.flux = flux

    def __mul__(self, field):
        if isinstance(field, _BoundaryPair):
            bpair = field
            return self.discr.lift_boundary_flux(self.flux, 
                    bpair.field, bpair.bfield, bpair.tag)
        else:
            return self.discr.lift_interior_flux(self.flux, field)

class _FluxMatrixOperator(object):
    def __init__(self, discr, flux):
        self.discr = discr
        self.flux = flux
        self.interior_op = self.discr.prepare_interior_flux_op(flux)

    def __mul__(self, field):
        if isinstance(field, _BoundaryPair):
            bpair = field
            return self.discr.lift_boundary_flux(self.flux, 
                    bpair.field, bpair.bfield, bpair.tag)
        else:
            return self.discr.apply_interior_flux_op(self.interior_op, field)




# operator binding functions --------------------------------------------------
def bind_nabla(discr):
    from pytools.arithmetic_container import ArithmeticList
    return ArithmeticList(
            [_DifferentiationOperator(discr, i) for i in range(discr.dimensions)]
            )

def bind_weak_nabla(discr):
    from pytools.arithmetic_container import ArithmeticList
    return ArithmeticList(
            [_WeakDifferentiationOperator(discr, i) for i in range(discr.dimensions)]
            )

def bind_mass_matrix(discr):
    return _MassMatrixOperator(discr)

def bind_inverse_mass_matrix(discr):
    return _InverseMassMatrixOperator(discr)

@work_with_arithmetic_containers
def bind_flux(*args, **kwargs):
    direct = True
    if "direct" in kwargs:
        direct = kwargs["direct"]
        del kwargs["direct"]

    if direct:
        return _DirectFluxOperator(*args, **kwargs)
    else:
        return _FluxMatrixOperator(*args, **kwargs)

def pair_with_boundary(field, bfield, tag=None):
    from pytools.arithmetic_container import ArithmeticList

    if isinstance(field, ArithmeticList):
        assert isinstance(bfield, ArithmeticList)
        return ArithmeticList([
            _BoundaryPair(sub_f, sub_bf, tag) for sub_f, sub_bf in zip(field, bfield)
            ])
    else:
        return _BoundaryPair(field, bfield, tag)

