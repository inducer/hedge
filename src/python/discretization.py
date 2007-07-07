import pylinear.array as num
import pylinear.computation as comp
from pytools.arithmetic_container import work_with_arithmetic_containers




class _ElementGroup(object):
    """Once fully filled, this structure has the following data members:

    - members: a list of hedge.mesh.Element instances in this group.-----------
    - local_discretization: an instance of hedge.element.Element.
    - maps: a list of hedge.tools.AffineMap instances mapping the
      unit element to the global element.
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




class Discretization:
    def __init__(self, mesh, local_discretization):
        self.mesh = mesh
        self.dimensions = local_discretization.dimensions

        self._build_maps_element_groups_and_points(local_discretization)
        self._calculate_local_matrices()
        self._find_face_data()
        self._build_face_groups()
        self._find_boundary_points_and_ranges()
        self._find_boundary_groups()

    # initialization ----------------------------------------------------------
    def _build_maps_element_groups_and_points(self, local_discretization):
        self.points = []
        from hedge._internal import ElementRanges

        eg = _ElementGroup()
        eg.members = self.mesh.elements
        eg.local_discretization = ldis = local_discretization
        eg.ranges = ElementRanges(0)
        eg.maps = []

        for el in self.mesh.elements:
            map = ldis.get_map_unit_to_global(
                        [self.mesh.points[vi] for vi in el.vertices])
            eg.maps.append(map)

            e_start = len(self.points)
            self.points += [map(node) for node in ldis.unit_nodes()]
            eg.ranges.append_range(e_start, len(self.points))

        eg.inverse_maps = [map.inverted() for map in eg.maps]

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

            eg.jacobians = \
                    num.array([abs(map.jacobian)  for map  in eg.maps])
            eg.inverse_jacobians = \
                    num.array([abs(imap.jacobian) for imap in eg.inverse_maps])

            eg.diff_coefficients = \
                    [ # runs over global differentiation coordinate
                            [ # runs over local differentiation coordinate
                                num.array([
                                    imap.matrix[loc_coord, glob_coord]
                                    for imap in eg.inverse_maps
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
            for el, map in zip(eg.members, eg.maps):
                el_faces = []
                for fi, (n, fj) in enumerate(zip(*ldis.face_normals_and_jacobians(map))):
                    f = Face()
                    f.h = map.jacobian/fj # same as sledge
                    f.face_jacobian = fj
                    f.element_id = el.id
                    f.face_id = fi
                    f.order = ldis.order
                    f.normal = n
                    el_faces.append(f)

                self.faces.append(el_faces)

    def _build_face_groups(self):
        from hedge._internal import FaceGroup
        fg = FaceGroup()

        face_number_map = {}
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

            findices_shuffled_n = ldis_l.shuffle_face_indices_to_match(
                    vertices_l, vertices_n, findices_n)

            for i, j in zip(findices_l, findices_shuffled_n):
                dist = self.points[estart_l+i]-self.points[estart_n+j]
                assert comp.norm_2(dist) < 1e-14

            fg.add_face(
                    [estart_l+i for i in findices_l],
                    [estart_n+i for i in findices_shuffled_n],
                    self.faces[e_l.id][fi_l])

        if len(fg):
            self.face_groups = [(fg, ldis_l.face_mass_matrix())]

            fg.connect_faces([
                    (face_number_map[local_face], face_number_map[neigh_face])
                    for local_face, neigh_face in self.mesh.both_interfaces()
                    ])
        else:
            self.face_groups = []
        
    def _find_boundary_points_and_ranges(self):
        """assign boundary points and face ranges, for each tag separately"""
        self.boundary_points = {}
        self.boundary_ranges = {}
        self.boundary_index_subsets = {}

        from hedge._internal import IndexSubset

        for tag, els_faces in self.mesh.tag_to_boundary.iteritems():
            tag_points = []
            tag_face_ranges = {}
            tag_index_subset = IndexSubset()
            point_idx = 0

            for ef in els_faces:
                el, face = ef

                (el_start, el_end), ldis = self.find_el_data(el.id)
                face_indices = ldis.face_indices()[face]

                f_start = len(tag_points)
                tag_points += [self.points[el_start+i] for i in face_indices]
                tag_face_ranges[ef] = (f_start, len(tag_points))
                for i in face_indices:
                    tag_index_subset.add_index(point_idx, el_start+i)
                    point_idx += 1

            self.boundary_points[tag] = tag_points
            self.boundary_ranges[tag] = tag_face_ranges
            self.boundary_index_subsets[tag] = tag_index_subset

    def _find_boundary_groups(self):
        from hedge._internal import FaceGroup
        self.boundary_groups = {}
        for tag, els_faces in self.mesh.tag_to_boundary.iteritems():
            fg = FaceGroup()
            ranges = self.boundary_ranges[tag]
            for face in els_faces:
                el, fl = face

                (estart_l, eend_l), ldis = self.find_el_data(el.id)
                findices_l = ldis.face_indices()[fl]
                fn_start, fn_end = ranges[face]

                fg.add_face(
                        [estart_l+i for i in findices_l],
                        range(fn_start, fn_end),
                        self.faces[el.id][fl])

            self.boundary_groups[tag] = [(fg, ldis.face_mass_matrix())]
                        
    # vector construction -----------------------------------------------------
    def volume_zeros(self):
        return num.zeros((len(self.points),))

    def interpolate_volume_function(self, f):
        return num.array([f(x) for x in self.points])

    def interpolate_tag_volume_function(self, f, tag=None):
        result = self.volume_zeros()
        for el in self.mesh.tag_to_elements[tag]:
            e_start, e_end = self.find_el_range(el.id)
            for i, pt in enumerate(self.points[e_start:e_end]):
                result[e_start+i] = f(pt)
        return result

    def boundary_zeros(self, tag=None):
        return num.zeros((len(self.boundary_points[tag]),))

    def interpolate_boundary_function(self, f, tag=None):
        return num.array([f(x) for x in self.boundary_points[tag]])

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
        target.begin(len(self.points), len(self.points))
        for eg in self.element_groups:
            perform_elwise_scaled_operator(
                    eg.ranges, eg.jacobians, eg.mass_matrix, target)
        target.finalize()

    def apply_mass_matrix(self, field):
        from hedge._internal import VectorTarget
        result = self.volume_zeros()
        self.perform_mass_operator(VectorTarget(field, result))
        return result

    def perform_inverse_mass_operator(self, target):
        from hedge._internal import perform_elwise_scaled_operator
        target.begin(len(self.points), len(self.points))
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

        target.begin(len(self.points), len(self.points))

        for eg in self.element_groups:
            for coeff, mat in zip(eg.diff_coefficients[coordinate], 
                    eg.differentiation_matrices):
                perform_elwise_scaled_operator(eg.ranges, coeff, mat, target)

        target.finalize()

    def perform_minv_st_operator(self, coordinate, target):
        from hedge._internal import perform_elwise_scaled_operator
        target.begin(len(self.points), len(self.points))
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
    def lift_interior_flux(self, flux, field):
        from hedge._internal import VectorTarget, perform_both_fluxes_operator
        from hedge.flux import ChainedFlux

        result = num.zeros_like(field)
        target = VectorTarget(field, result)
        target.begin(len(self.points), len(self.points))
        for fg, fmm in self.face_groups:
            perform_both_fluxes_operator(fg, fmm, ChainedFlux(flux), target)
        target.finalize()

        return result

    def lift_boundary_flux(self, flux, field, bfield, tag=None):
        from hedge._internal import \
                VectorTarget, \
                perform_local_flux_operator, \
                perform_neighbor_flux_operator
        from hedge.flux import ChainedFlux

        ch_flux = ChainedFlux(flux)

        result = num.zeros_like(field)

        target_local = VectorTarget(field, result)
        target_local.begin(len(self.points), len(self.points))
        for fg, fmm in self.boundary_groups[tag]:
            perform_local_flux_operator(fg, fmm, ch_flux, target_local)
        target_local.finalize()

        target_bdry = VectorTarget(bfield, result)
        target_bdry.begin(len(self.points), len(self.boundary_points[tag]))
        for fg, fmm in self.boundary_groups[tag]:
            perform_neighbor_flux_operator(fg, fmm, ch_flux, target_bdry)
        target_bdry.finalize()

        return result
    
    # misc stuff --------------------------------------------------------------
    def dt_factor(self, max_system_ev):
        distinct_ldis = set(eg.local_discretization for eg in self.element_groups)
        return 1/max_system_ev \
                * min(edata.dt_non_geometric_factor() for edata in distinct_ldis) \
                * min(min(eg.local_discretization.dt_geometric_factor(
                    [self.mesh.points[i] for i in el.vertices], map)
                    for el, map in zip(eg.members, eg.maps))
                    for eg in self.element_groups)

    @work_with_arithmetic_containers
    def volumize_boundary_field(self, bfield, tag=None):
        from hedge._internal import \
                VectorTarget, \
                perform_restriction

        result = self.volume_zeros(tag)

        target = VectorTarget(bfield, result)
        target.begin(len(self.points), len(self.boundary_points[tag]))
        perform_expansion(self.boundary_index_subsets[tag], target)
        target.finalize()

        return result

    @work_with_arithmetic_containers
    def boundarize_volume_field(self, field, tag=None):
        from hedge._internal import \
                VectorTarget, \
                perform_restriction

        result = self.boundary_zeros(tag)

        target = VectorTarget(field, result)
        target.begin(len(self.boundary_points[tag]), len(self.points))
        perform_restriction(self.boundary_index_subsets[tag], target)
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
                    pt = discr.points[i_pt]
                    mapped_pt = sym_map(pt)
                    for m_i_pt in range(mapped_start, mapped_stop):
                        if comp.norm_2(discr.points[m_i_pt] - mapped_pt) < threshold:
                            self.map[m_i_pt] = i_pt
                            break

        for i in range(len(discr.points)):
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

        el_start, el_end = discr.element_group[el.id]
        fl_indices = discr.element_map[el.id].face_indices()[fl]

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

class _FluxOperator(object):
    def __init__(self, discr, flux):
        self.discr = discr
        self.flux = flux

    def __mul__(self, field):
        return self.discr.lift_interior_flux(self.flux, field)

class _BoundaryFluxOperator(object):
    def __init__(self, discr, flux, tag=None):
        self.discr = discr
        self.flux = flux
        self.tag = tag

    def __mul__(self, bpair):
        return self.discr.lift_boundary_flux(self.flux, 
                bpair.field, bpair.bfield, 
                self.tag)




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
    return _FluxOperator(*args, **kwargs)

@work_with_arithmetic_containers
def bind_boundary_flux(*args, **kwargs):
    return _BoundaryFluxOperator(*args, **kwargs)

class BoundaryPair:
    def __init__(self, field, bfield):
        self.field = field
        self.bfield = bfield

def pair_with_boundary(field, bfield):
    from pytools.arithmetic_container import ArithmeticList

    if isinstance(field, ArithmeticList):
        assert isinstance(bfield, ArithmeticList)
        return ArithmeticList([
            BoundaryPair(sub_f, sub_bf) for sub_f, sub_bf in zip(field, bfield)
            ])
    else:
        return BoundaryPair(field, bfield)

