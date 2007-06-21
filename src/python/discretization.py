import pylinear.array as num
import pylinear.computation as comp




class Discretization:
    def __init__(self, mesh, edata):
        self.mesh = mesh
        self.edata = edata
        self.element_map = [edata] * len(self.mesh.elements)

        self._calculate_affine_maps()
        self._find_points_and_element_ranges()
        self._calculate_local_matrices()
        self._find_face_data()
        self._find_boundary_points_and_ranges()
        self._find_opposite_node_map()

    # initialization ----------------------------------------------------------
    def _calculate_affine_maps(self):
        self.maps = [
                self.element_map[el.id].get_map_unit_to_global(
                    [self.mesh.vertices[vi] for vi in el.vertices])
                for el in self.mesh.elements]
        self.inverse_maps = [map.inverted() for map in self.maps]
        if False:
            print edata.vandermonde().format(max_length=130, 
                    num_stringifier=lambda x: "%.3f" % x)
            print edata.inverse_mass_matrix().format(max_length=130, 
                    num_stringifier=lambda x: "%.2f" % x)
            print num.array(edata.unit_nodes())
            print num.array(edata.mass_matrix())
            print edata.face_mass_matrix().format(max_length=130, 
                    num_stringifier=lambda x: "%.3f" % x)

    def _find_points_and_element_ranges(self):
        self.points = []
        self.element_ranges = []
        for el, map in zip(self.mesh.elements, self.maps):
            e_start = len(self.points)
            self.points += [map(node) for node in self.element_map[el.id].unit_nodes()]
            self.element_ranges.append((e_start, len(self.points)))

    def _calculate_local_matrices(self):
        self.mass_mat = {}
        self.inverse_mass_mat = {}
        self.diff_mat = {}
        self.m_inv_s_t = {}
        for el in self.mesh.elements:
            this_edata = self.element_map[el.id]
            if this_edata not in self.diff_mat:
                mmat = self.mass_mat[this_edata] = this_edata.mass_matrix()
                immat = self.inverse_mass_mat[this_edata] = this_edata.inverse_mass_matrix()
                dmats = self.diff_mat[this_edata] = this_edata.differentiation_matrices()
                self.m_inv_s_t[this_edata] = [immat*d.T*mmat for d in dmats]

    def _find_face_data(self):
        from hedge.flux import Face
        self.faces = []
        for el, map in zip(self.mesh.elements, self.maps):
            el_faces = []
            for n, fj in zip(*self.element_map[el.id].face_normals_and_jacobians(map)):
                f = Face()
                f.h = self.maps[el.id].jacobian/fj # same as sledge
                f.face_jacobian = fj
                f.order = self.element_map[el.id].order
                f.normal = n
                el_faces.append(f)

            self.faces.append(el_faces)

    def _find_boundary_points_and_ranges(self):
        """assign boundary points and face ranges, for each tag separately"""
        self.boundary_points = {}
        self.boundary_ranges = {}
        for tag, els_faces in self.mesh.tag_to_boundary.iteritems():
            tag_points = []
            tag_face_ranges = {}
            for ef in els_faces:
                el, face = ef

                el_start, el_end = self.element_ranges[el.id]
                face_indices = self.element_map[el.id].face_indices()[face]

                f_start = len(tag_points)
                tag_points += [self.points[el_start+i] for i in face_indices]
                tag_face_ranges[ef] = (f_start, len(tag_points))

            self.boundary_points[tag] = tag_points
            self.boundary_ranges[tag] = tag_face_ranges

    def _find_opposite_node_map(self):
        self.opp_node_map = {}

        for local_face, neigh_face in self.mesh.both_interfaces():
            e_l, fi_l = local_face
            e_n, fi_n = neigh_face

            estart_l, eend_l = self.element_ranges[e_l.id]
            estart_n, eend_n = self.element_ranges[e_n.id]

            vertices_l = e_l.faces[fi_l]
            vertices_n = e_n.faces[fi_n]

            edata_l = self.element_map[e_l.id]
            edata_n = self.element_map[e_n.id]

            findices_l = edata_l.face_indices()[fi_l]
            findices_n = edata_n.face_indices()[fi_n]

            findices_shuffled_n = edata_l.shuffle_face_indices_to_match(
                    vertices_l, vertices_n, findices_n)

            for i, j in zip(findices_l, findices_shuffled_n):
                dist = self.points[estart_l+i]-self.points[estart_n+j]
                assert comp.norm_2(dist) < 1e-14

            self.opp_node_map[local_face] = [estart_n+j for j in findices_shuffled_n]
                        
    # vector construction -----------------------------------------------------
    def volume_zeros(self):
        return num.zeros((len(self.points),))

    def interpolate_volume_function(self, f, tag=None):
        return num.array([f(x) for x in self.points])

    def interpolate_tag_volume_function(self, f, tag=None):
        result = self.volume_zeros()
        for el in self.mesh.tag_to_elements[tag]:
            e_start, e_end = self.element_ranges[el.id]
            for i, pt in enumerate(self.points[e_start:e_end]):
                result[e_start+i] = f(pt)
        return result

    def boundary_zeros(self, tag=None):
        return num.zeros((len(self.boundary_points[tag]),))

    def interpolate_boundary_function(self, f, tag=None):
        return num.array([f(x) for x in self.boundary_points[tag]])

    # local operators ---------------------------------------------------------
    def apply_mass_matrix(self, field):
        result = num.zeros_like(field)
        for i_el, (map, (e_start, e_end)) in enumerate(zip(self.maps, self.element_ranges)):
            mmat = self.mass_mat[self.element_map[i_el]]
            result[e_start:e_end] = abs(map.jacobian)*mmat*field[e_start:e_end]
        return result

    def apply_inverse_mass_matrix(self, field):
        result = num.zeros_like(field)
        for i_el, (imap, (e_start, e_end)) in enumerate(zip(self.inverse_maps, self.element_ranges)):
            immat = self.inverse_mass_mat[self.element_map[i_el]]
            result[e_start:e_end] = abs(imap.jacobian)*immat*field[e_start:e_end]
        return result

    def _apply_diff_matrices(self, coordinate, field, matrices):
        from operator import add

        result = num.zeros_like(field)
        for i_el, imap in enumerate(self.inverse_maps):
            col = imap.matrix[:, coordinate]
            el_matrices = matrices[self.element_map[i_el]]
            e_start, e_end = self.element_ranges[i_el]
            local_field = field[e_start:e_end]
            result[e_start:e_end] = reduce(add, 
                    (dmat*coeff*local_field
                        for dmat, coeff in zip(el_matrices, col)))
        return result

    def differentiate(self, coordinate, field):
        return self._apply_diff_matrices(coordinate, field, self.diff_mat)

    def apply_stiffness_matrix_t(self, coordinate, field):
        return self._apply_diff_matrices(coordinate, field, self.m_inv_s_t)

    # flux computations -------------------------------------------------------
    def lift_face_values(self, flux, (el, fl), fl_values, fn_values, fl_indices, trace=False):
        face = self.faces[el.id][fl]

        fl_local_coeff = flux.local_coeff(face)
        fl_neighbor_coeff = flux.neighbor_coeff(face)

        edata = self.element_map[el.id]
        fl_contrib = face.face_jacobian * edata.face_mass_matrix() * \
                (fl_local_coeff*fl_values + fl_neighbor_coeff*fn_values)
        el_contrib = num.zeros((edata.node_count(),))

        for i, v in zip(fl_indices, fl_contrib):
            el_contrib[i] = v

        if trace:
            print "VALUES", el.id, fl, face.normal
            print "loc", fl_values
            print "neigh", fn_values
            print "comb", fl_local_coeff*fl_values + fl_neighbor_coeff*fn_values
            #print "test", 0.5 * (fl_values - fn_values)
            print fl_contrib
            #print "ELVALUES"
            #print el_contrib
            #print "fin", self.inverse_mass_mat[edata]*el_contrib/abs(self.maps[el.id].jacobian)

        return el_contrib

    def lift_face(self, flux, local_face, remote_face, field, result, trace):
        el, fl = local_face

        el_start, el_end = self.element_ranges[el.id]
        fl_indices = self.element_map[el.id].face_indices()[fl]

        onm = self.opp_node_map[local_face]
        fl_values = num.array([field[el_start+i] for i in fl_indices])
        fn_values = num.array([field[onm[i]] for i in range(len(fl_indices))])

        result[el_start:el_end] += \
                self.lift_face_values(flux, local_face, 
                        fl_values, fn_values, fl_indices, trace)

    def lift_interior_flux(self, flux, field):
        result = num.zeros_like(field)
        for local_face, neigh_face in self.mesh.both_interfaces():
            el, fl = local_face

            el_start, el_end = self.element_ranges[el.id]
            fl_indices = self.element_map[el.id].face_indices()[fl]

            onm = self.opp_node_map[local_face]
            fl_values = num.array([field[el_start+i] for i in fl_indices])
            fn_values = num.array([field[onm[i]] for i in range(len(fl_indices))])

            result[el_start:el_end] += \
                    self.lift_face_values(flux, local_face, 
                            fl_values, fn_values, fl_indices)
        return result
    
    def lift_boundary_flux(self, flux, field, bfield, tag=None):
        result = num.zeros_like(field)
        ranges = self.boundary_ranges[tag]

        for face in self.mesh.tag_to_boundary[tag]:
            el, fl = face

            el_start, el_end = self.element_ranges[el.id]
            fl_indices = self.element_map[el.id].face_indices()[fl]
            fn_start, fn_end = ranges[face]

            fl_values = num.array([field[el_start+i] for i in fl_indices])
            fn_values = bfield[fn_start:fn_end]

            result[el_start:el_end] += \
                    self.lift_face_values(flux, face, fl_values, fn_values, fl_indices)
        return result
    
    # misc stuff --------------------------------------------------------------
    def volumize_boundary_field(self, tag, bfield):
        result = self.volume_zeros()
        ranges = self.boundary_ranges[tag]

        for face in self.mesh.tag_to_boundary[tag]:
            el, fl = face

            el_start, el_end = self.element_ranges[el.id]
            fl_indices = self.element_map[el.id].face_indices()[fl]
            fn_start, fn_end = ranges[face]

            for i, fi in enumerate(fl_indices):
                result[el_start+fi] = bfield[fn_start+i]

        return result
    
    def find_element(self, idx):
        for i, (start, stop) in enumerate(self.element_ranges):
            if start <= idx < stop:
                return i
        raise ValueError, "not a valid dof index"
        
    def find_face(self, idx):
        el_id = self.find_element(idx)
        el_start, el_stop = self.element_ranges[el_id]
        for f_id, face_indices in enumerate(self.element_map[el_id].face_indices()):
            if idx-el_start in face_indices:
                return el_id, f_id, idx-el_start
        raise ValueError, "not a valid face dof index"

    def visualize_vtk(self, filename, fields=[], vectors=[]):
        from pyvtk import PolyData, PointData, VtkData, Scalars, Vectors
        import numpy

        def three_vector(x):
            if len(x) == 3:
                return x
            elif len(x) == 2:
                return x[0], x[1], 0.
            elif len(x) == 1:
                return x[0], 0, 0.

        points = [(x,y,0) for x,y in self.points]
        polygons = []
        for el_index, el in enumerate(self.mesh.elements):
            el_base = self.element_ranges[el_index][0]
            polygons += [[el_base+j for j in element] 
                    for element in self.element_map[el.id].generate_submesh_indices()]
        structure = PolyData(points=points, polygons=polygons)
        pdatalist = [
                Scalars(numpy.array(field), name=name, lookup_table="default") 
                for name, field in fields
                ] + [
                Vectors([three_vector(v) for v in field], name=name)
                for name, field in vectors]
        vtk = VtkData(structure, "Hedge visualization", PointData(*pdatalist))
        vtk.tofile(filename)




class SymmetryMap:
    def __init__(self, discr, sym_map, element_map, threshold=1e-13):
        self.discretization = discr

        complete_el_map = {}
        for i, j in element_map.iteritems():
            complete_el_map[i] = j
            complete_el_map[j] = i

        self.map = {}

        for i_el, (start, stop) in enumerate(discr.element_ranges):
            mapped_i_el = complete_el_map[i_el]
            mapped_start, mapped_stop = discr.element_ranges[mapped_i_el]
            for i_pt in range(start, stop):
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
    for i_el in range(len(discr.elements)):
        e_start, e_end = discr.element_ranges[i_el]
        result[e_start:e_end] = random.random()
    return result




def generate_ones_on_boundary(discr, tag):
    result = discr.volume_zeros()
    for face in discr.mesh.tag_to_boundary[tag]:
        el, fl = face

        el_start, el_end = discr.element_ranges[el.id]
        fl_indices = discr.element_map[el.id].face_indices()[fl]

        for i in fl_indices:
            result[el_start+i] = 1
    return result

