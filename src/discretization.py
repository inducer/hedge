import pylinear.array as num
import pylinear.computation as comp




class Discretization:
    def __init__(self, mesh, edata):
        self.mesh = mesh
        self.edata = edata
        self.element_map = [edata] * len(self.mesh.elements)

        # calculate the affine maps
        self.maps = [
                self.element_map[el.id].get_map_unit_to_global(
                    [mesh.vertices[vi] for vi in el.vertices])
                for el in mesh.elements]
        self.inv_maps = [map.inverted() for map in self.maps]
        if False:
            print edata.vandermonde().format(max_length=130, 
                    num_stringifier=lambda x: "%.3f" % x)
            print edata.inverse_mass_matrix().format(max_length=130, 
                    num_stringifier=lambda x: "%.2f" % x)
            print num.array(edata.unit_nodes())

        # find M^{-1} S^T
        from pylinear.operator import LUInverseOperator
        self.mass_mat = {}
        self.inv_mass_mat = {}
        self.diff_mat = {}
        self.m_inv_s_t = {}
        for el in self.mesh.elements:
            this_edata = self.element_map[el.id]
            if this_edata not in self.diff_mat:
                mmat = self.mass_mat[this_edata] = this_edata.mass_matrix()
                immat = self.inv_mass_mat[this_edata] = this_edata.inverse_mass_matrix()
                dmats = self.diff_mat[this_edata] = this_edata.differentiation_matrices()
                self.m_inv_s_t[this_edata] = [immat*d.T*mmat for d in dmats]
                #self.m_inv_s_t[this_edata] = [d.T for d in dmats]

        # find normals and face jacobians
        self.normals = []
        self.face_jacobians = []
        for el, map in zip(self.mesh.elements, self.maps):
            n, fj = self.element_map[el.id].face_normals_and_jacobians(map)
            self.normals.append(n)
            self.face_jacobians.append(fj)

        # find all points, assign element ranges
        self.points = []
        self.element_ranges = []
        for el, map in zip(self.mesh.elements, self.maps):
            e_start = len(self.points)
            self.points += [map(node) for node in self.element_map[el.id].unit_nodes()]
            self.element_ranges.append((e_start, len(self.points)))

        # assign boundary points and face ranges, for each tag separately
        self.boundary_points = {}
        self.boundary_ranges = {}
        for tag, els_faces in mesh.boundary_map.iteritems():
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

        # find opposite-node map in the interior
        self.opp_node_map = {}
        for face1, face2 in self.mesh.interfaces:
            e1, f1 = face1
            e2, f2 = face2

            e1_start, e1_end = self.element_ranges[e1.id]
            e2_start, e2_end = self.element_ranges[e2.id]

            f1_vertices = e1.faces[f1]
            f2_vertices = e2.faces[f2]

            e1data = self.element_map[e1.id]
            e2data = self.element_map[e2.id]

            f1_indices = e1data.face_indices()[f1]
            f2_indices = e2data.face_indices()[f2]

            f2_indices_for_1 = e1data.shuffle_face_indices_to_match(
                    f1_vertices, f2_vertices, f2_indices)

            for i, j in zip(f1_indices, f2_indices_for_1):
                dist = self.points[e1_start+i]-self.points[e2_start+j]
                assert comp.norm_2(dist) < 1e-12

                self.opp_node_map[e1_start+i] = e2_start+j
                self.opp_node_map[e2_start+j] = e1_start+i
                        
    def volume_zeros(self):
        return num.zeros((len(self.points),))

    def interpolate_volume_function(self, f):
        return num.array([f(x) for x in self.points])

    def boundary_zeros(self, tag):
        return num.zeros((len(self.boundary_points[tag]),))

    def interpolate_boundary_function(self, tag, f):
        return num.array([f(x) for x in self.boundary_points[tag]])

    def apply_mass_matrix(self, field):
        result = num.zeros_like(field)
        for i_el, map in enumerate(self.maps):
            mmat = self.mass_mat[self.element_map[i_el]]
            e_start, e_end = self.element_ranges[i_el]
            result[e_start:e_end] = abs(map.jacobian)*mmat*field[e_start:e_end]
        return result

    def _apply_diff_matrices(self, coordinate, field, matrices):
        from operator import add

        result = num.zeros_like(field)
        for i_el, imap in enumerate(self.inv_maps):
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

    def lift_face_values(self, flux, (el, fl), fl_values, fn_values, fl_indices):
        normal = self.normals[el.id][fl]
        fjac = self.face_jacobians[el.id][fl]

        fl_local_coeff = flux.local_coeff(normal)
        fl_neighbor_coeff = flux.neighbor_coeff(normal)

        edata = self.element_map[el.id]
        fl_contrib = fjac * edata.face_mass_matrix() * \
                (fl_local_coeff*fl_values + fl_neighbor_coeff*fn_values)
        el_contrib = num.zeros((edata.node_count(),))

        for i, v in zip(fl_indices, fl_contrib):
            el_contrib[i] = v

        if False and el.id == 78 and fl == 2:
            print "VALUES", el.id, fl, normal
            print fl_values
            print fn_values
            print fl_local_coeff*fl_values + fl_neighbor_coeff*fn_values
            print "test", 0.5 * (fl_values - fn_values)
            print fl_contrib
            print "ELVALUES"
            print el_contrib
            print "fin", self.inv_mass_mat[edata]*el_contrib/abs(self.maps[el.id].jacobian)

        return self.inv_mass_mat[edata]*el_contrib/abs(self.maps[el.id].jacobian)

    def lift_face(self, flux, local_face, field, result):
        el, fl = local_face

        el_start, el_end = self.element_ranges[el.id]
        fl_indices = self.element_map[el.id].face_indices()[fl]

        onm = self.opp_node_map
        fl_values = num.array([field[el_start+i] for i in fl_indices])
        fn_values = num.array([field[onm[el_start+i]] for i in fl_indices])

        result[el_start:el_end] += \
                self.lift_face_values(flux, local_face, 
                        fl_values, fn_values, fl_indices)

    def lift_interior_flux(self, flux, field):
        result = num.zeros_like(field)
        for face1, face2 in self.mesh.interfaces:
            self.lift_face(flux, face1, field, result)
            self.lift_face(flux, face2, field, result)
        return result
    
    def lift_boundary_flux(self, flux, field, bfield, tag):
        result = num.zeros_like(field)
        ranges = self.boundary_ranges[tag]

        for face in self.mesh.boundary_map[tag]:
            el, fl = face

            el_start, el_end = self.element_ranges[el.id]
            fl_indices = self.element_map[el.id].face_indices()[fl]
            fn_start, fn_end = ranges[face]

            fl_values = num.array([field[el_start+i] for i in fl_indices])
            fn_values = bfield[fn_start:fn_end]

            result[el_start:el_end] += \
                    self.lift_face_values(flux, face, 
                            fl_values, fn_values, fl_indices)
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
        pdatalist = []
        for name, field in fields:
            pdatalist.append(Scalars(numpy.array(field), name=name, lookup_table="default"))
        for name, field in vectors:
            pdatalist.append(Vectors([three_vector(v) for v in field], name=name))
        vtk = VtkData(structure, "Hedge visualization", PointData(*pdatalist))
        vtk.tofile(filename)




def generate_random_constant_on_elements(discr):
    result = discr.volume_zeros()
    import random
    for i_el in range(len(discr.elements)):
        e_start, e_end = discr.element_ranges[i_el]
        result[e_start:e_end] = random.random()
    return result




def generate_ones_on_boundary(discr, tag):
    result = discr.volume_zeros()
    for face in discr.mesh.boundary_map[tag]:
        el, fl = face

        el_start, el_end = discr.element_ranges[el.id]
        fl_indices = discr.element_map[el.id].face_indices()[fl]

        for i in fl_indices:
            result[el_start+i] = 1
    return result

