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




from __future__ import division
import pylinear.array as num
import pylinear.computation as comp




class Element(object):
    def __init__(self, id, vertex_indices, all_vertices):
        self.id = id

        vertices = [all_vertices[v] for v in vertex_indices]        
        vertex_indices = self.vertex_indices = \
                self._reorder_vertices(vertex_indices, 
                        vertices)
        vertices = [all_vertices[v] for v in vertex_indices]        

        self.map = self.get_map_unit_to_global(vertices)
        self.inverse_map = self.map.inverted()
        self.face_normals, self.face_jacobians = \
                self.face_normals_and_jacobians(self.map)

    def _reorder_vertices(self, vertex_indices, vertices):
        return vertex_indices





class SimplicialElement(Element):
    @property
    def faces(self):
        return self.face_vertices(self.vertex_indices)

    @classmethod
    def get_map_unit_to_global(cls, vertices):
        """Return an affine map that maps the unit coordinates of the reference
        element to a global element at a location given by its `vertices'.
        """
        from hedge.tools import AffineMap

        mat = num.zeros((cls.dimensions, cls.dimensions))
        for i in range(cls.dimensions):
            mat[:,i] = (vertices[i+1] - vertices[0])/2

        from operator import add

        return AffineMap(mat, 
                reduce(add, vertices[1:])/2
                -(cls.dimensions-2)/2*vertices[0]
                )

    def contains_point(self, x):
        unit_coords = self.inverse_map(x)
        for xi in unit_coords:
            if xi < -1:
                return False
        return sum(unit_coords) < -(self.dimensions-2)




class Triangle(SimplicialElement):
    dimensions = 2

    @staticmethod
    def face_vertices(vertices):
        return [(vertices[0], vertices[1]), 
                (vertices[1], vertices[2]), 
                (vertices[0], vertices[2])
                ]

    def _reorder_vertices(self, vertex_indices, vertices):
        map = self.get_map_unit_to_global(vertices)
        vi = vertex_indices
        if map.jacobian > 0:
            return [vi[0], vi[2], vi[1]]
        else:
            return vi

    @staticmethod
    def face_normals_and_jacobians(affine_map):
        """Compute the normals and face jacobians of the unit element
        transformed according to `affine_map'.

        Returns a pair of lists [normals], [jacobians].
        """
        from hedge.tools import sign

        m = affine_map.matrix
        orient = sign(affine_map.jacobian)
        face1 = m[:,1] - m[:,0]
        raw_normals = [
                orient*num.array([m[1,0], -m[0,0]]),
                orient*num.array([face1[1], -face1[0]]),
                orient*num.array([-m[1,1], m[0,1]]),
                ]

        face_lengths = [comp.norm_2(fn) for fn in raw_normals]
        return [n/fl for n, fl in zip(raw_normals, face_lengths)], \
                face_lengths




class Tetrahedron(SimplicialElement):
    dimensions = 3

    @staticmethod
    def face_vertices(vertices):
        return [(vertices[0],vertices[1],vertices[2]), 
                (vertices[0],vertices[1],vertices[3]),
                (vertices[0],vertices[2],vertices[3]),
                (vertices[1],vertices[2],vertices[3]),
                ]

    def _reorder_vertices(self, vertex_indices, vertices):
        map = self.get_map_unit_to_global(vertices)
        vi = vertex_indices
        if map.jacobian > 0:
            return [vi[0], vi[1], vi[3], vi[2]]
        else:
            return vi

    @classmethod
    def face_normals_and_jacobians(cls, affine_map):
        """Compute the normals and face jacobians of the unit element
        transformed according to `affine_map'.

        Returns a pair of lists [normals], [jacobians].
        """
        from hedge.tools import normalize, sign

        face_orientations = [-1,1,-1,1]
        element_orientation = sign(affine_map.jacobian)

        def fj_and_normal(fo, pts):
            normal = (pts[1]-pts[0]) <<num.cross>> (pts[2]-pts[0])
            n_length = comp.norm_2(normal)

            # ||n_length|| is the area of the parallelogram spanned by the two
            # vectors above. Half of that is the area of the triangle we're interested
            # in. Next, the area of the unit triangle is two, so divide by two again.
            return element_orientation*fo*normal/n_length, n_length/4

        m = affine_map.matrix

        vertices = [
                m*num.array([-1,-1,-1]),
                m*num.array([+1,-1,-1]),
                m*num.array([-1,+1,-1]),
                m*num.array([-1,-1,+1]),
                ]

        # realize that zip(*something) is unzip(something)
        return zip(*[fj_and_normal(fo, pts) for fo, pts in
            zip(face_orientations, cls.face_vertices(vertices))])




class Mesh:
    """Information about the geometry and connectivity of a finite
    element mesh. (Note: no information about the discretization
    is stored here.)

    After construction, a Mesh instance has (at least) the following data 
    members:

    * points: list of Pylinear vectors of node coordinates

    * elements: list of Element instances

    * interfaces: a list of pairs 

        ((element instance 1, face index 1), (element instance 2, face index 2))

      enumerating elements bordering one another.  The relation "element 1 touches 
      element 2" is always reflexive, but this list will only contain one entry
      per element pair.

    * tag_to_boundary: a mapping of the form
      boundary_tag -> [(element instance, face index)])

      The boundary tag None always refers to the entire boundary.

    * tag_to_elements: a mapping of the form
      element_tag -> [element instances]

      The boundary tag None always refers to the entire domain.

    * periodicity: A list of tuples (minus_tag, plus_tag) or None
      indicating the tags of the boundaries to be matched together
      as periodic. There is one tuple per axis, so that for example
      a 3D mesh has three tuples.
    """

    def both_interfaces(self):
        for face1, face2 in self.interfaces:
            yield face1, face2
            yield face2, face1

    @property
    def bounding_box(self):
        try:
            return self._bounding_box
        except AttributeError:
            self._bounding_box = (
                    reduce(num.minimum, self.points),
                    reduce(num.maximum, self.points),
                    )
            return self._bounding_box




class ConformalMesh(Mesh):
    """A mesh whose elements' faces exactly match up with one another.

    See the Mesh class for data members provided by this class.
    """

    def __init__(self, points, elements, 
            boundary_tagger=lambda fvi, el, fn: [], 
            element_tagger=lambda el: [],
            periodicity=None
            ):
        """Construct a simplical mesh.

        points is an iterable of vertex coordinates, given as vectors.

        elements is an iterable of tuples of indices into points,
          giving element endpoints.

        boundary_tagger is a function that takes the arguments
          (set_of_face_vertex_indices, element, face_number)
          It returns a list of tags that apply to this surface.

        element_tagger is a function that takes the arguments
          (element) and returns the a list of tags that apply
          to that element.

        periodicity is either None or is a list of tuples
          just like the one documented for the `periodicity'
          member of class Mesh.

        Face indices follow the convention for the respective element,
        such as Triangle or Tetrahedron, in this module.
        """
        if len(points) == 0:
            raise ValueError, "mesh contains no points"

        dim = len(points[0])
        if dim == 2:
            el_class = Triangle
        elif dim == 3:
            el_class = Tetrahedron
        else:
            raise ValueError, "%d-dimensional meshes are unsupported" % dim

        # build points and elements
        self.points = [num.asarray(v) for v in points]
        self.elements = [el_class(id, vert_indices, self.points) 
            for id, vert_indices in enumerate(elements)]

        # tag elements
        self.tag_to_elements = {None: []}
        for el in self.elements:
            for el_tag in element_tagger(el):
                self.tag_to_elements.setdefault(el_tag, []).append(el)
            self.tag_to_elements[None].append(el)
        
        # build connectivity
        if periodicity is None:
            periodicity = dim*[None]
        assert len(periodicity) == dim

        self._build_connectivity(boundary_tagger, periodicity)

    def transform(self, map):
        self.points = [map(x) for x in self.points]

    def _build_connectivity(self, boundary_tagger, periodicity):
        # create face_map, which is a mapping of
        # (vertices on a face) -> 
        #  [(element, face_idx) for elements bordering that face]
        face_map = {}
        for el in self.elements:
            for fid, face_vertices in enumerate(el.faces):
                face_map.setdefault(frozenset(face_vertices), []).append((el, fid))

        # build non-periodic connectivity structures
        self.interfaces = []
        self.tag_to_boundary = {None: []}
        for face_vertices, els_faces in face_map.iteritems():
            if len(els_faces) == 2:
                self.interfaces.append(els_faces)
            elif len(els_faces) == 1:
                el, face = els_faces[0]
                for btag in boundary_tagger(face_vertices, el, face):
                    self.tag_to_boundary.setdefault(btag, []) \
                            .append(els_faces[0])
                self.tag_to_boundary[None].append(els_faces[0])
            else:
                raise RuntimeError, "face can at most border two elements"

        # add periodicity-induced connectivity
        from pytools import flatten

        self.periodicity = periodicity

        self.periodic_face_vertex_map = {}

        for axis, axis_periodicity in enumerate(periodicity):
            if axis_periodicity is not None:
                # find faces on +-axis boundaries
                minus_tag, plus_tag = axis_periodicity
                minus_faces = self.tag_to_boundary[minus_tag]
                plus_faces = self.tag_to_boundary[plus_tag]

                # find vertex indices and points on these faces
                minus_vertex_indices = list(set(flatten(el.faces[face] 
                    for el, face in minus_faces)))
                plus_vertex_indices = list(set(flatten(el.faces[face] 
                    for el, face in plus_faces)))

                minus_z_points = [self.points[pi] for pi in minus_vertex_indices]
                plus_z_points = [self.points[pi] for pi in plus_vertex_indices]

                # find a mapping from -axis to +axis vertices
                from hedge.tools import find_matching_vertices_along_axis

                minus_to_plus = find_matching_vertices_along_axis(
                        axis, minus_z_points, plus_z_points,
                        minus_vertex_indices, plus_vertex_indices)

                # establish face connectivity
                for minus_face in minus_faces:
                    minus_el, minus_fi = minus_face
                    minus_fvi = minus_el.faces[minus_fi]

                    plus_fvi = tuple(minus_to_plus[i] for i in minus_fvi)
                    plus_faces = face_map[frozenset(plus_fvi)]
                    assert len(plus_faces) == 1

                    plus_face = plus_faces[0]
                    self.interfaces.append([minus_face, plus_face])

                    plus_el, plus_fi = plus_face
                    native_plus_fvi = plus_el.faces[plus_fi]

                    self.periodic_face_vertex_map[minus_fvi] = plus_fvi, axis
                    self.periodic_face_vertex_map[native_plus_fvi] = native_plus_fvi, axis

                    self.tag_to_boundary[None].remove(plus_face)
                    self.tag_to_boundary[None].remove(minus_face)




def _tag_and_make_conformal_mesh(boundary_tagger, generated_mesh_info):
    from itertools import izip

    boundary_tags = dict(
            (frozenset(seg), 
                boundary_tagger(generated_mesh_info.points, seg))
                for seg, marker in izip(
                    generated_mesh_info.faces,
                    generated_mesh_info.face_markers)
                if marker == 1)





def make_single_element_mesh(a=-0.5, b=0.5, 
        boundary_tagger=lambda vertices, face_indices: []):
    n = 2
    node_dict = {}
    points = []
    points_1d = num.linspace(a, b, n)
    for j in range(n):
        for i in range(n):
            node_dict[i,j] = len(points)
            points.append(num.array([points_1d[i], points_1d[j]]))

    elements = [(
                node_dict[1,1],
                node_dict[0,1],
                node_dict[1,0],
                )]

    boundary_faces = [(3,1), (1,2), (2,3)]

    boundary_tags = dict(
            (frozenset(seg), 
                boundary_tagger(points, seg))
                for seg in  boundary_faces)

    return ConformalMesh(
            points,
            elements,
            boundary_tags)




def make_regular_square_mesh(a=-0.5, b=0.5, n=5, 
        boundary_tagger=lambda fvi, el, fn: []):
    node_dict = {}
    points = []
    points_1d = num.linspace(a, b, n)
    for j in range(n):
        for i in range(n):
            node_dict[i,j] = len(points)
            points.append(num.array([points_1d[i], points_1d[j]]))

    from random import shuffle
    def shuffled(l):
        result = list(l)
        shuffle(result)
        return result

    elements = []
    for i in range(n-1):
        for j in range(n-1):
            elements.append(shuffled((
                node_dict[i,j],
                node_dict[i+1,j],
                node_dict[i,j+1],
                )))
            elements.append(shuffled((
                node_dict[i+1,j+1],
                node_dict[i,j+1],
                node_dict[i+1,j],
                )))

    return ConformalMesh(points, elements, boundary_tagger)




def make_square_mesh(a=-0.5, b=0.5, max_area=4e-3, 
        boundary_tagger=lambda fvi, el, fn: []):
    def round_trip_connect(start, end):
        for i in range(start, end):
            yield i, i+1
        yield end, start

    def needs_refinement(vert_origin, vert_destination, vert_apex, area):
        return area > max_area

    points = [(a,a), (a,b), (b,b), (b,a)]
            
    import meshpy.triangle as triangle

    mesh_info = triangle.MeshInfo()
    mesh_info.set_points(points)
    mesh_info.set_facets(
            list(round_trip_connect(0, 3)),
            4*[1]
            )

    generated_mesh = triangle.build(mesh_info, 
            refinement_func=needs_refinement)
    return ConformalMesh(
            generated_mesh.points,
            generated_mesh.elements,
            boundary_tagger)




def make_disk_mesh(r=0.5, faces=50, max_area=4e-3, 
        boundary_tagger=lambda fvi, el, fn: []):
    from math import cos, sin, pi

    def round_trip_connect(start, end):
        for i in range(start, end):
            yield i, i+1
        yield end, start

    def needs_refinement(vert_origin, vert_destination, vert_apex, area):
        return area > max_area

    points = [(r*cos(angle), r*sin(angle))
            for angle in num.linspace(0, 2*pi, faces, endpoint=False)]
            
    import meshpy.triangle as triangle

    mesh_info = triangle.MeshInfo()
    mesh_info.set_points(points)
    mesh_info.set_facets(
            list(round_trip_connect(0, faces-1)),
            faces*[1]
            )

    generated_mesh = triangle.build(mesh_info, refinement_func=needs_refinement)

    return ConformalMesh(
            generated_mesh.points,
            generated_mesh.elements,
            boundary_tagger)




def make_ball_mesh(r=0.5, subdivisions=10, max_volume=None,
        boundary_tagger=lambda fvi, el, fn: []):
    from math import pi, cos, sin
    from meshpy.tet import MeshInfo, build, generate_surface_of_revolution,\
            EXT_OPEN

    dphi = pi/subdivisions

    def truncate(r):
        if abs(r) < 1e-10:
            return 0
        else:
            return r

    rz = [(truncate(r*sin(i*dphi)), r*cos(i*dphi)) for i in range(subdivisions+1)]

    mesh_info = MeshInfo()
    points, facets = generate_surface_of_revolution(rz,
            closure=EXT_OPEN, radial_subdiv=subdivisions)

    mesh_info.set_points(points)
    mesh_info.set_facets(facets, [1 for i in range(len(facets))])
    generated_mesh = build(mesh_info, max_volume=max_volume)

    return ConformalMesh(
            generated_mesh.points,
            generated_mesh.elements,
            boundary_tagger)




MINUS_Z_MARKER = 1
PLUS_Z_MARKER = 2
SHELL_MARKER = 3




def _make_z_periodic_mesh(points, facets, tags, height, 
        max_volume, boundary_tagger):
    from meshpy.tet import MeshInfo, build

    mesh_info = MeshInfo()
    mesh_info.set_points(points)
    mesh_info.set_facets(facets, tags)

    mesh_info.pbc_groups.resize(1)
    pbcg = mesh_info.pbc_groups[0]

    pbcg.facet_marker_1 = MINUS_Z_MARKER
    pbcg.facet_marker_2 = PLUS_Z_MARKER

    trans_vector = num.array([0,0,height])
    pbcg.set_transform(translation=trans_vector)

    from pytools import flatten
    minus_z_point_indices = list(set(flatten(facet 
        for facet, tag in zip(facets, tags) if tag == MINUS_Z_MARKER)))
    plus_z_point_indices = list(set(flatten(facet 
        for facet, tag in zip(facets, tags) if tag == PLUS_Z_MARKER)))
    minus_z_points = [num.array(points[pi])
            for pi in minus_z_point_indices]
    plus_z_points = [num.array(points[pi])
            for pi in plus_z_point_indices]

    from hedge.tools import find_matching_vertices_along_axis
    minus_to_plus = find_matching_vertices_along_axis(
            2, minus_z_points, plus_z_points,
            minus_z_point_indices, plus_z_point_indices)

    pairs = pbcg.point_pairs
    pairs.resize(len(minus_to_plus))
    for i, pi in enumerate(minus_z_point_indices):
        pairs[i] = pi, minus_to_plus[pi]

    def zper_boundary_tagger(fvi, el, fn):
        result = boundary_tagger(fvi, el, fn)
        face_marker = fvi2fm[frozenset(fvi)]
        if face_marker == MINUS_Z_MARKER:
            result.append("minus_z")
        if face_marker == PLUS_Z_MARKER:
            result.append("plus_z")
        if face_marker == SHELL_MARKER:
            result.append("shell")
        return result

    generated_mesh = build(mesh_info, max_volume=max_volume)
    fvi2fm = generated_mesh.face_vertex_indices_to_face_marker
        
    return ConformalMesh(
            generated_mesh.points,
            generated_mesh.elements,
            zper_boundary_tagger,
            periodicity=[None, None, ("minus_z", "plus_z")])




def make_cylinder_mesh(radius=0.5, height=1, radial_subdivisions=10, 
        height_subdivisions=1, max_volume=None, periodic=False,
        boundary_tagger=lambda fvi, el, fn: []):
    from math import pi, cos, sin
    from meshpy.tet import MeshInfo, build, generate_surface_of_revolution, \
            EXT_OPEN

    dz = height/height_subdivisions
    rz = [(0,0)] \
            + [(radius, i*dz) for i in range(height_subdivisions+1)] \
            + [(0,height)]
    ring_tags = [MINUS_Z_MARKER] \
            + ((height_subdivisions)*[SHELL_MARKER]) \
            + [PLUS_Z_MARKER]

    points, facets, tags = generate_surface_of_revolution(rz,
            closure=EXT_OPEN, radial_subdiv=radial_subdivisions,
            ring_tags=ring_tags)

    assert len(facets) == len(tags)

    if periodic:
        return _make_z_periodic_mesh(
                points, facets, tags,
                height=height, 
                max_volume=max_volume,
                boundary_tagger=boundary_tagger)
    else:
        mesh_info = MeshInfo()
        mesh_info.set_points(points)
        mesh_info.set_facets(facets, tags)

        generated_mesh = build(mesh_info, max_volume=max_volume)

        return ConformalMesh(
                generated_mesh.points,
                generated_mesh.elements,
                boundary_tagger)




def make_box_mesh(dimensions=(1,1,1), max_volume=None, periodic=False,
        boundary_tagger=lambda fvi, el, fn: []):
    from math import pi, cos, sin
    from meshpy.tet import MeshInfo, build, generate_extrusion,\
            EXT_CLOSE_IN_Z

    d = dimensions
    base_shape = [
            (-d[0]/2, -d[1]/2),
            (+d[0]/2, -d[1]/2),
            (+d[0]/2, +d[1]/2),
            (-d[0]/2, +d[1]/2),
            ]
    rz = [(0,0), (1,0), (1,d[2]), (0,d[2])]
    ring_tags = [MINUS_Z_MARKER, SHELL_MARKER, PLUS_Z_MARKER]

    points, facets, tags = generate_extrusion(rz, base_shape, closure=EXT_CLOSE_IN_Z,
            ring_tags=ring_tags)

    def add_d_half_to_x_and_y((x,y,z)):
        return (x+d[0]/2, y+d[1]/2, z)

    points = [add_d_half_to_x_and_y(p) for p in points]

    if periodic:
        return _make_z_periodic_mesh(points, facets, tags, 
                height=dimensions[2], 
                max_volume=max_volume,
                boundary_tagger=boundary_tagger)
    else:
        mesh_info = MeshInfo()
        mesh_info.set_points(points)
        mesh_info.set_facets(facets, tags)

        generated_mesh = build(mesh_info, max_volume=max_volume)

        return ConformalMesh(
                generated_mesh.points,
                generated_mesh.elements,
                boundary_tagger)

