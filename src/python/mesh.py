from __future__ import division
import pylinear.array as num
import pylinear.computation as comp




class Element(object):
    def __init__(self, id, vertex_indices, tag, all_vertices):
        self.id = id

        vertices = [all_vertices[v] for v in vertex_indices]        
        vertex_indices = self.vertex_indices = \
                self._reorder_vertices(vertex_indices, 
                        vertices)
        vertices = [all_vertices[v] for v in vertex_indices]        

        self.tag = tag
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
        if map.jacobian < 0:
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
        if map.jacobian < 0:
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

    - points: list of Pylinear vectors of node coordinates
    - elements: list of Element instances
    - interfaces: a list of pairs 

        ((element instance 1, face index 1), (element instance 2, face index 2))

      enumerating elements bordering one another.  The relation "element 1 touches 
      element 2" is always reflexive, but this list will only contain one entry
      per element pair.
    - tag_to_boundary: a mapping of the form
      boundary_tag -> [(element instance, face index)])
    - tag_to_elements: a mapping of the form
      element_tag -> [element instances]
    """

    def both_interfaces(self):
        for face1, face2 in self.interfaces:
            yield face1, face2
            yield face2, face1




class ConformalMesh(Mesh):
    """A mesh whose elements' faces exactly match up with one another.

    See the Mesh class for data members provided by this class.
    """

    def __init__(self, points, elements, boundary_tags={}, element_tags={}):
        """Construct a simplical mesh.

        points is an iterable of vertex coordinates, given as 2-vectors.
        elements is an iterable of tuples of indices into points,
          giving element endpoints.
        boundary_tags is a map from sets of face vertices, indicating 
          face endpoints, into user-defined boundary tags. This map
          need not be complete. On unmentioned boundaries, the tag
          "None" will be assumed.
        element_tags is a map from element numbers into user-defined
          element tags. This map need not contain an entry for every
          element, the tag "None" will be assumed.
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

        self.points = [num.asarray(v) for v in points]
        self.elements = [el_class(id, vert_indices, 
            element_tags.get(id), 
            self.points) 
            for id, vert_indices in enumerate(elements)]
        self._build_connectivity(boundary_tags)

        self.tag_to_elements = {}
        for el in self.elements:
            self.tag_to_elements.setdefault(el.tag, []).append(el)

    def transform(self, map):
        self.points = [map(x) for x in self.points]

    def _build_connectivity(self, boundary_tags):
        # create face_map, which is a mapping of
        # (vertices on a face) -> [(element, face_idx) for elements bordering that face]
        face_map = {}
        for el in self.elements:
            for fid, face_vertices in enumerate(el.faces):
                face_map.setdefault(frozenset(face_vertices), []).append((el, fid))

        # build connectivity structures
        self.interfaces = []
        self.tag_to_boundary = {}
        for face_vertices, els_faces in face_map.iteritems():
            if len(els_faces) == 2:
                self.interfaces.append(els_faces)
            elif len(els_faces) == 1:
                btag = boundary_tags.get(face_vertices)
                self.tag_to_boundary.setdefault(btag, [])\
                        .append(els_faces[0])
            else:
                raise RuntimeError, "face can at most border two elements"




def _tag_and_make_conformal_mesh(boundary_tagger, generated_mesh_info):
    from itertools import izip

    boundary_tags = dict(
            (frozenset(seg), 
                boundary_tagger(generated_mesh_info.points, seg))
                for seg, marker in izip(
                    generated_mesh_info.faces,
                    generated_mesh_info.face_markers)
                if marker == 1)

    return ConformalMesh(
            generated_mesh_info.points,
            generated_mesh_info.elements,
            boundary_tags)




def make_single_element_mesh(a=-0.5, b=0.5, 
        boundary_tagger=lambda vertices, face_indices: None):
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
        boundary_tagger=lambda vertices, face_indices: None):
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

    boundary_faces = []

    for i in range(n-1):
        boundary_faces.append((node_dict[i  ,0  ], node_dict[i+1,0  ]))
        boundary_faces.append((node_dict[i  ,n-1], node_dict[i+1,n-1]))
        boundary_faces.append((node_dict[0  ,i  ], node_dict[0  ,i+1]))
        boundary_faces.append((node_dict[n-1,i  ], node_dict[n-1,i+1]))

    boundary_tags = dict(
            (frozenset(seg), 
                boundary_tagger(points, seg))
                for seg in  boundary_faces)

    return ConformalMesh(
            points,
            elements,
            boundary_tags)





def make_square_mesh(a=-0.5, b=0.5, max_area=4e-3, 
        boundary_tagger=lambda vertices, face_indices: None):
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
    mesh_info.set_faces(
            list(round_trip_connect(0, 3)),
            4*[1]
            )

    return _tag_and_make_conformal_mesh(
            boundary_tagger,
            triangle.build(mesh_info, refinement_func=needs_refinement))




def make_disk_mesh(r=0.5, faces=50, max_area=4e-3, 
        boundary_tagger=lambda vertices, face_indices: None):
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
    mesh_info.set_faces(
            list(round_trip_connect(0, faces-1)),
            faces*[1]
            )

    generated_mesh_info = triangle.build(mesh_info, refinement_func=needs_refinement)

    return _tag_and_make_conformal_mesh(
            boundary_tagger,
            triangle.build(mesh_info, refinement_func=needs_refinement))

