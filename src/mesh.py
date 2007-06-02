import pylinear.array as num




class Element(object):
    def __init__(self, id, vertices):
        self.id = id
        self.vertices = vertices





class Triangle(Element):
    @property
    def faces(self):
        return [(self.vertices[i], self.vertices[(i+1)%3])
                for i in range(3)]




class Mesh:
    """Information about the geometry and connectivity of a finite
    element mesh. (Note: no information about the discretization
    is stored here.)

    After construction, a Mesh instance has (at least) the following data 
    members:

    - vertices: list of Pylinear vectors of node coordinates
    - elements: list of Element instances
    - interfaces: a list of pairs 

        ((element instance 1, face index 1), (element instance 2, face index 2))

      enumerating elements bordering one another.  The relation "element 1 touches 
      element 2" is always reflexive, but this list will only contain one entry
      per element pair.
    - boundary_map: a mapping of the form
      boundary_tag -> [(element instance, face index)])
    """




class ConformalMesh(Mesh):
    """A mesh whose elements' faces exactly match up with one another.

    See the Mesh class for data members provided by this class.
    """

    def __init__(self, vertices, elements, boundary_tags):
        """Construct a simplical mesh.

        vertices is an iterable of vertex coordinates, given as 2-vectors.
        elements is an iterable of tuples of indices into vertices,
          giving element endpoints.
        bdry_tags is a map from sets of face vertices, indicating face endpoints,
          into user-defined boundary tags.
        Face indices follow the convention for the respective element,
        such as Triangle or Tetrahedron, in this module.
        """
        self.vertices = [num.asarray(v) for v in vertices]
        self.elements = [Triangle(id, tri) for id, tri in enumerate(elements)]
        self._build_connectivity(boundary_tags)

    def _build_connectivity(self, boundary_tags):
        # create face_map, which is a mapping of
        # (vertices on a face) -> [(element, face_idx) for elements bordering that face]
        face_map = {}
        for el in self.elements:
            for fid, face_vertices in enumerate(el.faces):
                face_map.setdefault(frozenset(face_vertices), []).append((el, fid))

        # build connectivity structures
        self.interfaces = []
        self.boundary_map = {}
        for face_vertices, els_faces in face_map.iteritems():
            if len(els_faces) == 2:
                self.interfaces.append(els_faces)
            elif len(els_faces) == 1:
                try:
                    btag = boundary_tags[face_vertices]
                except KeyError:
                    for vi in face_vertices:
                        print self.vertices[vi]
                    raise
                self.boundary_map.setdefault(btag, []).append(els_faces[0])
            else:
                raise RuntimeError, "face can at most border two elements"




def make_disk_mesh(r=0.5, segments=50, max_area=4e-3):
    from math import cos, sin, pi

    def circle_points(r=0.5, segments=50):
        for angle in num.linspace(0, 2*pi, 50, endpoint=False):
            yield r*cos(angle), r*sin(angle)

    def round_trip_connect(start, end):
        for i in range(start, end):
            yield i, i+1
        yield end, start

    def needs_refinement(vert_origin, vert_destination, vert_apex, area):
        return area > max_area

    points = circle_points(r=r, segments=segments)

    import meshpy.triangle as triangle

    mesh_info = triangle.MeshInfo()
    mesh_info.set_points(list(circle_points()))
    mesh_info.set_segments(
            list(round_trip_connect(0, segments-1)),
            segments*[1]
            )

    generated_mesh_info = triangle.build(mesh_info, refinement_func=needs_refinement)

    from itertools import izip

    return ConformalMesh(
            generated_mesh_info.points,
            generated_mesh_info.elements,
            dict((frozenset(seg), "dirichlet") for seg, marker in izip(
                generated_mesh_info.segments,
                generated_mesh_info.segment_markers)
                if marker == 1))


