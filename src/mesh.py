class Element:
    pass




class Triangle(Element):
    def __init__(self, id, vertices):
        self.id = id
        self.vertices = vertices
        vertices.sort()

    def _faces(self):
        for i in range(3):
            yield (self.vertices[i], self.vertices[(i+1)%3])
    faces = property(_faces)




class SimplicalMesh:
    def __init__(self, vertices, triangle_vertices, bdry_tags):
        """Construct a simplical mesh.

        vertices is an iterable of vertex coordinates, given as 2-vectors.
        triangle_vertices is an iterable of triples of indices into vertices,
          giving triangle endpoints.
        bdry_tags is a map from pairs of indices into vertices to (user-defined)
          identifiers of boundaries.
        """
        self.vertices = list(vertices)
        self.elements = [Triangle(id, tri) for id, tri in enumerate(triangle_vertices)]
        self._build_connectivity()

    def _interfaces(self):
        return self._unique_interfaces
    interfaces = property(_interfaces)

    def _boundaries(self):
        return self._boundary_faces
    boundaries = property(_boundaries)

    def _build_connectivity(self):
        self._face_map = {}
        for el in self.elements:
            for fid, face in enumerate(el.faces):
                self._face_map.setdefault(face, []).append((el, fid))

        self._neighbor_map = {}
        self._unique_interfaces = []
        self._boundary_faces = []
        for face, els_faces in self._face_map.iteritems():
            if len(elements) == 2:
                self._unique_interfaces.append(els_faces[0])
                self._neighbor_map[els_faces[0]] = els_faces[1]
                self._neighbor_map[els_faces[1]] = els_faces[0]
            elif len(elements) == 1:
                self_boundary_faces.append(els_faces[0])
            else:
                raise RuntimeError, "face can at most border two elements"

