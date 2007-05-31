import pylinear.array as num




class Element(object):
    pass




class Triangle(Element):
    def __init__(self, id, vertices):
        self.id = id
        self.vertices = vertices

        # After sorting, we still specify the same simplex.
        # This is done to make face orientation unique.
        vertices.sort()

    @property
    def faces(self):
        return [(self.vertices[i], self.vertices[(i+1)%3])
                for i in range(3)]




class SimplicalMesh:
    def __init__(self, vertices, elements, bdry_tags):
        """Construct a simplical mesh.

        vertices is an iterable of vertex coordinates, given as 2-vectors.
        elements is an iterable of tuples of indices into vertices,
          giving element endpoints.
        bdry_tags is a map from vertex indices, indicating face endpoints,
          into user-defined boundary tags.
        Face indices follow the convention for the respective element,
        such as Triangle or Tetrahedron, in this module.
        """
        self.vertices = [num.asarray(v) for v in vertices]
        self.elements = [Triangle(id, tri) for id, tri in enumerate(elements)]
        self._build_connectivity()

    @property
    def interfaces(self):
        return self._unique_interfaces

    @property
    def boundaries(self):
        return self._boundary_faces

    def _build_connectivity(self):
        self._face_map = {}
        for el in self.elements:
            for fid, face in enumerate(el.faces):
                self._face_map.setdefault(face, []).append((el, fid))

        self._neighbor_map = {}
        self._unique_interfaces = []
        self._boundary_faces = []
        for face, els_faces in self._face_map.iteritems():
            if len(els_faces) == 2:
                self._unique_interfaces.append(els_faces)
                self._neighbor_map[els_faces[0]] = els_faces[1]
                self._neighbor_map[els_faces[1]] = els_faces[0]
            elif len(els_faces) == 1:
                self._boundary_faces.append(els_faces[0])
            else:
                raise RuntimeError, "face can at most border two elements"

