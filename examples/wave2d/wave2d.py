import pylinear.array as num
from math import cos, sin, pi




def circle_points(r=0.5, segments=50):
    for angle in num.linspace(0, 2*pi, 50, endpoint=False):
        yield r*cos(angle), r*sin(angle)

def round_trip_connect(start, end):
    for i in range(start, end):
        yield i, i+1
    yield end, start

def make_mesh():
    segments = 50
    points = circle_points(segments=segments)

    import meshpy.triangle as triangle

    mesh_info = triangle.MeshInfo()
    mesh_info.set_points(list(circle_points()))
    mesh_info.set_segments(
            list(round_trip_connect(0, segments-1)),
            segments*[1]
            )

    generated_mesh_info = triangle.build(mesh_info)

    from hedge.mesh import SimplicalMesh
    from itertools import izip

    return SimplicalMesh(
            generated_mesh_info.points,
            generated_mesh_info.elements,
            dict((tuple(seg), "bdry") for seg, marker in izip(
                generated_mesh_info.segments,
                generated_mesh_info.segment_markers)
                if marker == 1))

def main() :
    mesh = make_mesh()

    from hedge.element import TriangularElement
    edata = TriangularElement(5)




    





if __name__ == "__main__":
    main()
