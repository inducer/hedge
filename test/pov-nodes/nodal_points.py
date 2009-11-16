from hedge.discretization.local import TetrahedronDiscretization as Tet
from pov import Sphere, Cylinder, File, Union, Texture, Pigment, \
        Camera, LightSource, Plane, Background, Finish
from numpy import array, ones

t = Tet(8)

ball_radius = 0.05
link_radius = 0.02

node_ids = t.node_tuples()
faces = [
        [node_ids[i] for i in face]
        for face in t.face_indices()]

nodes = [(n[0],n[2], n[1]) for n in t.equilateral_nodes()]
id_to_node = dict(zip(node_ids, nodes))

def get_ball_radius(nid):
    in_faces = len([f for f in faces if nid in f])
    if in_faces >= 2:
        return ball_radius * 1.333
    else:
        return ball_radius

def get_ball_color(nid):
    in_faces = len([f for f in faces if nid in f])
    if in_faces >= 2:
        return (1,0,1)
    else:
        return (0,0,1)

balls = Union(*[
    Sphere(node, get_ball_radius(nid), 
        Texture(Pigment(color=get_ball_color(nid)))
        )
    for nid, node in id_to_node.iteritems()
    ])

links = Union()

for nid in t.node_tuples():
    child_nids = []
    for i in range(len(nid)):
        nid2 = list(nid)
        nid2[i] += 1
        child_nids.append(tuple(nid2))

    def connect_nids(nid1, nid2):
        try:
            links.append(Cylinder(
                id_to_node[nid1],
                id_to_node[nid2],
                link_radius))
        except KeyError:
            pass

    for i, nid2 in enumerate(child_nids):
        connect_nids(nid, nid2)
        connect_nids(nid2, child_nids[(i+1)%len(child_nids)])

links.append(Texture(
    Pigment(color=(0.8,0.8,0.8,0.8)),
    Finish(
        ior=1.5,
        specular=1,
        ),
    ))

outf = File("nodes.pov")

Camera(location=0.65*array((4,0.8,-1)), look_at=(0,0.1,0)).write(outf)
LightSource(
        (10,5,0), 
        color=(1,1,1),
        ).write(outf)
Background(
        color=(1,1,1)
        ).write(outf)
if False:
    Plane(
            (0,1,0), min(n[1] for n in nodes)-ball_radius,
            Texture(Pigment(color=ones(3,)))
            ).write(outf)
balls.write(outf)
links.write(outf)
