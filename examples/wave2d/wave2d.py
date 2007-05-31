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

class Discretization:
    def __init__(self, mesh, edata):
        self.mesh = mesh
        self.edata = edata

        self.maps = [
                edata.get_map_unit_to_global(
                    [mesh.vertices[vi] for vi in el.vertices])
                for el in mesh.elements]
        self.inv_maps = [map.inverted() for map in self.maps]

        self.points = []
        unodes = edata.unit_nodes()
        for map in self.maps:
            self.points += [map(node) for node in unodes]

        self._dof_per_el = len(unodes)

        m = self.edata.mass_matrix()
        from pylinear.operator import LUInverseOperator
        self.m_inv_op = LUInverseOperator.make(m)
        self.minv_s_t = \
                [m <<num.solve>> d.T*m for d in self.edata.differentiation_matrices()]

        self.normals = []
        self.face_jacobians = []
        for map in self.maps:
            n, fj = edata.face_normals_and_jacobians(map)
            self.normals.append(n)
            self.face_jacobians.append(fj)
                        
    def zeros(self):
        return num.zeros((len(self.points),))

    def interpolate_body_function(self, f):
        return num.array([f(x) for x in self.points])

    def apply_stiffness_matrix_t(self, coordinate, field):
        result = self.zeros()

        from operator import add

        dpe = self._dof_per_el

        result = num.zeros_like(field)
        for i_el, imap in enumerate(self.inv_maps):
            col = imap.matrix[coordinate, :]
            e_start = i_el*dpe
            e_end = (i_el+1)*dpe
            local_field = field[e_start:e_end]
            result[e_start:e_end] = reduce(add, 
                    (dmat*coeff*local_field
                        for dmat, coeff in zip(self.minv_s_t, col)))
        return result

    def lift_flux(self, flux, field):
        result = num.zeros_like(field)
        fmm = self.edata.face_mass_matrix()
        face_indices = self.edata.face_indices()
        dpe = self._dof_per_el
        for (el1, face1), (el2, face2) in self.mesh.interfaces:
            e1base = el1.id * dpe
            e2base = el2.id * dpe
            f1indices = face_indices[face1]
            f2indices = face_indices[face2]
            f1values = num.array([field[e1base+i] for i in f1indices])
            f2values = num.array([field[e2base+i] for i in f2indices])

            f1_local_coeff = flux.local_coeff(self.normals[el1.id][face1])
            f2_local_coeff = flux.local_coeff(self.normals[el2.id][face2])
            f1_neighbor_coeff = flux.neighbor_coeff(self.normals[el2.id][face1])
            f2_neighbor_coeff = flux.neighbor_coeff(self.normals[el1.id][face2])

            # faces agree on orientation because we sort vertices by number
            assert  self.mesh.elements[el1.id].faces[face1] == \
                    self.mesh.elements[el2.id].faces[face2]

            f1contrib = fmm * (f1_local_coeff*f1values + f1_neighbor_coeff*f2values)
            f2contrib = fmm * (f2_local_coeff*f2values + f2_neighbor_coeff*f1values)

            e1contrib = num.zeros((dpe,))
            for i, v in zip(f1indices, f1contrib): e1contrib[i] = v
            e2contrib = num.zeros((dpe,))
            for i, v in zip(f2indices, f2contrib): e2contrib[i] = v

            for i, v in zip(f1indices, f1contrib):
                result[e1base:e1base+dpe] += \
                        self.m_inv_op(e1contrib)/self.maps[el1.id].jacobian
            for i, v in zip(f1indices, f1contrib):
                result[e2base:e2base+dpe] += \
                        self.m_inv_op(e2contrib)/self.maps[el2.id].jacobian
        return result




def dot(x, y): 
    return sum(xi*yi for xi, yi in zip(x,y))




def main() :
    from hedge.element import TriangularElement
    from hedge.timestep import RK4TimeStepper
    from pytools.arithmetic_container import ArithmeticList
    from pytools.stopwatch import Job
    from math import sin, cos

    discr = Discretization(make_mesh(), TriangularElement(5))
    # u, v1, v2
    fields = ArithmeticList([
        discr.interpolate_body_function(lambda x: sin(x[0])), 
        discr.zeros(), 
        discr.zeros()])

    dt = 1e-4
    nsteps = int(1/dt)

    class CentralNX:
        def local_coeff(self, normal):
            return 0.5*normal[0]
        def neighbor_coeff(self, normal):
            return 0.5*normal[0]

    class CentralNY:
        def local_coeff(self, normal):
            return 0.5*normal[1]
        def neighbor_coeff(self, normal):
            return 0.5*normal[1]

    central_nx = CentralNX()
    central_ny = CentralNY()

    def rhs(t, y):
        u = fields[0]
        v = fields[1:]

        return ArithmeticList([# rhs u
                -discr.apply_stiffness_matrix_t(0, v[0])
                -discr.apply_stiffness_matrix_t(1, v[1])
                +discr.lift_flux(central_nx, v[0])
                +discr.lift_flux(central_ny, v[1])
                ,
                # rhs v1
                -discr.apply_stiffness_matrix_t(0, u)
                +discr.lift_flux(central_nx, u)
                ,
                # rhs v2
                -discr.apply_stiffness_matrix_t(1, u)
                +discr.lift_flux(central_ny, u)
                ])

    stepper = RK4TimeStepper()
    for step in range(nsteps):
        job = Job("timestep")
        fields = stepper(fields, step*dt, dt, rhs)
        job.done()

if __name__ == "__main__":
    main()

