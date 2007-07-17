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




def _three_vector(x):
    if len(x) == 3:
        return x
    elif len(x) == 2:
        return x[0], x[1], 0.
    elif len(x) == 1:
        return x[0], 0, 0.





class VtkVisualizer:
    def __init__(self, discr):
        from pyvtk import PolyData

        points = [_three_vector(p) for p in discr.nodes]
        polygons = []

        for eg in discr.element_groups:
            ldis = eg.local_discretization
            for el, (el_start, el_stop) in zip(eg.members, eg.ranges):
                polygons += [[el_start+j for j in element] 
                        for element in ldis.generate_submesh_indices()]

        self.structure = PolyData(points=points, polygons=polygons)

    def __call__(self, filename, fields=[], vectors=[], description="Hedge visualization"):
        from pyvtk import PointData, VtkData, Scalars, Vectors
        import numpy

        pdatalist = [
                Scalars(numpy.array(field), name=name, lookup_table="default") 
                for name, field in fields
                ] + [
                Vectors([_three_vector(v) for v in zip(field)], name=name)
                for name, field in vectors]
        vtk = VtkData(self.structure, "Hedge visualization", PointData(*pdatalist))
        vtk.tofile(filename)




class SiloMeshData:
    def __init__(self, dim, points, element_groups):
        from pytools import flatten

        self.coords = flatten([p[d] for p in points] for d in range(dim))

        self.ndims = dim
        self.nodelist = []
        self.shapesizes = []
        self.shapecounts = []
        self.nshapetypes = 0
        self.nzones = 0

        for eg in element_groups:
            polygons = list(eg)

            if len(polygons):
                self.nodelist += flatten(polygons)
                self.shapesizes.append(len(polygons[0]))
                self.shapecounts.append(len(polygons))
                self.nshapetypes += 1
                self.nzones += len(polygons)

    def put_mesh(self, silo, zonelist_name, mesh_name, mesh_opts):
        # put zone list
        silo.put_zonelist(zonelist_name, self.nzones, self.ndims, self.nodelist,
                self.shapesizes, self.shapecounts)

        silo.put_ucdmesh(mesh_name, self.ndims, [], self.coords, self.nzones,
                zonelist_name, None, mesh_opts)


class SiloVisualizer:
    def __init__(self, discr):
        def generate_fine_elements(eg):
            ldis = eg.local_discretization
            smi = list(ldis.generate_submesh_indices())
            for el, (el_start, el_stop) in zip(eg.members, eg.ranges):
                for element in smi:
                    yield [el_start+j for j in element] 

        def generate_fine_element_groups():
            for eg in discr.element_groups:
                yield generate_fine_elements(eg)

        def generate_coarse_elements(eg):
            for el in eg.members:
                yield el.vertex_indices

        def generate_coarse_element_groups():
            for eg in discr.element_groups:
                yield generate_coarse_elements(eg)

        dim = discr.dimensions
        self.fine_mesh = SiloMeshData(dim, discr.nodes, generate_fine_element_groups())
        self.coarse_mesh = SiloMeshData(dim, discr.mesh.points, generate_coarse_element_groups())

    def add_to_silo(self, silo, fields=[], vectors=[], expressions=[],
            time=None, step=None, write_coarse_mesh=False):
        from hedge.silo import DB_NODECENT, DBOPT_DTIME, DBOPT_CYCLE

        # put mesh coordinates
        mesh_opts = {}
        if time is not None:
            mesh_opts[DBOPT_DTIME] = float(time)
        if step is not None:
            mesh_opts[DBOPT_CYCLE] = int(step)

        self.fine_mesh.put_mesh(silo, "finezonelist", "finemesh", mesh_opts)
        if write_coarse_mesh:
            self.coarse_mesh.put_mesh(silo, "coarsezonelist", "mesh", mesh_opts)

        # put data
        for name, field in fields:
            silo.put_ucdvar1(name, "finemesh", field, DB_NODECENT)
        for name, vec in vectors:
            silo.put_ucdvar(name, "finemesh", 
                    ["%s_comp%d" % (name, i) for i in range(len(vec))],
                    vec, DB_NODECENT)
        if expressions:
            silo.put_defvars("defvars", expressions)

