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

        points = [_three_vector(p) for p in discr.points]
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

    def put_mesh(self, db, zonelist_name, mesh_name, mesh_opts):
        # put zone list
        db.put_zonelist(zonelist_name, self.nzones, self.ndims, self.nodelist,
                self.shapesizes, self.shapecounts)

        db.put_ucdmesh(mesh_name, self.ndims, [], self.coords, self.nzones,
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
        self.fine_mesh = SiloMeshData(dim, discr.points, generate_fine_element_groups())
        self.coarse_mesh = SiloMeshData(dim, discr.mesh.points, generate_coarse_element_groups())

    def __call__(self, filename, fields=[], vectors=[], expressions=[],
            description="Hedge visualization", time=None, step=None,
            write_coarse_mesh=False):
        from hedge.silo import DBFile, \
                DB_CLOBBER, DB_LOCAL, DB_PDB, DB_NODECENT, \
                DBOPT_DTIME, DBOPT_CYCLE
        db = DBFile(filename, DB_CLOBBER, DB_LOCAL, description, DB_PDB)

        # put mesh coordinates
        mesh_opts = {}
        if time is not None:
            mesh_opts[DBOPT_DTIME] = float(time)
        if step is not None:
            mesh_opts[DBOPT_CYCLE] = int(step)

        self.fine_mesh.put_mesh(db, "finezonelist", "finemesh", mesh_opts)
        if write_coarse_mesh:
            self.coarse_mesh.put_mesh(db, "coarsezonelist", "mesh", mesh_opts)

        # put data
        for name, field in fields:
            db.put_ucdvar1(name, "finemesh", field, DB_NODECENT)
        for name, vec in vectors:
            db.put_ucdvar(name, "finemesh", 
                    ["%s_comp%d" % (name, i) for i in range(len(vec))],
                    vec, DB_NODECENT)
        if expressions:
            db.put_defvars("defvars", expressions)
        del db

