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




class SiloVisualizer:
    def __init__(self, discr):
        from pyvtk import PolyData
        from pytools import flatten

        self.coords = flatten(
                [p[d] for p in discr.points] for d in range(discr.dimensions))

        self.ndims = discr.dimensions
        self.nodelist = []
        self.shapesizes = []
        self.shapecounts = []
        self.nshapetypes = 0
        self.nzones = 0

        from hedge.element import TetrahedralElement

        for eg in discr.element_groups:
            polygons = []
            ldis = eg.local_discretization
            smi = list(ldis.generate_submesh_indices())
            for el, (el_start, el_stop) in zip(eg.members, eg.ranges):
                polygons += [[el_start+j for j in element] for element in smi]

            if len(polygons):
                self.nodelist += flatten(polygons)
                self.shapesizes.append(len(polygons[0]))
                self.shapecounts.append(len(polygons))
                self.nshapetypes += 1
                self.nzones += len(polygons)

    def __call__(self, filename, fields=[], vectors=[], expressions=[],
            description="Hedge visualization", time=None, step=None):
        from hedge.silo import DBFile, \
                DB_CLOBBER, DB_LOCAL, DB_PDB, DB_NODECENT, \
                DBOPT_DTIME, DBOPT_CYCLE
        db = DBFile(filename, DB_CLOBBER, DB_LOCAL, description, DB_PDB)

        # put zone list
        db.put_zonelist("zonelist", self.nzones, self.ndims, self.nodelist,
                self.shapesizes, self.shapecounts)

        # put mesh coordinates
        mesh_opts = {}
        if time is not None:
            mesh_opts[DBOPT_DTIME] = float(time)
        if step is not None:
            mesh_opts[DBOPT_CYCLE] = int(step)

        db.put_ucdmesh("mesh", self.ndims, [], self.coords, self.nzones,
                "zonelist", None, mesh_opts)

        # put data
        for name, field in fields:
            db.put_ucdvar1(name, "mesh", field, DB_NODECENT)
        for name, vec in vectors:
            db.put_ucdvar(name, "mesh", 
                    ["%s_comp%d" % (name, i) for i in range(len(vec))],
                    vec, DB_NODECENT)
        if expressions:
            db.put_defvars("defvars", expressions)
        del db

