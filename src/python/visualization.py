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




import hedge.tools 




# legacy vtk ------------------------------------------------------------------
def _three_vector(x):
    if len(x) == 3:
        return x
    elif len(x) == 2:
        return x[0], x[1], 0.
    elif len(x) == 1:
        return x[0], 0, 0.




class LegacyVtkFile:
    def __init__(self, pathname, structure, description="Hedge visualization"):
        self.pathname = pathname
        self.structure = structure
        self.description = description

        self.pointdata = []
        self.is_closed = False

    def __fin__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        if not self.is_closed:
            from pyvtk import PointData, VtkData
            vtk = VtkData(self.structure, 
                    self.description, 
                    PointData(*self.pointdata))
            vtk.tofile(self.pathname)
            self.is_closed = True
            



class LegacyVtkVisualizer:
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

    def make_file(self, pathname, pcontext=None):
        if pcontext is not None:
            if len(pcontext.ranks) > 1:
                raise RuntimeError, "Legacy VTK does not suport parallel visualization"
        return LegacyVtkFile(pathname+".vtk", self.structure)

    def add_data(self, vtkfile, scalars=[], vectors=[]):
        from pyvtk import Scalars, Vectors
        import numpy

        vtkfile.pointdata.extend(
                Scalars(numpy.array(field), name=name, lookup_table="default") 
                for name, field in scalars)
        vtkfile.pointdata.extend(
                Vectors([_three_vector(v) for v in zip(field)], name=name)
                for name, field in vectors)




# xml vtk ---------------------------------------------------------------------




class VtkFile(hedge.tools.Closable):
    def __init__(self, pathname, grid, filenames=None, compressor=None):
        hedge.tools.Closable.__init__(self)
        self.pathname = pathname
        self.grid = grid
        self.compressor = compressor

    def get_head_pathname(self):
        return self.pathname

    def do_close(self):
        from pytools import assert_not_a_file
        assert_not_a_file(self.pathname)

        from hedge.vtk import InlineXMLGenerator, AppendedDataXMLGenerator

        outf = file(self.pathname, "w")
        AppendedDataXMLGenerator(self.compressor)(self.grid).write(outf)
        #InlineXMLGenerator(self.compressor)(self.grid).write(outf)
        outf.close()





class ParallelVtkFile(VtkFile):
    def __init__(self, pathname, grid, index_pathname, pathnames=None, compressor=None):
        VtkFile.__init__(self, pathname, grid)
        self.index_pathname = index_pathname
        self.pathnames = pathnames
        self.compressor = compressor

    def get_head_pathname(self):
        return self.index_pathname

    def do_close(self):
        VtkFile.do_close(self)

        from hedge.vtk import ParallelXMLGenerator

        outf = file(self.index_pathname, "w")
        ParallelXMLGenerator(self.pathnames)(self.grid).write(outf)
        outf.close()







class VtkVisualizer(hedge.tools.Closable):
    def __init__(self, discr, pcontext=None, basename=None, compressor=None):
        hedge.tools.Closable.__init__(self)

        from pytools import assert_not_a_file

        if basename is not None:
            self.pvd_name = basename+".pvd"
            assert_not_a_file(self.pvd_name)
        else:
            self.pvd_name = None

        self.pcontext = pcontext
        self.compressor = compressor

        if self.pcontext is None or self.pcontext.is_head_rank:
            self.timestep_to_pathnames = {}
        else:
            self.timestep_to_pathnames = None

        from hedge.vtk import UnstructuredGrid, DataArray, \
                VTK_TRIANGLE, VTK_TETRA, VF_LIST_OF_VECTORS
        from hedge.element import Triangle, Tetrahedron

        cells = []
        cell_types = []

        for eg in discr.element_groups:
            ldis = eg.local_discretization

            for el, (el_start, el_stop) in zip(eg.members, eg.ranges):
                smi = ldis.generate_submesh_indices()
                cells.extend([el_start+j for j in element] 
                    for element in smi)
                if isinstance(el, Triangle):
                    cell_types.extend([VTK_TRIANGLE] * len(smi))
                elif isinstance(el, Tetrahedron):
                    cell_types.extend([VTK_TETRA] * len(smi))
                else:
                    raise RuntimeError, "unsupported element type: %s" % type(el)

        self.grid = UnstructuredGrid(
                (len(discr.nodes), 
                    DataArray("points", discr.nodes, vector_format=VF_LIST_OF_VECTORS)),
                cells, cell_types)


    def update_pvd(self):
        if self.pvd_name and self.timestep_to_pathnames:
            from hedge.vtk import XMLRoot, XMLElement, make_vtkfile

            collection = XMLElement("Collection")

            vtkf = make_vtkfile(collection.tag, compressor=None)
            xmlroot = XMLRoot(vtkf)

            vtkf.add_child(collection)

            tsteps = self.timestep_to_pathnames.keys()
            tsteps.sort()
            for i, time in enumerate(tsteps):
                for part, pathname in enumerate(self.timestep_to_pathnames[time]):
                    collection.add_child(XMLElement(
                        "DataSet",
                        timestep=time, part=part, file=pathname))
            outf = open(self.pvd_name, "w")
            xmlroot.write(outf)
            outf.close()

    def do_close(self):
        self.update_pvd()

    def make_file(self, pathname):
        """FIXME

        An appropriate extension (including the dot) is automatically
        appended to `pathname'.
        """
        if self.pcontext is None or len(self.pcontext.ranks) == 1:
            return VtkFile(pathname+"."+self.grid.vtk_extension(), 
                    self.grid.copy(),
                    compressor=self.compressor
                    )
        else:
            filename_pattern = (
                    pathname + "-%05d." + self.grid.vtk_extension())
            if self.pcontext.is_head_rank:
                return ParallelVtkFile(
                        filename_pattern % self.pcontext.rank,
                        self.grid.copy(), 
                        index_pathname="%s.p%s" % (
                            pathname, self.grid.vtk_extension()),
                        pathnames=[
                            filename_pattern % rank for rank in self.pcontext.ranks],
                       compressor=self.compressor 
                       )
            else:
                return VtkFile(
                        filename_pattern % self.pcontext.rank, 
                        self.grid.copy(),
                        compressor=self.compressor
                        )

    def register_pathname(self, time, pathname):
        if time is not None and self.timestep_to_pathnames is not None:
            self.timestep_to_pathnames.setdefault(time, []).append(pathname)

            # When we are run under MPI and cancelled by Ctrl+C, destructors
            # do not get called. Therefore, we just spend the (hopefully negligible)
            # time to update the PVD index every few data additions.
            if len(self.timestep_to_pathnames) % 5 == 0:
                self.update_pvd()

    def add_data(self, visf, scalars=[], vectors=[], time=None, step=None):
        from hedge.vtk import DataArray
        for name, data in scalars:
            visf.grid.add_pointdata(DataArray(name, data))
        for name, data in vectors:
            visf.grid.add_pointdata(DataArray(name, data))

        self.register_pathname(time, visf.get_head_pathname())





# silo ------------------------------------------------------------------------
class SiloMeshData:
    def __init__(self, dim, points, element_groups):
        from pytools import flatten

        self.coords = list(flatten([p[d] for p in points] for d in range(dim)))

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
    def __init__(self, discr, pcontext=None):
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
        self.pcontext = pcontext

    def close(self):
        pass

    def make_file(self, pathname):
        """This function returns either a pylo.SiloFile or a
        pylo.ParallelSiloFile, depending on the ParallelContext
        under which we are running

        An extension of .silo is automatically appended to `pathname'.
        """
        if self.pcontext is None or len(self.pcontext.ranks) == 1:
            from pylo import SiloFile
            return SiloFile(pathname+".silo")
        else:
            from pylo import ParallelSiloFile
            return ParallelSiloFile(
                    pathname, 
                    self.pcontext.rank, self.pcontext.ranks)

    def add_data(self, silo, scalars=[], vectors=[], expressions=[],
            time=None, step=None, write_coarse_mesh=False):
        from pylo import DB_NODECENT, DBOPT_DTIME, DBOPT_CYCLE

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
        for name, field in scalars:
            silo.put_ucdvar1(name, "finemesh", field, DB_NODECENT)
        for name, vec in vectors:
            silo.put_ucdvar(name, "finemesh", 
                    ["%s_comp%d" % (name, i) for i in range(len(vec))],
                    vec, DB_NODECENT)
        if expressions:
            silo.put_defvars("defvars", expressions)




# tools -----------------------------------------------------------------------
def get_rank_partition(pcon, discr):
    vec = discr.volume_zeros()
    vec[:] = pcon.rank
    return vec
