"""Visualization for global DG functions. Supports VTK, Silo, etc."""

__copyright__ = "Copyright (C) 2007 Andreas Kloeckner"

__license__ = """
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see U{http://www.gnu.org/licenses/}.
"""




import hedge.tools 
import numpy




class Visualizer(object):
    pass




# legacy vtk ------------------------------------------------------------------
def _three_vector(x):
    if len(x) == 3:
        return x
    elif len(x) == 2:
        return x[0], x[1], 0.
    elif len(x) == 1:
        return x[0], 0, 0.




class LegacyVtkFile(object):
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
            



class LegacyVtkVisualizer(Visualizer):
    def __init__(self, discr):
        from pyvtk import PolyData

        points = [_three_vector(p) for p in discr.nodes]
        polygons = []

        for eg in discr.element_groups:
            ldis = eg.local_discretization
            for el, (el_start, el_stop) in zip(eg.members, eg.ranges):
                polygons += [[el_start+j for j in element] 
                        for element in ldis.get_submesh_indices()]

        self.structure = PolyData(points=points, polygons=polygons)

    def make_file(self, pathname, pcontext=None):
        if pcontext is not None:
            if len(pcontext.ranks) > 1:
                raise RuntimeError, "Legacy VTK does not suport parallel visualization"
        return LegacyVtkFile(pathname+".vtk", self.structure)

    def add_data(self, vtkfile, scalars=[], vectors=[], scale_factor=1):
        from pyvtk import Scalars, Vectors
        import numpy

        vtkfile.pointdata.extend(
                Scalars(numpy.array(scale_factor*field), 
                    name=name, lookup_table="default") 
                for name, field in scalars)
        vtkfile.pointdata.extend(
                Vectors([_three_vector(scale_factor*v) 
                    for v in zip(field)], name=name)
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






class VtkVisualizer(Visualizer, hedge.tools.Closable):
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
                VTK_LINE, VTK_TRIANGLE, VTK_TETRA, VF_LIST_OF_VECTORS
        from hedge.mesh import Interval, Triangle, Tetrahedron

        # For now, we use IntVector here because the Python allocator
        # is somewhat reluctant to return allocated chunks of memory
        # to the OS.
        from hedge._internal import IntVector
        cells = IntVector()
        cell_types = IntVector()

        for eg in discr.element_groups:
            ldis = eg.local_discretization
            smi = ldis.get_submesh_indices()

            cells.reserve(len(cells)+len(smi)*len(eg.members))
            for el, (el_start, el_stop) in zip(eg.members, eg.ranges):
                for element in smi:
                    for j in element:
                        cells.append(el_start+j)

            if ldis.geometry is Interval:
                vtk_eltype = VTK_LINE
            elif ldis.geometry is Triangle:
                vtk_eltype = VTK_TRIANGLE
            elif ldis.geometry is Tetrahedron:
                vtk_eltype = VTK_TETRA
            else:
                raise RuntimeError, "unsupported element type: %s" % ldis.geometry

            cell_types.extend([vtk_eltype] * len(smi) * len(eg.members))

        self.grid = UnstructuredGrid(
                (len(discr), 
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

    def add_data(self, visf, variables=[], scalars=[], vectors=[], time=None, step=None,
            scale_factor=1):
        if scalars or vectors:
            import warnings
            warnings.warn("`scalars' and `vectors' arguments are deprecated",
                    DeprecationWarning)
            variables = scalars + vectors

        from hedge.vtk import DataArray, VF_LIST_OF_COMPONENTS
        for name, field in variables:
            visf.grid.add_pointdata(DataArray(name, scale_factor*field,
                vector_format=VF_LIST_OF_COMPONENTS))

        self.register_pathname(time, visf.get_head_pathname())





# silo ------------------------------------------------------------------------
class SiloMeshData(object):
    def __init__(self, dim, coords, element_groups):
        self.coords = coords

        from pylo import IntVector
        self.ndims = dim
        self.nodelist = IntVector()
        self.shapesizes = IntVector()
        self.shapecounts = IntVector()
        self.nshapetypes = 0
        self.nzones = 0

        for nodelist_size_estimate, eg in element_groups:
            poly_count = 0
            poly_length = None
            self.nodelist.reserve(len(self.nodelist) + nodelist_size_estimate)
            for polygon in eg:
                prev_nodelist_len = len(self.nodelist)
                for i in polygon:
                    self.nodelist.append(i)
                poly_count += 1
                poly_length = len(self.nodelist) - prev_nodelist_len

            if poly_count:
                self.shapesizes.append(poly_length)
                self.shapecounts.append(poly_count)
                self.nshapetypes += 1
                self.nzones += poly_count

    def put_mesh(self, silo, zonelist_name, mesh_name, mesh_opts):
        # put zone list
        silo.put_zonelist(zonelist_name, self.nzones, self.ndims, self.nodelist,
                self.shapesizes, self.shapecounts)

        silo.put_ucdmesh(mesh_name, self.ndims, [], self.coords, self.nzones,
                zonelist_name, None, mesh_opts)




class SiloVisualizer(Visualizer):
    def __init__(self, discr, pcontext=None):
        def generate_fine_elements(eg):
            ldis = eg.local_discretization
            smi = ldis.get_submesh_indices()
            for el, (el_start, el_stop) in zip(eg.members, eg.ranges):
                for element in smi:
                    yield [el_start+j for j in element]

        def generate_fine_element_groups():
            for eg in discr.element_groups:
                ldis = eg.local_discretization
                smi = ldis.get_submesh_indices()
                nodelist_size_estimate = len(eg.members) * len(smi) * len(smi[0])
                yield nodelist_size_estimate, generate_fine_elements(eg)

        def generate_coarse_elements(eg):
            for el in eg.members:
                yield el.vertex_indices

        def generate_coarse_element_groups():
            for eg in discr.element_groups:
                if eg.members:
                    nodelist_size_estimate = len(eg.members) \
                            * len(eg.members[0].vertex_indices)
                else:
                    nodelist_size_estimate = 0

                yield nodelist_size_estimate, generate_coarse_elements(eg)

        self.dim = discr.dimensions
        if self.dim != 1:
            self.fine_mesh = SiloMeshData(self.dim, 
                    numpy.asarray(discr.nodes.T, order="C"),
                    generate_fine_element_groups())
            self.coarse_mesh = SiloMeshData(self.dim, 
                    numpy.asarray(discr.mesh.points.T, order="C"),
                    generate_coarse_element_groups())
        else:
            self.xvals = numpy.asarray(discr.nodes.T, order="C")
            
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

    def add_data(self, silo, variables=[], scalars=[], vectors=[], expressions=[],
            time=None, step=None, scale_factor=1):
        if scalars or vectors:
            import warnings
            warnings.warn("`scalars' and `vectors' arguments are deprecated",
                    DeprecationWarning)
            variables = scalars + vectors

        from pylo import DB_NODECENT, DBOPT_DTIME, DBOPT_CYCLE

        # put mesh coordinates
        mesh_opts = {}
        if time is not None:
            mesh_opts[DBOPT_DTIME] = float(time)
        if step is not None:
            mesh_opts[DBOPT_CYCLE] = int(step)

        if self.dim == 1:
            for name, field in variables:
                if isinstance(field, list) and len(field) > 1:
                    from warnings import warn
                    warn("Silo visualization does not support vectors in 1D, ignoring '%s'" % name)
                else:
                    if isinstance(field, list):
                        field = field[0]
                    silo.put_curve(name, self.xvals, scale_factor*field, mesh_opts)
        else:
            self.fine_mesh.put_mesh(silo, "finezonelist", "finemesh", mesh_opts)
            self.coarse_mesh.put_mesh(silo, "coarsezonelist", "mesh", mesh_opts)

            from hedge.tools import log_shape

            # put data
            for name, field in variables:
                ls = log_shape(field)
                if ls != () and ls[0] > 1:
                    assert len(ls) == 1
                    silo.put_ucdvar(name, "finemesh", 
                            ["%s_comp%d" % (name, i) 
                                for i in range(ls[0])],
                            scale_factor*field, DB_NODECENT)
                else:
                    if ls != ():
                        field = field[0]
                    silo.put_ucdvar1(name, "finemesh", scale_factor*field, DB_NODECENT)

        if expressions:
            silo.put_defvars("defvars", expressions)




# tools -----------------------------------------------------------------------
def get_rank_partition(pcon, discr):
    vec = discr.volume_zeros()
    vec[:] = pcon.rank
    return vec
