"""Generic support for new-style (XML) VTK visualization data files."""

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




import numpy




VTK_INT8 = "Int8"
VTK_UINT8 = "UInt8"
VTK_INT16 = "Int16"
VTK_UINT16 = "UInt16"
VTK_INT32 = "Int32"
VTK_UINT32 = "UInt32"
VTK_INT64 = "Int64"
VTK_UINT64 = "UInt64"
VTK_FLOAT32 = "Float32"
VTK_FLOAT64 = "Float64"

VTK_VERTEX = 1
VTK_POLY_VERTEX = 2
VTK_LINE = 3
VTK_POLY_LINE = 4
VTK_TRIANGLE = 5
VTK_TRIANGLE_STRIP = 6
VTK_POLYGON = 7
VTK_PIXEL = 8
VTK_QUAD = 9
VTK_TETRA = 10
VTK_VOXEL = 11
VTK_HEXAHEDRON = 12
VTK_WEDGE = 13
VTK_PYRAMID = 14

CELL_NODE_COUNT = {
        VTK_VERTEX: 1,
        # VTK_POLY_VERTEX: no a-priori size
        VTK_LINE: 2,
        # VTK_POLY_LINE: no a-priori size
        VTK_TRIANGLE: 3,
        # VTK_TRIANGLE_STRIP: no a-priori size
        # VTK_POLYGON: no a-priori size
        VTK_PIXEL: 4,
        VTK_QUAD: 4,
        VTK_TETRA: 4,
        VTK_VOXEL: 8,
        VTK_HEXAHEDRON: 8,
        VTK_WEDGE: 6,
        VTK_PYRAMID: 5,
        }


VF_LIST_OF_COMPONENTS = 0 # [[x0,y0,z0], [x1,y1,z1]
VF_LIST_OF_VECTORS = 1 # [[x0,x1], [y0,y1], [z0,z1]]




# Ah, the joys of home-baked non-compliant XML goodness.
class XMLElementBase(object):
    def __init__(self):
        self.children = []

    def copy(self, new_children=None):
        result = self.__class__(self.tag, self.attributes)
        if new_children is not None:
            result.children = new_children
        else:
            result.children = self.children
        return result

    def add_child(self, child):
        self.children.append(child)




class XMLElement(XMLElementBase):
    def __init__(self, tag, **attributes):
        XMLElementBase.__init__(self)
        self.tag = tag
        self.attributes = attributes

    def write(self, file):
        attr_string = "".join(
                " %s=\"%s\"" % (key,value) 
                for key,value in self.attributes.iteritems())
        if self.children:
            file.write("<%s%s>\n" % (self.tag, attr_string))
            for child in self.children:
                if isinstance(child, XMLElement):
                    child.write(file)
                else:
                    # likely a string instance, write it directly
                    file.write(child)
            file.write("</%s>\n" % self.tag)
        else:
            file.write("<%s%s/>\n" % (self.tag, attr_string))





class XMLRoot(XMLElementBase):
    def __init__(self, child=None):
        XMLElementBase.__init__(self)
        if child:
            self.add_child(child)

    def write(self, file):
        file.write("<?xml version=\"1.0\"?>\n")
        for child in self.children:
            if isinstance(child, XMLElement):
                child.write(file)
            else:
                # likely a string instance, write it directly
                file.write(child)



class EncodedBuffer:
    def encoder(self):
        """Return an identifier for the binary encoding used."""
        raise NotImplementedError

    def compressor(self):
        """Return an identifier for the compressor used, or None."""
        raise NotImplementedError

    def raw_buffer(self):
        """Reobtain the raw buffer string object that was used to
        construct this encoded buffer."""

        raise NotImplementedError

    def add_to_xml_element(self, xml_element):
        """Add encoded buffer to the given C{xml_element}.
        Return total size of encoded buffer in bytes."""

        raise NotImplementedError





class BinaryEncodedBuffer:
    def __init__(self, buffer):
        self.buffer = buffer

    def encoder(self):
        return "binary"

    def compressor(self):
        return None

    def raw_buffer(self):
        return self.buffer

    def add_to_xml_element(self, xml_element):
        raise NotImplementedError




class Base64EncodedBuffer:
    def __init__(self, buffer):
        from hedge._internal import bufferize_int32
        from base64 import b64encode
        self.b64header = b64encode(bufferize_int32([len(buffer)]))
        self.b64data = b64encode(buffer)

    def encoder(self):
        return "base64"

    def compressor(self):
        return None

    def raw_buffer(self):
        from base64 import b64decode
        return b64decode(self.b64data)

    def add_to_xml_element(self, xml_element):
        """Add encoded buffer to the given C{xml_element}.
        Return total size of encoded buffer in bytes."""

        xml_element.add_child(self.b64header)
        xml_element.add_child(self.b64data)

        return len(self.b64header) + len(self.b64data)




class Base64ZLibEncodedBuffer:
    def __init__(self, buffer):
        from hedge._internal import bufferize_int32
        from base64 import b64encode
        from zlib import compress
        comp_buffer = compress(buffer)
        comp_header = [1, len(buffer), len(buffer), len(comp_buffer)]
        self.b64header = b64encode(bufferize_int32(comp_header))
        self.b64data = b64encode(comp_buffer)

    def encoder(self):
        return "base64"

    def compressor(self):
        return "zlib"

    def raw_buffer(self):
        from base64 import b64decode
        from zlib import decompress
        return decompress(b64decode(self.b64data))

    def add_to_xml_element(self, xml_element):
        """Add encoded buffer to the given C{xml_element}.
        Return total size of encoded buffer in bytes."""

        xml_element.add_child(self.b64header)
        xml_element.add_child(self.b64data)

        return len(self.b64header) + len(self.b64data)






class DataArray(object):
    def __init__(self, name, container, typehint=None, vector_padding=3, 
            vector_format=VF_LIST_OF_COMPONENTS, components=None):
        self.name = name

        if isinstance(container, DataArray):
            self.type = container.type
            self.components = container.components
            self.encoded_buffer = container.encoded_buffer
            return

        from hedge._internal import \
                bufferize_vector, \
                bufferize_list_of_vectors, \
                bufferize_list_of_components, \
                bufferize_int32, \
                bufferize_uint8

        def vec_type(vec):
            # FIXME
            return VTK_FLOAT64

        from hedge._internal import IntVector

        if isinstance(container, numpy.ndarray):
            if container.dtype == object:
                if vector_format == VF_LIST_OF_COMPONENTS:
                    ctr = list(container)
                    if len(ctr) > 1:
                        while len(ctr) < vector_padding:
                            ctr.append(None)
                        self.type = vec_type(ctr[0])
                    else:
                        self.type = VTK_FLOAT64

                    self.components = len(ctr)
                    buffer =  bufferize_list_of_components(ctr, len(ctr[0]))

                elif vector_format == VF_LIST_OF_VECTORS:
                    self.type = vec_type(container[0])
                    self.components = components or len(container[0])
                    if self.components < vector_padding:
                        self.components = vector_padding
                    buffer =  bufferize_list_of_vectors(container, self.components)

                else:
                    raise TypeError, "unrecognized vector format"
            else:
                if len(container.shape) > 1:
                    if vector_format == VF_LIST_OF_COMPONENTS:
                        container = container.T.copy()

                    assert len(container.shape) == 2, "numpy vectors of rank >2 are not supported"
                    assert container.strides[1] == container.itemsize, "2D numpy arrays must be row-major"
                    if vector_padding > container.shape[1]:
                        container = numpy.asarray(numpy.hstack((
                                container, 
                                numpy.zeros((
                                    container.shape[0], 
                                    vector_padding-container.shape[1],
                                    ),
                                    container.dtype))), order="C")
                    self.components = container.shape[1]
                else:
                    self.components = 1
                self.type = vec_type(container)
                buffer = bufferize_vector(container)

        elif isinstance(container, IntVector):
            self.components = 1
            if typehint == VTK_UINT8:
                self.type = VTK_UINT8
                buffer = bufferize_uint8(container)
            elif typehint == VTK_INT32: 
                self.type = VTK_INT32
                buffer = bufferize_int32(container)
            else:
                raise ValueError, "unsupported typehint"


                if typehint is not None:
                    assert typehint == self.type
        else:
            raise ValueError, "cannot convert object of type `%s' to DataArray" % container

        self.encoded_buffer = BinaryEncodedBuffer(buffer)

    def get_encoded_buffer(self, encoder, compressor):
        have_encoder = self.encoded_buffer.encoder()
        have_compressor = self.encoded_buffer.compressor()

        if (encoder, compressor) != (have_encoder, have_compressor):
            raw_buf = self.encoded_buffer.raw_buffer()

            # avoid having three copies of the buffer around temporarily
            del self.encoded_buffer

            if (encoder, compressor) == ("binary", None):
                self.encoded_buffer = BinaryEncodedBuffer(raw_buf)
            elif (encoder, compressor) == ("base64", None):
                self.encoded_buffer = Base64EncodedBuffer(raw_buf)
            elif (encoder, compressor) == ("base64", "zlib"):
                self.encoded_buffer = Base64ZLibEncodedBuffer(raw_buf)
            else:
                self.encoded_buffer = BinaryEncodedBuffer(raw_buf)
                raise ValueError, "invalid encoder/compressor pair"

            have_encoder = self.encoded_buffer.encoder()
            have_compressor = self.encoded_buffer.compressor()

            assert (encoder, compressor) == (have_encoder, have_compressor)

        return self.encoded_buffer

    def encode(self, compressor, xml_element):
        ebuf = self.get_encoded_buffer("base64", compressor)
        return ebuf.add_to_xml_element(xml_element)

    def invoke_visitor(self, visitor):
        return visitor.gen_data_array(self)




class UnstructuredGrid(object):
    def __init__(self, points, cells, cell_types):
        self.point_count = len(points)
        self.cell_count = len(cells)

        self.point_count, self.points = points
        assert self.points.name == "points"

        try:
            self.cell_count, self.cell_connectivity, \
                    self.cell_offsets = cells
        except:
            self.cell_count = len(cell_types)

            def cumsum(container):
                run_sum = 0
                for i in range(len(container)):
                    run_sum += container[i]
                    container[i] = run_sum
                return container

            from hedge._internal import IntVector
            offsets = cumsum(
                    IntVector(CELL_NODE_COUNT[ct] for ct in cell_types)
                    )

            self.cell_connectivity = DataArray("connectivity", cells, VTK_INT32)
            self.cell_offsets = DataArray("offsets", offsets, VTK_INT32)

        self.cell_types = DataArray("types", cell_types, VTK_UINT8)

        self.pointdata = []
        self.celldata = []

    def copy(self):
        return UnstructuredGrid(
                (self.point_count, self.points),
                (self.cell_count, self.cell_connectivity,
                    self.cell_offsets), 
                self.cell_types)

    def vtk_extension(self):
        return "vtu"

    def invoke_visitor(self, visitor):
        return visitor.gen_unstructured_grid(self)

    def add_pointdata(self, data_array):
        self.pointdata.append(data_array)





def make_vtkfile(filetype, compressor):
    import sys
    if sys.byteorder == "little":
        bo = "LittleEndian"
    else:
        bo = "BigEndian"

    kwargs = {}
    if compressor == "zlib":
        kwargs["compressor"] = "vtkZLibDataCompressor"

    return XMLElement("VTKFile", type=filetype, version="0.1", byte_order=bo, **kwargs)




class XMLGenerator(object):
    def __init__(self, compressor=None):
        if compressor == "zlib":
            try:
                import zlib
            except ImportError:
                compress = False
        elif compressor is None:
            pass
        else:
            raise ValueError, "Invalid compressor name `%s'" % compressor

        self.compressor = compressor

    def __call__(self, vtkobj):
        child = self.rec(vtkobj)
        vtkf = make_vtkfile(child.tag, self.compressor)
        vtkf.add_child(child)
        return XMLRoot(vtkf)

    def rec(self, vtkobj):
        return vtkobj.invoke_visitor(self)





class InlineXMLGenerator(XMLGenerator):
    def gen_unstructured_grid(self, ugrid):
        el = XMLElement("UnstructuredGrid")
        piece = XMLElement("Piece", 
                NumberOfPoints=ugrid.point_count, NumberOfCells=ugrid.cell_count)
        el.add_child(piece)

        pointdata = XMLElement("PointData")
        piece.add_child(pointdata)
        for data_array in ugrid.pointdata:
            pointdata.add_child(self.rec(data_array))

        points = XMLElement("Points")
        piece.add_child(points)
        points.add_child(self.rec(ugrid.points))

        cells = XMLElement("Cells")
        piece.add_child(cells)
        cells.add_child(self.rec(ugrid.cell_connectivity))
        cells.add_child(self.rec(ugrid.cell_offsets))
        cells.add_child(self.rec(ugrid.cell_types))

        return el

    def gen_data_array(self, data):
        el = XMLElement("DataArray", type=data.type, Name=data.name, 
                NumberOfComponents=data.components, format="binary")
        data.encode(self.compressor, el)
        el.add_child("\n")
        return el




class AppendedDataXMLGenerator(InlineXMLGenerator):
    def __init__(self, compressor=None):
        InlineXMLGenerator.__init__(self, compressor)

        self.base64_len = 0
        self.app_data = XMLElement("AppendedData", encoding="base64")
        self.app_data.add_child("_")

    def __call__(self, vtkobj):
        xmlroot = XMLGenerator.__call__(self, vtkobj)
        self.app_data.add_child("\n")
        xmlroot.children[0].add_child(self.app_data)
        return xmlroot

    def gen_data_array(self, data):
        el = XMLElement("DataArray", type=data.type, Name=data.name, 
                NumberOfComponents=data.components, format="appended", 
                offset=self.base64_len)

        self.base64_len += data.encode(self.compressor, self.app_data)

        return el




class ParallelXMLGenerator(XMLGenerator):
    def __init__(self, pathnames):
        XMLGenerator.__init__(self, compressor=None)

        self.pathnames = pathnames

    def gen_unstructured_grid(self, ugrid):
        el = XMLElement("PUnstructuredGrid")

        pointdata = XMLElement("PPointData")
        el.add_child(pointdata)
        for data_array in ugrid.pointdata:
            pointdata.add_child(self.rec(data_array))

        points = XMLElement("PPoints")
        el.add_child(points)
        points.add_child(self.rec(ugrid.points))

        cells = XMLElement("PCells")
        el.add_child(cells)
        cells.add_child(self.rec(ugrid.cell_connectivity))
        cells.add_child(self.rec(ugrid.cell_offsets))
        cells.add_child(self.rec(ugrid.cell_types))

        for pn in self.pathnames:
            el.add_child(XMLElement("Piece", Source=pn))

        return el

    def gen_data_array(self, data):
        from hedge._internal import bufferize_int32
        el = XMLElement("PDataArray", type=data.type, Name=data.name, 
                NumberOfComponents=data.components)
        return el





