"""Interface with Nvidia CUDA."""

from __future__ import division

__copyright__ = "Copyright (C) 2008 Andreas Kloeckner"

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
import numpy.linalg as la
from pytools import memoize_method




class StructField(object):
    def __init__(self, name, dtype):
        self.name = name
        self.dtype = numpy.dtype(dtype)

    def struct_format(self):
        return self.dtype.char

    @staticmethod
    def ctype(dtype):
        if dtype == numpy.int32:
            return "int"
        elif dtype == numpy.uint32:
            return "unsigned int"
        elif dtype == numpy.int16:
            return "short int"
        elif dtype == numpy.uint16:
            return "short unsigned int"
        elif dtype == numpy.int8:
            return "signed char"
        elif dtype == numpy.uint8:
            return "unsigned char"
        elif dtype == numpy.intp or dtype == numpy.uintp:
            return "void *"
        elif dtype == numpy.float32:
            return "float"
        elif dtype == numpy.float64:
            return "double"
        else:
            raise ValueError, "unable to map dtype '%s'" % dtype

    def cdecl(self):
        return "  %s %s;\n" % (self.ctype(self.dtype), self.name)
    
    def struct_format(self):
        return self.dtype.char

    def prepare(self, arg):
        return [arg]

    def __len__(self):
        return self.dtype.itemsize




class ArrayStructField(StructField):
    def __init__(self, name, dtype, count):
        StructField.__init__(self, name, dtype)
        self.count = count

    def struct_format(self):
        return "%d%s" % (self.count, StructField.struct_format(self))

    def cdecl(self):
        return "  %s %s[%d];\n" % (self.ctype(self.dtype), self.name, self.count)

    def prepare(self, arg):
        assert len(arg) == self.count
        return arg

    def __len__(self):
        return self.count * StructField.__len__(self)




class Struct(object):
    def __init__(self, fields):
        self.fields = fields

    def make(self, **kwargs):
        import struct
        data = []
        for f in self.fields:
            data.extend(f.prepare(kwargs[f.name]))
        return struct.pack(self.struct_format(), *data)

    @memoize_method
    def struct_format(self):
        return "".join(f.struct_format() for f in self.fields)

    @memoize_method
    def cdecl(self, name):
        return "struct %s\n{\n%s};\n" % (name, "".join(f.cdecl() for f in self.fields))

    def __len__(self):
        a = sum(len(f) for f in self.fields)
        from struct import calcsize
        b = calcsize(self.struct_format())
        assert a == b
        return a





