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




def dtype_to_ctype(dtype):
    dtype = numpy.dtype(dtype)
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




class Generable(object):
    def __str__(self):
        return "\n".join(self.generate())

class Declarator(Generable):
    def generate(self, with_semicolon=True):
        tp_lines, tp_decl = self.get_decl_pair()
        tp_lines = list(tp_lines)
        for line in tp_lines[:-1]:
            yield line
        sc = ";"
        if not with_semicolon:
            sc = ""
        if tp_decl is None:
            yield "%s%s" % (tp_lines[-1], sc)
        else:
            yield "%s %s%s" % (tp_lines[-1], tp_decl, sc)

    def inline(self, with_semicolon=True):
        tp_lines, tp_decl = self.get_decl_pair()
        tp_lines = " ".join(tp_lines)
        if tp_decl is None:
            return tp_lines
        else:
            return "%s %s" % (tp_lines, tp_decl)

class POD(Declarator):
    def __init__(self, dtype, name):
        self.dtype = numpy.dtype(dtype)
        self.name = name
        
    def get_decl_pair(self):
        return [dtype_to_ctype(self.dtype)], self.name

    def struct_args(self, data):
        return [data]

    def struct_format(self):
        return self.dtype.char

class Value(Declarator):
    def __init__(self, typename, name):
        self.typename = typename
        self.name = name
        
    def get_decl_pair(self):
        return [self.typename], self.name

    def struct_args(self, data):
        raise RuntimeError, "named-type values can't be put into structs"

    def struct_format(self):
        raise RuntimeError, "named-type values have no struct format"




class NestedDeclarator(Declarator):
    def __init__(self, subdecl):
        self.subdecl = subdecl

    @property 
    def name(self):
        return self.subdecl.name

    def struct_format(self):
        return self.subdecl.struct_format()

    def struct_args(self, data):
        return self.subdecl.struct_args(data)

    def get_decl_pair(self):
        return self.subdecl.get_decl_pair()

class DeclSpecifier(NestedDeclarator):
    def __init__(self, subdecl, spec):
        NestedDeclarator.__init__(self, subdecl)
        self.spec = spec

    def get_decl_pair(self):
        def add_spec(sub_it):
            it = iter(sub_it)
            try:
                yield "%s %s" % (self.spec, it.next())
            except StopIteration:
                pass

            for line in it:
                yield line

        sub_tp, sub_decl = self.subdecl.get_decl_pair()
        return add_spec(sub_tp), sub_decl

class Typedef(DeclSpecifier):
    def __init__(self, subdecl):
        DeclSpecifier.__init__(self, subdecl, "typedef")

class Static(DeclSpecifier):
    def __init__(self, subdecl):
        DeclSpecifier.__init__(self, subdecl, "static")

class CudaGlobal(DeclSpecifier):
    def __init__(self, subdecl):
        DeclSpecifier.__init__(self, subdecl, "__global__")

class CudaDevice(DeclSpecifier):
    def __init__(self, subdecl):
        DeclSpecifier.__init__(self, subdecl, "__device__")

class CudaShared(DeclSpecifier):
    def __init__(self, subdecl):
        DeclSpecifier.__init__(self, subdecl, "__shared__")

class CudaConstant(DeclSpecifier):
    def __init__(self, subdecl):
        DeclSpecifier.__init__(self, subdecl, "__constant__")



class Const(NestedDeclarator):
    def get_decl_pair(self):
        sub_tp, sub_decl = self.subdecl.get_decl_pair()
        return sub_tp, ("const %s" % sub_decl)

class Pointer(NestedDeclarator):
    def __init__(self, subdecl, count=None):
        NestedDeclarator.__init__(self, subdecl)
        self.count = count

    def get_decl_pair(self):
        sub_tp, sub_decl = self.subdecl.get_decl_pair()
        return sub_tp, ("*%s" % sub_decl)

    def struct_args(self, data):
        return data

    def struct_format(self):
        return "P"

class ArrayOf(NestedDeclarator):
    def __init__(self, subdecl, count=None):
        NestedDeclarator.__init__(self, subdecl)
        self.count = count

    def get_decl_pair(self):
        sub_tp, sub_decl = self.subdecl.get_decl_pair()
        if self.count is None:
            count_str = ""
        else:
            count_str = str(self.count)
        return sub_tp, ("%s[%s]" % (sub_decl, count_str))

    def struct_args(self, data):
        return data

    def struct_format(self):
        if self.count is None:
            return "P"
        else:
            return "%d%s" % (self.count, self.subdecl.struct_format())




class FunctionDeclaration(NestedDeclarator):
    def __init__(self, subdecl, arg_decls):
        NestedDeclarator.__init__(self, subdecl)
        self.arg_decls = arg_decls

    def get_decl_pair(self):
        sub_tp, sub_decl = self.subdecl.get_decl_pair()

        return sub_tp, ("%s(%s)" % (
            sub_decl, 
            ", ".join(ad.inline() for ad in self.arg_decls)))

    def struct_args(self, data):
        raise RuntimeError, "function pointers can't be put into structs"

    def struct_format(self):
        raise RuntimeError, "function pointers have no struct format"




class Struct(Declarator):
    def __init__(self, tpname, fields, declname=None, debug=False):
        self.tpname = tpname
        self.fields = fields
        self.declname = declname
        self.debug = debug

    def get_decl_pair(self):
        def get_tp():
            if self.tpname is not None:
                yield "struct %s" % self.tpname
            else:
                yield "struct" 
            yield "{"
            for f in self.fields:
                for f_line in f.generate():
                    yield "  " + f_line
            yield "}"
        return get_tp(), self.declname
        
    def make(self, **kwargs):
        import struct
        data = []
        for f in self.fields:
            data.extend(f.struct_args(kwargs[f.name]))
        if self.debug:
            print self
            print data
            print self.struct_format()
            print kwargs
            print [ord(i) for i in struct.pack(self.struct_format(), *data)]
            raw_input()
        return struct.pack(self.struct_format(), *data)

    @memoize_method
    def struct_format(self):
        return "".join(decl.struct_format() for decl in self.fields)

    @memoize_method
    def __len__(self):
        from struct import calcsize
        return calcsize(self.struct_format())




# control flow/statement stuff ------------------------------------------------
class If(Generable):
    def __init__(self, condition, then_, else_=None):
        self.condition = condition
        self.then_ = then_
        self.else_ = else_

    def generate(self):
        yield "if (%s)" % self.condition

        if isinstance(self.then_, Block):
            for line in self.then_.generate():
                yield line
        else:
            for line in self.then_.generate():
                yield "  "+line

        if self.else_ is not None:
            yield "else"
            if isinstance(self.else_, Block):
                for line in self.else_.generate():
                    yield line
            else:
                for line in self.else_.generate():
                    yield "  "+line

class While(Generable):
    def __init__(self, condition, body):
        self.condition = condition
        self.body = body

    def generate(self):
        yield "while (%s)" % self.condition

        if isinstance(self.body, Block):
            for line in self.body.generate():
                yield line
        else:
            for line in self.body.generate():
                yield "  "+line

class For(Generable):
    def __init__(self, start, condition, end, body):
        self.start = start
        self.condition = condition
        self.end = end
        self.body = body

    def generate(self):
        yield "for (%s; %s; %s)" % (self.start, self.condition, self.end)

        if isinstance(self.body, Block):
            for line in self.body.generate():
                yield line
        else:
            for line in self.body.generate():
                yield "  "+line

class DoWhile(Generable):
    def __init__(self, condition, body):
        self.condition = condition
        self.body = body

    def generate(self):
        yield "do"
        if isinstance(self.body, Block):
            for line in self.body.generate():
                yield line
        else:
            for line in self.body.generate():
                yield "  "+line
        yield "while (%s)" % self.condition

def make_multiple_ifs(conditions_and_blocks, base=None):
    for cond, block in conditions_and_blocks[::-1]:
        base = If(cond, block, base)
    return base




# simple statements -----------------------------------------------------------
class Define(Generable):
    def __init__(self, symbol, value):
        self.symbol = symbol
        self.value = value

    def generate(self):
        yield "#define %s %s" % (self.symbol, self.value)

class Pragma(Generable):
    def __init__(self, value):
        self.value = value

    def generate(self):
        yield "#pragma %s" % (self.value)

class Statement(Generable):
    def __init__(self, text):
        self.text = text

    def generate(self):
        yield self.text+";"

class Assign(Generable):
    def __init__(self, lvalue, rvalue):
        self.lvalue = lvalue
        self.rvalue = rvalue

    def generate(self):
        yield "%s = %s;" % (self.lvalue, self.rvalue)

class Line(Generable):
    def __init__(self, text=""):
        self.text = text

    def generate(self):
        yield self.text

class Comment(Generable):
    def __init__(self, text):
        self.text = text

    def generate(self):
        yield "/* %s */" % self.text



# initializers ----------------------------------------------------------------
class Initializer(Generable):
    def __init__(self, vdecl, data):
        self.vdecl = vdecl
        self.data = data

    def generate(self):
        tp_lines, tp_decl = self.vdecl.get_decl_pair()
        tp_lines = list(tp_lines)
        for line in tp_lines[:-1]:
            yield line
        yield "%s %s = %s;" % (tp_lines[-1], tp_decl, self.data)

def Constant(vdecl, data):
    return Initializer(Const(vdecl), data)

class ArrayInitializer(Generable):
    def __init__(self, vdecl, data):
        self.vdecl = vdecl
        self.data = data

    def generate(self):
        for v_line in self.vdecl.generate(with_semicolon=False):
            yield v_line
        yield "  = { %s };" % (", ".join(str(item) for item in self.data))

class FunctionBody(Generable):
    def __init__(self, fdecl, body):
        self.fdecl = fdecl
        self.body = body

    def generate(self):
        for f_line in self.fdecl.generate(with_semicolon=False):
            yield f_line
        for b_line in self.body.generate():
            yield b_line





# block -----------------------------------------------------------------------
class Block(Generable):
    def __init__(self, contents=[]):
        self.contents = contents[:]

    def generate(self):
        yield "{"
        for item in self.contents:
            for item_line in item.generate():
                yield "  " + item_line
        yield "}"

    def append(self, data):
        self.contents.append(data)

    def extend(self, data):
        self.contents.extend(data)
        
    def extend_log_block(self, descr, data):
        self.contents.append(Comment(descr))
        self.contents.extend(data)
        self.contents.append(Line())
        
class Module(Block):
    def generate(self):
        for c in self.contents:
            for line in c.generate():
                yield line





def _test():
    s = Struct("yuck", [
        POD(numpy.float32, "h", ),
        POD(numpy.float32, "order"),
        POD(numpy.float32, "face_jacobian"),
        ArrayOf(POD(numpy.float32, "normal"), 17),
        POD(numpy.uint16, "a_base"),
        POD(numpy.uint16, "b_base"),
        CudaGlobal(POD(numpy.uint8, "a_ilist_number")),
        POD(numpy.uint8, "b_ilist_number"),
        POD(numpy.uint8, "bdry_flux_number"), # 0 if not on boundary
        POD(numpy.uint8, "reserved"),
        POD(numpy.uint32, "b_global_base"),
        ])
    f_decl = FunctionDeclaration(POD(numpy.uint16, "get_num"), [ 
        POD(numpy.uint8, "reserved"),
        POD(numpy.uint32, "b_global_base"),
        ])
    f_body = FunctionBody(f_decl, Block([
        POD(numpy.uint32, "i"),
        For("i = 0", "i < 17", "++i",
            If("a > b",
                Assign("a", "b"),
                Block([
                    Assign("a", "b-1", "+="),
                    Break(),
                    ])
                ),
            ),
        BlankLine(),
        Comment("all done"),
        ]))
    print s
    print f_body




if __name__ == "__main__":
    _test()
