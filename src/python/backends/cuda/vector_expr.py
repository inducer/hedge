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
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
import pymbolic.mapper.substitutor




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




class DefaultingSubstitutionMapper(
        pymbolic.mapper.substitutor.SubstitutionMapper):
    def handle_unsupported_expression(self, expr):
        result = self.subst_func(expr)
        if result is not None:
            return result
        else:
            return expr

class CompiledVectorExpression(object):
    def __init__(self, vec_expr, type_getter, result_dtype, 
            stream=None, allocator=drv.mem_alloc):
        """
        @arg type_getter: A function `expr -> (is_vector, dtype)`, where
          C{is_vector} is a C{bool} determining whether C{expr} evaluates to 
          an aggregate over whose entries the kernel should iterate, and 
          C{dtype} is the numpy dtype of the expression.
        """
        self.result_dtype = result_dtype
        self.stream = stream
        self.allocator = allocator

        from pymbolic import get_dependencies
        deps = get_dependencies(vec_expr, 
                include_subscripts=True,
                include_lookups=True,
                include_calls="descend_args")

        self.vector_exprs = [dep for dep in deps if type_getter(dep)[0]]
        self.scalar_exprs = [dep for dep in deps if not type_getter(dep)[0]]
        vector_names = ["v%d" % i for i in range(len(self.vector_exprs))]
        scalar_names = ["s%d" % i for i in range(len(self.scalar_exprs))]

        from pymbolic import substitute, var
        var_i = var("i")
        subst_map = dict(
                list(zip(self.vector_exprs, [var(vecname)[var_i]
                    for vecname in vector_names]))
                +list(zip(self.scalar_exprs, 
                    [var(scaname) for scaname in scalar_names])))

        def subst_func(expr):
            try:
                return subst_map[expr]
            except KeyError:
                return None

        subst_expr = DefaultingSubstitutionMapper(subst_func)(vec_expr)

        from pymbolic.mapper.stringifier import PREC_NONE, PREC_SUM
        from pymbolic.mapper.c_code import CCodeMapper

        def get_c_declarator(name, is_vector, dtype):
            if is_vector:
                return "%s *%s" % (dtype_to_ctype(dtype), name)
            else:
                return "%s %s" % (dtype_to_ctype(dtype), name)
            
        from hedge.tools import is_obj_array
        if is_obj_array(subst_expr):
            args = [get_c_declarator("_result%d" % i, True, result_dtype)
                    for i in range(len(subst_expr))]
            exprs = subst_expr
        else:
            args = [get_c_declarator("_result0", True, result_dtype)]
            exprs = [subst_expr]

        self.result_count = len(exprs)
        self.subst_expr = subst_expr
        
        code_mapper = CCodeMapper()
        expr_codes = [code_mapper(e, PREC_NONE) for e in exprs]
        code_lines = [
            "%s = %s;" % (
                get_c_declarator(
                    "%s%d" % (code_mapper.cse_prefix, i), 
                    False, result_dtype),
                cse)
            for i, cse in enumerate(code_mapper.cses)
            ] + [
            "_result%d[i] = %s;" % (i, ec)
            for i, ec in enumerate(expr_codes)]

        args.extend(
                get_c_declarator(var_name, *type_getter(var_expr)) 
                for var_expr, var_name in zip(
                    self.vector_exprs+self.scalar_exprs, 
                    vector_names+scalar_names))

        #print ",".join(args)
        #print "\n".join(code_lines)

        from pycuda._kernel import get_scalar_kernel
        self.kernel = get_scalar_kernel(", ".join(args),
                "\n".join(code_lines),
                name="vector_expression",
                keep=True)

        from pymbolic.mapper.flop_counter import FlopCounter
        self.flop_count = FlopCounter()(subst_expr)

    def __call__(self, evaluate_subexpr, add_timer=None):
        vectors = [evaluate_subexpr(vec_expr) for vec_expr in self.vector_exprs]
        scalars = [evaluate_subexpr(scal_expr) for scal_expr in self.scalar_exprs]

        from pytools import single_valued
        shape = single_valued(vec.shape for vec in vectors)

        assert self.result_count > 0
        from hedge.tools import make_obj_array
        results = [gpuarray.empty(shape, self.result_dtype, self.stream, self.allocator)
                for i in range(self.result_count)]

        size = results[0].size
        self.kernel.set_block_shape(*results[0]._block)
        args = ([r.gpudata for r in results]
                +[v.gpudata for v in vectors]
                +scalars
                +[size])

        if add_timer is not None:
            add_timer(vectors[0].size,  self,
                    self.kernel.prepared_timed_call(vectors[0]._grid, *args))
        else:
            self.kernel.prepared_async_call(vectors[0]._grid, self.stream, *args)

        from hedge.tools import is_obj_array, make_obj_array
        if is_obj_array(self.subst_expr):
            return make_obj_array(results)
        else:
            return results[0]




if __name__ == "__main__":
    import pycuda.autoinit
    from pymbolic import parse
    expr = parse("2*x+3*y+4*z")
    print expr
    cexpr = CompiledVectorExpression(expr, 
            lambda expr: (True, numpy.float32),
            numpy.float32)
    from pymbolic import var
    print cexpr({
        var("x"): gpuarray.arange(5),
        var("y"): gpuarray.arange(5),
        var("z"): gpuarray.arange(5),
        })

