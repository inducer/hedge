"""Base facility for compiled vector expressions."""

from __future__ import division

__copyright__ = "Copyright (C) 2009 Andreas Kloeckner"

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
import pymbolic.mapper.substitutor
import hedge.optemplate




class DefaultingSubstitutionMapper(
        pymbolic.mapper.substitutor.SubstitutionMapper,
        hedge.optemplate.IdentityMapperMixin):
    def handle_unsupported_expression(self, expr):
        result = self.subst_func(expr)
        if result is not None:
            return result
        else:
            pymbolic.mapper.substitutor.SubstitutionMapper.handle_unsupported_expression(
                    self, expr)

class CompiledVectorExpressionBase(object):
    def __init__(self, vec_expr, type_getter, result_dtype):
        """
        @arg type_getter: A function `expr -> (is_vector, dtype)`, where
          C{is_vector} is a C{bool} determining whether C{expr} evaluates to 
          an aggregate over whose entries the kernel should iterate, and 
          C{dtype} is the numpy dtype of the expression.
        """
        self.result_dtype = result_dtype

        from hedge.optemplate import DependencyMapper
        deps = DependencyMapper(
                include_subscripts=True,
                include_lookups=True,
                include_calls="descend_args")(vec_expr)

        self.vector_exprs = [dep for dep in deps if type_getter(dep)[0]]
        self.scalar_exprs = [dep for dep in deps if not type_getter(dep)[0]]
        vector_names = ["v%d" % i for i in range(len(self.vector_exprs))]
        scalar_names = ["s%d" % i for i in range(len(self.scalar_exprs))]

        from pymbolic import var
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

        from pymbolic.mapper.stringifier import PREC_NONE
        from pymbolic.mapper.c_code import CCodeMapper

        elwise = self.elementwise_mod

        from hedge.tools import is_obj_array
        if is_obj_array(subst_expr):
            args = [elwise.VectorArg(result_dtype, "_result%d" % i)
                    for i in range(len(subst_expr))]
            exprs = subst_expr
        else:
            args = [elwise.VectorArg(result_dtype, "_result0")]
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

        def get_arg_spec(name, is_vector, dtype):
            if is_vector:
                return elwise.VectorArg(dtype, name)
            else:
                return elwise.ScalarArg(dtype, name)

        args.extend(
                get_arg_spec(var_name, *type_getter(var_expr)) 
                for var_expr, var_name in zip(
                    self.vector_exprs+self.scalar_exprs, 
                    vector_names+scalar_names))

        #print ",".join(args)
        #print "\n".join(code_lines)

        self.make_kernel(args, "\n".join(code_lines))

        from pymbolic.mapper.flop_counter import FlopCounter
        self.flop_count = FlopCounter()(subst_expr)
