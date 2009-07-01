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
from pytools import memoize_method, Record




class DefaultingSubstitutionMapper(
        pymbolic.mapper.substitutor.SubstitutionMapper,
        hedge.optemplate.IdentityMapperMixin):
    def map_operator_binding(self, expr):
        result = self.subst_func(expr)
        if result is None:
            raise ValueError("operator binding may not survive "
                    "vector expression subsitution")

        return result

    def handle_unsupported_expression(self, expr):
        result = self.subst_func(expr)
        if result is not None:
            return result
        else:
            pymbolic.mapper.substitutor.SubstitutionMapper.handle_unsupported_expression(
                    self, expr)





class ConstantGatherMapper(
        hedge.optemplate.CombineMapper,
        hedge.optemplate.CollectorMixin,
        hedge.optemplate.OperatorReducerMixin):
    def map_algebraic_leaf(self, expr):
        return set()

    def map_constant(self, expr):
        return set([expr])

    def map_operator(self, expr):
        return set()




class KernelRecord(Record):
    pass




class CompiledVectorExpressionBase(object):
    def __init__(self, vec_exprs, vec_names, is_vector_func, result_dtype_getter):
        assert len(vec_exprs) == len(vec_names)

        self.vec_exprs = vec_exprs
        self.vec_names = vec_names

        self.is_vector_func = is_vector_func
        self.result_dtype_getter = result_dtype_getter

        from hedge.optemplate import DependencyMapper
        from operator import or_
        from pymbolic import var

        dep_mapper = DependencyMapper(
                include_subscripts=True,
                include_lookups=True,
                include_calls="descend_args")
        deps = reduce(or_, (dep_mapper(ve) for ve in vec_exprs))

        deps -= set(var(vn) for vn in vec_names)

        self.vector_deps = [dep for dep in deps if is_vector_func(dep)]
        self.scalar_deps = [dep for dep in deps if not is_vector_func(dep)]
        self.vector_dep_names = ["v%d" % i for i in range(len(self.vector_deps))]
        self.scalar_dep_names = ["s%d" % i for i in range(len(self.scalar_deps))]

        self.constant_dtypes = [
                numpy.array(const).dtype
                for ve in vec_exprs
                for const in ConstantGatherMapper()(ve)]

        var_i = var("i")
        subst_map = dict(
                list(zip(self.vector_deps, [var(vecname)[var_i]
                    for vecname in self.vector_dep_names]))
                +list(zip(self.scalar_deps,
                    [var(scaname) for scaname in self.scalar_dep_names]))
                +[(var(vn), var(vn)[var_i]) for vn in vec_names]
                )

        def subst_func(expr):
            try:
                return subst_map[expr]
            except KeyError:
                return None

        self.exprs = [
                DefaultingSubstitutionMapper(subst_func)(ve)
                for ve in vec_exprs]

    @memoize_method
    def get_kernel(self, vector_dtypes, scalar_dtypes):
        from pymbolic.mapper.stringifier import PREC_NONE
        from pymbolic.mapper.c_code import CCodeMapper

        elwise = self.elementwise_mod

        result_dtype = self.result_dtype_getter(
                dict(zip(self.vector_deps, vector_dtypes)),
                dict(zip(self.scalar_deps, scalar_dtypes)),
                self.constant_dtypes)

        from hedge.tools import is_obj_array
        args = [elwise.VectorArg(result_dtype, vn)
                for vn in self.vec_names]

        code_mapper = CCodeMapper(constant_mapper=lambda c: "double(%r)" % c)
        expr_codes = [code_mapper(e, PREC_NONE) for e in self.exprs]

        # common subexpressions have been taken care of by the compiler
        assert not code_mapper.cses

        code_lines = [
            "%s[i] = %s;" % (vn, ec)
            for vn, ec in zip(self.vec_names, expr_codes)]

        args.extend(
                elwise.VectorArg(dtype, name)
                for dtype, name in zip(vector_dtypes, self.vector_dep_names))
        args.extend(
                elwise.ScalarArg(dtype, name)
                for dtype, name in zip(scalar_dtypes, self.scalar_dep_names))

        return KernelRecord(
                kernel=self.make_kernel_internal(args, "\n".join(code_lines)),
                result_dtype=result_dtype)
