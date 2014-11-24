# -*- coding: utf-8 -*-
"""Base facility for C generation from vector expressions."""

from __future__ import division

__copyright__ = "Copyright (C) 2009 Andreas Kloeckner"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
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

    def map_jacobian(self, expr):
        result = self.subst_func(expr)
        if result is not None:
            return result
        else:
            pymbolic.mapper.substitutor.SubstitutionMapper.handle_unsupported_expression(
                    self, expr)

    map_forward_metric_derivative = map_jacobian
    map_inverse_metric_derivative = map_jacobian

    def map_scalar_parameter(self, expr):
        return pymbolic.mapper.substitutor.SubstitutionMapper.map_variable(
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




class VectorExpressionInfo(Record):
    __slots__ = ["name", "expr", "do_not_return"]




def simple_result_dtype_getter(vector_dtype_map, scalar_dtype_map, const_dtypes):
    from pytools import common_dtype, match_precision

    result = common_dtype(vector_dtype_map.values())

    scalar_dtypes = scalar_dtype_map.values() + const_dtypes
    if scalar_dtypes:
        prec_matched_scalar_dtype = match_precision(
                common_dtype(scalar_dtypes),
                dtype_to_match=result)
        result = common_dtype([result, prec_matched_scalar_dtype])

    return result




class CompiledVectorExpressionBase(object):
    def __init__(self, vec_expr_info_list, result_dtype_getter):
        self.result_dtype_getter = result_dtype_getter

        from hedge.optemplate.primitives import ScalarParameter
        from hedge.optemplate.mappers import (
                DependencyMapper, GeometricFactorCollector)

        from operator import or_
        from pymbolic import var

        dep_mapper = DependencyMapper(
                include_subscripts=True,
                include_lookups=True,
                include_calls="descend_args")
        gfc = GeometricFactorCollector()

        # gather all dependencies

        deps = (reduce(or_, (dep_mapper(vei.expr) for vei in vec_expr_info_list))
                | reduce(or_, (gfc(vei.expr) for vei in vec_expr_info_list)))

        # We're compiling a batch of vector expressions, some of which may
        # depend on results generated in this same batch. These dependencies
        # are also captured above, but they're not genuine external dependencies.
        # Hence we remove them here:

        deps -= set(var(vei.name) for vei in vec_expr_info_list)

        from pytools import partition

        def is_vector_pred(dep):
            return not isinstance(dep, ScalarParameter)

        vdeps, sdeps  = partition(is_vector_pred, deps)

        vdeps = [(str(vdep), vdep) for vdep in vdeps]
        sdeps = [(str(sdep), sdep) for sdep in sdeps]
        vdeps.sort()
        sdeps.sort()
        self.vector_deps = [vdep for key, vdep in vdeps]
        self.scalar_deps = [sdep for key, sdep in sdeps]

        self.vector_dep_names = ["hedge_v%d" % i for i in range(len(self.vector_deps))]
        self.scalar_dep_names = ["hedge_s%d" % i for i in range(len(self.scalar_deps))]

        self.constant_dtypes = [
                numpy.array(const).dtype
                for vei in vec_expr_info_list
                for const in ConstantGatherMapper()(vei.expr)]

        var_i = var("i")
        subst_map = dict(
                list(zip(self.vector_deps, [var(vecname).index(var_i)
                    for vecname in self.vector_dep_names]))
                +list(zip(self.scalar_deps,
                    [var(scaname) for scaname in self.scalar_dep_names]))
                +[(var(vei.name), var(vei.name).index(var_i))
                    for vei in vec_expr_info_list
                    if not vei.do_not_return])

        def subst_func(expr):
            try:
                return subst_map[expr]
            except KeyError:
                return None

        self.vec_expr_info_list = [
                vei.copy(expr=DefaultingSubstitutionMapper(subst_func)(vei.expr))
                for vei in vec_expr_info_list]

        self.result_vec_expr_info_list = [
                vei for vei in vec_expr_info_list if not vei.do_not_return]

    @memoize_method
    def result_names(self):
        return [rvei.name for rvei in self.result_vec_expr_info_list]

    @memoize_method
    def get_kernel(self, vector_dtypes, scalar_dtypes):
        from pymbolic.mapper.stringifier import PREC_NONE
        from pymbolic.mapper.c_code import CCodeMapper

        elwise = self.elementwise_mod

        result_dtype = self.result_dtype_getter(
                dict(zip(self.vector_deps, vector_dtypes)),
                dict(zip(self.scalar_deps, scalar_dtypes)),
                self.constant_dtypes)

        args = [elwise.VectorArg(result_dtype, vei.name)
                for vei in self.vec_expr_info_list
                if not vei.do_not_return]

        def real_const_mapper(num):
            # Make sure we do not generate integers or doubles by accident.
            # Oh, C and your broken division semantics.

            r = repr(num)
            if "." not in r or result_dtype == numpy.float32:
                from pytools import to_uncomplex_dtype
                from cgen import dtype_to_ctype
                return "%s(%s)" % (dtype_to_ctype(
                        to_uncomplex_dtype(result_dtype)), r)
            else:
                return r

        code_mapper = CCodeMapper(constant_mapper=real_const_mapper)

        code_lines = []
        for vei in self.vec_expr_info_list:
            expr_code = code_mapper(vei.expr, PREC_NONE)
            if vei.do_not_return:
                from cgen import dtype_to_ctype
                code_lines.append(
                        "%s %s = %s;" % (
                            dtype_to_ctype(result_dtype), vei.name, expr_code))
            else:
                code_lines.append(
                        "%s[i] = %s;" % (vei.name, expr_code))

        # common subexpressions have been taken care of by the compiler
        assert not code_mapper.cse_names

        args.extend(
                elwise.VectorArg(dtype, name)
                for dtype, name in zip(vector_dtypes, self.vector_dep_names))
        args.extend(
                elwise.ScalarArg(dtype, name)
                for dtype, name in zip(scalar_dtypes, self.scalar_dep_names))

        return KernelRecord(
                kernel=self.make_kernel_internal(args, "\n".join(code_lines)),
                result_dtype=result_dtype)
