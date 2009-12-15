"""C code generation for vector expressions."""

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
import codepy.elementwise
from hedge.backends.vector_expr import CompiledVectorExpressionBase




class CompiledVectorExpression(CompiledVectorExpressionBase):
    elementwise_mod = codepy.elementwise

    def __init__(self, vec_expr_info_list, result_dtype_getter,
            toolchain=None, wait_on_error=False):
        CompiledVectorExpressionBase.__init__(self,
                vec_expr_info_list, result_dtype_getter)

        self.toolchain = toolchain
        self.wait_on_error = wait_on_error

    def make_kernel_internal(self, args, instructions):
        return self.elementwise_mod.ElementwiseKernel(
                args, instructions, name="vector_expression",
                toolchain=self.toolchain,
                wait_on_error=self.wait_on_error)

    def __call__(self, evaluate_subexpr, stats_callback=None):
        vectors = [evaluate_subexpr(vec_expr) 
                for vec_expr in self.vector_deps]
        scalars = [evaluate_subexpr(scal_expr) 
                for scal_expr in self.scalar_deps]

        from pytools import single_valued
        shape = single_valued(vec.shape for vec in vectors)

        kernel_rec = self.get_kernel(
                tuple(v.dtype for v in vectors),
                tuple(s.dtype for s in scalars))

        results = [numpy.empty(shape, kernel_rec.result_dtype)
                for vei in self.result_vec_expr_info_list]

        size = results[0].size
        args = (results+vectors+scalars)

        if stats_callback is not None:
            timer = stats_callback(size, self)
            sub_timer = timer.start_sub_timer()
            kernel_rec.kernel(*args)
            sub_timer.stop().submit()
        else:
            kernel_rec.kernel(*args)

        return results




if __name__ == "__main__":
    test_dtype = numpy.float32

    import pycuda.autoinit
    from pymbolic import parse
    expr = parse("2*x+3*y+4*z")
    print expr
    cexpr = CompiledVectorExpression(expr, 
            lambda expr: (True, test_dtype),
            test_dtype)

    from pymbolic import var
    ctx = {
        var("x"): numpy.arange(5, dtype=test_dtype),
        var("y"): numpy.arange(5, dtype=test_dtype),
        var("z"): numpy.arange(5, dtype=test_dtype),
        }

    print cexpr(lambda expr: ctx[expr])
