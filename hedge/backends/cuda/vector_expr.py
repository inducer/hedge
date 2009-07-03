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
import pycuda.elementwise
from hedge.backends.vector_expr import CompiledVectorExpressionBase




class CompiledVectorExpression(CompiledVectorExpressionBase):
    elementwise_mod = pycuda.elementwise

    def __init__(self, vec_expr_info_list,
            is_vector_pred, result_dtype_getter, 
            stream=None, allocator=drv.mem_alloc):
        CompiledVectorExpressionBase.__init__(self, 
                vec_expr_info_list, is_vector_pred, result_dtype_getter)

        self.stream = stream
        self.allocator = allocator

    def make_kernel_internal(self, args, instructions):
        from pycuda.elementwise import get_elwise_kernel
        return get_elwise_kernel(args, instructions, name="vector_expression")

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

        from hedge.tools import make_obj_array
        results = [gpuarray.empty(
            shape, kernel_rec.result_dtype, self.allocator)
            for expr in self.result_vec_expr_info_list]

        size = results[0].size
        kernel_rec.kernel.set_block_shape(*results[0]._block)
        args = ([r.gpudata for r in results]
                +[v.gpudata for v in vectors]
                +scalars
                +[size])

        if stats_callback is not None:
            stats_callback(size,  self,
                    kernel_rec.kernel.prepared_timed_call(vectors[0]._grid, *args))
        else:
            kernel_rec.kernel.prepared_async_call(vectors[0]._grid, self.stream, *args)

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
        var("x"): gpuarray.arange(5, dtype=test_dtype),
        var("y"): gpuarray.arange(5, dtype=test_dtype),
        var("z"): gpuarray.arange(5, dtype=test_dtype),
        }

    print cexpr(lambda expr: ctx[expr])

