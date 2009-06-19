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

    def __init__(self, vec_expr, type_getter, result_dtype, 
            stream=None, allocator=drv.mem_alloc):
        CompiledVectorExpressionBase.__init__(self, 
                vec_expr, type_getter, result_dtype)

        self.stream = stream
        self.allocator = allocator

    def make_kernel_internal(self, args, instructions):
        from pycuda.elementwise import get_elwise_kernel
        self.kernel = get_elwise_kernel(args, instructions, name="vector_expression")

    def __call__(self, evaluate_subexpr, stats_callback=None):
        vectors = [evaluate_subexpr(vec_expr) for vec_expr in self.vector_exprs]
        scalars = [evaluate_subexpr(scal_expr) for scal_expr in self.scalar_exprs]

        from pytools import single_valued
        shape = single_valued(vec.shape for vec in vectors)
        single_valued(vec.dtype for vec in vectors)

        assert self.result_count > 0
        from hedge.tools import make_obj_array
        results = [gpuarray.empty(shape, self.result_dtype, self.allocator)
                for i in range(self.result_count)]

        size = results[0].size
        self.kernel.set_block_shape(*results[0]._block)
        args = ([r.gpudata for r in results]
                +[v.gpudata for v in vectors]
                +scalars
                +[size])

        if stats_callback is not None:
            stats_callback(size,  self,
                    self.kernel.prepared_timed_call(vectors[0]._grid, *args))
        else:
            self.kernel.prepared_async_call(vectors[0]._grid, self.stream, *args)

        from hedge.tools import is_obj_array
        if is_obj_array(self.subst_expr):
            return make_obj_array(results)
        else:
            return results[0]




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

