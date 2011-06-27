"""CUDA code generation for vector expressions."""

from __future__ import division

__copyright__ = "Copyright (C) 2008 Andreas Kloeckner"

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
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
import pycuda.elementwise
from hedge.backends.vector_expr import CompiledVectorExpressionBase




class CompiledVectorExpression(CompiledVectorExpressionBase):
    elementwise_mod = pycuda.elementwise

    def __init__(self, vec_expr_info_list, result_dtype_getter, 
            stream=None, allocator=drv.mem_alloc):
        CompiledVectorExpressionBase.__init__(self, 
                vec_expr_info_list, result_dtype_getter)

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
