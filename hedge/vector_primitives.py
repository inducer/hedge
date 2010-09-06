"""Backend support for optimized linear combinations,  for timestepping."""

from __future__ import division

__copyright__ = "Copyright (C) 2007 Andreas Kloeckner"

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




# {{{ linear combinations -----------------------------------------------------
class ObjectArrayLinearCombinationWrapper(object):
    def __init__(self, scalar_kernel):
        self.scalar_kernel = scalar_kernel

    def __call__(self, *args):
        from pytools import indices_in_shape, single_valued

        oa_shape = single_valued(ary.shape for fac, ary in args)
        result = numpy.zeros(oa_shape, dtype=object)

        for i in indices_in_shape(oa_shape):
            args_i = [(fac, ary[i]) for fac, ary in args]
            result[i] = self.scalar_kernel(*args_i)

        return result




class UnoptimizedLinearCombiner(object):
    def __init__(self, result_dtype, scalar_dtype):
        self.result_type = result_dtype.type

    def __call__(self, *args):
        return sum(self.result_type(fac)*vec for fac, vec in args)




class NumpyLinearCombiner(object):
    def __init__(self, result_dtype, scalar_dtype, sample_vec, arg_count):
        self.result_dtype = result_dtype
        self.shape = sample_vec.shape

        from codepy.elementwise import \
                make_linear_comb_kernel_with_result_dtype
        self.kernel = make_linear_comb_kernel_with_result_dtype(
                result_dtype,
                (scalar_dtype,)*arg_count,
                (sample_vec.dtype,)*arg_count)

    def __call__(self, *args):
        result = numpy.empty(self.shape, self.result_dtype)

        from pytools import flatten
        self.kernel(result, *tuple(flatten(args)))

        return result




class CUDALinearCombiner:
    def __init__(self, result_dtype, scalar_dtype, sample_vec, arg_count,
            pool=None):
        from pycuda.elementwise import get_linear_combination_kernel
        self.vector_dtype = sample_vec.dtype
        self.result_dtype = result_dtype
        self.shape = sample_vec.shape
        self.block = sample_vec._block
        self.grid = sample_vec._grid
        self.mem_size = sample_vec.mem_size

        self.kernel, _ = get_linear_combination_kernel(
                arg_count*((False, scalar_dtype, self.vector_dtype),),
                result_dtype)

        if pool:
            self.allocator = pool.allocate
        else:
            self.allocator = None

    def __call__(self, *args):
        import pycuda.gpuarray as gpuarray
        result = gpuarray.empty(self.shape, self.result_dtype,
                allocator=self.allocator)

        knl_args = []
        for fac, vec in args:
            if vec.dtype != self.vector_dtype:
                raise TypeError("unexpected vector type in CUDA linear combination")

            knl_args.append(fac)
            knl_args.append(vec.gpudata)

        knl_args.append(result.gpudata)
        knl_args.append(self.mem_size)

        self.kernel.set_block_shape(*self.block)
        self.kernel.prepared_async_call(self.grid, None, *knl_args)

        return result

# }}}

# {{{ inner product -----------------------------------------------------------
class ObjectArrayInnerProductWrapper(object):
    def __init__(self, scalar_kernel):
        self.scalar_kernel = scalar_kernel

    def __call__(self, a, b):
        from pytools import indices_in_shape

        assert a.shape == b.shape

        result = 0
        for i in indices_in_shape(a.shape):
            result += self.scalar_kernel(a[i], b[i])

        return result

# }}}

# {{{ maximum norm ------------------------------------------------------------

class ObjectArrayMaximumNormWrapper(object):
    def __init__(self, scalar_kernel):
        self.scalar_kernel = scalar_kernel

    def __call__(self, a):
        from pytools import indices_in_shape

        # assumes nonempty, which is reasonable
        return max(
                abs(self.scalar_kernel(a[i]))
                for i in indices_in_shape(a.shape))

# }}}



# {{{ vector primitive factory

class VectorPrimitiveFactory(object):
    def make_special_linear_combiner(self, result_dtype, scalar_dtype, sample_vec, arg_count):
        return None

    def make_linear_combiner(self, result_dtype, scalar_dtype, sample_vec, arg_count):
        """
        :param result_dtype: dtype of the desired result.
        :param scalar_dtype: dtype of the scalars.
        :param sample_vec: must match states and right hand sides in shape, object
          array composition, and dtypes.
        :returns: a function that accepts `arg_count` arguments
          *((factor0, vec0), (factor1, vec1), ...)* and returns
          `factor0*vec0 + factor1*vec1`.
        """
        from hedge.tools import is_obj_array
        sample_is_obj_array = is_obj_array(sample_vec)

        if sample_is_obj_array:
            sample_vec = sample_vec[0]

        if isinstance(sample_vec, numpy.ndarray) and sample_vec.dtype != object:
            kernel = NumpyLinearCombiner(result_dtype, scalar_dtype, sample_vec,
                    arg_count)
        else:
            kernel = self.make_special_linear_combiner(
                    result_dtype, scalar_dtype, sample_vec, arg_count)

            if kernel is None:
                from warnings import warn
                warn("using unoptimized linear combination routine")
                kernel = UnoptimizedLinearCombiner(result_dtype, scalar_dtype)

        if sample_is_obj_array:
            kernel = ObjectArrayLinearCombinationWrapper(kernel)

        return kernel

    def make_special_inner_product(self, sample_vec):
        return None

    def make_inner_product(self, sample_vec):
        from hedge.tools import is_obj_array
        sample_is_obj_array = is_obj_array(sample_vec)

        if sample_is_obj_array:
            sample_vec = sample_vec[0]

        if isinstance(sample_vec, numpy.ndarray) and sample_vec.dtype != object:
            kernel = numpy.dot
        else:
            kernel = self.make_special_inner_product(sample_vec)

            if kernel is None:
                raise RuntimeError("could not find an inner product routine for "
                        "the given sample vector")

        if sample_is_obj_array:
            kernel = ObjectArrayInnerProductWrapper(kernel)

        return kernel

    def make_maximum_norm(self, sample_vec):
        from hedge.tools import is_obj_array
        sample_is_obj_array = is_obj_array(sample_vec)

        if sample_is_obj_array:
            sample_vec = sample_vec[0]

        if isinstance(sample_vec, numpy.ndarray) and sample_vec.dtype != object:
            def kernel(vec):
                return la.norm(vec, numpy.inf)
            kernel = numpy.max
        else:
            kernel = self.make_special_maximum_norm(sample_vec)

            if kernel is None:
                raise RuntimeError("could not find a maximum norm routine for "
                        "the given sample vector")

        if sample_is_obj_array:
            kernel = ObjectArrayMaximumNormWrapper(kernel)

        return kernel





class CUDAVectorPrimitiveFactory(VectorPrimitiveFactory):
    def __init__(self, discr=None):
        self.discr = discr

    def make_special_linear_combiner(self, *args, **kwargs):
        my_kwargs = kwargs.copy()
        kwargs["pool"] = self.discr.pool
        return CUDALinearCombiner(*args, **kwargs)

    def make_special_inner_product(self, sample_vec):
        from pycuda.gpuarray import GPUArray

        if isinstance(sample_vec, GPUArray):
            if self.discr is None:
                def kernel(a, b):
                    from pycuda.gpuarray import dot
                    return dot(a, b).get()

                return kernel
            else:
                return self.discr.nodewise_dot_product

    def make_special_maximum_norm(self, sample_vec):
        from pycuda.gpuarray import GPUArray

        if isinstance(sample_vec, GPUArray):
            if self.discr is None:
                def my_max(vec):
                    from pycuda.gpuarray import max as my_max
                    return my_max(vec).get()
            else:
                my_max = self.discr.nodewise_max

            def kernel(a):
                from pycuda.cumath import fabs
                return my_max(fabs(a))

            return kernel

# }}}




# vim: foldmethod=marker
