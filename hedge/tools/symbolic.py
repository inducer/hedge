"""Symbolic math helpers."""

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
from pymbolic.primitives import Variable

from decorator import decorator




@decorator
def memoize_method_with_obj_array_args(method, instance, *args):
    """This decorator manages to memoize functions that
    take object arrays (which are mutable, but are assumed
    to never change) as arguments.
    """
    dicname = "_memoize_dic_"+method.__name__

    new_args = []
    for arg in args:
        if isinstance(arg, numpy.ndarray) and arg.dtype == object:
            new_args.append(tuple(arg))
        else:
            new_args.append(arg)
    new_args = tuple(new_args)

    try:
        return getattr(instance, dicname)[new_args]
    except AttributeError:
        result = method(instance, *args)
        setattr(instance, dicname, {new_args: result})
        return result
    except KeyError:
        result = method(instance, *args)
        getattr(instance,dicname)[new_args] = result
        return result




from pymbolic.primitives import make_common_subexpression




class CFunction(Variable):
    """A symbol representing a C-level function, to be used as the function
    argument of :class:`pymbolic.primitives.Call`.
    """
    def stringifier(self):
        from hedge.optemplate import StringifyMapper
        return StringifyMapper

    def get_mapper_method(self, mapper):
        return mapper.map_c_function




def flat_end_sin(x):
    from hedge.optemplate.primitives import CFunction
    from pymbolic.primitives import IfPositive
    from math import pi
    return IfPositive(-pi/2-x,
            -1, IfPositive(x-pi/2, 1, CFunction("sin")(x)))





def smooth_ifpos(crit, right, left, width):
    from math import pi
    return 0.5*((left+right)
            +(right-left)*flat_end_sin(
                pi/2/width * crit))
