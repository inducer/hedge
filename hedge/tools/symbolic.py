"""Symbolic math helpers."""

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
from pymbolic.primitives import Variable




def make_common_subexpression(field, prefix=None):
    from pytools.obj_array import log_shape
    from hedge.tools import is_zero
    from pymbolic.primitives import CommonSubexpression

    ls = log_shape(field)
    if ls != ():
        from pytools import indices_in_shape
        result = numpy.zeros(ls, dtype=object)

        for i in indices_in_shape(ls):
            if prefix is not None:
                component_prefix = prefix+"_".join(str(i_i) for i_i in i)
            else:
                component_prefix = None

            if is_zero(field[i]):
                result[i] = 0
            else:
                result[i] = CommonSubexpression(field[i], component_prefix)

        return result
    else:
        if is_zero(field):
            return 0
        else:
            return CommonSubexpression(field, prefix)





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
