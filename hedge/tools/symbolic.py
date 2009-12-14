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




def make_common_subexpression(field, prefix=None):
    from hedge.tools import log_shape

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




