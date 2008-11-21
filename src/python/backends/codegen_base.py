"""Just-in-time compiling backend."""

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




import pymbolic.mapper.stringifier




class FluxToCodeMapperBase(pymbolic.mapper.stringifier.SimplifyingSortingStringifyMapper):
    def map_power(self, expr, enclosing_prec):
        from pymbolic.mapper.stringifier import PREC_NONE
        from pymbolic.primitives import is_constant
        if is_constant(expr.exponent):
            if expr.exponent == 0:
                return "1"
            elif expr.exponent == 1:
                return self.rec(expr.base, enclosing_prec)
            elif expr.exponent == 2:
                return self.rec(expr.base*expr.base, enclosing_prec)
            else:
                return ("pow(%s, %s)" 
                        % (self.rec(expr.base, PREC_NONE), 
                        self.rec(expr.exponent, PREC_NONE)))
                return self.rec(expr.base*expr.base, enclosing_prec)
        else:
            return ("pow(%s, %s)" 
                    % (self.rec(expr.base, PREC_NONE), 
                    self.rec(expr.exponent, PREC_NONE)))

    def map_if_positive(self, expr, enclosing_prec):
        from pymbolic.mapper.stringifier import PREC_NONE
        return "(%s > 0 ? %s : %s)" % (
                self.rec(expr.criterion, PREC_NONE),
                self.rec(expr.then, PREC_NONE),
                self.rec(expr.else_, PREC_NONE),
                )


