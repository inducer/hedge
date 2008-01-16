"""Logging utilities."""

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



from pytools.log import LogQuantity




class VariableL2Norm(LogQuantity):
    """Log the L2 norm of a variable in a scope."""

    def __init__(self, discr, scope, varname, name=None, indices=None,
            unit="1", description=None):
        """Construct the L2 norm logger.

        @arg discr: a L{Discretization} to which the variable belongs.
        @arg scope: the scope in which the variable may be looked up.
          You may obtain the current local scope by calling 
          C{locals()}.
        @arg varname: the name under which the variable is looked up 
          in the C{scope}.
        @arg name: the name reported to the C{LogManager}.
        @arg indices: A C{slice} or a single index indicating the subset
          of C{varname} of which to take the L2 norm.
        @arg unit: the unit of measure for the L2 norm.
        @arg description: A description fed to the C{LogManager}.
        """
        self.scope = scope
        self.varname = varname
        if name is None:
            name = "l2_%s" % varname
        LogQuantity.__init__(self, name, unit, description)

        self.indices = None

        self.mass_op = discr.mass_operator

    @property
    def default_aggregator(self): 
        from pytools import norm_2
        return norm_2

    def __call__(self):
        var = self.scope[self.varname]
        if self.indices is not None:
            var = var[self.indices]

        from math import sqrt
        if isinstance(var, list):
            return sqrt(dot(var, self.mass_op*var))
        else:
            return sqrt(var*(self.mass_op*var))
