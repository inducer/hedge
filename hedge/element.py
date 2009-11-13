"""Local function space representation. (deprecated)"""

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



import hedge.discretization.local
import warnings

warnings.warn("hedge.element is deprecated. Use hedge.discretization.local instead.")

Element = hedge.discretization.local.LocalDiscretization
IntervalElement = hedge.discretization.local.IntervalDiscretization
TriangularElement = hedge.discretization.local.TriangleDiscretization
TetrahedralElement = hedge.discretization.local.TetrahedronDiscretization
