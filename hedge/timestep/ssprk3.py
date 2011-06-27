"""Strongly Stability Preserving third-order RK ODE timestepper."""

from __future__ import division

__copyright__ = "Copyright (C) 2009 Scott Field"

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
from hedge.timestep.runge_kutta import SSP3TimeStepper as SSPRK3TimeStepperBase



class SSPRK3TimeStepper(SSPRK3TimeStepperBase):
    def __init__(self, *args, **kwargs):
        from warnings import warn
        warn("hedge.timestep.ssprk3 is deprecated. Use the generic SSP support "
                "in hedge.timestep.runge_kutta instead.")

        SSPRK3TimeStepperBase.__init__(self, *args, **kwargs)
