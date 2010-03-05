# -*- coding: utf8 -*-

"""ODE solvers: timestepping support, such as Runge-Kutta, Adams-Bashforth, etc."""

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



from hedge.timestep.runge_kutta import RK4TimeStepper, LSRK4TimeStepper
from hedge.timestep.ab import AdamsBashforthTimeStepper
from hedge.timestep.ssprk3 import SSPRK3TimeStepper




def times_and_steps(max_dt_getter, taken_dt_getter=None,
        start_time=0, final_time=None, max_steps=None, 
        logmgr=None):
    """Generate tuples *(step, t, recommended_dt)* to control a timestep loop.
    The controlled simulation may decide to take a smaller timestep, and
    indicate so through the use of *taken_dt_getter*.

    :param max_dt_getter: *None* or a function of time obtaining the maximal
      admissible timestep. The timestep yielded as *recommended_dt* is
      less or equal to the value returned by this function.
    :param taken_dt_getter: if not *None*, this argumentless function is used to
      obtain the time step actually taken.
    :param logmgr: An instance of :class:`pytools.log.LogManager` (or None).
      This routine will then take care of telling the log manager about
      time step sizes.
    :param max_steps: Maximum number of steps taken. A "step" is one 
      execution of the loop body.
    """
    if final_time is None and max_steps is None:
        raise ValueError("at least one of final_time and max_steps "
                "must be specified")

    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be positive")

    if final_time is not None and start_time > final_time:
        raise ValueError("final_time is before start_time")

    t = start_time
    step = 0

    final_step = False
    while True:
        if final_step:
            break

        if max_steps is not None and step > max_steps:
            break

        next_dt = max_dt_getter(t)

        if final_time is not None and t + next_dt >= final_time:
            next_dt = final_time - t
            final_step = True

        if logmgr is not None:
            logmgr.tick_before()

        yield step, t, next_dt

        if taken_dt_getter is not None:
            taken_dt = taken_dt_getter()
        else:
            taken_dt = next_dt

        if logmgr is not None:
            from pytools.log import set_dt
            set_dt(logmgr, taken_dt)
            logmgr.tick_after()

        step += 1
        t += taken_dt
