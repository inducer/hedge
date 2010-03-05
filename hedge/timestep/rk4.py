from warnings import warn
warn("hedge.timestep.rk4 is now spelled hedge.timestep.runge_kutta",
        DeprecationWarning)

from hedge.timestep.runge_kutta import *
