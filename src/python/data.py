"""Representations for given data, such as initial and boundary 
conditions and source terms."""

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




import hedge.mesh




# helpers ---------------------------------------------------------------------
class _ConstantFunctionContainer:
    def __init__(self, value):
        self.value = value

    @property
    def shape(self):
        return self.value.shape

    def __call__(self, x):
        return self.value





# interpolation wrappers ------------------------------------------------------
class IGivenFunction(object):
    """Abstract interface for obtaining interpolants of I{time-independent} 
    functions.
    """

    def volume_interpolant(self, discr):
        """Return the volume interpolant of this function with respect to
        the L{Discretization} C{discr}.
        """
        raise NotImplementedError

    def boundary_interpolant(self, discr, tag=hedge.mesh.TAG_ALL):
        """Return the boundary interpolant of this function with respect to
        the L{Discretization} discr at the boundary tagged with C{tag}.
        """
        raise NotImplementedError




class ITimeDependentGivenFunction(object):
    """Abstract interface for obtaining interpolants of I{time-dependent} 
    functions.
    """

    def volume_interpolant(self, t, discr):
        """Return the volume interpolant of this function with respect to
        the L{Discretization} discr at time {t}.
        """
        raise NotImplementedError

    def boundary_interpolant(self, t, discr, tag=hedge.mesh.TAG_ALL):
        """Return the boundary interpolant of this function with respect to
        the L{Discretization} discr at time C{t} at the boundary tagged with
        C{tag}.
        """
        raise NotImplementedError




class GivenFunction(IGivenFunction):
    """Adapter for a function M{f(x)} into an L{IGivenFunction}.
    """
    def __init__(self, f):
        """Initialize the caches and store the function C{f}.

        @param f: a function mapping space to a scalar value.
          If f.target_dimensions exists and equals M{n}, then f maps into an
          M{n}-dimensional vector space instead.
        """
        from weakref import WeakKeyDictionary

        self.f = f

        self.volume_cache = WeakKeyDictionary()
        self.boundary_cache = WeakKeyDictionary()

    def volume_interpolant(self, discr):
        try:
            return self.volume_cache[discr]
        except KeyError:
            result = discr.interpolate_volume_function(self.f)
            self.volume_cache[discr] = result
            return result

    def boundary_interpolant(self, discr, tag=hedge.mesh.TAG_ALL):
        try:
            return self.boundary_cache[discr][tag]
        except KeyError:
            tag_cache = self.boundary_cache.setdefault(discr, {})
            result = discr.interpolate_boundary_function(self.f, tag)
            tag_cache[tag] = result
            return result




class ConstantGivenFunction(GivenFunction):
    """A L{GivenFunction} that has a constant value on all space.
    """
    def __init__(self, value=0):
        self.value = value

        GivenFunction.__init__(self, _ConstantFunctionContainer(value))




class GivenVolumeInterpolant(IGivenFunction):
    """A constant-valued L{GivenFunction}.
    """
    def __init__(self, discr, interpolant):
        self.discr = discr
        self.interpolant = interpolant

    def volume_interpolant(self, discr):
        if discr != self.discr:
            raise ValueError, "cross-interpolation between discretizations not supported"
        return self.interpolant

    def boundary_interpolant(self, discr, tag=hedge.mesh.TAG_ALL):
        if discr != self.discr:
            raise ValueError, "cross-interpolation between discretizations not supported"
        return discr.boundarize_volume_field(self.interpolant, tag)





class TimeConstantGivenFunction(ITimeDependentGivenFunction):
    """Adapts a L{GivenFunction} to have a (formal) time-dependency, being constant
    over all time.
    """
    def __init__(self, gf):
        self.gf = gf

    def volume_interpolant(self, t, discr):
        return self.gf.volume_interpolant(discr)

    def boundary_interpolant(self, t, discr, tag=hedge.mesh.TAG_ALL):
        return self.gf.boundary_interpolant(discr, tag)





def make_tdep_constant(x):
    return TimeConstantGivenFunction(ConstantGivenFunction(x))




class TimeHarmonicGivenFunction(ITimeDependentGivenFunction):
    """Adapts an L{IGivenFunction} to have a harmonic time-dependency.
    """
    def __init__(self, gf, omega, phase=0):
        self.gf = gf
        self.omega = omega
        self.phase = phase

    def volume_interpolant(self, t, discr):
        from math import sin
        return sin(omega*t+phase)*self.gf.volume_interpolant(discr)

    def boundary_interpolant(self, t, discr, tag=hedge.mesh.TAG_ALL):
        from math import sin
        return sin(omega*t+phase)*self.gf.boundary_interpolant(discr, tag)





class TimeIntervalGivenFunction(ITimeDependentGivenFunction):
    """Adapts an L{IGivenFunction} to depend on time by "turning it on"
    for the time interval [on_time, off_time), and having it be zero
    the rest of the time.
    """

    def __init__(self, gf, on_time=0, off_time=1):
        self.gf = gf
        self.on_time = on_time
        self.off_time = off_time
        assert on_time <= off_time



    def volume_interpolant(self, t, discr):
        if self.on_time <= t < self.off_time:
            return self.gf.volume_interpolant(discr)
        else:
            # FIXME: not optimal
            # difficult part here is to match shape
            return 0*self.gf.volume_interpolant(discr)

    def boundary_interpolant(self, t, discr, tag=hedge.mesh.TAG_ALL):
        if self.on_time <= t < self.off_time:
            return self.gf.boundary_interpolant(discr, tag)
        else:
            # FIXME: not optimal
            # difficult part here is to match shape
            return 0*self.gf.boundary_interpolant(discr, tag)





class TimeDependentGivenFunction(ITimeDependentGivenFunction):
    """Adapts a function M{f(x,t)} into the L{GivenFunction} framework.
    """
    def __init__(self, f):
        self.f = f

    def volume_interpolant(self, t, discr):
        return discr.interpolate_volume_function(lambda x: self.f(x,t))

    def boundary_interpolant(self, t, discr, tag=hedge.mesh.TAG_ALL):
        return discr.interpolate_boundary_function(lambda x: self.f(x, t), tag)
