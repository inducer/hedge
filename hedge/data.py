"""Representations for given data, such as initial and boundary
conditions and source terms."""

__copyright__ = "Copyright (C) 2007 Andreas Kloeckner"

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
from pytools import memoize_method

from warnings import warn
warn("hedge.data is deprecated and should no longer be used. Instead, "
        "use bare hedge expressions that get inserted into operators",
        DeprecationWarning)


# {{{ helpers

class _ConstantFunctionContainer:
    def __init__(self, value):
        self.value = value

    @property
    def shape(self):
        return self.value.shape

    def __call__(self, x, el):
        return self.value

# }}}


# {{{ abstract interfaces

class IGivenFunction(object):
    """Abstract interface for obtaining interpolants of I{time-independent}
    functions.
    """

    def volume_interpolant(self, discr):
        """Return the volume interpolant of this function with respect to
        the :class:`hedge.discretization.Discretization` *discr*.
        """
        raise NotImplementedError

    def boundary_interpolant(self, discr, tag):
        """Return the boundary interpolant of this function with respect to
        the :class:`hedge.discretization.Discretization` *discr* at the
        boundary tagged with *tag*.
        """
        raise NotImplementedError




class ITimeDependentGivenFunction(object):
    """Abstract interface for obtaining interpolants of time-dependent
    functions.
    """

    def volume_interpolant(self, t, discr):
        """Return the volume interpolant of this function with respect to
        the :class:`hedge.discretization.Discretization` discr at time {t}.
        """
        raise NotImplementedError

    def boundary_interpolant(self, t, discr, tag):
        """Return the boundary interpolant of this function with respect to
        the :class:`hedge.discretization.Discretization` *discr* at time *t*
        at the boundary tagged with *tag*.
        """
        raise NotImplementedError




class IFieldDependentGivenFunction(object):
    """Abstract interface for obtaining interpolants of functions
    that depend on both time and system state.
    """

    def volume_interpolant(self, t, fields, discr):
        """Return the volume interpolant of this function with respect to
        the :class:`hedge.discretization.Discretization` discr at time {t}.
        """
        raise NotImplementedError

    def boundary_interpolant(self, t, fields, discr, tag):
        """Return the boundary interpolant of this function with respect to
        the :class:`hedge.discretization.Discretization` *discr* at time *t*
        at the boundary tagged with *tag*.
        """
        raise NotImplementedError

# }}}

# {{{ time-independent data ---------------------------------------------------
class GivenFunction(IGivenFunction):
    """Adapter for a function :math:`f(x)` into an :class:`IGivenFunction`.
    """
    def __init__(self, f):
        """Initialize the caches and store the function :math:`f`.

        :param f: a valid argument to 
          :meth:`hedge.discretization.Discretization.interpolate_volume_function`.
        """
        from weakref import WeakKeyDictionary

        self.f = f

        self.volume_cache = WeakKeyDictionary()
        self.boundary_cache = WeakKeyDictionary()

    def volume_interpolant(self, discr):
        try:
            return self.volume_cache[discr]
        except KeyError:
            result = discr.interpolate_volume_function(self.f, 
                    dtype=discr.default_scalar_type)
            self.volume_cache[discr] = result
            return result

    def boundary_interpolant(self, discr, tag):
        try:
            return self.boundary_cache[discr][tag]
        except KeyError:
            tag_cache = self.boundary_cache.setdefault(discr, {})
            result = discr.interpolate_boundary_function(self.f, tag,
                    dtype=discr.default_scalar_type)
            tag_cache[tag] = result
            return result




class ConstantGivenFunction(GivenFunction):
    """A :class:`GivenFunction` that has a constant value on all space.
    """
    def __init__(self, value=0):
        self.value = value

        GivenFunction.__init__(self, _ConstantFunctionContainer(value))




class GivenVolumeInterpolant(IGivenFunction):
    """A constant-valued :class:`GivenFunction`.
    """
    def __init__(self, discr, interpolant):
        self.discr = discr
        self.interpolant = interpolant

    def volume_interpolant(self, discr):
        if discr != self.discr:
            raise ValueError("cross-interpolation between discretizations "
                    "not supported")
        return self.interpolant

    def boundary_interpolant(self, discr, tag):
        if discr != self.discr:
            raise ValueError("cross-interpolation between discretizations "
                    "not supported")
        return discr.boundarize_volume_field(self.interpolant, tag)

# }}}

# {{{ time-depedent data ------------------------------------------------------

class TimeConstantGivenFunction(ITimeDependentGivenFunction):
    """Adapts a :class:`GivenFunction` to have a (formal) time-dependency,
    being constant over all time.
    """
    def __init__(self, gf):
        self.gf = gf

    def volume_interpolant(self, t, discr):
        return self.gf.volume_interpolant(discr)

    def boundary_interpolant(self, t, discr, tag):
        return self.gf.boundary_interpolant(discr, tag)





def make_tdep_constant(x):
    return TimeConstantGivenFunction(ConstantGivenFunction(x))




def make_tdep_given(x):
    return TimeConstantGivenFunction(GivenFunction(x))




class TimeHarmonicGivenFunction(ITimeDependentGivenFunction):
    """Modulates an :class:`ITimeDependentGivenFunction` by a sine
    in time.
    """
    def __init__(self, gf, omega, phase=0):
        self.gf = gf
        self.omega = omega
        self.phase = phase

    def volume_interpolant(self, t, discr):
        from math import sin
        return sin(self.omega * t + self.phase) \
                * self.gf.volume_interpolant(t, discr)

    def boundary_interpolant(self, t, discr, tag):
        from math import sin
        return sin(self.omega * t + self.phase)\
                * self.gf.boundary_interpolant(t, discr, tag)





class TimeIntervalGivenFunction(ITimeDependentGivenFunction):
    """Adapts an :class:`ITimeDependentGivenFunction` to depend on time by 
    "turning it on" for the time interval :math:`[\\text{on\\_time}, \\text{off\\_time})`, 
    and having it be zero the rest of the time.
    """

    def __init__(self, gf, on_time=0, off_time=1):
        self.gf = gf
        self.on_time = on_time
        self.off_time = off_time
        assert on_time <= off_time

    def volume_interpolant(self, t, discr):
        if self.on_time <= t < self.off_time:
            return self.gf.volume_interpolant(t, discr)
        else:
            # FIXME: not optimal
            # difficult part here is to match shape
            return 0 * self.gf.volume_interpolant(t, discr)

    def boundary_interpolant(self, t, discr, tag):
        if self.on_time <= t < self.off_time:
            return self.gf.boundary_interpolant(t, discr, tag)
        else:
            # FIXME: not optimal
            # difficult part here is to match shape
            return 0 * self.gf.boundary_interpolant(t, discr, tag)




class TimeDependentGivenFunction(ITimeDependentGivenFunction):
    """Adapts a function :math:`f(x,t)` into the
    :class:`GivenFunction` framework.
    """
    def __init__(self, f):
        self.f = f

    class ConstantWrapper:
        def __init__(self, f, t):
            """Adapt a function :math:`f(x, el, t)` in such a way that
            it can be fed to `interpolate_*_function()`. In particular,
            preserve the `shape` attribute.
            """
            self.f = f
            self.t = t

        @property
        def shape(self):
            return self.f.shape

        def __call__(self, x, el):
            return self.f(x, el, self.t)

    def volume_interpolant(self, t, discr):
        return discr.interpolate_volume_function(
                self.ConstantWrapper(self.f, t))

    def boundary_interpolant(self, t, discr, tag):
        return discr.interpolate_boundary_function(
                self.ConstantWrapper(self.f, t), tag)

# }}}

# {{{ compiled initial/boundary data ------------------------------------------
class CompiledExpressionData(
        ITimeDependentGivenFunction,
        IFieldDependentGivenFunction,
        ):
    def __init__(self, expressions_getter, arg_count=0):
        self.expressions_getter = expressions_getter
        self.arg_count = arg_count

    @memoize_method
    def make_func(self, discr, boundary_tag=None):
        from pymbolic import var

        def make_vec(basename):
            from hedge.tools import make_obj_array
            return make_obj_array(
                    [var("%s%d" % (basename, i)) for i in range(self.dimensions)])

        from hedge.optemplate.primitives import ScalarParameter
        from hedge.optemplate.tools import make_sym_vector

        x = make_sym_vector("x", discr.dimensions)
        fields = make_sym_vector("fields", self.arg_count)
        exprs = self.expressions_getter(
                t=ScalarParameter("t"), x=x, fields=fields)

        from hedge.optemplate.mappers.type_inference import (
                type_info, NodalRepresentation)
        type_hints = {}
        if boundary_tag is not None:
            my_vec_type = type_info.BoundaryVector(
                    boundary_tag, NodalRepresentation())
        else:
            my_vec_type = type_info.VolumeVector(
                    NodalRepresentation())

        for x_i in x:
            type_hints[x_i] = my_vec_type

        for f_i in fields:
            type_hints[f_i] = my_vec_type

        return discr.compile(exprs, type_hints=type_hints)

    def __call__(self, discr, t, fields, x, make_empty):
        result = self.make_func(discr)(
                t=numpy.float64(t), x=x, fields=fields)

        # make sure we return no scalars in the result
        from pytools.obj_array import log_shape, is_obj_array
        if is_obj_array(result):
            from pytools import indices_in_shape
            from hedge.optemplate.tools import is_scalar
            for i in indices_in_shape(log_shape(result)):
                if is_scalar(result[i]):
                    result[i] = make_empty().fill(result[i])

        return result

    @memoize_method
    def get_volume_nodes(self, discr):
        from hedge.tools import make_obj_array
        return discr.convert_volume(
                make_obj_array([
                    numpy.array(discr.nodes[:, i],
                        dtype=discr.default_scalar_type)
                    for i in range(discr.dimensions)]),
                kind=discr.compute_kind)

    def volume_interpolant(self, *args):
        if len(args) == 3:
            t, fields, discr = args
        elif len(args) == 2:
            t, discr = args
            fields = []
        else:
            raise TypeError("invalid arguments to "
                    "CompiledExpressionData.volume_interpolant")
        return self(discr, t, fields, self.get_volume_nodes(discr),
                 discr.volume_empty)

    def get_boundary_nodes(self, discr, tag):
        from hedge.tools import make_obj_array
        bnodes = discr.get_boundary(tag).nodes
        nodes = discr.convert_boundary(
                make_obj_array([
                    numpy.array(bnodes[:, i],
                        dtype=discr.default_scalar_type)
                        for i in range(discr.dimensions)]),
                tag, kind=discr.compute_kind)
        return nodes

    def boundary_interpolant(self, *args):
        if len(args) == 4:
            t, fields, discr, tag = args
        elif len(args) == 3:
            t, discr, tag = args
            fields = []
        else:
            raise TypeError("invalid arguments to "
                    "CompiledExpressionData.boundary_interpolant")

        return self(discr, t, fields,
                self.get_boundary_nodes(discr, tag),
                lambda: discr.boundary_empty(tag))

# }}}

# vim: foldmethod=marker
