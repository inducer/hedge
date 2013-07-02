"""Timeseries data gathering sensors."""

from __future__ import division

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


from pytools.log import LogQuantity, MultiLogQuantity
import numpy as np


def axis_name(axis):
    if axis == 0:
        return "x"
    elif axis == 1:
        return "y"
    elif axis == 2:
        return "z"
    else:
        raise RuntimeError("invalid axis index")


class Integral(LogQuantity):
    """Log the volume integral of a variable in a scope."""

    def __init__(self, getter, discr, name=None,
            unit="1", description=None):
        """Construct the integral logger.

        :param getter: a callable that returns the value of which to
          take the integral.
        :param discr: a :class:`hedge.discretization.Discretization`
            to which the variable belongs.
        :param name: the name reported to the :class:`pytools.log.LogManager`.
        :param unit: the unit of measure for the log quantity.
        :param description: A description fed to the :class:`pytools.log.LogManager`.
        """
        self.getter = getter

        if name is None:
            try:
                name = "int_%s" % self.getter.name()
            except AttributeError:
                raise ValueError("must specify a name")

        LogQuantity.__init__(self, name, unit, description)

        self.discr = discr

    @property
    def default_aggregator(self):
        return sum

    def __call__(self):
        var = self.getter()

        from hedge.tools import log_shape

        if len(log_shape(var)) == 1:
            return sum(
                    self.discr.integral(np.abs(v))
                    for v in var)
        else:
            return self.discr.integral(var)


class LpNorm(LogQuantity):
    """Log the Lp norm of a variable in a scope."""

    def __init__(self, getter, discr, p=2, name=None,
            unit="1", description=None):
        """Construct the Lp norm logger.

        :param getter: a callable that returns the value of which to
          take the norm.
        :param discr: a :class:`hedge.discretization.Discretization`
            to which the variable belongs.
        :param p: the power of the norm.
        :param name: the name reported to the :class:`pytools.log.LogManager`.
        :param unit: the unit of measure for the log quantity.
        :param description: A description fed to the :class:`pytools.log.LogManager`.
        """

        self.getter = getter
        self.discr = discr
        self.p = p

        if name is None:
            try:
                name = "l%d_%s" % (int(p), self.getter.name())
            except AttributeError:
                raise ValueError("must specify a name")

        LogQuantity.__init__(self, name, unit, description)

    @property
    def default_aggregator(self):
        from pytools import norm_inf, Norm

        if self.p == np.Inf:
            return norm_inf
        else:
            return Norm(self.p)

    def __call__(self):
        var = self.getter()
        return self.discr.norm(var, self.p)


# {{{ electromagnetic quantities

class EMFieldGetter(object):
    """Makes E and H field accessible as self.e and self.h from a variable lookup.
    To be used with the EM log quantities in this module."""
    def __init__(self, discr, maxwell_op, fgetter):
        self.discr = discr
        self.maxwell_op = maxwell_op
        self.fgetter = fgetter

    @property
    def e(self):
        fields = self.fgetter()
        e, h = self.maxwell_op.split_eh(fields)
        return e

    @property
    def h(self):
        fields = self.fgetter()
        e, h = self.maxwell_op.split_eh(fields)
        return h


class ElectricFieldEnergy(LogQuantity):
    def __init__(self, fields, name="W_el"):
        LogQuantity.__init__(self, name, "J", "Energy of the electric field")
        self.fields = fields

    @property
    def default_aggregator(self):
        from pytools import norm_2
        return norm_2

    def __call__(self):
        max_op = self.fields.maxwell_op

        e = self.fields.e
        d = max_op.epsilon * e

        from hedge.tools import ptwise_dot
        energy_density = 1/2*(ptwise_dot(1, 1, e, d))
        return self.fields.discr.integral(energy_density)


class MagneticFieldEnergy(LogQuantity):
    def __init__(self, fields, name="W_mag"):
        LogQuantity.__init__(self, name, "J", "Energy of the magnetic field")
        self.fields = fields

    @property
    def default_aggregator(self):
        from pytools import norm_2
        return norm_2

    def __call__(self):
        max_op = self.fields.maxwell_op

        h = self.fields.h
        b = max_op.mu * h

        from hedge.tools import ptwise_dot
        energy_density = 1/2*(ptwise_dot(1, 1, h, b))
        return self.fields.discr.integral(energy_density)


class EMFieldMomentum(MultiLogQuantity):
    def __init__(self, fields, c0, names=None):
        if names is None:
            names = ["p%s_field" % axis_name(i)
                    for i in range(3)]

        vdim = len(names)

        MultiLogQuantity.__init__(self, names,
            units=["N*s"] * vdim,
            descriptions=["Field Momentum"] * vdim)

        self.fields = fields
        self.c0 = c0

        e_subset = fields.maxwell_op.get_eh_subset()[0:3]
        h_subset = fields.maxwell_op.get_eh_subset()[3:6]

        from hedge.tools import SubsettableCrossProduct
        self.poynting_cross = SubsettableCrossProduct(
                op1_subset=e_subset,
                op2_subset=h_subset,
                )

    def __call__(self):
        e = self.fields.e
        h = self.fields.h

        poynting_s = self.poynting_cross(e, h)

        momentum_density = poynting_s/self.c0**2
        return self.fields.discr.integral(momentum_density)


class EMFieldDivergenceD(LogQuantity):
    def __init__(self, maxwell_op, fields, name="divD"):
        LogQuantity.__init__(self, name, "C", "Integral over div D")

        self.fields = fields

        from hedge.models.nd_calculus import DivergenceOperator
        div_op = DivergenceOperator(maxwell_op.dimensions,
                maxwell_op.get_eh_subset()[:3])
        self.bound_div_op = div_op.bind(self.fields.discr)

    def __call__(self):
        max_op = self.fields.maxwell_op
        d = max_op.epsilon * self.fields.e
        div_d = self.bound_div_op(d)

        return self.fields.discr.integral(div_d)


class EMFieldDivergenceB(MultiLogQuantity):
    def __init__(self, maxwell_op, fields, names=None):
        self.fields = fields

        from hedge.models.nd_calculus import DivergenceOperator
        self.div_op = DivergenceOperator(maxwell_op.dimensions,
                maxwell_op.get_eh_subset()[3:]).bind(self.fields.discr)

        if names is None:
            names = ["divB", "err_divB_l1"]

        MultiLogQuantity.__init__(self,
                names=names,
                units=["T/m", "T/m"],
                descriptions=["Integral over div B", "Integral over |div B|"])

    def __call__(self):
        max_op = self.fields.maxwell_op
        b = max_op.mu * self.fields.h
        div_b = self.div_op(b)

        return [self.fields.discr.integral(div_b),
                self.fields.discr.integral(abs(div_b))]


def add_em_energies(mgr, maxwell_op, fields):
    mgr.add_quantity(ElectricFieldEnergy(fields))
    mgr.add_quantity(MagneticFieldEnergy(fields))


def add_em_quantities(mgr, maxwell_op, fields):
    add_em_energies(mgr, maxwell_op, fields)
    mgr.add_quantity(EMFieldMomentum(fields, maxwell_op.c))
    mgr.add_quantity(EMFieldDivergenceD(maxwell_op, fields))
    mgr.add_quantity(EMFieldDivergenceB(maxwell_op, fields))

# }}}

# vim: fdm=marker
