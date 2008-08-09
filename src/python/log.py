"""Logging utilities."""

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



from pytools.log import LogQuantity, MultiLogQuantity
import numpy
import numpy.linalg as la




def axis_name(axis):
    if axis == 0: return "x"
    elif axis == 1: return "y"
    elif axis == 2: return "z"
    else: raise RuntimeError, "invalid axis index"




class Integral(LogQuantity):
    """Log the volume integral of a variable in a scope."""

    def __init__(self, getter, discr, name=None, 
            unit="1", description=None):
        """Construct the integral logger.

        @arg getter: a callable that returns the value of which to 
          take the integral.
        @arg discr: a L{Discretization} to which the variable belongs.
        @arg name: the name reported to the C{LogManager}.
        @arg unit: the unit of measure for the log quantity.
        @arg description: A description fed to the C{LogManager}.
        """
        self.getter = getter

        if name is None:
            try:
                name = "int_%s" % self.getter.name()
            except AttributeError:
                raise ValueError, "must specify a name"

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
                    self.discr.integral(numpy.abs(v))
                    for v in var)
        else:
            return self.discr.integral(var)




class LpNorm(LogQuantity):
    """Log the Lp norm of a variable in a scope."""

    def __init__(self, getter, discr, p=2, name=None, 
            unit="1", description=None):
        """Construct the Lp norm logger.

        @arg getter: a callable that returns the value of which to 
          take the norm.
        @arg discr: a L{Discretization} to which the variable belongs.
        @arg p: the power of the norm.
        @arg name: the name reported to the C{LogManager}.
        @arg unit: the unit of measure for the log quantity.
        @arg description: A description fed to the C{LogManager}.
        """
        self.getter = getter
        self.discr = discr
        self.p = p

        if name is None:
            try:
                name = "l%d_%s" % (int(p), self.getter.name())
            except AttributeError:
                raise ValueError, "must specify a name"

        LogQuantity.__init__(self, name, unit, description)

    @property
    def default_aggregator(self): 
        from pytools import norm_inf, Norm

        if self.p == numpy.Inf:
            return norm_inf
        else:
            return Norm(self.p)

    def __call__(self):
        var = self.getter()

        from hedge.tools import log_shape
        return self.discr.norm(var, self.p)




# electromagnetic quantities --------------------------------------------------
class EMFieldGetter(object):
    """Makes E and H field accessible as self.e and self.h from a variable lookup.
    To be used with the EM log quantities in this module."""
    def __init__(self, maxwell_op, scope, varname):
        self.maxwell_op = maxwell_op
        self.scope = scope
        self.varname = varname

    @property
    def e(self):
        fields = self.scope[self.varname]
        e, h = self.maxwell_op.split_eh(fields)
        return e

    @property
    def h(self):
        fields = self.scope[self.varname]
        e, h = self.maxwell_op.split_eh(fields)
        return h




class EMFieldEnergy(LogQuantity):
    def __init__(self, fields, name="W_field"):
        LogQuantity.__init__(self, name, "J", "Field Energy")
        self.fields = fields

    @property
    def default_aggregator(self): 
        from pytools import norm_2
        return norm_2

    def __call__(self):
        max_op = self.fields.maxwell_op

        e = self.fields.e
        h = self.fields.h
        d = max_op.epsilon * e
        b = max_op.mu * h

        from hedge.tools import ptwise_dot
        energy_density = 1/2*(
                ptwise_dot(1, 1, e, d) 
                + ptwise_dot(1, 1, h, b))

        return max_op.discr.integral(energy_density)




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
        max_op = self.fields.maxwell_op

        e = self.fields.e
        h = self.fields.h

        poynting_s = self.poynting_cross(e, h)

        momentum_density = poynting_s/self.c0**2
        return max_op.discr.integral(momentum_density)




class EMFieldDivergenceD(LogQuantity):
    def __init__(self, maxwell_op, fields, name="divD"):
        LogQuantity.__init__(self, name, "C", "Integral over div D")

        self.fields = fields
        self.discr = self.fields.maxwell_op.discr

        from hedge.pde import DivergenceOperator
        self.div_op = DivergenceOperator(self.discr,
                maxwell_op.get_eh_subset()[:3])

    def __call__(self):
        max_op = self.fields.maxwell_op
        d = max_op.epsilon * self.fields.e
        div_d = self.div_op(d)
        
        return self.discr.integral(div_d)




class EMFieldDivergenceB(MultiLogQuantity):
    def __init__(self, maxwell_op, fields, names=None):
        self.fields = fields
        self.discr = self.fields.maxwell_op.discr

        from hedge.pde import DivergenceOperator
        self.div_op = DivergenceOperator(self.discr,
                maxwell_op.get_eh_subset()[3:])

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
        
        return [self.discr.integral(div_b), 
                self.discr.integral(numpy.absolute(div_b))]




def add_em_quantities(mgr, maxwell_op, fields):
    mgr.add_quantity(EMFieldEnergy(fields))
    mgr.add_quantity(EMFieldMomentum(fields, maxwell_op.c))
    mgr.add_quantity(EMFieldDivergenceD(maxwell_op, fields))
    mgr.add_quantity(EMFieldDivergenceB(maxwell_op, fields))
