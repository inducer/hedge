"""Operator template language: primitives."""

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




import numpy
import pymbolic.primitives
import hedge.mesh





# {{{ variables ---------------------------------------------------------------
from hedge.tools.symbolic import CFunction

Field = pymbolic.primitives.Variable

def make_field(var_or_string):
    if not isinstance(var_or_string, pymbolic.primitives.Expression):
        return Field(var_or_string)
    else:
        return var_or_string




class ScalarParameter(pymbolic.primitives.Variable):
    """A placeholder for a user-supplied scalar variable."""

    def stringifier(self):
        from hedge.optemplate import StringifyMapper
        return StringifyMapper

    def get_mapper_method(self, mapper):
        return mapper.map_scalar_parameter

# }}}

# {{{ technical helpers -------------------------------------------------------
class OperatorBinding(pymbolic.primitives.AlgebraicLeaf):
    def __init__(self, op, field):
        self.op = op
        self.field = field

    def stringifier(self):
        from hedge.optemplate import StringifyMapper
        return StringifyMapper

    def get_mapper_method(self, mapper):
        return mapper.map_operator_binding

    def __getinitargs__(self):
        return self.op, self.field

    def is_equal(self, other):
        from hedge.tools import field_equal
        return (other.__class__ == self.__class__
                and other.op == self.op
                and field_equal(other.field, self.field))

    def get_hash(self):
        from hedge.tools import hashable_field
        return hash((self.__class__, self.op, hashable_field(self.field)))




class PrioritizedSubexpression(pymbolic.primitives.CommonSubexpression):
    """When the optemplate-to-code transformation is performed,
    prioritized subexpressions  work like common subexpression in
    that they are assigned their own separate identifier/register
    location. In addition to this behavior, prioritized subexpressions
    are evaluated with a settable priority, allowing the user to
    expedite or delay the evaluation of the subexpression.
    """

    def __init__(self, child, priority=0):
        pymbolic.primitives.CommonSubexpression.__init__(self, child)
        self.priority = priority

    def __getinitargs__(self):
        return (self.child, self.priority)

    def get_extra_properties(self):
        return {"priority": self.priority}




class BoundaryPair(pymbolic.primitives.AlgebraicLeaf):
    """Represents a pairing of a volume and a boundary field, used for the
    application of boundary fluxes.
    """

    def __init__(self, field, bfield, tag=hedge.mesh.TAG_ALL):
        self.field = field
        self.bfield = bfield
        self.tag = tag

    def get_mapper_method(self, mapper):
        return mapper.map_boundary_pair

    def stringifier(self):
        from hedge.optemplate.mappers import StringifyMapper
        return StringifyMapper

    def __getinitargs__(self):
        return (self.field, self.bfield, self.tag)

    def get_hash(self):
        from hedge.tools import hashable_field

        return hash((self.__class__,
            hashable_field(self.field),
            hashable_field(self.bfield),
            self.tag))

    def is_equal(self, other):
        from hedge.tools import field_equal
        return (self.__class__ == other.__class__
                and field_equal(other.field,  self.field)
                and field_equal(other.bfield, self.bfield)
                and other.tag == self.tag)


# }}}

# {{{ geometry data -----------------------------------------------------------
class BoundaryNormalComponent(pymbolic.primitives.AlgebraicLeaf):
    def __init__(self, tag, axis):
        self.tag = tag
        self.axis = axis

    def stringifier(self):
        from hedge.optemplate.mappers import StringifyMapper
        return StringifyMapper

    def get_hash(self):
        return hash((self.__class__, self.tag, self.axis))

    def is_equal(self, other):
        return (other.__class__ == self.__class__
                and other.tag == self.tag
                and other.axis == self.axis)

    def get_mapper_method(self, mapper):
        return mapper.map_normal_component

    def __getinitargs__(self):
        return (self.tag, self.axis)




def make_normal(tag, dimensions):
    return numpy.array([BoundaryNormalComponent(tag, i)
        for i in range(dimensions)], dtype=object)




class Jacobian(pymbolic.primitives.AlgebraicLeaf):
    def stringifier(self):
        from hedge.optemplate.mappers import StringifyMapper
        return StringifyMapper

    def get_hash(self):
        return hash(self.__class__)

    def is_equal(self, other):
        return (other.__class__ == self.__class__)

    def get_mapper_method(self, mapper):
        return mapper.map_jacobian

    def __getinitargs__(self):
        return ()




class ForwardMetricDerivative(pymbolic.primitives.AlgebraicLeaf):
    """
    Pointwise metric derivatives representing

    .. math::
    
        \frac{d x_{\mathtt{xyz\_axis}} }{d r_{\mathtt{rst\_axis}} }
    """

    def __init__(self, xyz_axis, rst_axis):
        self.xyz_axis = xyz_axis
        self.rst_axis = rst_axis

    def stringifier(self):
        from hedge.optemplate.mappers import StringifyMapper
        return StringifyMapper

    def get_hash(self):
        return hash((self.__class__, self.xyz_axis, self.rst_axis))

    def is_equal(self, other):
        return (other.__class__ == self.__class__
                and other.xyz_axis == self.xyz_axis
                and other.rst_axis == self.rst_axis
                )

    def get_mapper_method(self, mapper):
        return mapper.map_forward_metric_derivative

    def __getinitargs__(self):
        return (self.xyz_axis, self.rst_axis)




class InverseMetricDerivative(pymbolic.primitives.AlgebraicLeaf):
    """
    Pointwise metric derivatives representing

    .. math::
    
        \frac{d r_{\mathtt{rst\_axis}} }{d x_{\mathtt{xyz\_axis}} }
    """

    def __init__(self, rst_axis, xyz_axis, ):
        self.rst_axis = rst_axis
        self.xyz_axis = xyz_axis

    def stringifier(self):
        from hedge.optemplate.mappers import StringifyMapper
        return StringifyMapper

    def get_hash(self):
        return hash((self.__class__, self.rst_axis, self.xyz_axis))

    def is_equal(self, other):
        return (other.__class__ == self.__class__
                and other.rst_axis == self.rst_axis
                and other.xyz_axis == self.xyz_axis
                )

    def get_mapper_method(self, mapper):
        return mapper.map_inverse_metric_derivative

    def __getinitargs__(self):
        return (self.rst_axis, self.xyz_axis)

# }}}




# vim: foldmethod=marker
