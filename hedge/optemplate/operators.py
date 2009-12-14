"""Building blocks and mappers for operator expression trees."""

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




# operators -------------------------------------------------------------------
class Operator(pymbolic.primitives.Leaf):
    def stringifier(self):
        from hedge.optemplate import StringifyMapper
        return StringifyMapper

    def __call__(self, expr):
        from hedge.tools import with_object_array_or_scalar, is_zero
        def bind_one(subexpr):
            if is_zero(subexpr):
                return subexpr
            else:
                return OperatorBinding(self, subexpr)

        return with_object_array_or_scalar(bind_one, expr)

    def apply(self, discr, field):
        from hedge.optemplate import Field
        return discr.compile(self * Field("f"))(f=field)




class StatelessOperator(Operator):
    def __getinitargs__(self):
        return ()

    def get_hash(self):
        return hash(self.__class__)

    def is_equal(self, other):
        return other.__class__ == self.__class__




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




# diff operators --------------------------------------------------------------
class DiffOperatorBase(Operator):
    def __init__(self, xyz_axis):
        Operator.__init__(self)

        self.xyz_axis = xyz_axis

    def __getinitargs__(self):
        return (self.xyz_axis,)

    def get_hash(self):
        return hash((self.__class__, self.xyz_axis))

    def is_equal(self, other):
        return (other.__class__ == self.__class__
                and other.xyz_axis == self.xyz_axis)

class StrongFormDiffOperatorBase(DiffOperatorBase):
    pass

class WeakFormDiffOperatorBase(DiffOperatorBase):
    pass

class StiffnessOperator(StrongFormDiffOperatorBase):
    @staticmethod
    def matrices(element_group):
        return element_group.stiffness_matrices

    @staticmethod
    def coefficients(element_group):
        return element_group.stiffness_coefficients

    def get_mapper_method(self, mapper):
        return mapper.map_stiffness

class DifferentiationOperator(StrongFormDiffOperatorBase):
    @staticmethod
    def matrices(element_group):
        return element_group.differentiation_matrices

    @staticmethod
    def coefficients(element_group):
        return element_group.diff_coefficients

    def get_mapper_method(self, mapper):
        return mapper.map_diff

class StiffnessTOperator(WeakFormDiffOperatorBase):
    @staticmethod
    def matrices(element_group):
        return element_group.stiffness_t_matrices

    @staticmethod
    def coefficients(element_group):
        return element_group.stiffness_coefficients

    def get_mapper_method(self, mapper):
        return mapper.map_stiffness_t

class MInvSTOperator(WeakFormDiffOperatorBase):
    @staticmethod
    def matrices(element_group):
        return element_group.minv_st

    @staticmethod
    def coefficients(element_group):
        return element_group.diff_coefficients

    def get_mapper_method(self, mapper):
        return mapper.map_minv_st





def DiffOperatorVector(els):
    from hedge.tools import join_fields
    return join_fields(*els)




# elementwise operators -------------------------------------------------------
class ElementwiseOperator(Operator):
    @staticmethod
    def matrix(element_group):
        raise NotImplementedError

    @staticmethod
    def coefficients(element_group):
        raise NotImplementedError

    def get_mapper_method(self, mapper):
        return mapper.map_elementwise




# mass operators --------------------------------------------------------------
class MassOperatorBase(ElementwiseOperator, StatelessOperator):
    pass




class MassOperator(MassOperatorBase):
    @staticmethod
    def matrix(element_group):
        return element_group.mass_matrix

    @staticmethod
    def coefficients(element_group):
        return element_group.jacobians

    def get_mapper_method(self, mapper):
        return mapper.map_mass

class InverseMassOperator(MassOperatorBase):
    @staticmethod
    def matrix(element_group):
        return element_group.inverse_mass_matrix

    @staticmethod
    def coefficients(element_group):
        return element_group.inverse_jacobians

    def get_mapper_method(self, mapper):
        return mapper.map_inverse_mass





# filter operator -------------------------------------------------------------
class FilterOperator:
    def __init__(self, mode_response_func):
        """Construct a filter.

        :param discr: The :class:`Discretization` for which the filter is to be
          constructed.
        :param mode_response_func: A function mapping
          ``(mode_tuple, local_discretization)`` to a float indicating the
          factor by which this mode is to be multiplied after filtering.
          (For example an instance of 
          :class:`ExponentialFilterResponseFunction`.
        """
        self.discr = discr
        self.mode_response_func = mode_response_func

    def __call__(self, vec):
        return self.discr.apply_element_local_matrix(
                self.get_filter_matrix, vec,
                prepared_data_store=self.prepared_data_store)

    def get_filter_matrix(self, eg):
        ldis = eg.local_discretization

        node_count = ldis.node_count()

        filter_coeffs = [self.mode_response_func(mid, ldis)
            for mid in ldis.generate_mode_identifiers()]

        # build filter matrix
        vdm = ldis.vandermonde()
        from hedge.tools import leftsolve
        from numpy import dot
        mat = numpy.asarray(
            leftsolve(vdm,
                dot(vdm, numpy.diag(filter_coeffs))),
            order="C")

        return mat




# misc operators --------------------------------------------------------------
class ElementwiseMaxOperator(StatelessOperator):
    def get_mapper_method(self, mapper):
        return mapper.map_elementwise_max




class BoundarizeOperator(Operator):
    def __init__(self, tag):
        self.tag = tag

    def get_hash(self):
        return hash((self.__class__, self.tag))

    def is_equal(self, other):
        return (other.__class__ == self.__class__
                and other.tag == self.tag)

    def get_mapper_method(self, mapper):
        return mapper.map_boundarize

    def __getinitargs__(self):
        return (self.tag,)




class FluxExchangeOperator(Operator):
    """An operator that results in the sending and receiving of
    boundary information for its argument fields.
    """

    def __init__(self, idx, rank):
        self.index = idx
        self.rank = rank

    def __getinitargs__(self):
        return (self.index, self.rank)

    def get_hash(self):
        return hash((self.__class__, self.index, self.rank))

    def is_equal(self, other):
        return (other.__class__ == self.__class__
                and other.index == self.index
                and other.rank == self.rank)

    def get_mapper_method(self, mapper):
        return mapper.map_flux_exchange




# flux-like operators ---------------------------------------------------------
class FluxOperatorBase(Operator):
    def __init__(self, flux):
        Operator.__init__(self)
        self.flux = flux

    def __getinitargs__(self):
        return (self.flux, )

    def get_hash(self):
        return hash((self.__class__, self.flux))

    def is_equal(self, other):
        return (self.__class__ == other.__class__
                and self.flux == other.flux)

    def __call__(self, arg):
        from hedge.tools import is_obj_array
        from hedge.optemplate import Field
        if isinstance(arg, Field) or is_obj_array(arg):
            return OperatorBinding(self, arg)
        else:
            return Operator.__mul__(self, arg)

    def __mul__(self, arg):
        from warnings import warn
        warn("Multiplying by a flux operator is deprecated. "
                "Use the less ambiguous parenthesized syntax instead.",
                DeprecationWarning, stacklevel=2)
        return self.__call__(arg)



class FluxOperator(FluxOperatorBase):
    def get_mapper_method(self, mapper):
        return mapper.map_flux



class LiftingFluxOperator(FluxOperatorBase):
    def get_mapper_method(self, mapper):
        return mapper.map_lift



class VectorFluxOperator(object):
    def __init__(self, fluxes):
        self.fluxes = fluxes

    def __call__(self, arg):
        if isinstance(arg, int) and arg == 0:
            return 0
        from hedge.tools import make_obj_array
        return make_obj_array(
                [OperatorBinding(FluxOperator(f), arg)
                    for f in self.fluxes])

    def __mul__(self, arg):
        from warnings import warn
        warn("Multiplying by a vector flux operator is deprecated. "
                "Use the less ambiguous parenthesized syntax instead.",
                DeprecationWarning, stacklevel=2)
        return self.__call__(arg)
