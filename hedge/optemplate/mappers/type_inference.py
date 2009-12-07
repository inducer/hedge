"""Operator template type inference."""

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



import pymbolic.mapper
from pymbolic.mapper import CSECachingMapperMixin




class type_info:
    class _TypeInfo:
        def unify(self, other):
            """
            .. note::

                If no change of type results, unify must return self.
            """
            if self != other:
                raise TypeError("type '%s' and '%s' cannot be unified" 
                        % (self, other))

    class _StatelessTypeInfo(_TypeInfo):
        def __getinitargs__(self):
            return()

    class Scalar(_StatelessTypeInfo):
        pass

    class VolumeVectorBase(object):
        pass

    class QuadratureVectorBase(object):
        def __init__(self, quadrature_tag):
            self.quadrature_tag = quadrature_tag

    class VolumeVector(_StatelessTypeInfo, VolumeVectorBase):
        pass

    class QuadratureVolumeVector(_TypeInfo, VolumeVectorBase,
            QuadratureVectorBase):
        pass

    class FaceVector(_StatelessTypeInfo):
        pass

    class BoundaryVectorBase(object):
        def __init__(self, boundary_tag):
            self.boundary_tag = boundary_tag

    class BoundaryVector(_TypeInfo, BoundaryVectorBase):
        pass

    class QuadratureBoundaryVector(_TypeInfo, BoundaryVectorBase,
            QuadratureVectorBase):
        def __init__(self, boundary_tag, quadrature_tag):
            BoundaryVectorBase.__init__(self, boundary_tag)
            self.quadrature_tag = quadrature_tag




class TypeDict(object):
    def __init__(self):
        self.container = {}
        self.change_flag = False

    def __getitem__(self, expr):
        try:
            return self.container[expr]
        except KeyError:
            return None

    def __setitem__(self, expr, new_tp):
        assert new_tp is not None
        try:
            old_tp = self.container[expr]
        except KeyError:
            self.container[expr] = new_tp
            self.change_flag = True
        else:
            tp = old_tp.unify(new_tp)
            if tp is not old_tp:
                self.change_flag = True
            self.container[expr] = tp




class TypeInferrer(pymbolic.mapper.RecursiveMapper,
        CSECachingMapperMixin):
    def __call__(self, expr):
        typedict = TypeDict()

        while True:
            typedict.change_flag = False
            tp = pymbolic.mapper.RecursiveMapper.__call__(self, typedict)
            if tp is not None:
                typedict[expr] = tp

            if not typedict.change_flag:
                # nothing has changed any more, type information has 'converged'
                break

        for tdv in typedict.itervalues:
            if 

        return typedict

    def rec(self, expr, typedict):
        tp = pymbolic.mapper.RecursiveMapper.rec(self, expr, typedict)
        if tp is not None:
            typedict[expr] = tp
        return tp

    def map_sum(self, expr, typedict):
        tp = None

        for child in expr.children:
            if tp is None:
                tp = self.rec(child, typedict)
            else:
                typedict[child] = tp
                self.rec(child, typedict)

        return tp

    def map_product(self, expr, typedict):
        tp = None

        non_scalar_exprs = []

        for child in expr.children:
            if tp is None:
                tp = self.rec(child, typedict)
                if tp == type_info.Scalar:
                    tp = None
                else:
                    non_scalar_exprs.append(child)
            else:
                other_tp = self.rec(expr, typedict)

                if other_tp != type_info.Scalar:
                    non_scalar_exprs.append(child)
                    tp = tp.unify(other_tp)

        for child in non_scalar_exprs:
            typedict[child] = tp

        return tp

    def map_operator_binding(self, expr, typedict):
        from hedge.optemplate.operator import (
                DiffOperatorBase, MassOperatorBase, ElementwiseMaxOperator,
                BoundarizeOperator, FluxExchangeOperator,
                FluxOperatorBase)

        if isinstance(expr.op, (DiffOperatorBase, MassOperatorBase, 
            ElementwiseMaxOperator)):
            typedict[expr.field] = type_info.VolumeVectorBase()
            return self.rec(expr, typedict))
        elif isinstance(expr.op, BoundarizeOperator):
            typedict[expr.field] = type_info.BoundaryVectorBase(expr.op.tag)
            return self.rec(expr, typedict))
        elif isinstance(expr.op, FluxExchangeOperator):
            raise NotImplementedError
        elif isinstance(expr.op, FluxOperatorBase):
            raise NotImplementedError

    def map_variable(self, expr, typedict):
        return None

    def map_scalar_parameter(self, expr, typedict):
        return type_info.Scalar()

    def map_normal_component(self, expr, typedict):
        return type_info.BoundaryVectorBase(expr.tag)

    def map_common_subexpression_uncached(self, expr, typedict):
        return typedict[expr.child]
