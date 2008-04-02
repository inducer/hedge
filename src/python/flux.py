"""Building blocks for flux computation. Flux compilation."""

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




import numpy
import hedge._internal as _internal
from pytools import monkeypatch_method
import pymbolic
import pymbolic.mapper.collector




FluxFace = _internal.FluxFace




# c++ fluxes debug dumping ----------------------------------------------------
@monkeypatch_method(_internal.ConstantFlux)
def __str__(self):
    return "ConstantFlux(%s)" % self.value
@monkeypatch_method(_internal.NormalFlux)
def __str__(self):
    return "NormalFlux(%d)" % self.axis
@monkeypatch_method(_internal.PenaltyFlux)
def __str__(self):
    return "PenaltyFlux(%s)" % self.power
@monkeypatch_method(_internal.SumFlux)
def __str__(self):
    return "SumFlux(%s, %s)" % (self.operand1, self.operand2)
@monkeypatch_method(_internal.ProductFlux)
def __str__(self):
    return "ProductFlux(%s, %s)" % (self.operand1, self.operand2)
@monkeypatch_method(_internal.NegativeFlux)
def __str__(self):
    return "NegativeFlux(%s)" % self.operand
@monkeypatch_method(_internal.ChainedFlux)
def __str__(self):
    #return "ChainedFlux(%s)" % self.child
    return str(self.child)





# python fluxes ---------------------------------------------------------------
class Flux(pymbolic.primitives.AlgebraicLeaf, _internal.Flux):
    def stringify(self, enclosing_prec, use_repr_for_constants=False):
        return FluxStringifyMapper(use_repr_for_constants)(self, enclosing_prec)

    def perform(self, face_group, which_faces, fmm, target):
        _internal.ChainedFlux(self).perform(face_group, which_faces, fmm, target)




class FieldComponent(Flux):
    def __init__(self, index, is_local):
        self.index = index
        self.is_local = is_local

    def __eq__(self, other):
        return (isinstance(other, FieldComponent) 
                and self.index == other.index
                and self.is_local == other.is_local
                )

    def __getinitargs__(self):
        return self.index, self.is_local

    def __hash__(self):
        return 0x7371afcd ^ hash(self.index) ^ hash(self.is_local)

    def get_mapper_method(self, mapper):
        return mapper.map_field_component




class Normal(Flux):
    def __init__(self, axis):
        self.axis = axis

    def __getinitargs__(self):
        return self.axis,

    def __eq__(self, other):
        return isinstance(other, Normal) and self.axis == other.axis

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return 0xa931ff ^ hash(self.axis)

    def get_mapper_method(self, mapper):
        return mapper.map_normal




class PenaltyTerm(Flux):
    def __init__(self, power=1):
        self.power = power

    def __getinitargs__(self):
        return self.power,

    def __eq__(self, other):
        return isinstance(other, PenaltyTerm) and self.power == other.power

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return 0x9f194 ^ hash(self.power)

    def get_mapper_method(self, mapper):
        return mapper.map_penalty_term




class IfPositive(Flux):
    def __init__(self, criterion, then, else_):
        self.criterion = criterion
        self.then = then
        self.else_ = else_

        if FluxDependencyMapper(composite_leaves=True)(criterion):
            raise ValueError("criterion part of IfPositive may not depend on field values")

    def __getinitargs__(self):
        return self.criterion, self.then, self.else_

    def __eq__(self, other):
        return (isinstance(other, IfPositive)
                and self.criterion == other.criterion
                and self.then == other.then
                and self.else_ == other.else_)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return 0xfa5814 ^ hash(self.criterion) ^ hash(self.then) ^ hash(self.else_)

    def get_mapper_method(self, mapper):
        return mapper.map_if_positive




def make_normal(dimensions):
    from hedge.tools import amap
    return amap(Normal,  range(dimensions))





class FluxZeroPlaceholder(object):
    @property
    def int(self):
        return 0

    @property
    def ext(self):
        return 0

    @property
    def avg(self):
        return 0




class FluxScalarPlaceholder(object):
    def __init__(self, component=0):
        self.component = component

    @property
    def int(self):
        return FieldComponent(self.component, True)

    @property
    def ext(self):
        return FieldComponent(self.component, False)

    @property
    def avg(self):
        return 0.5*(self.int+self.ext)




class FluxVectorPlaceholder(object):
    def __init__(self, components=None, scalars=None):
        if not (components or scalars):
            raise ValueError, "either components or scalars must be specified"
        if components and scalars:
            raise ValueError, "only one of components and scalars may be specified"

        if components:
            self.scalars = [
                    FluxScalarPlaceholder(i) 
                    for i in range(components)]
        else:
            self.scalars = scalars

    def __len__(self):
        return len(self.scalars)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.scalars[idx]
        else:
            return FluxVectorPlaceholder(scalars=self.scalars.__getitem__(idx))

    @property
    def int(self):
        return numpy.array([scalar.int for scalar in self.scalars])

    @property
    def ext(self):
        return numpy.array([scalar.ext for scalar in self.scalars])

    @property
    def avg(self):
        return numpy.array([scalar.avg for scalar in self.scalars])




# internal flux wrangling -----------------------------------------------------
class FluxStringifyMapper(pymbolic.mapper.stringifier.StringifyMapper):
    def map_field_component(self, expr, enclosing_prec):
        if expr.is_local:
            return "Int[%d]" % expr.index
        else:
            return "Ext[%d]" % expr.index

    def map_normal(self, expr, enclosing_prec):
        return "Normal(%d)" % expr.axis

    def map_penalty_term(self, expr, enclosing_prec):
        return "Penalty(%s)" % (expr.power)

    def map_if_positive(self, expr, enclosing_prec):
        return "IfPositive(%s, %s, %s)" % (expr.criterion, expr.then, expr.else_)




class FluxNormalizationMapper(pymbolic.mapper.collector.TermCollector):
    def map_constant_flux(self, expr):
        if expr.local_c == expr.neighbor_c:
            return expr.local_c
        else:
            return expr

    def map_if_positive(self, expr):
        return IfPositive(
                self.rec(expr.criterion),
                self.rec(expr.then),
                self.rec(expr.else_),
                )



        
def normalize_flux(flux):
    from pymbolic import expand, flatten
    from pymbolic.mapper.constant_folder import CommutativeConstantFoldingMapper
    return CommutativeConstantFoldingMapper()(FluxNormalizationMapper()(
        flatten(expand(flux))))




class FluxDependencyMapper(pymbolic.mapper.dependency.DependencyMapper):
    def map_field_component(self, expr):
        return set([expr])

    def map_normal(self, expr):
        return set()

    def map_penalty_term(self, expr):
        return set()

    def map_if_positive(self, expr):
        return self.rec(expr.criterion) | self.rec(expr.then) | self.rec(expr.else_)




class FluxCompilationMapper(pymbolic.mapper.RecursiveMapper):
    def handle_unsupported_expression(self, expr):
        if isinstance(expr, _internal.Flux):
            return expr
        else:
            pymbolic.mapper.RecursiveMapper.\
                    handle_unsupported_expression(self, expr)

    def map_constant(self, expr):
        return _internal.ConstantFlux(expr)

    def map_sum(self, expr):
        return reduce(lambda f1, f2: _internal.SumFlux(
                    _internal.ChainedFlux(f1), 
                    _internal.ChainedFlux(f2)),
                (self.rec(c) for c in expr.children))

    def map_product(self, expr):
        return reduce(
                lambda f1, f2: _internal.ProductFlux(
                    _internal.ChainedFlux(f1), 
                    _internal.ChainedFlux(f2)),
                (self.rec(c) for c in expr.children))

    def map_negation(self, expr):
        return _internal.NegativeFlux(_internal.ChainedFlux(self.rec(expr.child)))

    def map_power(self, expr):
        base = self.rec(expr.base)
        result = base

        chain_base = _internal.ChainedFlux(base)

        assert isinstance(expr.exponent, int)

        for i in range(1, expr.exponent):
            result = _internal.ProductFlux(_internal.ChainedFlux(result), chain_base)

        return result

    def map_normal(self, expr):
        return _internal.NormalFlux(expr.axis)

    def map_penalty_term(self, expr):
        return _internal.PenaltyFlux(expr.power)

    def map_if_positive(self, expr):
        return _internal.IfPositiveFlux(
                _internal.ChainedFlux(self.rec(expr.criterion)),
                _internal.ChainedFlux(self.rec(expr.then)),
                _internal.ChainedFlux(self.rec(expr.else_)),
                )




class FluxDifferentiationMapper(pymbolic.mapper.differentiator.DifferentiationMapper):
    def map_field_component(self, expr):
        if expr == self.variable:
            return 1
        else:
            return 0

    def map_normal(self, expr):
        return 0

    def map_penalty_term(self, expr):
        return 0

    def map_if_positive(self, expr):
        return IfPositive(
                expr.criterion,
                self.rec(expr.then),
                self.rec(expr.else_),
                )




def compile_flux(flux):
    from hedge.tools import is_obj_array
    if is_obj_array(flux):
        return numpy.array([compile_flux(subflux) for subflux in flux])

    def compile_scalar_single_dep_flux(flux):
        return FluxCompilationMapper()(flux)

    def compile_scalar_flux(flux):
        def in_fields_cmp(a, b):
            return cmp(a.index, b.index) \
                    or cmp(a.is_local, b.is_local)

        in_fields = list(FluxDependencyMapper(composite_leaves=True)(flux))

        # check that all in_fields are FieldComponents
        for in_field in in_fields:
            if not isinstance(in_field, FieldComponent):
                raise ValueError, "flux depends on invalid term `%s'" % str(in_field)
            
        in_fields.sort(in_fields_cmp)

        result = []

        if in_fields:
            max_in_field = max(in_field.index for in_field in in_fields)

            # find d<flux> / d<in_fields>
            in_derivatives = dict(
                    ((in_field.index, in_field.is_local),
                    normalize_flux(FluxDifferentiationMapper(in_field)(flux)))
                    for in_field in in_fields)

            # check for (invalid) nonlinearity
            for i, deriv in in_derivatives.iteritems():
                if FluxDependencyMapper()(deriv):
                    raise ValueError, "Flux is nonlinear in component %d" % i

            for in_field_idx in range(max_in_field+1):
                int = in_derivatives.get((in_field_idx, True), 0)
                ext = in_derivatives.get((in_field_idx, False), 0)

                if int or ext:
                    result.append(
                            (in_field_idx, 
                                compile_scalar_single_dep_flux(int),
                                compile_scalar_single_dep_flux(ext),))

        try:
            flux._compiled = result
        except AttributeError:
            pass # ok if we can't set cache

        return result

    try:
        return flux._compiled
    except AttributeError:
        return compile_scalar_flux(flux)
