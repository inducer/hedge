# Hedge - the Hybrid'n'Easy DG Environment
# Copyright (C) 2007 Andreas Kloeckner
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.




import hedge._internal as _internal
from pytools.arithmetic_container import ArithmeticList, \
        work_with_arithmetic_containers
import pymbolic




Face = _internal.Face




# c++ fluxes debug dumping ----------------------------------------------------
def _constant_str(self):
    return "ConstantFlux(%s)" % self.value
def _normal_str(self):
    return "NormalFlux(%d)" % self.axis
def _penalty_str(self):
    return "PenaltyFlux(%s)" % self.power
def _penalty_str(self):
    return "PenaltyFlux(%s)" % self.power
def _sum_str(self):
    return "SumFlux(%s, %s)" % (self.operand1, self.operand2)
def _product_str(self):
    return "ProductFlux(%s, %s)" % (self.operand1, self.operand2)
def _negative_str(self):
    return "NegativeFlux(%s)" % self.operand
def _chained_str(self):
    #return "ChainedFlux(%s)" % self.child
    return str(self.child)

_internal.ConstantFlux.__str__ = _constant_str
_internal.NormalFlux.__str__ = _normal_str
_internal.PenaltyFlux.__str__ = _penalty_str
_internal.SumFlux.__str__ = _sum_str
_internal.ProductFlux.__str__ = _product_str
_internal.NegativeFlux.__str__ = _negative_str
_internal.ChainedFlux.__str__ = _chained_str


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
    def __init__(self, power):
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





def make_normal(dimensions):
    return ArithmeticList([Normal(i) for i in range(dimensions)])





class FluxScalarPlaceholder:
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

    def jump(self, dimensions):
        return make_normal(dimensions) * (self.int-self.ext)




class FluxVectorPlaceholder:
    def __init__(self, components=None, indices=None):
        if not (components or indices):
            raise ValueError, "either components or indices must be specified"

        if components:
            self.indices = range(components)
        else:
            self.indices = indices

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return FluxScalarPlaceholder(self.indices[idx])
        else:
            return FluxVectorPlaceholder(indices=self.indices.__getitem__(idx))

    @property
    def int(self):
        return ArithmeticList(FieldComponent(i, True) for i in self.indices)

    @property
    def ext(self):
        return ArithmeticList(FieldComponent(i, False) for i in self.indices)

    @property
    def avg(self):
        return 0.5*(self.int+self.ext)

    @property
    def jump(self):
        return dot(make_normal(len(self.indices)), self.int-self.ext)




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





class FluxNormalizationMapper(pymbolic.mapper.IdentityMapper):
    def map_constant_flux(self, expr):
        if expr.local_c == expr.neighbor_c:
            return expr.local_c
        else:
            return expr

    def map_sum(self, expr):
        from pymbolic.primitives import \
                flattened_sum, flattened_product, \
                Product, is_constant

        terms2coeff = {}

        def add_term(coeff, terms):
            terms2coeff[terms] = terms2coeff.get(terms, 0)  + coeff

        for kid in expr.children:
            kid = self.rec(kid)

            if isinstance(kid, Product):
                pkids = kid.children
            else:
                pkids = [kid]

            coeffs = []
            terms = set()
            for p_kid in pkids:
                if is_constant(p_kid):
                    coeffs.append(p_kid)
                else:
                    terms.add(p_kid)

            add_term(flattened_product(coeffs), frozenset(terms))

        return flattened_sum(
                coeffs*flattened_product(terms)
                for terms, coeffs in terms2coeff.iteritems())




    def map_product(self, expr):
        from pymbolic.primitives import flattened_product, is_constant

        constants = []
        rest = []

        for kid in expr.children:
            kid = self.rec(kid)

            if is_constant(kid):
                constants.append(kid)
            else:
                rest.append(kid)

        from operator import mul
        constant = reduce(mul, constants, 1)

        return flattened_product([constant] + rest)




        
def normalize_flux(flux):
    from pymbolic import expand, flatten
    return FluxNormalizationMapper()(flatten(expand(flux)))




class FluxDependencyMapper(pymbolic.mapper.dependency.DependencyMapper):
    def map_field_component(self, expr):
        return set([expr])

    def map_normal(self, expr):
        return set()

    def map_penalty_term(self, expr):
        return set()




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
        return reduce(lambda f1, f2: _internal.make_SumFlux(f1, f2),
                (self.rec(c) for c in expr.children))

    def map_product(self, expr):
        return reduce(lambda f1, f2: _internal.make_ProductFlux(f1, f2),
                (self.rec(c) for c in expr.children))

    def map_negation(self, expr):
        return _internal.make_NegativeFlux(self.rec(expr.child))

    def map_power(self, expr):
        base = self.rec(expr.base)
        result = base

        assert isinstance(expr.exponent, int)

        for i in range(1, expr.exponent):
            result = _internal.make_ProductFlux(result, base)

        return result

    def map_normal(self, expr):
        return _internal.NormalFlux(expr.axis)

    def map_penalty_term(self, expr):
        return _internal.PenaltyTerm(expr.power)





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




@work_with_arithmetic_containers
def compile_flux(flux):
    def compile_scalar_single_dep_flux(flux):
        return FluxCompilationMapper()(flux)
        #if not flux:
            #return None
        #else:
            #return FluxCompilationMapper()(normalize_flux(flux))

    def compile_scalar_flux(flux):
        def in_fields_cmp(a, b):
            return cmp(a.index, b.index) \
                    or cmp(a.is_local, b.is_local)

        in_fields = list(FluxDependencyMapper()(flux))
        in_fields.sort(in_fields_cmp)

        max_in_field = max(in_field.index for in_field in in_fields)

        in_derivatives = dict(
                ((in_field.index, in_field.is_local),
                normalize_flux(FluxDifferentiationMapper(in_field)(flux)))
                for in_field in in_fields)

        for i, deriv in in_derivatives.iteritems():
            if FluxDependencyMapper()(deriv):
                raise ValueError, "Flux is nonlinear in component %d" % i

        flux._compiled = []
        for in_field_idx in range(max_in_field+1):
            int = in_derivatives.get((in_field_idx, True), 0)
            ext = in_derivatives.get((in_field_idx, False), 0)

            if int or ext:
                flux._compiled.append(
                        (in_field_idx, 
                            compile_scalar_single_dep_flux(int),
                            compile_scalar_single_dep_flux(ext),))

        return flux._compiled

    try:
        return flux._compiled
    except AttributeError:
        return compile_scalar_flux(flux)
