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
which_faces = _internal.which_faces




# c++ fluxes debug dumping ----------------------------------------------------
def _constant_str(self):
    return "ConstantFlux(%s, %s)" % (self.local_c, self.neighbor_c)
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
class Flux(pymbolic.primitives.AlgebraicLeaf):
    def stringify(self, enclosing_prec, use_repr_for_constants=False):
        return FluxStringifyMapper(use_repr_for_constants)(self, enclosing_prec)




class ConstantFlux(Flux):
    def __init__(self, local_c, neighbor_c=None):
        if neighbor_c is None:
            neighbor_c = local_c
        self.local_c = local_c
        self.neighbor_c = neighbor_c

    def __add__(self, other):
        from pymbolic.primitives import is_constant

        if isinstance(other, ConstantFlux):
            return ConstantFlux(
                    self.local_c + other.local_c,
                    self.neighbor_c + other.neighbor_c)
        elif is_constant(other):
            return ConstantFlux( self.local_c + other, self.neighbor_c + other)
        else:
            return Flux.__add__(self, other)

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, ConstantFlux):
            return ConstantFlux(
                    self.local_c - other.local_c,
                    self.neighbor_c - other.neighbor_c)
        else:
            return Flux.__sub__(self, other)

    def __rsub__(self, other):
        if isinstance(other, ConstantFlux):
            return ConstantFlux(
                    other.local_c - self.local_c,
                    other.neighbor_c - self.neighbor_c)
        else:
            return Flux.__rsub__(self, other)

    def __neg__(self, other):
        return ConstantFlux(- self.local_c, - self.neighbor_c)

    def __mul__(self, other):
        try:
            otherfloat = float(other)
        except:
            return Flux.__mul__(self, other)
        else:
            return ConstantFlux(otherfloat*self.local_c, otherfloat*self.neighbor_c)

    def __eq__(self, other):
        return (isinstance(other, ConstantFlux) and
                self.local_c == other.local_c and
                self.neighbor_c == other.neighbor_c)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return 0xdedbeef ^ hash(self.local_c) ^ hash(self.neighbor_c)

    def __getinitargs__(self):
        return self.local_c, self.neighbor_c

    def get_mapper_method(self, mapper):
        return mapper.map_constant_flux




class NormalFlux(Flux):
    def __init__(self, axis):
        self.axis = axis

    def __getinitargs__(self):
        return self.axis,

    def __eq__(self, other):
        return isinstance(other, NormalFlux) and self.axis == other.axis

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return 0xa931ff ^ hash(self.axis)

    def get_mapper_method(self, mapper):
        return mapper.map_normal_flux



class PenaltyFlux(Flux):
    def __init__(self, power):
        self.power = power

    def __getinitargs__(self):
        return self.power,

    def __eq__(self, other):
        return isinstance(other, PenaltyFlux) and self.power == other.power

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return 0x9f194 ^ hash(self.power)

    def get_mapper_method(self, mapper):
        return mapper.map_penalty_flux





def make_normal(dim):
    return ArithmeticList([NormalFlux(i) for i in range(dim)])




zero = ConstantFlux(0)
local = ConstantFlux(1, 0)
neighbor = ConstantFlux(0, 1)
average = ConstantFlux(0.5, 0.5)




class FluxIdentityMapper(pymbolic.mapper.IdentityMapper):
    def map_constant_flux(self, expr, *args, **kwargs):
        return expr.__class__(expr.local_c, expr.neighbor_c)

    def map_normal_flux(self, expr, *args, **kwargs):
        return expr.__class__(expr.axis)

    def map_penalty_flux(self, expr, *args, **kwargs):
        return expr.__class__(expr.power)




class FluxNormalizationMapper(FluxIdentityMapper):
    pass




class FluxStringifyMapper(pymbolic.mapper.stringifier.StringifyMapper):
    def map_constant_flux(self, expr, enclosing_prec):
        return "Const(%s, %s)" % (expr.local_c, expr.neighbor_c)

    def map_normal_flux(self, expr, enclosing_prec):
        return "Normal(%d)" % expr.axis

    def map_penalty_flux(self, expr, enclosing_prec):
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
                if is_constant(p_kid) or isinstance(p_kid, ConstantFlux):
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
        c_fluxes = []
        rest = []

        for kid in expr.children:
            kid = self.rec(kid)

            if is_constant(kid):
                constants.append(kid)
            elif isinstance(kid, ConstantFlux):
                c_fluxes.append(kid)
            else:
                rest.append(kid)

        from operator import mul
        constant = reduce(mul, constants, 1)

        if c_fluxes:
            if len(c_fluxes) > 1:
                raise RuntimeError, "Multiplying two nontrivial constant fluxes is not going to do what you want"
            c_flux = c_fluxes[0]
            c_flux *= constant
            return flattened_product([c_flux] + rest)
        else:
            return flattened_product([constant] + rest)




        
def normalize_flux(flux):
    from pymbolic import expand, flatten
    return FluxNormalizationMapper()(flatten(expand(flux)))




class FluxCompilationMapper(pymbolic.mapper.RecursiveMapper):
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

    def map_constant_flux(self, expr):
        return _internal.ConstantFlux(expr.local_c, expr.neighbor_c)

    def map_normal_flux(self, expr):
        return _internal.NormalFlux(expr.axis)

    def map_penalty_flux(self, expr):
        return _internal.PenaltyFlux(expr.power)




def compile_flux(flux):
    try:
        return flux._compiled
    except AttributeError:
        flux._compiled = FluxCompilationMapper()(normalize_flux(flux))
        return flux._compiled


