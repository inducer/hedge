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
import pymbolic.primitives
import pymbolic.mapper.collector
import pymbolic.mapper.expander
import pymbolic.mapper.flattener
import pymbolic.mapper.substitutor
import pymbolic.mapper.constant_folder
import pymbolic.mapper.flop_counter




FluxFace = _internal.FluxFace




# python fluxes ---------------------------------------------------------------
class Flux(pymbolic.primitives.AlgebraicLeaf):
    def stringifier(self):
        return FluxStringifyMapper




class FluxScalarParameter(pymbolic.primitives.Variable):
    def __init__(self, name, is_complex=False):
        pymbolic.primitives.Variable.__init__(self, name)
        self.is_complex = is_complex

    def get_mapper_method(self, mapper):
        return mapper.map_scalar_parameter




class FieldComponent(Flux):
    def __init__(self, index, is_local):
        self.index = index
        self.is_local = is_local

    def is_equal(self, other):
        return (isinstance(other, FieldComponent) 
                and self.index == other.index
                and self.is_local == other.is_local
                )

    def __getinitargs__(self):
        return self.index, self.is_local

    def get_hash(self):
        return hash((
                self.__class__,
                self.index,
                self.is_local))

    def get_mapper_method(self, mapper):
        return mapper.map_field_component




class Normal(Flux):
    def __init__(self, axis):
        self.axis = axis

    def __getinitargs__(self):
        return self.axis,

    def is_equal(self, other):
        return isinstance(other, Normal) and self.axis == other.axis

    def get_hash(self):
        return hash((
                self.__class__,
                self.axis))

    def get_mapper_method(self, mapper):
        return mapper.map_normal




class PenaltyTerm(Flux):
    def __init__(self, power=1):
        self.power = power

    def __getinitargs__(self):
        return self.power,

    def is_equal(self, other):
        return isinstance(other, PenaltyTerm) and self.power == other.power

    def get_hash(self):
        return hash((
                self.__class__,
                self.power))

    def get_mapper_method(self, mapper):
        return mapper.map_penalty_term




class IfPositive(Flux):
    def __init__(self, criterion, then, else_):
        self.criterion = criterion
        self.then = then
        self.else_ = else_

    def __getinitargs__(self):
        return self.criterion, self.then, self.else_

    def is_equal(self, other):
        return (isinstance(other, IfPositive)
                and self.criterion == other.criterion
                and self.then == other.then
                and self.else_ == other.else_)

    def get_hash(self):
        return hash((
                self.__class__,
                self.criterion,
                self.then,
                self.else_))

    def get_mapper_method(self, mapper):
        return mapper.map_if_positive




class FluxFunctionSymbol(pymbolic.primitives.FunctionSymbol):
    pass

class Abs(FluxFunctionSymbol):
    arg_count = 1

class Max(FluxFunctionSymbol):
    arg_count = 2

class Min(FluxFunctionSymbol):
    arg_count = 2

flux_abs = Abs()
flux_max = Max()
flux_min = Min()



def norm(v):
    return numpy.dot(v, v)**0.5





def make_normal(dimensions):
    return numpy.array([Normal(i) for i in range(dimensions)], dtype=object)





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
        if not (components is not None or scalars is not None):
            raise ValueError, "either components or scalars must be specified"
        if components is not None and scalars is not None:
            raise ValueError, "only one of components and scalars may be specified"

        # make them arrays for the better indexing
        if components:
            self.scalars = numpy.array([
                    FluxScalarPlaceholder(i) 
                    for i in range(components)])
        else:
            self.scalars = numpy.array(scalars)

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
class FluxIdentityMapperMixin(object):
    def map_field_component(self, expr):
        return expr

    def map_normal(self, expr):
        return expr

    def map_penalty_term(self, expr):
        return expr.__class__(self.rec(expr.power))

    def map_if_positive(self, expr):
        return expr.__class__(
                self.rec(expr.criterion),
                self.rec(expr.then),
                self.rec(expr.else_),
                )

    def map_scalar_parameter(self, expr):
        return expr




class FluxIdentityMapper(
        pymbolic.mapper.IdentityMapper, 
        FluxIdentityMapperMixin):
    pass

class FluxSubstitutionMapper(pymbolic.mapper.substitutor.SubstitutionMapper,
        FluxIdentityMapperMixin):
    def map_field_component(self, expr):
        result = self.subst_func(expr)
        if result is not None:
            return result
        else:
            return expr

    map_normal = map_field_component




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




class FluxFlattenMapper(pymbolic.mapper.flattener.FlattenMapper,
        FluxIdentityMapperMixin):
    pass





class FluxDependencyMapper(pymbolic.mapper.dependency.DependencyMapper):
    def map_field_component(self, expr):
        return set([expr])

    def map_normal(self, expr):
        return set()

    def map_penalty_term(self, expr):
        return set()

    def map_if_positive(self, expr):
        return self.rec(expr.criterion) | self.rec(expr.then) | self.rec(expr.else_)

    def map_scalar_parameter(self, expr):
        return set([expr])





class FluxTermCollector(pymbolic.mapper.collector.TermCollector,
        FluxIdentityMapperMixin):
    pass




class FluxAllDependencyMapper(FluxDependencyMapper):
    def map_normal(self, expr):
        return set([expr])

    def map_penalty_term(self, expr):
        return set([expr])




class FluxNormalizationMapper(pymbolic.mapper.collector.TermCollector,
        FluxIdentityMapperMixin):
    def get_dependencies(self, expr):
        return FluxAllDependencyMapper()(expr)

    def map_constant_flux(self, expr):
        if expr.local_c == expr.neighbor_c:
            return expr.local_c
        else:
            return expr




class FluxCCFMapper(pymbolic.mapper.constant_folder.CommutativeConstantFoldingMapper,
        FluxIdentityMapperMixin):
    def is_constant(self, expr):
        return not bool(FluxAllDependencyMapper()(expr))




class FluxExpandMapper(pymbolic.mapper.expander.ExpandMapper,
        FluxIdentityMapperMixin):
    def __init__(self):
        pymbolic.mapper.expander.ExpandMapper.__init__(self,
                FluxNormalizationMapper())




class FluxFlipper(FluxIdentityMapper):
    def map_normal(self, expr):
        return -expr

    def map_field_component(self, expr):
        return expr.__class__(expr.index, not expr.is_local)





class FluxFlopCounter(pymbolic.mapper.flop_counter.FlopCounter):
    def map_penalty_term(self, expr):
        return 0

    def map_normal(self, expr):
        return 0

    def map_field_component(self, expr):
        return 0

    def map_if_positive(self, expr):
        return self.rec(expr.criterion) + max(
                self.rec(expr.then),
                self.rec(expr.else_))

    def map_function_symbol(self, expr):
        return 1

    def map_scalar_parameter(self, expr):
        return 0





def normalize_flux(flux):
    return FluxCCFMapper()(FluxNormalizationMapper()(
        FluxFlattenMapper()(FluxExpandMapper()(flux))))
