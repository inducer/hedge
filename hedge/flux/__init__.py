"""Building blocks for flux computation. Flux compilation."""

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
import pymbolic.primitives
import pymbolic.mapper.collector
import pymbolic.mapper.expander
import pymbolic.mapper.flattener
import pymbolic.mapper.substitutor
import pymbolic.mapper.constant_folder
import pymbolic.mapper.flop_counter




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
    def __init__(self, index, is_interior):
        self.index = index
        self.is_interior = is_interior

    def is_equal(self, other):
        return (isinstance(other, FieldComponent)
                and self.index == other.index
                and self.is_interior == other.is_interior
                )

    def __getinitargs__(self):
        return self.index, self.is_interior

    def get_hash(self):
        return hash((
                self.__class__,
                self.index,
                self.is_interior))

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




class _StatelessFlux(Flux):
    def __getinitargs__(self):
        return ()

    def is_equal(self, other):
        return isinstance(other, self.__class__)

    def get_hash(self):
        return hash(self.__class__)




class _SidedFlux(Flux):
    def __init__(self, is_interior=True):
        self.is_interior = is_interior

    def __getinitargs__(self):
        return (is_interior,)

    def is_equal(self, other):
        return (isinstance(other, self.__class__)
                and self.is_interior == other.is_interior)

    def get_hash(self):
        return hash((self.__class__, self.is_interior))




class FaceJacobian(_StatelessFlux):
    def get_mapper_method(self, mapper):
        return mapper.map_face_jacobian

class ElementJacobian(_SidedFlux):
    def get_mapper_method(self, mapper):
        return mapper.map_element_jacobian

class ElementOrder(_SidedFlux):
    def get_mapper_method(self, mapper):
        return mapper.map_element_order

class LocalMeshSize(_StatelessFlux):
    def get_mapper_method(self, mapper):
        return mapper.map_local_mesh_size

def make_penalty_term(power=1):
    from pymbolic.primitives import CommonSubexpression
    return CommonSubexpression(
            (ElementOrder()**2/LocalMeshSize())**power,
            "penalty")




PenaltyTerm = make_penalty_term




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





class FluxConstantPlaceholder(object):
    def __init__(self, constant):
        self.constant = constant

    @property
    def int(self):
        return self.constant

    @property
    def ext(self):
        return self.constant

    @property
    def avg(self):
        return self.constant




class FluxZeroPlaceholder(FluxConstantPlaceholder):
    def __init__(self):
        FluxConstantPlaceholder.__init__(self, 0)





class FluxScalarPlaceholder(object):
    def __init__(self, component=0):
        self.component = component

    def __str__(self):
        return "FSP(%d)" % self.component

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
            raise ValueError("either components or scalars must be specified")
        if components is not None and scalars is not None:
            raise ValueError("only one of components and scalars "
                    "may be specified")

        # make them arrays for the better indexing
        if components:
            self.scalars = numpy.array([
                    FluxScalarPlaceholder(i)
                    for i in range(components)])
        else:
            self.scalars = numpy.array(scalars)

    def __len__(self):
        return len(self.scalars)

    def __str__(self):
        return "%s(%s)" % (self.__class__.__name__, self.scalars)

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
        # a leaf
        return expr

    map_element_jacobian = map_normal
    map_face_jacobian = map_normal
    map_element_order = map_normal
    map_local_mesh_size = map_normal
    map_c_function = map_normal

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
        if expr.is_interior:
            return "Int[%d]" % expr.index
        else:
            return "Ext[%d]" % expr.index

    def map_normal(self, expr, enclosing_prec):
        return "Normal(%d)" % expr.axis

    def map_element_jacobian(self, expr, enclosing_prec):
        return "ElJac"

    def map_face_jacobian(self, expr, enclosing_prec):
        return "FJac"

    def map_element_order(self, expr, enclosing_prec):
        return "N"

    def map_local_mesh_size(self, expr, enclosing_prec):
        return "h"




class PrettyFluxStringifyMapper(
        pymbolic.mapper.stringifier.CSESplittingStringifyMapperMixin,
        FluxStringifyMapper):
    pass



class PrettyFluxStringifyMapper(
        pymbolic.mapper.stringifier.CSESplittingStringifyMapperMixin,
        FluxStringifyMapper):
    pass



class FluxFlattenMapper(pymbolic.mapper.flattener.FlattenMapper,
        FluxIdentityMapperMixin):
    pass





class FluxDependencyMapper(pymbolic.mapper.dependency.DependencyMapper):
    def map_field_component(self, expr):
        return set([expr])

    def map_normal(self, expr):
        return set()

    map_element_jacobian = map_normal
    map_face_jacobian = map_normal
    map_element_order = map_normal
    map_local_mesh_size = map_normal
    map_c_function = map_normal

    def map_scalar_parameter(self, expr):
        return set([expr])





class FluxTermCollector(pymbolic.mapper.collector.TermCollector,
        FluxIdentityMapperMixin):
    pass




class FluxAllDependencyMapper(FluxDependencyMapper):
    def map_normal(self, expr):
        return set([expr])

    def map_element_order(self, expr):
        return set([expr])

    def map_local_mesh_size(self, expr):
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
    def map_field_component(self, expr):
        return expr.__class__(expr.index, not expr.is_interior)

    def map_normal(self, expr):
        return -expr

    def map_element_jacobian(self, expr):
        return expr.__class__(not expr.is_interior)

    map_element_order = map_element_jacobian






class FluxFlopCounter(pymbolic.mapper.flop_counter.FlopCounter):
    def map_normal(self, expr):
        return 0

    map_element_jacobian = map_normal
    map_face_jacobian = map_normal
    map_element_order = map_normal
    map_local_mesh_size = map_normal

    def map_field_component(self, expr):
        return 0

    def map_function_symbol(self, expr):
        return 1

    def map_c_function(self, expr):
        return 1

    def map_scalar_parameter(self, expr):
        return 0
