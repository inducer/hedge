"""Lazy evaluation support infastructure."""

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
import pymbolic.mapper.stringifier
import pymbolic.mapper.evaluator
import pymbolic.mapper.dependency
import pymbolic.mapper.constant_folder
import hedge.mesh




def make_common_subexpression(fields): 
    from hedge.tools import with_object_array_or_scalar
    from pymbolic.primitives import CommonSubexpression
    return with_object_array_or_scalar(CommonSubexpression, fields)




class Field(pymbolic.primitives.Variable):
    pass




def make_field(var_or_string):
    if not isinstance(var_or_string, pymbolic.primitives.Expression):
        return Field(var_or_string)
    else:
        return var_or_string




class BoundaryNormalComponent(pymbolic.primitives.AlgebraicLeaf):
    def __init__(self, tag, axis):
        self.tag = tag
        self.axis = axis

    def stringifier(self):
        return StringifyMapper

    def get_hash(self):
        return hash((self.__class__, self.tag, self.axis))

    def is_equal(self, other):
        return (other.__class__ == self.__class__
                and other.tag == self.tag
                and other.axis == self.axis)

    def get_mapper_method(self, mapper): 
        return mapper.map_normal_component




def make_normal(tag, dimensions):
    return numpy.array([BoundaryNormalComponent(tag, i) 
        for i in range(dimensions)], dtype=object)





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




# operators -------------------------------------------------------------------
class Operator(pymbolic.primitives.Leaf):
    def stringifier(self):
        return StringifyMapper

    def __call__(self, *args, **kwargs):
        # prevent lazy-eval semantics from kicking in
        raise RuntimeError, "symbolic operators are not callable"

    def apply(self, discr, field):
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

class DifferentiationOperator(DiffOperatorBase):
    @staticmethod
    def matrices(element_group): 
        return element_group.differentiation_matrices

    @staticmethod
    def coefficients(element_group): 
        return element_group.diff_coefficients

    def get_mapper_method(self, mapper): 
        return mapper.map_diff

class MInvSTOperator(DiffOperatorBase):
    @staticmethod
    def matrices(element_group): 
        return element_group.minv_st

    @staticmethod
    def coefficients(element_group): 
        return element_group.diff_coefficients

    def get_mapper_method(self, mapper): 
        return mapper.map_minv_st

class StiffnessOperator(DiffOperatorBase):
    @staticmethod
    def matrices(element_group): 
        return element_group.stiffness_matrices

    @staticmethod
    def coefficients(element_group): 
        return element_group.stiffness_coefficients

    def get_mapper_method(self, mapper): 
        return mapper.map_stiffness

class StiffnessTOperator(DiffOperatorBase):
    @staticmethod
    def matrices(element_group): 
        return element_group.stiffness_t_matrices

    @staticmethod
    def coefficients(element_group): 
        return element_group.stiffness_coefficients

    def get_mapper_method(self, mapper): 
        return mapper.map_stiffness_t





def DiffOperatorVector(els):
    from hedge.tools import join_fields
    return join_fields(*els)
    

    

# mass operators --------------------------------------------------------------
class MassOperatorBase(StatelessOperator):
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

    def __mul__(self, arg):
        from hedge.tools import is_obj_array
        if isinstance(arg, Field) or is_obj_array(arg):
            return OperatorBinding(self, arg)
        else:
            return Operator.__mul__(self, arg)




class FluxOperator(FluxOperatorBase):
    def get_mapper_method(self, mapper): 
        return mapper.map_flux



class LiftingFluxOperator(FluxOperatorBase):
    def get_mapper_method(self, mapper): 
        return mapper.map_lift



class VectorFluxOperator(object):
    def __init__(self, fluxes):
        self.fluxes = fluxes

    def __mul__(self, arg):
        if isinstance(arg, int) and arg == 0:
            return 0
        from hedge.tools import make_obj_array
        return make_obj_array(
                [OperatorBinding(FluxOperator(f), arg)
                    for f in self.fluxes])
                




# other parts of an operator template -----------------------------------------
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
        




def pair_with_boundary(field, bfield, tag=hedge.mesh.TAG_ALL):
    if tag is hedge.mesh.TAG_NONE:
        return 0
    else:
        return BoundaryPair(field, bfield, tag)




# convenience functions -------------------------------------------------------
def make_vector_field(name, components):
    from hedge.tools import join_fields
    vfld = pymbolic.primitives.Variable(name)
    return join_fields(*[vfld[i] for i in range(components)])




def get_flux_operator(flux):
    """Return a flux operator that can be multiplied with
    a volume field to obtain the lifted interior fluxes
    or with a boundary pair to obtain the lifted boundary
    flux.
    """
    from hedge.tools import is_obj_array

    if is_obj_array(flux):
        return VectorFluxOperator(flux)
    else:
        return FluxOperator(flux)




def make_nabla(dim):
    from hedge.tools import make_obj_array
    return make_obj_array(
            [DifferentiationOperator(i) for i in range(dim)])

def make_minv_stiffness_t(dim):
    from hedge.tools import make_obj_array
    return make_obj_array(
        [MInvSTOperator(i) for i in range(dim)])

def make_stiffness(dim):
    from hedge.tools import make_obj_array
    return make_obj_array(
        [StiffnessOperator(i) for i in range(dim)])

def make_stiffness_t(dim):
    from hedge.tools import make_obj_array
    return make_obj_array(
        [StiffnessTOperator(i) for i in range(dim)])




# mappers ---------------------------------------------------------------------
class LocalOpReducerMixin(object):
    """Reduces calls to mapper methods for all local differentiation
    operators to a single mapper method, and likewise for mass 
    operators.
    """
    def map_diff(self, expr, *args, **kwargs):
        return self.map_diff_base(expr, *args, **kwargs)

    def map_minv_st(self, expr, *args, **kwargs):
        return self.map_diff_base(expr, *args, **kwargs)

    def map_stiffness(self, expr, *args, **kwargs):
        return self.map_diff_base(expr, *args, **kwargs)

    def map_stiffness_t(self, expr, *args, **kwargs):
        return self.map_diff_base(expr, *args, **kwargs)

    def map_mass(self, expr, *args, **kwargs):
        return self.map_mass_base(expr, *args, **kwargs)

    def map_inverse_mass(self, expr, *args, **kwargs):
        return self.map_mass_base(expr, *args, **kwargs)




class FluxOpReducerMixin(object):
    """Reduces calls to mapper methods for all flux 
    operators to a smaller number of mapper methods.
    """
    def map_flux(self, expr, *args, **kwargs):
        return self.map_flux_base(expr, *args, **kwargs)

    def map_lift(self, expr, *args, **kwargs):
        return self.map_flux_base(expr, *args, **kwargs)




class OperatorReducerMixin(LocalOpReducerMixin, FluxOpReducerMixin):
    """Reduces calls to *any* operator mapping function to just one."""
    def map_diff_base(self, expr, *args, **kwargs):
        return self.map_operator(expr, *args, **kwargs)

    map_mass_base = map_diff_base
    map_flux_base = map_diff_base
    map_elementwise_max = map_diff_base
    map_boundarize = map_diff_base
    map_flux_exchange = map_diff_base




class CombineMapperMixin(object):
    def map_operator_binding(self, expr):
        return self.combine([self.rec(expr.op), self.rec(expr.field)])

    def map_boundary_pair(self, expr):
        return self.combine([self.rec(expr.field), self.rec(expr.bfield)])




class CombineMapper(CombineMapperMixin, pymbolic.mapper.CombineMapper):
    pass




class IdentityMapperMixin(LocalOpReducerMixin, FluxOpReducerMixin):
    def map_operator_binding(self, expr, *args, **kwargs):
        assert not isinstance(self, BoundOpMapperMixin), \
                "IdentityMapper instances cannot be combined with " \
                "the BoundOpMapperMixin"

        return expr.__class__(
                self.rec(expr.op, *args, **kwargs),
                self.rec(expr.field, *args, **kwargs))

    def map_boundary_pair(self, expr, *args, **kwargs):
        assert not isinstance(self, BoundOpMapperMixin), \
                "IdentityMapper instances cannot be combined with " \
                "the BoundOpMapperMixin"

        return expr.__class__(
                self.rec(expr.field, *args, **kwargs),
                self.rec(expr.bfield, *args, **kwargs),
                expr.tag)

    def map_mass_base(self, expr, *args, **kwargs):
        assert not isinstance(self, BoundOpMapperMixin), \
                "IdentityMapper instances cannot be combined with " \
                "the BoundOpMapperMixin"

        # it's a leaf--no changing children
        return expr

    map_diff_base = map_mass_base
    map_flux_base = map_mass_base
    map_elementwise_max = map_mass_base
    map_boundarize = map_mass_base
    map_flux_exchange = map_mass_base

    map_normal_component = map_mass_base




class DependencyMapper(
        CombineMapperMixin, 
        pymbolic.mapper.dependency.DependencyMapper, 
        OperatorReducerMixin):
    def __init__(self, 
            include_operator_bindings=True,
            composite_leaves=None,
            **kwargs):
        if composite_leaves == False:
            include_operator_bindings = False
        if composite_leaves == True:
            include_operator_bindings = True

        pymbolic.mapper.dependency.DependencyMapper.__init__(self,
                composite_leaves=composite_leaves, **kwargs)

        self.include_operator_bindings = include_operator_bindings

    def map_operator_binding(self, expr):
        if self.include_operator_bindings:
            return set([expr])
        else:
            return CombineMapperMixin.map_operator_binding(self, expr)

    def map_operator(self, expr):
        return set()

    def map_normal_component(self, expr):
        return set()



class CommutativeConstantFoldingMapper(
        pymbolic.mapper.constant_folder.CommutativeConstantFoldingMapper,
        IdentityMapperMixin):
    def is_constant(self, expr):
        return not bool(DependencyMapper()(expr))




class IdentityMapper(
        IdentityMapperMixin, 
        pymbolic.mapper.IdentityMapper):
    pass





class StringifyMapper(pymbolic.mapper.stringifier.StringifyMapper):
    def map_boundary_pair(self, expr, enclosing_prec):
        return "BPair(%s, %s, %s)" % (expr.field, expr.bfield, repr(expr.tag))

    def map_diff(self, expr, enclosing_prec):
        return "Diff%d" % expr.xyz_axis

    def map_minv_st(self, expr, enclosing_prec):
        return "MInvST%d" % expr.xyz_axis

    def map_stiffness(self, expr, enclosing_prec):
        return "Stiff%d" % expr.xyz_axis

    def map_stiffness_t(self, expr, enclosing_prec):
        return "StiffT%d" % expr.xyz_axis

    def map_mass(self, expr, enclosing_prec):
        return "M"

    def map_inverse_mass(self, expr, enclosing_prec):
        return "InvM"

    def map_flux(self, expr, enclosing_prec):
        return "Flux(%s)" % expr.flux

    def map_lift(self, expr, enclosing_prec):
        return "Lift(%s)" % expr.flux

    def map_elementwise_max(self, expr, enclosing_prec):
        return "ElWMax"

    def map_boundarize(self, expr, enclosing_prec):
        return "Boundarize<tag=%s>" % expr.tag

    def map_flux_exchange(self, expr, enclosing_prec):
        return "FExch<idx=%d,rank=%d>" % (expr.index, expr.rank)

    def map_normal_component(self, expr, enclosing_prec):
        return "Normal<tag=%s>[%d]" % (expr.tag, expr.axis)

    def map_operator_binding(self, expr, enclosing_prec):
        return "<%s>(%s)" % (expr.op, expr.field)





class NoCSEStringifyMapper(StringifyMapper):
    def map_common_subexpression(self, expr, enclosing_prec):
        return self.rec(expr.child, enclosing_prec)




class BoundOpMapperMixin(object):
    def map_operator_binding(self, expr, *args, **kwargs):
        return expr.op.get_mapper_method(self)(expr.op, expr.field, *args, **kwargs)




class EmptyFluxKiller(IdentityMapper):
    def __init__(self, discr):
        IdentityMapper.__init__(self)
        self.discr = discr

    def map_operator_binding(self, expr):
        if (isinstance(expr.op, (
            FluxOperatorBase,
            LiftingFluxOperator)) 
            and 
            isinstance(expr.field, BoundaryPair)
            and
            len(self.discr.get_boundary(expr.field.tag).nodes) == 0):
            return 0
        else:
            return IdentityMapper.map_operator_binding(self, expr)



        
class OperatorBinder(IdentityMapper):
    def map_product(self, expr):
        if len(expr.children) == 0:
            return expr

        from pymbolic.primitives import flattened_product
        first = expr.children[0]
        if isinstance(first, Operator):
            return OperatorBinding(first, 
                    self.rec(flattened_product(expr.children[1:])))
        else:
            return first * self.rec(flattened_product(expr.children[1:]))




class _InnerInverseMassContractor(pymbolic.mapper.RecursiveMapper):
    def map_constant(self, expr):
        return OperatorBinding(
                InverseMassOperator(),
                expr)

    def map_algebraic_leaf(self, expr):
        return OperatorBinding(
                InverseMassOperator(),
                expr)

    def map_operator_binding(self, binding):
        if isinstance(binding.op, MassOperator):
            return binding.field
        elif isinstance(binding.op, StiffnessOperator):
            return OperatorBinding(
                    DifferentiationOperator(binding.op.xyz_axis),
                    binding.field)
        elif isinstance(binding.op, StiffnessTOperator):
            return OperatorBinding(
                    MInvSTOperator(binding.op.xyz_axis),
                    binding.field)
        elif isinstance(binding.op, FluxOperator):
            return OperatorBinding(
                    LiftingFluxOperator(binding.op.flux),
                    binding.field)
        else:
            return OperatorBinding(
                InverseMassOperator(),
                binding)

    def map_sum(self, expr):
        return expr.__class__(tuple(self.rec(child) for child in expr.children))

    def map_product(self, expr):
        def is_scalar(expr):
            return isinstance(expr, (int, float, complex))

        from pytools import len_iterable
        nonscalar_count = len_iterable(ch 
                for ch in expr.children
                if not is_scalar(ch))

        if nonscalar_count > 1:
            # too complicated, don't touch it
            return expr
        else:
            def do_map(expr):
                if is_scalar(expr):
                    return expr
                else:
                    return self.rec(expr)
            return expr.__class__(tuple(
                do_map(child) for child in expr.children))




        
class InverseMassContractor(IdentityMapper):
    # assumes all operators to be bound

    def map_boundary_pair(self, bp):
        return BoundaryPair(self.rec(bp.field), self.rec(bp.bfield), bp.tag)

    def map_operator_binding(self, binding):
        # we only care about bindings of inverse mass operators
        if not isinstance(binding.op, InverseMassOperator):
            return binding.__class__(binding.op,
                    self.rec(binding.field))
        else:
            return  _InnerInverseMassContractor()(binding.field)




# BC-to-flux rewriting --------------------------------------------------------
class BCToFluxRewriter(IdentityMapper):
    """Operates on L{FluxOperator} instances bound to L{BoundaryPair}s. If the
    boundary pair's C{bfield} is an expression of what's available in the
    C{field}, we can avoid fetching the data for the explicit boundary
    condition and just substitute the C{bfield} expression into the flux. This
    mapper does exactly that.  
    """

    def map_operator_binding(self, expr):
        if not (isinstance(expr.op, FluxOperator)
                and isinstance(expr.field, BoundaryPair)):
            return IdentityMapper.map_operator_binding(self, expr)

        bpair = expr.field
        vol_field = bpair.field
        bdry_field = bpair.bfield
        flux = expr.op.flux

        bdry_dependencies = DependencyMapper(
                    include_calls="descend_args",
                    include_operator_bindings=True)(bdry_field)
        
        vol_dependencies = DependencyMapper(
                include_operator_bindings=True)(vol_field)

        vol_bdry_intersection = bdry_dependencies & vol_dependencies
        if vol_bdry_intersection:
            raise RuntimeError("Variables are being used as both "
                    "boundary and volume quantities: %s" 
                    % ", ".join(str(v) for v in vol_bdry_intersection))
  
        # Step 1: Find maximal flux-evaluable subexpression of bounary field
        # in given BoundaryPair.

        class MaxBoundaryFluxEvaluableExpressionFinder(
                IdentityMapper, OperatorReducerMixin):
            def __init__(self, vol_expr_list):
                self.vol_expr_list = vol_expr_list
                self.vol_expr_to_idx = dict((vol_expr, idx) 
                        for idx, vol_expr in enumerate(vol_expr_list))

                self.bdry_expr_list = []
                self.bdry_expr_to_idx = {}

            def register_boundary_expr(self, expr):
                try:
                    return self.bdry_expr_to_idx[expr]
                except KeyError:
                    idx = len(self.bdry_expr_to_idx)
                    self.bdry_expr_to_idx[expr] = idx
                    self.bdry_expr_list.append(expr)
                    return idx

            def register_volume_expr(self, expr):
                try:
                    return self.vol_expr_to_idx[expr]
                except KeyError:
                    idx = len(self.vol_expr_to_idx)
                    self.vol_expr_to_idx[expr] = idx
                    self.vol_expr_list.append(expr)
                    return idx

            def map_normal_component(self, expr):
                if expr.tag != bpair.tag:
                    raise RuntimeError("BoundaryNormalComponent and BoundaryPair "
                            "do not agree about boundary tag: %s vs %s" 
                            % (expr.tag, bpair.tag))

                from hedge.flux import Normal
                return Normal(expr.axis)

            def map_variable(self, expr):
                from hedge.flux import FieldComponent
                return FieldComponent(
                        self.register_boundary_expr(expr), 
                        is_local=False)

            map_subscript = map_variable

            def map_operator_binding(self, expr):
                from hedge.flux import FieldComponent
                if isinstance(expr.op, BoundarizeOperator):
                    if expr.op.tag != bpair.tag:
                        raise RuntimeError("BoundarizeOperator and BoundaryPair "
                                "do not agree about boundary tag: %s vs %s" 
                                % (expr.op.tag, bpair.tag))

                    return FieldComponent(
                            self.register_volume_expr(expr.field), 
                            is_local=True)
                elif isinstance(expr.op, FluxExchangeOperator):
                    from hedge.mesh import TAG_RANK_BOUNDARY
                    op_tag = TAG_RANK_BOUNDARY(expr.op.rank)
                    if bpair.tag != op_tag:
                        raise RuntimeError("BoundarizeOperator and FluxExchangeOperator "
                                "do not agree about boundary tag: %s vs %s" 
                                % (op_tag, bpair.tag))
                    return FieldComponent(
                            self.register_boundary_expr(expr), 
                            is_local=False)
                else:
                    raise RuntimeError("Found '%s' in a boundary term. "
                            "To the best of my knowledge, no hedge operator applies "
                            "directly to boundary data, so this is likely in error."
                            % expr.op)

        from hedge.tools import is_obj_array
        if not is_obj_array(vol_field):
            vol_field = [vol_field]

        mbfeef = MaxBoundaryFluxEvaluableExpressionFinder(vol_field)
        new_bdry_field = mbfeef(bdry_field)

        # Step II: Substitute the new_bdry_field into the flux.
        from hedge.flux import FluxSubstitutionMapper, FieldComponent

        def sub_bdry_into_flux(expr):
            if isinstance(expr, FieldComponent) and not expr.is_local:
                if expr.index == 0 and not is_obj_array(bdry_field):
                    return new_bdry_field
                else:
                    return new_bdry_field[expr.index]
            else:
                return None

        new_flux = FluxSubstitutionMapper(
                sub_bdry_into_flux)(flux)

        from hedge.tools import is_zero
        if is_zero(new_flux):
            return 0
        else:
            return OperatorBinding(
                    FluxOperator(new_flux), BoundaryPair(
                        numpy.array(mbfeef.vol_expr_list, dtype=object), 
                        numpy.array(mbfeef.bdry_expr_list, dtype=object), 
                        bpair.tag))




# collecting ------------------------------------------------------------------
class CollectorMixin(LocalOpReducerMixin, FluxOpReducerMixin):
    def combine(self, values):
        from pytools import flatten
        return set(flatten(values))

    def map_constant(self, bpair):
        return set()

    def map_mass_base(self, expr):
        return set()
    
    def map_diff_base(self, expr):
        return set()

    def map_flux_base(self, expr):
        return set()

    def map_variable(self, expr):
        return set()

    def map_elementwise_max(self, expr, *args, **kwargs):
        return set()

    def map_normal_component(self, expr):
        return set()

    



class FluxCollector(CollectorMixin, CombineMapper):
    def map_operator_binding(self, expr):
        if isinstance(expr.op, (
            FluxOperatorBase)):
            result = set([expr])
        else:
            result = set()

        return result | self.rec(expr.field)




class BoundaryTagCollector(CollectorMixin, CombineMapper):
    def map_boundary_pair(self, bpair):
        return set([bpair.tag])




class BoundOperatorCollector(CollectorMixin, CombineMapper):
    def __init__(self, op_class):
        self.op_class = op_class

    def map_operator_binding(self, expr):
        if isinstance(expr.op, self.op_class):
            result = set([expr])
        else:
            result = set()

        return result | self.rec(expr.field)



# evaluation ------------------------------------------------------------------
class Evaluator(pymbolic.mapper.evaluator.EvaluationMapper):
    def map_boundary_pair(self, bp):
        return BoundaryPair(self.rec(bp.field), self.rec(bp.bfield), bp.tag)

