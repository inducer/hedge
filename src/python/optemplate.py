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




import pymbolic.primitives
import pymbolic.mapper.stringifier
import pymbolic.mapper.evaluator
import pymbolic.mapper.dependency
import pymbolic.mapper.constant_folder
import hedge.mesh




def make_common_subexpression(expr):
    from pymbolic.primitives import CommonSubexpression

    from hedge.tools import is_obj_array, make_obj_array
    if is_obj_array(expr):
        from hedge.tools import make_obj_array
        return make_obj_array([
            CommonSubexpression(e_i) for e_i in expr])
    else:
        return CommonSubexpression(expr)




# -----------------------------------------------------------------------------
class Field(pymbolic.primitives.Variable):
    pass

# operators -------------------------------------------------------------------
class Operator(pymbolic.primitives.Leaf):
    def stringifier(self):
        return StringifyMapper

    def __call__(self, *args, **kwargs):
        # prevent lazy-eval semantics from kicking in
        raise RuntimeError, "symbolic operators are not callable"

    def apply(self, discr, field):
        return discr.compile(self * Field("f"))(f=field)




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
class MassOperatorBase(Operator):
    def get_hash(self):
        return hash(self.__class__)

    def is_equal(self, other):
        return other.__class__ == self.__class__




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



class FluxCoefficientOperator(Operator):
    """Results in a volume-global vector with data along the faces,
    obtained by computing the flux and applying the face mass matrix.
    """
    def __init__(self, int_coeff, ext_coeff):
        Operator.__init__(self)
        self.int_coeff = int_coeff
        self.ext_coeff = ext_coeff

    def get_mapper_method(self, mapper): 
        return mapper.map_flux_coefficient

    def __getinitargs__(self):
        return (self.int_coeff, self.ext_coeff)

    def get_hash(self):
        return hash((self.__class__, self.int_coeff, self.ext_coeff))



class LiftingFluxCoefficientOperator(Operator):
    """Results in a volume-global vector with data along the faces,
    obtained by computing the flux and applying the face mass matrix
    and the inverse volume mass matrix.
    """
    def __init__(self, int_coeff, ext_coeff):
        Operator.__init__(self)
        self.int_coeff = int_coeff
        self.ext_coeff = ext_coeff

    def get_mapper_method(self, mapper): 
        return mapper.map_lift_coefficient

    def __getinitargs__(self):
        return (self.int_coeff, self.ext_coeff)

    def get_hash(self):
        return hash((self.__class__, self.int_coeff, self.ext_coeff))




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
    from hedge.tools import is_obj_array, make_obj_array

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

    def map_flux_coefficient(self, expr, *args, **kwargs):
        return self.map_flux_coeff_base(expr, *args, **kwargs)

    def map_lift_coefficient(self, expr, *args, **kwargs):
        return self.map_flux_coeff_base(expr, *args, **kwargs)




class OperatorReducerMixin(LocalOpReducerMixin, FluxOpReducerMixin):
    """Reduces calls to *any* operator mapping function to just one."""
    def map_diff_base(self, expr, *args, **kwargs):
        return self.map_operator(expr, *args, **kwargs)

    def map_mass_base(self, expr, *args, **kwargs):
        return self.map_operator(expr, *args, **kwargs)

    def map_flux_base(self, expr, *args, **kwargs):
        return self.map_operator(expr, *args, **kwargs)

    def map_flux_coeff_base(self, expr, *args, **kwargs):
        return self.map_operator(expr, *args, **kwargs)





class CombineMapperMixin(object):
    def map_operator_binding(self, expr):
        return self.combine([self.rec(expr.op), self.rec(expr.field)])

    def map_boundary_pair(self, expr):
        return self.combine([self.rec(expr.field), self.rec(expr.bfield)])




class CombineMapper(CombineMapperMixin, pymbolic.mapper.CombineMapper):
    pass




class IdentityMapperMixin(LocalOpReducerMixin, FluxOpReducerMixin):
    def map_operator_binding(self, expr, *args, **kwargs):
        return expr.__class__(
                self.rec(expr.op, *args, **kwargs),
                self.rec(expr.field, *args, **kwargs))

    def map_boundary_pair(self, expr, *args, **kwargs):
        return expr.__class__(
                self.rec(expr.field, *args, **kwargs),
                self.rec(expr.bfield, *args, **kwargs),
                expr.tag)

    def map_mass_base(self, expr, *args, **kwargs):
        # it's a leaf--no changing children
        return expr

    def map_diff_base(self, expr, *args, **kwargs):
        # it's a leaf--no changing children
        return expr

    def map_diff_base(self, expr, *args, **kwargs):
        # it's a leaf--no changing children
        return expr

    def map_flux_base(self, expr, *args, **kwargs):
        # it's a leaf--no changing children
        return expr

    def map_flux_coeff_base(self, expr, *args, **kwargs):
        # it's a leaf--no changing children
        return expr




class DependencyMapper(
        CombineMapperMixin, 
        pymbolic.mapper.dependency.DependencyMapper, 
        OperatorReducerMixin):
    def map_operator(self, expr):
        return set()




class CommutativeConstantFoldingMapper(pymbolic.mapper.constant_folder.CommutativeConstantFoldingMapper,
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

    def map_flux_coefficient(self, expr, enclosing_prec):
        return "FluxCoeff(int=%s, ext=%s)" % (expr.int_coeff, expr.ext_coeff)

    def map_lift_coefficient(self, expr, enclosing_prec):
        return "LiftFluxCoeff(int=%s, ext=%s)" % (expr.int_coeff, expr.ext_coeff)

    def map_operator_binding(self, expr, enclosing_prec):
        return "<%s>(%s)" % (expr.op, expr.field)




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
            LiftingFluxOperator,
            FluxCoefficientOperator,
            LiftingFluxCoefficientOperator)) 
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
        elif isinstance(binding.op, FluxCoefficientOperator):
            return OperatorBinding(
                    LiftingFluxCoefficientOperator(
                        binding.op.int_coeff, binding.op.ext_coeff),
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




class FluxDecomposer(IdentityMapper):
    """Replaces each L{FluxOperator} in an operator template
    with a sum of L{FluxCoefficientOperator}s.
    """
    # assumes all flux operators to be bound

    def compile_coefficient(self, coeff):
        return coeff

    @staticmethod
    def _subscript(field, idx, is_scalar):
        if is_scalar:
            return field
        else:
            return field[idx]

    def _map_inner_flux(self, analyzed_flux, field):
        from hedge.tools import log_shape
        lsf = log_shape(field)
        is_scalar = lsf == ()
        if not is_scalar:
            assert len(lsf) == 1

        from pymbolic import flattened_sum
        return flattened_sum(
                OperatorBinding(
                    FluxCoefficientOperator(
                        self.compile_coefficient(int_flux),
                        self.compile_coefficient(ext_flux),
                        ),
                    self._subscript(field, idx, is_scalar)
                    )
                    for idx, int_flux, ext_flux in analyzed_flux)

    def _map_bdry_flux(self, analyzed_flux, field, bfield, tag):
        class ZeroVector:
            dtype = 0
            def __getitem__(self, idx):
                return 0

        from hedge.tools import log_shape
        lsf = log_shape(field)
        blsf = log_shape(bfield)

        is_scalar = lsf == () and blsf == ()
        if not is_scalar:
            assert len(lsf) == 1

            if isinstance(bfield, int) and bfield == 0:
                bfield = ZeroVector()
            elif isinstance(field, int) and field == 0:
                field = ZeroVector()
            else:
                print analyzed_flux
                print field, bfield
                print lsf, blsf
                assert lsf == blsf

        from pymbolic import flattened_sum
        return flattened_sum(
                OperatorBinding(
                    FluxCoefficientOperator(
                        self.compile_coefficient(int_flux),
                        self.compile_coefficient(ext_flux),
                        ),
                    BoundaryPair(
                        self._subscript(field, idx, is_scalar),
                        self._subscript(bfield, idx, is_scalar),
                        tag)
                    )
                    for idx, int_flux, ext_flux in analyzed_flux)


    def map_operator_binding(self, binding):
        # we only care about bindings of flux operators
        if isinstance(binding.op, FluxOperator):
            from hedge.flux import analyze_flux
            if isinstance(binding.field, BoundaryPair):
                bpair = binding.field
                return self._map_bdry_flux(
                        analyze_flux(binding.op.flux), 
                        bpair.field,
                        bpair.bfield,
                        bpair.tag)
            else:
                return self._map_inner_flux(
                        analyze_flux(binding.op.flux), 
                        binding.field)
        elif isinstance(binding.op, (FluxCoefficientOperator, LiftingFluxCoefficientOperator)):
            return OperatorBinding(
                    binding.op.__class__(
                        self.compile_coefficient(binding.op.int_coeff),
                        self.compile_coefficient(binding.op.ext_coeff),
                        ),
                    self.rec(binding.field)
                    )
        else:
            return binding.__class__(binding.op,
                    self.rec(binding.field))




# BC-to-flux rewriting --------------------------------------------------------
class BCToFluxRewriter(IdentityMapper):
    """Operates on L{FluxOperator} (note: not L{FluxCoefficientOperator})
    instances bound to L{BoundaryPair}s. If the boundary pair's C{bfield} is
    an expression of what's available in the C{field}, we can avoid fetching
    the data for the explicit boundary condition and just substitute the C{bfield}
    expression into the flux. This mapper does exactly that.
    """
    def map_operator_binding(self, expr):
        if not (isinstance(expr.op, FluxOperator)
                and isinstance(expr.field, BoundaryPair)):
            return IdentityMapper.map_operator_binding(self, expr)

        bpair = expr.field
        vol_field = bpair.field
        bdry_field = bpair.bfield
        flux = expr.op.flux

        class NormalToleratingDependencyMapper(DependencyMapper):
            # normals may occur in substitutable BCs, as a convenience.
            def map_normal(self, expr):
                return set([expr])

        from hedge.flux import Normal
        bdry_dependencies = NormalToleratingDependencyMapper(
                    include_calls="descend_args")(bdry_field)
        
        bdry_deps_without_normals = set(
                d for d in bdry_dependencies 
                if not isinstance(d, Normal))

        from hedge.tools import is_obj_array
        from hedge.flux import FieldComponent
        if is_obj_array(vol_field):
            vol_field_expr_translation = dict(
                    (vol_field_expr, FieldComponent(i, is_local=True))
                    for i, vol_field_expr in enumerate(vol_field))
        else:
            vol_field_expr_translation = {
                    vol_field: FieldComponent(0, is_local=True)}

        # If we can't completely eliminate the explicit boundary condition,
        # we might as well not try.
        if not bdry_deps_without_normals <= set(vol_field_expr_translation.keys()):
            if len(bdry_deps_without_normals) < len(bdry_dependencies):
                raise RuntimeError( 
                        "cannot use hedge.flux.Normal in non-substitutable flux expression")

            return IdentityMapper.map_operator_binding(self, expr)

        # Step I: Substitute the above volume terms into bdry_field 
        # as FieldComponent() instances.
        
        from pymbolic.mapper.substitutor import SubstitutionMapper

        class FieldIntoBdrySubstitutionMapper(
                SubstitutionMapper,
                IdentityMapperMixin):
            def map_normal(self, expr):
                return expr

        def sub_field_into_bdry(expr):
            try:
                return vol_field_expr_translation[expr]
            except KeyError:
                return None

        new_bdry_field = FieldIntoBdrySubstitutionMapper(
                sub_field_into_bdry)(bdry_field)

        # Step II: Substitute the new_bdry_field into the flux.
        from hedge.flux import FluxSubstitutionMapper

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
        
        result = OperatorBinding(
                FluxOperator(new_flux), BoundaryPair(vol_field, 0, bpair.tag))
        return result





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

    



class FluxCollector(CollectorMixin, CombineMapper):
    def map_operator_binding(self, expr):
        if isinstance(expr.op, (
            FluxOperatorBase, 
            FluxCoefficientOperator,
            LiftingFluxCoefficientOperator)):
            return set([expr])
        else:
            return set()




class BoundaryTagCollector(CollectorMixin, CombineMapper):
    def map_boundary_pair(self, bpair):
        return set([bpair.tag])




class DiffOpCollector(CollectorMixin, CombineMapper):
    def map_operator_binding(self, expr):
        if isinstance(expr.op, DiffOperatorBase):
            return set([expr])
        else:
            return set()




# evaluation ------------------------------------------------------------------
class Evaluator(pymbolic.mapper.evaluator.EvaluationMapper):
    def map_boundary_pair(self, bp):
        return BoundaryPair(self.rec(bp.field), self.rec(bp.bfield), bp.tag)

