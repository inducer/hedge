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
import hedge.mesh




# -----------------------------------------------------------------------------
class Field(pymbolic.primitives.Variable):
    pass

def make_vector_field(name, components):
    from hedge.tools import join_fields
    vfld = pymbolic.primitives.Variable(name)
    return join_fields(*[vfld[i] for i in range(components)])




# operators -------------------------------------------------------------------
class Operator(pymbolic.primitives.Leaf):
    def __init__(self, discr):
        self.discr = discr

    def stringifier(self):
        return StringifyMapper

    def __call__(self, *args, **kwargs):
        # prevent lazy-eval semantics from kicking in
        raise RuntimeError, "symbolic operators are not callable"

    def apply(self, field):
        return self.discr.compile(self * Field("f"))(f=field)




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




# diff operators --------------------------------------------------------------
class DiffOperatorBase(Operator):
    def __init__(self, discr, xyz_axis):
        Operator.__init__(self, discr)

        self.xyz_axis = xyz_axis

    def __getinitargs__(self):
        return self.discr, self.xyz_axis

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





# flux-like operators ---------------------------------------------------------
class FluxOperator(Operator):
    def __init__(self, discr, flux):
        Operator.__init__(self, discr)
        self.flux = flux

    def __mul__(self, arg):
        from hedge.tools import is_obj_array
        if isinstance(arg, Field) or is_obj_array(arg):
            return OperatorBinding(self, arg)
        else:
            return Operator.__mul__(self, arg)

    def get_mapper_method(self, mapper): 
        return mapper.map_flux




class FluxCoefficientOperator(Operator):
    """Results in a volume-global vector with data along the faces,
    obtained by computing the flux and applying the face mass matrix.
    """
    def __init__(self, discr, int_coeff, ext_coeff):
        Operator.__init__(self, discr)
        self.int_coeff = int_coeff
        self.ext_coeff = ext_coeff

    def get_mapper_method(self, mapper): 
        return mapper.map_flux_coefficient

    def __getinitargs__(self):
        return (self.discr, self.int_coeff, self.ext_coeff)



class LiftingFluxCoefficientOperator(Operator):
    """Results in a volume-global vector with data along the faces,
    obtained by computing the flux and applying the face mass matrix
    and the inverse volume mass matrix.
    """
    def __init__(self, discr, int_coeff, ext_coeff):
        Operator.__init__(self, discr)
        self.int_coeff = int_coeff
        self.ext_coeff = ext_coeff

    def get_mapper_method(self, mapper): 
        return mapper.map_lift_coefficient

    def __getinitargs__(self):
        return (self.discr, self.int_coeff, self.ext_coeff)




class VectorFluxOperator(object):
    def __init__(self, discr, fluxes):
        self.discr = discr
        self.fluxes = fluxes

    def __mul__(self, arg):
        if isinstance(arg, int) and arg == 0:
            return 0
        from hedge.tools import make_obj_array
        return make_obj_array(
                [OperatorBinding(FluxOperator(self.discr, f), arg)
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

    def __hash__(self):
        from pytools import hash_combine
        return hash_combine(self.__class__, self.field, self.bfield, self.tag)




def pair_with_boundary(field, bfield, tag=hedge.mesh.TAG_ALL):
    if tag is hedge.mesh.TAG_NONE:
        return 0
    else:
        return BoundaryPair(field, bfield, tag)




# mappers ---------------------------------------------------------------------
class IdentityMapper(pymbolic.mapper.IdentityMapper):
    def map_operator_binding(self, expr, *args, **kwargs):
        return expr.__class__(
                self.rec(expr.op, *args, **kwargs),
                self.rec(expr.field, *args, **kwargs))

    def map_boundary_pair(self, expr, *args, **kwargs):
        return expr.__class__(
                self.rec(expr.field, *args, **kwargs),
                self.rec(expr.bfield, *args, **kwargs),
                expr.tag)





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

    def map_flux_coefficient(self, expr, enclosing_prec):
        return "FluxCoeff(int=%s, ext=%s)" % (expr.int_coeff, expr.ext_coeff)

    def map_lift_coefficient(self, expr, enclosing_prec):
        return "LiftFluxCoeff(int=%s, ext=%s)" % (expr.int_coeff, expr.ext_coeff)

    def map_operator_binding(self, expr, enclosing_prec):
        return "<%s>(%s)" % (expr.op, expr.field)




class BoundOpMapperMixin(object):
    def map_operator_binding(self, expr, *args, **kwargs):
        return expr.op.get_mapper_method(self)(expr.op, expr.field, *args, **kwargs)




class LocalOpReducerMixin(object):
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




class OperatorBinder(IdentityMapper):
    def handle_unsupported_expression(self, expr):
        return expr

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
    def __init__(self, discr):
        self.discr = discr

    def map_constant(self, expr):
        return OperatorBinding(
                InverseMassOperator(self.discr),
                expr)

    def map_algebraic_leaf(self, expr):
        return OperatorBinding(
                InverseMassOperator(self.discr),
                expr)

    def map_operator_binding(self, binding):
        if isinstance(binding.op, MassOperator):
            return binding.field
        elif isinstance(binding.op, StiffnessOperator):
            return OperatorBinding(
                    DifferentiationOperator(self.discr, binding.op.xyz_axis),
                    binding.field)
        elif isinstance(binding.op, StiffnessTOperator):
            return OperatorBinding(
                    MInvSTOperator(self.discr, binding.op.xyz_axis),
                    binding.field)
        elif isinstance(binding.op, FluxCoefficientOperator):
            return OperatorBinding(
                    LiftingFluxCoefficientOperator(
                        self.discr, 
                        binding.op.int_coeff, binding.op.ext_coeff),
                    binding.field)
        else:
            return OperatorBinding(
                InverseMassOperator(self.discr),
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



        
class InverseMassContractor(pymbolic.mapper.IdentityMapper):
    # assumes all operators to be bound

    def map_boundary_pair(self, bp):
        return BoundaryPair(self.rec(bp.field), self.rec(bp.bfield), bp.tag)

    def map_operator_binding(self, binding):
        # we only care about bindings of inverse mass operators
        if not isinstance(binding.op, InverseMassOperator):
            return binding.__class__(binding.op,
                    self.rec(binding.field))
        else:
            return  _InnerInverseMassContractor(binding.op.discr)(binding.field)




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

    def _map_inner_flux(self, discr, analyzed_flux, field):
        from hedge.tools import log_shape
        lsf = log_shape(field)
        is_scalar = lsf == ()
        if not is_scalar:
            assert len(lsf) == 1

        from pymbolic import flattened_sum
        return flattened_sum(
                OperatorBinding(
                    FluxCoefficientOperator(discr,
                        self.compile_coefficient(int_flux),
                        self.compile_coefficient(ext_flux),
                        ),
                    self._subscript(field, idx, is_scalar)
                    )
                    for idx, int_flux, ext_flux in analyzed_flux)

    def _map_bdry_flux(self, discr, analyzed_flux, field, bfield, tag):
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
                assert lsf == blsf

        from pymbolic import flattened_sum
        return flattened_sum(
                OperatorBinding(
                    FluxCoefficientOperator(discr,
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
                return self._map_bdry_flux(binding.op.discr,
                        analyze_flux(binding.op.flux), 
                        bpair.field,
                        bpair.bfield,
                        bpair.tag)
            else:
                return self._map_inner_flux(
                        binding.op.discr,
                        analyze_flux(binding.op.flux), 
                        binding.field)
        elif isinstance(binding.op, (FluxCoefficientOperator, LiftingFluxCoefficientOperator)):
            return OperatorBinding(
                    binding.op.__class__(
                        binding.op.discr,
                        self.compile_coefficient(binding.op.int_coeff),
                        self.compile_coefficient(binding.op.ext_coeff),
                        ),
                    self.rec(binding.field)
                    )
        else:
            return binding.__class__(binding.op,
                    self.rec(binding.field))




class Evaluator(pymbolic.mapper.evaluator.EvaluationMapper):
    def map_boundary_pair(self, bp):
        return BoundaryPair(self.rec(bp.field), self.rec(bp.bfield), bp.tag)
