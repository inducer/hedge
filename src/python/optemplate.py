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




class OperatorBinding(pymbolic.primitives.Leaf):
    def __init__(self, op, field):
        self.op = op
        self.field = field

    def stringifier(self):
        return StringifyMapper

    def get_mapper_method(self, mapper): 
        return mapper.map_operator_binding




# diff operators --------------------------------------------------------------
class DiffOperatorBase(Operator):
    def __init__(self, discr, xyz_axis):
        Operator.__init__(self, discr)

        self.xyz_axis = xyz_axis

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
    

    

#class DiffOperatorVector(pymbolic.primitives.Vector):
    #pass




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





# flux operators --------------------------------------------------------------
class FluxOperator(Operator):
    def __init__(self, discr, flux):
        Operator.__init__(self, discr)
        self.flux = flux

    def get_mapper_method(self, mapper): 
        return mapper.map_flux




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
class _BoundaryPair(pymbolic.primitives.Leaf):
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




def pair_with_boundary(field, bfield, tag=hedge.mesh.TAG_ALL):
    if tag is hedge.mesh.TAG_NONE:
        return 0
    else:
        return _BoundaryPair(field, bfield, tag)




# mappers ---------------------------------------------------------------------
class StringifyMapper(pymbolic.mapper.stringifier.StringifyMapper):
    def map_boundary_pair(self, expr, enclosing_prec):
        return "BPair(%s, %s, %s)" % (expr.field, expr.bfield, repr(expr.tag))

    def map_diff(self, expr, enclosing_prec):
        return "Diff(%d)" % expr.xyz_axis

    def map_minv_st(self, expr, enclosing_prec):
        return "MInvST(%d)" % expr.xyz_axis

    def map_stiffness(self, expr, enclosing_prec):
        return "Stiff(%d)" % expr.xyz_axis

    def map_stiffness_t(self, expr, enclosing_prec):
        return "StiffT(%d)" % expr.xyz_axis

    def map_mass(self, expr, enclosing_prec):
        return "M"

    def map_inverse_mass(self, expr, enclosing_prec):
        return "InvM"

    def map_flux(self, expr, enclosing_prec):
        return "Flux(%s)" % expr.flux

    def map_operator_binding(self, expr, enclosing_prec):
        return "<%s>(%s)" % (expr.op, expr.field)




class LocalOpReducerMixin(object):
    def map_diff(self, expr, *args, **kwargs):
        return self.map_diff_base(expr, *args, **kwargs)

    def map_minv_st(self, expr, enclosing_prec):
        return self.map_diff_base(expr, *args, **kwargs)

    def map_stiffness(self, expr, enclosing_prec):
        return self.map_diff_base(expr, *args, **kwargs)

    def map_stiffness_t(self, expr, enclosing_prec):
        return self.map_diff_base(expr, *args, **kwargs)

    def map_mass(self, expr, enclosing_prec):
        return self.map_mass_base(expr, *args, **kwargs)

    def map_inverse_mass(self, expr, enclosing_prec):
        return self.map_mass_base(expr, *args, **kwargs)




class BoundOpMapperMixin(object):
    def map_operator_binding(self, expr):
        return expr.op.get_mapper_method(self)(expr.op, expr.field)




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




class OperatorBinder(pymbolic.mapper.IdentityMapper):
    def map_product(self, expr):
        def generate_new_children():
            it = iter(expr.children)
            while True:
                try:
                    child = it.next()
                except StopIteration:
                    break

                if isinstance(child, Operator):
                    try:
                        operand = it.next()
                    except StopIteration:
                        raise ValueError("no operand for to bind '%s'" % child)

                    yield OperatorBinding(child, self.rec(operand))
                else:
                    yield self.rec(child)

        from pymbolic.primitives import flattened_product
        return flattened_product(generate_new_children())




class Evaluator(pymbolic.mapper.evaluator.EvaluationMapper):
    def map_boundary_pair(self, bp):
        return BoundaryPair(self.rec(bp.field), self.rec(bp.bfield), bp.tag)

