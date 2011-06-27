"""Operator template mappers."""

from __future__ import division

__copyright__ = "Copyright (C) 2008 Andreas Kloeckner"

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
import pymbolic.mapper.stringifier
import pymbolic.mapper.evaluator
import pymbolic.mapper.dependency
import pymbolic.mapper.substitutor
import pymbolic.mapper.constant_folder
import pymbolic.mapper.flop_counter
from pymbolic.mapper import CSECachingMapperMixin




# {{{ mixins ------------------------------------------------------------------
class LocalOpReducerMixin(object):
    """Reduces calls to mapper methods for all local differentiation
    operators to a single mapper method, and likewise for mass
    operators.
    """
    # {{{ global differentiation
    def map_diff(self, expr, *args, **kwargs):
        return self.map_diff_base(expr, *args, **kwargs)

    def map_minv_st(self, expr, *args, **kwargs):
        return self.map_diff_base(expr, *args, **kwargs)

    def map_stiffness(self, expr, *args, **kwargs):
        return self.map_diff_base(expr, *args, **kwargs)

    def map_stiffness_t(self, expr, *args, **kwargs):
        return self.map_diff_base(expr, *args, **kwargs)

    def map_quad_stiffness_t(self, expr, *args, **kwargs):
        return self.map_diff_base(expr, *args, **kwargs)
    # }}}

    # {{{ global mass
    def map_mass_base(self, expr, *args, **kwargs):
        return self.map_elementwise_linear(expr, *args, **kwargs)

    def map_mass(self, expr, *args, **kwargs):
        return self.map_mass_base(expr, *args, **kwargs)

    def map_inverse_mass(self, expr, *args, **kwargs):
        return self.map_mass_base(expr, *args, **kwargs)

    def map_quad_mass(self, expr, *args, **kwargs):
        return self.map_mass_base(expr, *args, **kwargs)
    # }}}

    # {{{ reference differentiation
    def map_ref_diff(self, expr, *args, **kwargs):
        return self.map_ref_diff_base(expr, *args, **kwargs)

    def map_ref_stiffness_t(self, expr, *args, **kwargs):
        return self.map_ref_diff_base(expr, *args, **kwargs)

    def map_ref_quad_stiffness_t(self, expr, *args, **kwargs):
        return self.map_ref_diff_base(expr, *args, **kwargs)
    # }}}

    # {{{ reference mass
    def map_ref_mass_base(self, expr, *args, **kwargs):
        return self.map_elementwise_linear(expr, *args, **kwargs)

    def map_ref_mass(self, expr, *args, **kwargs):
        return self.map_ref_mass_base(expr, *args, **kwargs)

    def map_ref_inverse_mass(self, expr, *args, **kwargs):
        return self.map_ref_mass_base(expr, *args, **kwargs)

    def map_ref_quad_mass(self, expr, *args, **kwargs):
        return self.map_ref_mass_base(expr, *args, **kwargs)
    # }}}




class FluxOpReducerMixin(object):
    """Reduces calls to mapper methods for all flux
    operators to a smaller number of mapper methods.
    """
    def map_flux(self, expr, *args, **kwargs):
        return self.map_flux_base(expr, *args, **kwargs)

    def map_bdry_flux(self, expr, *args, **kwargs):
        return self.map_flux_base(expr, *args, **kwargs)

    def map_quad_flux(self, expr, *args, **kwargs):
        return self.map_flux_base(expr, *args, **kwargs)

    def map_quad_bdry_flux(self, expr, *args, **kwargs):
        return self.map_flux_base(expr, *args, **kwargs)



class OperatorReducerMixin(LocalOpReducerMixin, FluxOpReducerMixin):
    """Reduces calls to *any* operator mapping function to just one."""
    def map_diff_base(self, expr, *args, **kwargs):
        return self.map_operator(expr, *args, **kwargs)

    map_ref_diff_base = map_diff_base
    map_elementwise_linear = map_diff_base
    map_flux_base = map_diff_base
    map_elementwise_max = map_diff_base
    map_boundarize = map_diff_base
    map_flux_exchange = map_diff_base
    map_quad_grid_upsampler = map_diff_base
    map_quad_int_faces_grid_upsampler = map_diff_base
    map_quad_bdry_grid_upsampler = map_diff_base




class CombineMapperMixin(object):
    def map_operator_binding(self, expr):
        return self.combine([self.rec(expr.op), self.rec(expr.field)])

    def map_boundary_pair(self, expr):
        return self.combine([self.rec(expr.field), self.rec(expr.bfield)])




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

    def map_elementwise_linear(self, expr, *args, **kwargs):
        assert not isinstance(self, BoundOpMapperMixin), \
                "IdentityMapper instances cannot be combined with " \
                "the BoundOpMapperMixin"

        # it's a leaf--no changing children
        return expr

    def map_scalar_parameter(self, expr, *args, **kwargs):
        # it's a leaf--no changing children
        return expr
    map_c_function = map_scalar_parameter
    map_jacobian = map_scalar_parameter
    map_inverse_metric_derivative = map_scalar_parameter
    map_forward_metric_derivative = map_scalar_parameter

    map_mass_base = map_elementwise_linear
    map_ref_mass_base = map_elementwise_linear
    map_diff_base = map_elementwise_linear
    map_ref_diff_base = map_elementwise_linear
    map_flux_base = map_elementwise_linear
    map_elementwise_max = map_elementwise_linear
    map_boundarize = map_elementwise_linear
    map_flux_exchange = map_elementwise_linear
    map_quad_grid_upsampler = map_elementwise_linear
    map_quad_int_faces_grid_upsampler = map_elementwise_linear
    map_quad_bdry_grid_upsampler = map_elementwise_linear

    map_normal_component = map_elementwise_linear




class BoundOpMapperMixin(object):
    def map_operator_binding(self, expr, *args, **kwargs):
        return expr.op.get_mapper_method(self)(expr.op, expr.field, *args, **kwargs)



# }}}

# {{{ basic mappers -----------------------------------------------------------
class CombineMapper(CombineMapperMixin, pymbolic.mapper.CombineMapper):
    pass




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

    def map_scalar_parameter(self, expr):
        return set([expr])

    def map_normal_component(self, expr):
        return set()

    map_jacobian = map_normal_component
    map_forward_metric_derivative = map_normal_component
    map_inverse_metric_derivative = map_normal_component


class FlopCounter(
        CombineMapperMixin,
        pymbolic.mapper.flop_counter.FlopCounter):
    def map_operator_binding(self, expr):
        return self.rec(expr.field)

    def map_scalar_parameter(self, expr):
        return 0

    def map_c_function(self, expr):
        return 1

    def map_normal_component(self, expr):
        return 0

    map_jacobian = map_normal_component
    map_forward_metric_derivative = map_normal_component
    map_inverse_metric_derivative = map_normal_component





class IdentityMapper(
        IdentityMapperMixin,
        pymbolic.mapper.IdentityMapper):
    pass





class SubstitutionMapper(pymbolic.mapper.substitutor.SubstitutionMapper,
        IdentityMapperMixin):
    pass




class CSERemover(IdentityMapper):
    def map_common_subexpression(self, expr):
        return self.rec(expr.child)

# }}}

# {{{ operator binder ---------------------------------------------------------
class OperatorBinder(CSECachingMapperMixin, IdentityMapper):
    map_common_subexpression_uncached = \
            IdentityMapper.map_common_subexpression

    def map_product(self, expr):
        if len(expr.children) == 0:
            return expr

        from pymbolic.primitives import flattened_product, Product
        from hedge.optemplate import Operator, OperatorBinding

        first = expr.children[0]
        if isinstance(first, Operator):
            prod = flattened_product(expr.children[1:])
            if isinstance(prod, Product) and len(prod.children) > 1:
                from warnings import warn
                warn("Binding '%s' to more than one "
                        "operand in a product is ambiguous - "
                        "use the parenthesized form instead."
                        % first)
            return OperatorBinding(first, self.rec(prod))
        else:
            return first * self.rec(flattened_product(expr.children[1:]))

# }}}

# {{{ operator specializer ----------------------------------------------------
class OperatorSpecializer(CSECachingMapperMixin, IdentityMapper):
    """Guided by a typedict obtained through type inference (i.e. by
    :class:`hedge.optemplate.mappers.type_inference.TypeInferrrer`),
    substitutes more specialized operators for generic ones.

    For example, if type inference has determined that a differentiation
    operator is applied to an argument on a quadrature grid, this
    differentiation operator is then swapped out for a *quadrature*
    differentiation operator.
    """

    def __init__(self, typedict):
        """
        :param typedict: generated by
        :class:`hedge.optemplate.mappers.type_inference.TypeInferrer`.
        """
        self.typedict = typedict

    map_common_subexpression_uncached = \
            IdentityMapper.map_common_subexpression

    def map_operator_binding(self, expr):
        from hedge.optemplate.primitives import BoundaryPair

        from hedge.optemplate.operators import (
                MassOperator,
                QuadratureMassOperator,
                ReferenceMassOperator,
                ReferenceQuadratureMassOperator,

                StiffnessTOperator,
                QuadratureStiffnessTOperator,
                ReferenceStiffnessTOperator,
                ReferenceQuadratureStiffnessTOperator,

                QuadratureGridUpsampler, QuadratureBoundaryGridUpsampler,
                FluxOperatorBase, FluxOperator, QuadratureFluxOperator,
                BoundaryFluxOperator, QuadratureBoundaryFluxOperator,
                BoundarizeOperator)

        from hedge.optemplate.mappers.type_inference import (
                type_info, QuadratureRepresentation)

        # {{{ figure out field type
        try:
            field_type = self.typedict[expr.field]
        except TypeError:
            # numpy arrays are not hashable
            # has_quad_operand remains unset

            assert isinstance(expr.field, numpy.ndarray)
        else:
            try:
                field_repr_tag = field_type.repr_tag
            except AttributeError:
                # boundary pairs are not assigned types
                assert isinstance(expr.field, BoundaryPair)
                has_quad_operand = False
            else:
                has_quad_operand = isinstance(field_repr_tag,
                            QuadratureRepresentation)
        # }}}

        # Based on where this is run in the optemplate processing pipeline,
        # we may encounter both reference and non-reference operators.

        # {{{ elementwise operators
        if isinstance(expr.op, MassOperator) and has_quad_operand:
            return QuadratureMassOperator(
                    field_repr_tag.quadrature_tag)(self.rec(expr.field))

        elif isinstance(expr.op, ReferenceMassOperator) and has_quad_operand:
            return ReferenceQuadratureMassOperator(
                    field_repr_tag.quadrature_tag)(self.rec(expr.field))

        elif (isinstance(expr.op, StiffnessTOperator) and has_quad_operand):
            return QuadratureStiffnessTOperator(
                    expr.op.xyz_axis, field_repr_tag.quadrature_tag) \
                    (self.rec(expr.field))

        elif (isinstance(expr.op, ReferenceStiffnessTOperator)
                and has_quad_operand):
            return ReferenceQuadratureStiffnessTOperator(
                    expr.op.xyz_axis, field_repr_tag.quadrature_tag) \
                    (self.rec(expr.field))

        elif (isinstance(expr.op, QuadratureGridUpsampler)
                and isinstance(field_type, type_info.BoundaryVectorBase)):
            # potential shortcut:
            #if (isinstance(expr.field, OperatorBinding)
                    #and isinstance(expr.field.op, BoundarizeOperator)):
                #return QuadratureBoundarizeOperator(
                        #expr.field.op.tag, expr.op.quadrature_tag)(
                                #self.rec(expr.field.field))

            return QuadratureBoundaryGridUpsampler(
                    expr.op.quadrature_tag, field_type.boundary_tag)(expr.field)
        # }}}

        elif isinstance(expr.op, BoundarizeOperator) and has_quad_operand:
            raise TypeError("BoundarizeOperator cannot be applied to "
                    "quadrature-based operands--use QuadUpsample(Boundarize(...))")

        # {{{ flux operator specialization
        elif isinstance(expr.op, FluxOperatorBase):
            from pytools.obj_array import with_object_array_or_scalar

            repr_tag_cell = [None]

            def process_flux_arg(flux_arg):
                arg_repr_tag = self.typedict[flux_arg].repr_tag
                if repr_tag_cell[0] is None:
                    repr_tag_cell[0] = arg_repr_tag
                else:
                    # An error for this condition is generated by
                    # the type inference pass.

                    assert arg_repr_tag == repr_tag_cell[0]

            is_boundary = isinstance(expr.field, BoundaryPair)
            if is_boundary:
                bpair = expr.field
                with_object_array_or_scalar(process_flux_arg, bpair.field)
                with_object_array_or_scalar(process_flux_arg, bpair.bfield)
            else:
                with_object_array_or_scalar(process_flux_arg, expr.field)

            is_quad = isinstance(repr_tag_cell[0], QuadratureRepresentation)
            if is_quad:
                assert not expr.op.is_lift
                quad_tag = repr_tag_cell[0].quadrature_tag

            new_fld = self.rec(expr.field)
            flux = expr.op.flux

            if is_boundary:
                if is_quad:
                    return QuadratureBoundaryFluxOperator(
                            flux, quad_tag, bpair.tag)(new_fld)
                else:
                    return BoundaryFluxOperator(flux, bpair.tag)(new_fld)
            else:
                if is_quad:
                    return QuadratureFluxOperator(flux, quad_tag)(new_fld)
                else:
                    return FluxOperator(flux, expr.op.is_lift)(new_fld)
        # }}}

        else:
            return IdentityMapper.map_operator_binding(self, expr)

    def map_normal_component(self, expr):
        from hedge.optemplate.mappers.type_inference import (
                NodalRepresentation)

        expr_type = self.typedict[expr]
        if not isinstance(
                expr_type.repr_tag,
                NodalRepresentation):
            from hedge.optemplate.primitives import (
                    BoundaryNormalComponent)

            # for now, parts of this are implemented.
            raise NotImplementedError("normal components on quad. grids")

            return BoundaryNormalComponent(
                    expr.boundary_tag, expr.axis, 
                    expr_type.repr_tag.quadrature_tag)

        # a leaf, doesn't change
        return expr

# }}}

# {{{ global-to-reference mapper ----------------------------------------------

class GlobalToReferenceMapper(CSECachingMapperMixin, IdentityMapper):
    """Maps operators that apply on the global function space down to operators on
    reference elements, together with explicit multiplication by geometric factors.
    """

    def __init__(self, dimensions):
        CSECachingMapperMixin.__init__(self)
        IdentityMapper.__init__(self)

        self.dimensions = dimensions

    map_common_subexpression_uncached = \
            IdentityMapper.map_common_subexpression

    def map_operator_binding(self, expr):
        from hedge.optemplate.primitives import (
                Jacobian, InverseMetricDerivative)

        from hedge.optemplate.operators import (
                MassOperator,
                ReferenceMassOperator,
                QuadratureMassOperator,
                ReferenceQuadratureMassOperator,

                StiffnessOperator,

                StiffnessTOperator,
                ReferenceStiffnessTOperator,
                QuadratureStiffnessTOperator,
                ReferenceQuadratureStiffnessTOperator,

                InverseMassOperator, ReferenceInverseMassOperator,
                DifferentiationOperator, ReferenceDifferentiationOperator,

                MInvSTOperator)

        # Global-to-reference is run after operator specialization, so
        # if we encounter non-quadrature operators here, we know they
        # must be nodal.

        def rewrite_derivative(ref_class, field, quadrature_tag=None,
                with_jacobian=True):
            if quadrature_tag is not None:
                diff_kwargs = dict(quadrature_tag=quadrature_tag)
            else:
                diff_kwargs = {}

            rec_field = self.rec(field)
            if with_jacobian:
                rec_field = Jacobian(quadrature_tag) * rec_field
            return sum(InverseMetricDerivative(None, rst_axis, expr.op.xyz_axis) *
                    ref_class(rst_axis, **diff_kwargs)(rec_field)
                    for rst_axis in range(self.dimensions))

        if isinstance(expr.op, MassOperator):
            return ReferenceMassOperator()(
                    Jacobian(None) * self.rec(expr.field))

        elif isinstance(expr.op, QuadratureMassOperator):
            return ReferenceQuadratureMassOperator(
                    expr.op.quadrature_tag)(
                    Jacobian(expr.op.quadrature_tag) * self.rec(expr.field))

        elif isinstance(expr.op, InverseMassOperator) :
            return ReferenceInverseMassOperator()(
                1/Jacobian(None) * self.rec(expr.field))

        elif isinstance(expr.op, StiffnessOperator) :
            return ReferenceMassOperator()(Jacobian(None) *
                    self.rec(
                        DifferentiationOperator(expr.op.xyz_axis)(expr.field)))

        elif isinstance(expr.op, DifferentiationOperator):
            return rewrite_derivative(
                    ReferenceDifferentiationOperator,
                    expr.field, with_jacobian=False)

        elif isinstance(expr.op, StiffnessTOperator):
            return rewrite_derivative(
                    ReferenceStiffnessTOperator,
                    expr.field)

        elif isinstance(expr.op, QuadratureStiffnessTOperator):
            return rewrite_derivative(
                    ReferenceQuadratureStiffnessTOperator,
                    expr.field, quadrature_tag=expr.op.quadrature_tag)

        elif isinstance(expr.op, MInvSTOperator):
            return self.rec(
                    InverseMassOperator()(
                        StiffnessTOperator(expr.op.xyz_axis)(expr.field)))

        else:
            return IdentityMapper.map_operator_binding(self, expr)

# }}}

# {{{ stringification ---------------------------------------------------------
class StringifyMapper(pymbolic.mapper.stringifier.StringifyMapper):
    def _format_btag(self, tag):
        from hedge.mesh import SYSTEM_TAGS
        if tag in SYSTEM_TAGS:
            return tag.__name__
        else:
            return repr(tag)

    def __init__(self, constant_mapper=str, flux_stringify_mapper=None):
        pymbolic.mapper.stringifier.StringifyMapper.__init__(
                self, constant_mapper=constant_mapper)

        if flux_stringify_mapper is None:
            from hedge.flux import FluxStringifyMapper
            flux_stringify_mapper = FluxStringifyMapper()

        self.flux_stringify_mapper = flux_stringify_mapper

    def map_boundary_pair(self, expr, enclosing_prec):
        from pymbolic.mapper.stringifier import PREC_NONE
        return "BPair(%s, %s, %s)" % (
                self.rec(expr.field, PREC_NONE),
                self.rec(expr.bfield, PREC_NONE),
                self._format_btag(expr.tag))

    # {{{ global differentiation
    def map_diff(self, expr, enclosing_prec):
        return "Diffx%d" % expr.xyz_axis

    def map_minv_st(self, expr, enclosing_prec):
        return "MInvSTx%d" % expr.xyz_axis

    def map_stiffness(self, expr, enclosing_prec):
        return "Stiffx%d" % expr.xyz_axis

    def map_stiffness_t(self, expr, enclosing_prec):
        return "StiffTx%d" % expr.xyz_axis

    def map_quad_stiffness_t(self, expr, enclosing_prec):
        return "Q[%s]StiffTx%d" % (
                expr.quadrature_tag, expr.xyz_axis)
    # }}}

    # {{{ global mass
    def map_mass(self, expr, enclosing_prec):
        return "M"

    def map_inverse_mass(self, expr, enclosing_prec):
        return "InvM"

    def map_quad_mass(self, expr, enclosing_prec):
        return "Q[%s]M" % expr.quadrature_tag
    # }}}

    # {{{ reference differentiation
    def map_ref_diff(self, expr, enclosing_prec):
        return "Diffr%d" % expr.rst_axis

    def map_ref_stiffness_t(self, expr, enclosing_prec):
        return "StiffTr%d" % expr.rst_axis

    def map_ref_quad_stiffness_t(self, expr, enclosing_prec):
        return "Q[%s]StiffTr%d" % (
                expr.quadrature_tag, expr.rst_axis)
    # }}}

    # {{{ reference mass
    def map_ref_mass(self, expr, enclosing_prec):
        return "RefM"

    def map_ref_inverse_mass(self, expr, enclosing_prec):
        return "RefInvM"

    def map_ref_quad_mass(self, expr, enclosing_prec):
        return "RefQ[%s]M" % expr.quadrature_tag
    # }}}

    def map_elementwise_linear(self, expr, enclosing_prec):
        return "ElWLin:%s" % expr.__class__.__name__

    # {{{ flux
    def map_flux(self, expr, enclosing_prec):
        from pymbolic.mapper.stringifier import PREC_NONE
        return "%s(%s)" % (
                expr.get_flux_or_lift_text(),
                self.flux_stringify_mapper(expr.flux, PREC_NONE))

    def map_bdry_flux(self, expr, enclosing_prec):
        from pymbolic.mapper.stringifier import PREC_NONE
        return "B[%s]%s(%s)" % (
                self._format_btag(expr.boundary_tag),
                expr.get_flux_or_lift_text(),
                self.flux_stringify_mapper(expr.flux, PREC_NONE))

    def map_quad_flux(self, expr, enclosing_prec):
        from pymbolic.mapper.stringifier import PREC_NONE
        return "Q[%s]%s(%s)" % (
                expr.quadrature_tag,
                expr.get_flux_or_lift_text(),
                self.flux_stringify_mapper(expr.flux, PREC_NONE))

    def map_quad_bdry_flux(self, expr, enclosing_prec):
        from pymbolic.mapper.stringifier import PREC_NONE
        return "Q[%s]B[%s]%s(%s)" % (
                expr.quadrature_tag,
                self._format_btag(expr.boundary_tag),
                expr.get_flux_or_lift_text(),
                self.flux_stringify_mapper(expr.flux, PREC_NONE))

    def map_whole_domain_flux(self, expr, enclosing_prec):
        # used from hedge.backends.cuda.optemplate
        if expr.is_lift:
            opname = "WLift"
        else:
            opname = "WFlux"

        from pymbolic.mapper.stringifier import PREC_NONE
        return "%s(%s)" % (opname,
                self.rec(expr.rebuild_optemplate(), PREC_NONE))
    # }}}

    def map_elementwise_max(self, expr, enclosing_prec):
        return "ElWMax"

    def map_boundarize(self, expr, enclosing_prec):
        return "Boundarize<tag=%s>" % expr.tag

    def map_flux_exchange(self, expr, enclosing_prec):
        from pymbolic.mapper.stringifier import PREC_NONE
        return "FExch<idx=%s,rank=%d>(%s)" % (expr.index, expr.rank,
                ", ".join(self.rec(arg, PREC_NONE) for arg in expr.arg_fields))

    # {{{ geometry data
    def map_normal_component(self, expr, enclosing_prec):
        if expr.quadrature_tag is None:
            return ("Normal<tag=%s>[%d]" 
                    % (expr.boundary_tag, expr.axis))
        else:
            return ("Q[%s]Normal<tag=%s>[%d]" 
                    % (expr.quadrature_tag, expr.boundary_tag, expr.axis))

    def map_jacobian(self, expr, enclosing_prec):
        if expr.quadrature_tag is None:
            return "Jac"
        else:
            return "JacQ[%s]" % expr.quadrature_tag

    def map_forward_metric_derivative(self, expr, enclosing_prec):
        result = "dx%d/dr%d" % (expr.xyz_axis, expr.rst_axis)
        if expr.quadrature_tag is not None:
            result += "Q[%s]" % expr.quadrature_tag
        return result

    def map_inverse_metric_derivative(self, expr, enclosing_prec):
        result = "dr%d/dx%d" % (expr.rst_axis, expr.xyz_axis)
        if expr.quadrature_tag is not None:
            result += "Q[%s]" % expr.quadrature_tag
        return result

    # }}}

    def map_operator_binding(self, expr, enclosing_prec):
        from pymbolic.mapper.stringifier import PREC_NONE
        return "<%s>(%s)" % (
                self.rec(expr.op, PREC_NONE),
                self.rec(expr.field, PREC_NONE))

    def map_c_function(self, expr, enclosing_prec):
        return expr.name

    def map_scalar_parameter(self, expr, enclosing_prec):
        return "ScalarPar[%s]" % expr.name

    def map_quad_grid_upsampler(self, expr, enclosing_prec):
        return "ToQ[%s]" % expr.quadrature_tag

    def map_quad_int_faces_grid_upsampler(self, expr, enclosing_prec):
        return "ToIntFaceQ[%s]" % expr.quadrature_tag

    def map_quad_bdry_grid_upsampler(self, expr, enclosing_prec):
        return "ToBdryQ[%s,%s]" % (expr.quadrature_tag, expr.boundary_tag)




class PrettyStringifyMapper(
        pymbolic.mapper.stringifier.CSESplittingStringifyMapperMixin,
        StringifyMapper):
    def __init__(self):
        pymbolic.mapper.stringifier.CSESplittingStringifyMapperMixin.__init__(self)
        StringifyMapper.__init__(self)

        self.flux_to_number = {}
        self.flux_string_list = []

        self.bc_to_number = {}
        self.bc_string_list = []

        from hedge.flux import PrettyFluxStringifyMapper
        self.flux_stringify_mapper = PrettyFluxStringifyMapper()

    def get_flux_number(self, flux):
        try:
            return self.flux_to_number[flux]
        except KeyError:
            from pymbolic.mapper.stringifier import PREC_NONE
            str_flux = self.flux_stringify_mapper(flux, PREC_NONE)

            flux_number = len(self.flux_to_number)
            self.flux_string_list.append(str_flux)
            self.flux_to_number[flux] = flux_number
            return flux_number

    def map_boundary_pair(self, expr, enclosing_prec):
        try:
            bc_number = self.bc_to_number[expr]
        except KeyError:
            from pymbolic.mapper.stringifier import PREC_NONE
            str_bc = StringifyMapper.map_boundary_pair(self, expr, PREC_NONE)

            bc_number = len(self.bc_to_number)
            self.bc_string_list.append(str_bc)
            self.bc_to_number[expr] = bc_number

        return "BC%d@%s" % (bc_number, expr.tag)

    def map_operator_binding(self, expr, enclosing_prec):
        from hedge.optemplate import BoundarizeOperator
        if isinstance(expr.op, BoundarizeOperator):
            from pymbolic.mapper.stringifier import PREC_CALL, PREC_SUM
            return self.parenthesize_if_needed(
                    "%s@%s" % (
                        self.rec(expr.field, PREC_CALL),
                        expr.op.tag),
                    enclosing_prec, PREC_SUM)
        else:
            return StringifyMapper.map_operator_binding(
                    self, expr, enclosing_prec)

    def get_bc_strings(self):
        return ["BC%d : %s" % (i, bc_str)
                for i, bc_str in enumerate(self.bc_string_list)]

    def get_flux_strings(self):
        return ["Flux%d : %s" % (i, flux_str)
                for i, flux_str in enumerate(self.flux_string_list)]

    def map_flux(self, expr, enclosing_prec):
        return "%s%d" % (
                expr.get_flux_or_lift_text(),
                self.get_flux_number(expr.flux))

    def map_bdry_flux(self, expr, enclosing_prec):
        return "B[%s]%s%d" % (
                expr.boundary_tag,
                expr.get_flux_or_lift_text(),
                self.get_flux_number(expr.flux))

    def map_quad_flux(self, expr, enclosing_prec):
        return "Q[%s]%s%d" % (
                expr.quadrature_tag,
                expr.get_flux_or_lift_text(),
                self.get_flux_number(expr.flux))

    def map_quad_bdry_flux(self, expr, enclosing_prec):
        return "Q[%s]B[%s]%s%d" % (
                expr.quadrature_tag,
                expr.boundary_tag,
                expr.get_flux_or_lift_text(),
                self.get_flux_number(expr.flux))




class NoCSEStringifyMapper(StringifyMapper):
    def map_common_subexpression(self, expr, enclosing_prec):
        return self.rec(expr.child, enclosing_prec)




# }}}

# {{{ quadrature support ------------------------------------------------------
class QuadratureUpsamplerRemover(CSECachingMapperMixin, IdentityMapper):
    def __init__(self, quad_min_degrees, do_warn=True):
        IdentityMapper.__init__(self)
        CSECachingMapperMixin.__init__(self)
        self.quad_min_degrees = quad_min_degrees
        self.do_warn = do_warn

    map_common_subexpression_uncached = \
            IdentityMapper.map_common_subexpression

    def map_operator_binding(self, expr):
        from hedge.optemplate.operators import (
                QuadratureGridUpsampler,
                QuadratureInteriorFacesGridUpsampler,
                QuadratureBoundaryGridUpsampler)

        if isinstance(expr.op, (QuadratureGridUpsampler,
            QuadratureInteriorFacesGridUpsampler,
            QuadratureBoundaryGridUpsampler)):
            try:
                min_degree = self.quad_min_degrees[expr.op.quadrature_tag]
            except KeyError:
                if self.do_warn:
                    from warnings import warn
                    warn("No minimum degree for quadrature tag '%s' specified--"
                            "falling back to nodal evaluation" % expr.op.quadrature_tag)
                return self.rec(expr.field)
            else:
                if min_degree is None:
                    return self.rec(expr.field)
                else:
                    return expr.op(self.rec(expr.field))
        else:
            return IdentityMapper.map_operator_binding(self, expr)




class QuadratureDetector(CSECachingMapperMixin, CombineMapper):
    """For a given expression, this mapper returns the upsampling
    operator in effect at the root of the expression, or *None*
    if there isn't one.
    """
    class QuadStatusNotKnown:
        pass

    map_common_subexpression_uncached = \
            CombineMapper.map_common_subexpression

    def combine(self, values):
        from pytools import single_valued
        return single_valued([
            v for v in values if v is not self.QuadStatusNotKnown])

    def map_variable(self, expr):
        return None

    def map_constant(self, expr):
        return self.QuadStatusNotKnown

    def map_operator_binding(self, expr):
        from hedge.optemplate.operators import (
                DiffOperatorBase, FluxOperatorBase,
                MassOperatorBase,
                QuadratureGridUpsampler,
                QuadratureInteriorFacesGridUpsampler)

        if isinstance(expr.op, (
            DiffOperatorBase, FluxOperatorBase,
            MassOperatorBase)):
            return None
        elif isinstance(expr.op, (QuadratureGridUpsampler,
            QuadratureInteriorFacesGridUpsampler)):
            return expr.op
        else:
            return CombineMapper.map_operator_binding(self, expr)




class QuadratureUpsamplerChanger(CSECachingMapperMixin, IdentityMapper):
    """This mapper descends the expression tree, down to each 
    quadrature-consuming operator (diff, mass) along each branch.
    It then change
    """
    def __init__(self, desired_quad_op):
        IdentityMapper.__init__(self)
        CSECachingMapperMixin.__init__(self)

        self.desired_quad_op = desired_quad_op

    map_common_subexpression_uncached = \
            IdentityMapper.map_common_subexpression

    def map_operator_binding(self, expr):
        from hedge.optemplate.operators import (
                DiffOperatorBase, FluxOperatorBase,
                MassOperatorBase,
                QuadratureGridUpsampler,
                QuadratureInteriorFacesGridUpsampler)

        if isinstance(expr.op, (
            DiffOperatorBase, FluxOperatorBase,
            MassOperatorBase)):
            return expr
        elif isinstance(expr.op, (QuadratureGridUpsampler,
            QuadratureInteriorFacesGridUpsampler)):
            return self.desired_quad_op(expr.field)
        else:
            return IdentityMapper.map_operator_binding(self, expr)

# }}}

# {{{ simplification / optimization -------------------------------------------
class CommutativeConstantFoldingMapper(
        pymbolic.mapper.constant_folder.CommutativeConstantFoldingMapper,
        IdentityMapperMixin):

    def __init__(self):
        pymbolic.mapper.constant_folder.CommutativeConstantFoldingMapper.__init__(self)
        self.dep_mapper = DependencyMapper()

    def is_constant(self, expr):
        return not bool(self.dep_mapper(expr))

    def map_operator_binding(self, expr):
        field = self.rec(expr.field)

        from hedge.tools import is_zero
        if is_zero(field):
            return 0

        from hedge.optemplate.operators import FluxOperatorBase
        from hedge.optemplate.primitives import BoundaryPair

        if isinstance(expr.op, FluxOperatorBase):
            if isinstance(field, BoundaryPair):
                return self.remove_zeros_from_boundary_flux(expr.op, field)
            else:
                return self.remove_zeros_from_interior_flux(expr.op, field)
        else:
            return expr.op(field)

    # {{{ remove zeros from interior flux
    def remove_zeros_from_interior_flux(self, op, vol_field):
        from pytools.obj_array import is_obj_array
        if not is_obj_array(vol_field):
            vol_field = [vol_field]

        from hedge.flux import FieldComponent
        subst_map = {}

        from hedge.tools import is_zero, make_obj_array

        # process volume field
        new_vol_field = []
        new_idx = 0
        for i, flux_arg in enumerate(vol_field):
            flux_arg = self.rec(flux_arg)

            if is_zero(flux_arg):
                subst_map[FieldComponent(i, is_interior=True)] = 0
                subst_map[FieldComponent(i, is_interior=False)] = 0
            else:
                new_vol_field.append(flux_arg)
                subst_map[FieldComponent(i, is_interior=True)] = \
                        FieldComponent(new_idx, is_interior=True)
                subst_map[FieldComponent(i, is_interior=False)] = \
                        FieldComponent(new_idx, is_interior=False)
                new_idx += 1

        # substitute results into flux
        def sub_flux(expr):
            return subst_map.get(expr, expr)

        from hedge.flux import FluxSubstitutionMapper
        new_flux = FluxSubstitutionMapper(sub_flux)(op.flux)

        if is_zero(new_flux):
            return 0
        else:
            return type(op)(new_flux, *op.__getinitargs__()[1:])(
                    make_obj_array(new_vol_field))

    # }}}

    # {{{ remove zeros from boundary flux
    def remove_zeros_from_boundary_flux(self, op, bpair):
        vol_field = bpair.field
        bdry_field = bpair.bfield
        from pytools.obj_array import is_obj_array
        if not is_obj_array(vol_field):
            vol_field = [vol_field]
        if not is_obj_array(bdry_field):
            bdry_field = [bdry_field]

        from hedge.flux import FieldComponent
        subst_map = {}

        # process volume field
        from hedge.tools import is_zero, make_obj_array
        new_vol_field = []
        new_idx = 0
        for i, flux_arg in enumerate(vol_field):
            fc = FieldComponent(i, is_interior=True)
            flux_arg = self.rec(flux_arg)

            if is_zero(flux_arg):
                subst_map[fc] = 0
            else:
                new_vol_field.append(flux_arg)
                subst_map[fc] = FieldComponent(new_idx, is_interior=True)
                new_idx += 1


        # process boundary field
        new_bdry_field = []
        new_idx = 0
        for i, flux_arg in enumerate(bdry_field):
            fc = FieldComponent(i, is_interior=False)
            flux_arg = self.rec(flux_arg)

            if is_zero(flux_arg):
                subst_map[fc] = 0
            else:
                new_bdry_field.append(flux_arg)
                subst_map[fc] = FieldComponent(new_idx, is_interior=False)
                new_idx += 1

        # substitute results into flux
        def sub_flux(expr):
            return subst_map.get(expr, expr)

        from hedge.flux import FluxSubstitutionMapper
        new_flux = FluxSubstitutionMapper(sub_flux)(op.flux)

        if is_zero(new_flux):
            return 0
        else:
            from hedge.optemplate.primitives import BoundaryPair
            return type(op)(new_flux, *op.__getinitargs__()[1:])(
                    BoundaryPair(
                        make_obj_array(new_vol_field),
                        make_obj_array(new_bdry_field),
                        bpair.tag))

    # }}}




class EmptyFluxKiller(CSECachingMapperMixin, IdentityMapper):
    def __init__(self, mesh):
        IdentityMapper.__init__(self)
        self.mesh = mesh

    map_common_subexpression_uncached = \
            IdentityMapper.map_common_subexpression

    def map_operator_binding(self, expr):
        from hedge.optemplate import BoundaryFluxOperatorBase

        if (isinstance(expr.op, BoundaryFluxOperatorBase) and
            len(self.mesh.tag_to_boundary.get(expr.op.boundary_tag, [])) == 0):
            return 0
        else:
            return IdentityMapper.map_operator_binding(self, expr)




class _InnerDerivativeJoiner(pymbolic.mapper.RecursiveMapper):
    def map_operator_binding(self, expr, derivatives):
        from hedge.optemplate import DifferentiationOperator

        if isinstance(expr.op, DifferentiationOperator):
            derivatives.setdefault(expr.op, []).append(expr.field)
            return 0
        else:
            return DerivativeJoiner()(expr)

    def map_common_subexpression(self, expr, derivatives):
        # no use preserving these if we're moving derivatives around
        return self.rec(expr.child, derivatives)

    def map_sum(self, expr, derivatives):
        from pymbolic.primitives import flattened_sum
        return flattened_sum(tuple(
            self.rec(child, derivatives) for child in expr.children))

    def map_product(self, expr, derivatives):
        from hedge.optemplate.tools import is_scalar
        from pytools import partition
        scalars, nonscalars = partition(is_scalar, expr.children)

        if len(nonscalars) != 1:
            return DerivativeJoiner()(expr)
        else:
            from pymbolic import flattened_product
            factor = flattened_product(scalars)
            nonscalar, = nonscalars

            sub_derivatives = {}
            nonscalar = self.rec(nonscalar, sub_derivatives)
            def do_map(expr):
                if is_scalar(expr):
                    return expr
                else:
                    return self.rec(expr, derivatives)

            for operator, operands in sub_derivatives.iteritems():
                for operand in operands:
                    derivatives.setdefault(operator, []).append(
                            factor*operand)

            return factor*nonscalar

    def map_constant(self, expr, *args):
        return DerivativeJoiner()(expr)

    def map_scalar_parameter(self, expr, *args):
        return DerivativeJoiner()(expr)

    def map_if_positive(self, expr, *args):
        return DerivativeJoiner()(expr)

    def map_power(self, expr, *args):
        return DerivativeJoiner()(expr)

    # these two are necessary because they're forwarding targets
    def map_algebraic_leaf(self, expr, *args):
        return DerivativeJoiner()(expr)

    def map_quotient(self, expr, *args):
        return DerivativeJoiner()(expr)




class DerivativeJoiner(CSECachingMapperMixin, IdentityMapper):
    """Joins derivatives:

    .. math::

        \frac{\partial A}{\partial x} + \frac{\partial B}{\partial x}
        \rightarrow
        \frac{\partial (A+B)}{\partial x}.
    """
    map_common_subexpression_uncached = \
            IdentityMapper.map_common_subexpression

    def map_sum(self, expr):
        idj = _InnerDerivativeJoiner()

        def invoke_idj(expr):
            sub_derivatives = {}
            result = idj(expr, sub_derivatives)
            if not sub_derivatives:
                return expr
            else:
                for operator, operands in sub_derivatives.iteritems():
                    derivatives.setdefault(operator, []).extend(operands)

                return result

        derivatives = {}
        new_children = [invoke_idj(child)
                for child in expr.children]

        for operator, operands in derivatives.iteritems():
            new_children.insert(0, operator(
                sum(self.rec(operand) for operand in operands)))

        from pymbolic.primitives import flattened_sum
        return flattened_sum(new_children)




class _InnerInverseMassContractor(pymbolic.mapper.RecursiveMapper):
    def __init__(self, outer_mass_contractor):
        self.outer_mass_contractor = outer_mass_contractor
        self.extra_operator_count = 0

    def map_constant(self, expr):
        from hedge.tools import is_zero
        from hedge.optemplate import InverseMassOperator, OperatorBinding

        if is_zero(expr):
            return 0
        else:
            return OperatorBinding(
                    InverseMassOperator(),
                    self.outer_mass_contractor(expr))

    def map_algebraic_leaf(self, expr):
        from hedge.optemplate import InverseMassOperator, OperatorBinding

        return OperatorBinding(
                InverseMassOperator(),
                self.outer_mass_contractor(expr))

    def map_operator_binding(self, binding):
        from hedge.optemplate import (
                MassOperator, StiffnessOperator, StiffnessTOperator,
                DifferentiationOperator,
                MInvSTOperator, InverseMassOperator,
                FluxOperator, BoundaryFluxOperator)

        if isinstance(binding.op, MassOperator):
            return binding.field
        elif isinstance(binding.op, StiffnessOperator):
            return DifferentiationOperator(binding.op.xyz_axis)(
                    self.outer_mass_contractor(binding.field))
        elif isinstance(binding.op, StiffnessTOperator):
            return MInvSTOperator(binding.op.xyz_axis)(
                    self.outer_mass_contractor(binding.field))
        elif isinstance(binding.op, FluxOperator):
            assert not binding.op.is_lift

            return FluxOperator(binding.op.flux, is_lift=True)(
                    self.outer_mass_contractor(binding.field))
        elif isinstance(binding.op, BoundaryFluxOperator):
            assert not binding.op.is_lift

            return BoundaryFluxOperator(binding.op.flux,
                        binding.op.boundary_tag, is_lift=True)(
                    self.outer_mass_contractor(binding.field))
        else:
            self.extra_operator_count += 1
            return InverseMassOperator()(
                self.outer_mass_contractor(binding))

    def map_sum(self, expr):
        return expr.__class__(tuple(self.rec(child) for child in expr.children))

    def map_product(self, expr):
        from hedge.optemplate import (
                InverseMassOperator, OperatorBinding, ScalarParameter)

        def is_scalar(expr):
            return isinstance(expr, (int, float, complex, ScalarParameter))

        from pytools import len_iterable
        nonscalar_count = len_iterable(ch
                for ch in expr.children
                if not is_scalar(ch))

        if nonscalar_count > 1:
            # too complicated, don't touch it
            self.extra_operator_count += 1
            return InverseMassOperator()(
                    self.outer_mass_contractor(expr))
        else:
            def do_map(expr):
                if is_scalar(expr):
                    return expr
                else:
                    return self.rec(expr)
            return expr.__class__(tuple(
                do_map(child) for child in expr.children))





class InverseMassContractor(CSECachingMapperMixin, IdentityMapper):
    # assumes all operators to be bound
    map_common_subexpression_uncached = \
            IdentityMapper.map_common_subexpression

    def map_boundary_pair(self, bp):
        from hedge.optemplate import BoundaryPair
        return BoundaryPair(self.rec(bp.field), self.rec(bp.bfield), bp.tag)

    def map_operator_binding(self, binding):
        # we only care about bindings of inverse mass operators
        from hedge.optemplate import InverseMassOperator

        if isinstance(binding.op, InverseMassOperator):
            iimc = _InnerInverseMassContractor(self)
            proposed_result = iimc(binding.field)
            if iimc.extra_operator_count > 1:
                # We're introducing more work than we're saving.
                # Don't perform the simplification
                return binding.op(self.rec(binding.field))
            else:
                return proposed_result
        else:
            return binding.op(self.rec(binding.field))




# }}}

# {{{ error checker -----------------------------------------------------------
class ErrorChecker(CSECachingMapperMixin, IdentityMapper):
    map_common_subexpression_uncached = \
            IdentityMapper.map_common_subexpression

    def __init__(self, mesh):
        self.mesh = mesh

    def map_operator_binding(self, expr):
        from hedge.optemplate import DiffOperatorBase

        if isinstance(expr.op, DiffOperatorBase):
            if (self.mesh is not None
                    and expr.op.xyz_axis >= self.mesh.dimensions):
                raise ValueError("optemplate tries to differentiate along a "
                        "non-existent axis (e.g. Z in 2D)")

        # FIXME: Also check fluxes
        return IdentityMapper.map_operator_binding(self, expr)

    def map_normal(self, expr):
        if self.mesh is not None and expr.axis >= self.mesh.dimensions:
            raise ValueError("optemplate tries to differentiate along a "
                    "non-existent axis (e.g. Z in 2D)")

        return expr




# }}}

# {{{ collectors for various optemplate components --------------------------------
class CollectorMixin(OperatorReducerMixin, LocalOpReducerMixin, FluxOpReducerMixin):
    def combine(self, values):
        from pytools import flatten
        return set(flatten(values))

    def map_constant(self, bpair):
        return set()

    map_operator = map_constant
    map_variable = map_constant
    map_normal_component = map_constant
    map_jacobian = map_constant
    map_forward_metric_derivative = map_constant
    map_inverse_metric_derivative = map_constant
    map_scalar_parameter = map_constant
    map_c_function = map_constant

    def map_whole_domain_flux(self, expr):
        result = set()

        for ii in expr.interiors:
            result.update(self.rec(ii.field_expr))

        for bi in expr.boundaries:
            result.update(self.rec(bi.bpair.field))
            result.update(self.rec(bi.bpair.bfield))

        return result




class FluxCollector(CSECachingMapperMixin, CollectorMixin, CombineMapper):
    map_common_subexpression_uncached = \
            CombineMapper.map_common_subexpression

    def map_operator_binding(self, expr):
        from hedge.optemplate import FluxOperatorBase

        if isinstance(expr.op, FluxOperatorBase):
            result = set([expr])
        else:
            result = set()

        return result | CombineMapper.map_operator_binding(self, expr)

    def map_whole_domain_flux(self, wdflux):
        result = set([wdflux])

        for intr in wdflux.interiors:
            result |= self.rec(intr.field_expr)
        for bdry in wdflux.boundaries:
            result |= self.rec(bdry.bpair)

        return result




class BoundaryTagCollector(CollectorMixin, CombineMapper):
    def map_boundary_pair(self, bpair):
        return set([bpair.tag])




class GeometricFactorCollector(CollectorMixin, CombineMapper):
    def map_jacobian(self, expr):
        return set([expr])

    map_forward_metric_derivative = map_jacobian
    map_inverse_metric_derivative = map_jacobian




class BoundOperatorCollector(CSECachingMapperMixin, CollectorMixin, CombineMapper):
    def __init__(self, op_class):
        self.op_class = op_class

    map_common_subexpression_uncached = \
            CombineMapper.map_common_subexpression

    def map_operator_binding(self, expr):
        if isinstance(expr.op, self.op_class):
            result = set([expr])
        else:
            result = set()

        return result | CombineMapper.map_operator_binding(self, expr)

class FluxExchangeCollector(CSECachingMapperMixin, CollectorMixin, CombineMapper):
    map_common_subexpression_uncached = \
            CombineMapper.map_common_subexpression

    def map_flux_exchange(self, expr):
        return set([expr])

# }}}

# {{{ evaluation --------------------------------------------------------------
class Evaluator(pymbolic.mapper.evaluator.EvaluationMapper):
    def map_boundary_pair(self, bp):
        from hedge.optemplate.primitives import BoundaryPair
        return BoundaryPair(self.rec(bp.field), self.rec(bp.bfield), bp.tag)
# }}}

# {{{ boundary combiner (used by CUDA backend) --------------------------------
class BoundaryCombiner(CSECachingMapperMixin, IdentityMapper):
    """Combines inner fluxes and boundary fluxes into a
    single, whole-domain operator of type
    :class:`hedge.optemplate.operators.WholeDomainFluxOperator`.
    """
    def __init__(self, mesh):
        self.mesh = mesh

    map_common_subexpression_uncached = \
            IdentityMapper.map_common_subexpression

    def gather_one_wdflux(self, expressions):
        from hedge.optemplate.primitives import OperatorBinding, BoundaryPair
        from hedge.optemplate.operators import WholeDomainFluxOperator

        interiors = []
        boundaries = []
        is_lift = None
        # Since None is a valid value of quadrature tags, use
        # the empty list to symbolize "not known", and add to
        # list once something is known.
        quad_tag = []

        rest = []

        for ch in expressions:
            from hedge.optemplate.operators import FluxOperatorBase
            if (isinstance(ch, OperatorBinding)
                    and isinstance(ch.op, FluxOperatorBase)):
                skip = False

                my_is_lift = ch.op.is_lift

                if is_lift is None:
                    is_lift = my_is_lift
                else:
                    if is_lift != my_is_lift:
                        skip = True

                from hedge.optemplate.operators import \
                        QuadratureFluxOperatorBase

                if isinstance(ch.op, QuadratureFluxOperatorBase):
                    my_quad_tag = ch.op.quadrature_tag
                else:
                    my_quad_tag = None

                if quad_tag:
                    if quad_tag[0] != my_quad_tag:
                        skip = True
                else:
                    quad_tag.append(my_quad_tag)

                if skip:
                    rest.append(ch)
                    continue

                if isinstance(ch.field, BoundaryPair):
                    bpair = self.rec(ch.field)
                    if self.mesh.tag_to_boundary.get(bpair.tag, []):
                        boundaries.append(WholeDomainFluxOperator.BoundaryInfo(
                            flux_expr=ch.op.flux,
                            bpair=bpair))
                else:
                    interiors.append(WholeDomainFluxOperator.InteriorInfo(
                            flux_expr=ch.op.flux,
                            field_expr=self.rec(ch.field)))
            else:
                rest.append(ch)

        if interiors or boundaries:
            wdf = WholeDomainFluxOperator(
                    is_lift=is_lift,
                    interiors=interiors,
                    boundaries=boundaries,
                    quadrature_tag=quad_tag[0])
        else:
            wdf = None
        return wdf, rest

    def map_operator_binding(self, expr):
        from hedge.optemplate.operators import FluxOperatorBase
        if isinstance(expr.op, FluxOperatorBase):
            wdf, rest = self.gather_one_wdflux([expr])
            assert not rest
            return wdf
        else:
            return IdentityMapper \
                    .map_operator_binding(self, expr)

    def map_sum(self, expr):
        # FIXME: With flux joining now in the compiler, this is
        # probably now unnecessary.

        from pymbolic.primitives import flattened_sum

        result = 0
        expressions = expr.children
        while True:
            wdf, expressions = self.gather_one_wdflux(expressions)
            if wdf is not None:
                result += wdf
            else:
                return result + flattened_sum(self.rec(r_i) for r_i in expressions)
# }}}



# vim: foldmethod=marker
