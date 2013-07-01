"""Building blocks and mappers for operator expression trees."""

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


import numpy as np
import numpy.linalg as la
import pymbolic.primitives
from pytools import Record, memoize_method


# {{{ base classes

class Operator(pymbolic.primitives.Leaf):
    def stringifier(self):
        from hedge.optemplate import StringifyMapper
        return StringifyMapper

    def __call__(self, expr):
        from pytools.obj_array import with_object_array_or_scalar
        from hedge.tools import is_zero

        def bind_one(subexpr):
            if is_zero(subexpr):
                return subexpr
            else:
                from hedge.optemplate.primitives import OperatorBinding
                return OperatorBinding(self, subexpr)

        return with_object_array_or_scalar(bind_one, expr)

    @memoize_method
    def bind(self, discr):
        from hedge.optemplate import Field
        bound_op = discr.compile(self(Field("f")))

        def apply_op(field):
            from hedge.tools import with_object_array_or_scalar
            return with_object_array_or_scalar(lambda f: bound_op(f=f), field)

        return apply_op

    def apply(self, discr, field):
        return self.bind(discr)(field)

    def get_hash(self):
        return hash((self.__class__,) + (self.__getinitargs__()))

    def is_equal(self, other):
        return self.__class__ == other.__class__ and \
                self.__getinitargs__() == other.__getinitargs__()


class StatelessOperator(Operator):
    def __getinitargs__(self):
        return ()

# }}}


# {{{ sum, integral, max

class NodalReductionOperator(StatelessOperator):
    pass


class NodalSum(NodalReductionOperator):
    mapper_method = intern("map_nodal_sum")


class NodalMax(NodalReductionOperator):
    mapper_method = intern("map_nodal_max")


class NodalMin(NodalReductionOperator):
    mapper_method = intern("map_nodal_min")

# }}}


# {{{ differentiation operators

# {{{ global differentiation

class DiffOperatorBase(Operator):
    def __init__(self, xyz_axis):
        Operator.__init__(self)

        self.xyz_axis = xyz_axis

    def __getinitargs__(self):
        return (self.xyz_axis,)

    def preimage_ranges(self, eg):
        return eg.ranges

    def equal_except_for_axis(self, other):
        return (type(self) == type(other)
                # first argument is always the axis
                and self.__getinitargs__()[1:] == other.__getinitargs__()[1:])


class StrongFormDiffOperatorBase(DiffOperatorBase):
    pass


class WeakFormDiffOperatorBase(DiffOperatorBase):
    pass


class StiffnessOperator(StrongFormDiffOperatorBase):
    mapper_method = intern("map_stiffness")


class DifferentiationOperator(StrongFormDiffOperatorBase):
    mapper_method = intern("map_diff")


class StiffnessTOperator(WeakFormDiffOperatorBase):
    mapper_method = intern("map_stiffness_t")


class MInvSTOperator(WeakFormDiffOperatorBase):
    mapper_method = intern("map_minv_st")


class QuadratureStiffnessTOperator(DiffOperatorBase):
    """
    .. note::

        This operator is purely for internal use. It is inserted
        by :class:`hedge.optemplate.mappers.OperatorSpecializer`
        when a :class:`StiffnessTOperator` is applied to a quadrature
        field, and then eliminated by
        :class:`hedge.optemplate.mappers.GlobalToReferenceMapper`
        in favor of operators on the reference element.
    """

    def __init__(self, xyz_axis, quadrature_tag):
        DiffOperatorBase.__init__(self, xyz_axis)
        self.quadrature_tag = quadrature_tag

    def __getinitargs__(self):
        return (self.xyz_axis, self.quadrature_tag)

    mapper_method = intern("map_quad_stiffness_t")


def DiffOperatorVector(els):
    from hedge.tools import join_fields
    return join_fields(*els)

# }}}


# {{{ reference-element differentiation

class ReferenceDiffOperatorBase(Operator):
    def __init__(self, rst_axis):
        Operator.__init__(self)

        self.rst_axis = rst_axis

    def __getinitargs__(self):
        return (self.rst_axis,)

    def preimage_ranges(self, eg):
        return eg.ranges

    def equal_except_for_axis(self, other):
        return (type(self) == type(other)
                # first argument is always the axis
                and self.__getinitargs__()[1:] == other.__getinitargs__()[1:])


class ReferenceDifferentiationOperator(ReferenceDiffOperatorBase):
    @staticmethod
    def matrices(element_group):
        return element_group.differentiation_matrices

    mapper_method = intern("map_ref_diff")


class ReferenceStiffnessTOperator(ReferenceDiffOperatorBase):
    @staticmethod
    def matrices(element_group):
        return element_group.stiffness_t_matrices

    mapper_method = intern("map_ref_stiffness_t")


class ReferenceQuadratureStiffnessTOperator(ReferenceDiffOperatorBase):
    """
    .. note::

        This operator is purely for internal use. It is inserted
        by :class:`hedge.optemplate.mappers.OperatorSpecializer`
        when a :class:`StiffnessTOperator` is applied to a quadrature field.
    """

    def __init__(self, rst_axis, quadrature_tag):
        ReferenceDiffOperatorBase.__init__(self, rst_axis)
        self.quadrature_tag = quadrature_tag

    def __getinitargs__(self):
        return (self.rst_axis, self.quadrature_tag)

    mapper_method = intern("map_ref_quad_stiffness_t")

    def preimage_ranges(self, eg):
        return eg.quadrature_info[self.quadrature_tag].ranges

    def matrices(self, element_group):
        return element_group.quadrature_info[self.quadrature_tag] \
                .ldis_quad_info.stiffness_t_matrices()

# }}}

# }}}


# {{{ elementwise operators

class ElementwiseLinearOperator(Operator):
    def matrix(self, element_group):
        raise NotImplementedError

    def coefficients(self, element_group):
        return None

    mapper_method = intern("map_elementwise_linear")


class ElementwiseMaxOperator(StatelessOperator):
    mapper_method = intern("map_elementwise_max")


# {{{ quadrature upsamplers

class QuadratureGridUpsampler(Operator):
    """In a user-specified optemplate, this operator can be used to interpolate
    volume and boundary data to their corresponding quadrature grids.

    In pre-processing, the boundary quad interpolation is specialized to
    a separate operator, :class:`QuadratureBoundaryGridUpsampler`.
    """
    def __init__(self, quadrature_tag):
        self.quadrature_tag = quadrature_tag

    def __getinitargs__(self):
        return (self.quadrature_tag,)

    mapper_method = intern("map_quad_grid_upsampler")


class QuadratureInteriorFacesGridUpsampler(Operator):
    """Interpolates nodal volume data to interior face data on a quadrature
    grid.

    Note that the "interior faces" grid includes faces lying opposite to the
    boundary.
    """
    def __init__(self, quadrature_tag):
        self.quadrature_tag = quadrature_tag

    def __getinitargs__(self):
        return (self.quadrature_tag,)

    mapper_method = intern("map_quad_int_faces_grid_upsampler")


class QuadratureBoundaryGridUpsampler(Operator):
    """
    .. note::

        This operator is purely for internal use. It is inserted
        by :class:`hedge.optemplate.mappers.OperatorSpecializer`
        when a :class:`MassOperator` is applied to a quadrature field.
    """
    def __init__(self, quadrature_tag, boundary_tag):
        self.quadrature_tag = quadrature_tag
        self.boundary_tag = boundary_tag

    def __getinitargs__(self):
        return (self.quadrature_tag, self.boundary_tag)

    mapper_method = intern("map_quad_bdry_grid_upsampler")

# }}}


# {{{ various elementwise linear operators

class FilterOperator(ElementwiseLinearOperator):
    def __init__(self, mode_response_func):
        """
        :param mode_response_func: A function mapping
          ``(mode_tuple, local_discretization)`` to a float indicating the
          factor by which this mode is to be multiplied after filtering.
          (For example an instance of
          :class:`ExponentialFilterResponseFunction`.
        """
        self.mode_response_func = mode_response_func

    def __getinitargs__(self):
        return (self.mode_response_func,)

    def matrix(self, eg):
        ldis = eg.local_discretization

        filter_coeffs = [self.mode_response_func(mid, ldis)
            for mid in ldis.generate_mode_identifiers()]

        # build filter matrix
        vdm = ldis.vandermonde()
        from hedge.tools import leftsolve
        mat = np.asarray(
            leftsolve(vdm,
                np.dot(vdm, np.diag(filter_coeffs))),
            order="C")

        return mat


class OnesOperator(ElementwiseLinearOperator, StatelessOperator):
    def matrix(self, eg):
        ldis = eg.local_discretization

        node_count = ldis.node_count()
        return np.ones((node_count, node_count), dtype=np.float64)


class AveragingOperator(ElementwiseLinearOperator, StatelessOperator):
    def matrix(self, eg):
        # average matrix, so that AVE*fields = cellaverage(fields)
        # see Hesthaven and Warburton page 227

        mmat = eg.local_discretization.mass_matrix()
        standard_el_vol = np.sum(np.dot(mmat, np.ones(mmat.shape[0])))
        avg_mat_row = np.sum(mmat, 0)/standard_el_vol

        avg_mat = np.zeros((np.size(avg_mat_row), np.size(avg_mat_row)))
        avg_mat[:] = avg_mat_row
        return avg_mat


class InverseVandermondeOperator(ElementwiseLinearOperator, StatelessOperator):
    def matrix(self, eg):
        return np.asarray(
                la.inv(eg.local_discretization.vandermonde()),
                order="C")


class VandermondeOperator(ElementwiseLinearOperator, StatelessOperator):
    def matrix(self, eg):
        return np.asarray(
                eg.local_discretization.vandermonde(),
                order="C")

# }}}

# }}}


# {{{ mass operators

class MassOperatorBase(ElementwiseLinearOperator, StatelessOperator):
    pass


class MassOperator(MassOperatorBase):
    @staticmethod
    def matrix(element_group):
        return element_group.mass_matrix

    @staticmethod
    def coefficients(element_group):
        return element_group.jacobians

    mapper_method = intern("map_mass")


class InverseMassOperator(MassOperatorBase):
    @staticmethod
    def matrix(element_group):
        return element_group.inverse_mass_matrix

    @staticmethod
    def coefficients(element_group):
        return element_group.inverse_jacobians

    mapper_method = intern("map_inverse_mass")


class QuadratureMassOperator(Operator):
    """
    .. note::

        This operator is purely for internal use. It is inserted
        by :class:`hedge.optemplate.mappers.OperatorSpecializer`
        when a :class:`StiffnessTOperator` is applied to a quadrature
        field, and then eliminated by
        :class:`hedge.optemplate.mappers.GlobalToReferenceMapper`
        in favor of operators on the reference element.
    """

    def __init__(self, quadrature_tag):
        self.quadrature_tag = quadrature_tag

    def __getinitargs__(self):
        return (self.quadrature_tag,)

    mapper_method = intern("map_quad_mass")


class ReferenceQuadratureMassOperator(Operator):
    """
    .. note::

        This operator is purely for internal use. It is inserted
        by :class:`hedge.optemplate.mappers.OperatorSpecializer`
        when a :class:`MassOperator` is applied to a quadrature field.
    """

    def __init__(self, quadrature_tag):
        self.quadrature_tag = quadrature_tag

    def __getinitargs__(self):
        return (self.quadrature_tag,)

    mapper_method = intern("map_ref_quad_mass")


class ReferenceMassOperatorBase(MassOperatorBase):
    pass


class ReferenceMassOperator(ReferenceMassOperatorBase):
    @staticmethod
    def matrix(element_group):
        return element_group.mass_matrix

    @staticmethod
    def coefficients(element_group):
        return None

    mapper_method = intern("map_ref_mass")


class ReferenceInverseMassOperator(ReferenceMassOperatorBase):
    @staticmethod
    def matrix(element_group):
        return element_group.inverse_mass_matrix

    @staticmethod
    def coefficients(element_group):
        return None

    mapper_method = intern("map_ref_inverse_mass")

# }}}


# {{{ boundary-related operators

class BoundarizeOperator(Operator):
    def __init__(self, tag):
        self.tag = tag

    def __getinitargs__(self):
        return (self.tag,)

    mapper_method = intern("map_boundarize")


class FluxExchangeOperator(pymbolic.primitives.AlgebraicLeaf):
    """An operator that results in the sending and receiving of
    boundary information for its argument fields.
    """

    def __init__(self, idx, rank, arg_fields):
        self.index = idx
        self.rank = rank
        self.arg_fields = arg_fields

        # only tuples are hashable
        if not isinstance(arg_fields, tuple):
            raise TypeError("FluxExchangeOperator: arg_fields must be a tuple")

    def __getinitargs__(self):
        return (self.index, self.rank, self.arg_fields)

    def get_hash(self):
        return hash((self.__class__, self.index, self.rank, self.arg_fields))

    mapper_method = intern("map_flux_exchange")

    def is_equal(self, other):
        return self.__class__ == other.__class__ and \
                self.__getinitargs__() == other.__getinitargs__()

# }}}


# {{{ flux-like operators

class FluxOperatorBase(Operator):
    def __init__(self, flux, is_lift=False):
        Operator.__init__(self)
        self.flux = flux
        self.is_lift = is_lift

    def get_flux_or_lift_text(self):
        if self.is_lift:
            return "Lift"
        else:
            return "Flux"

    def repr_op(self):
        """Return an equivalent operator with the flux expression set to 0."""
        return type(self)(0, *self.__getinitargs__()[1:])

    def __call__(self, arg):
        # override to suppress apply-operator-to-each-operand
        # behavior from superclass

        from hedge.optemplate.primitives import OperatorBinding
        return OperatorBinding(self, arg)

    def __mul__(self, arg):
        from warnings import warn
        warn("Multiplying by a flux operator is deprecated. "
                "Use the less ambiguous parenthesized syntax instead.",
                DeprecationWarning, stacklevel=2)
        return self.__call__(arg)


class QuadratureFluxOperatorBase(FluxOperatorBase):
    pass


class BoundaryFluxOperatorBase(FluxOperatorBase):
    pass


class FluxOperator(FluxOperatorBase):
    def __getinitargs__(self):
        return (self.flux, self.is_lift)

    mapper_method = intern("map_flux")


class BoundaryFluxOperator(BoundaryFluxOperatorBase):
    """
    .. note::

        This operator is purely for internal use. It is inserted
        by :class:`hedge.optemplate.mappers.OperatorSpecializer`
        when a :class:`FluxOperator` is applied to a boundary field.
    """
    def __init__(self, flux, boundary_tag, is_lift=False):
        FluxOperatorBase.__init__(self, flux, is_lift)
        self.boundary_tag = boundary_tag

    def __getinitargs__(self):
        return (self.flux, self.boundary_tag, self.is_lift)

    mapper_method = intern("map_bdry_flux")


class QuadratureFluxOperator(QuadratureFluxOperatorBase):
    """
    .. note::

        This operator is purely for internal use. It is inserted
        by :class:`hedge.optemplate.mappers.OperatorSpecializer`
        when a :class:`FluxOperator` is applied to a quadrature field.
    """

    def __init__(self, flux, quadrature_tag):
        FluxOperatorBase.__init__(self, flux, is_lift=False)

        self.quadrature_tag = quadrature_tag

    def __getinitargs__(self):
        return (self.flux, self.quadrature_tag)

    mapper_method = intern("map_quad_flux")


class QuadratureBoundaryFluxOperator(
        QuadratureFluxOperatorBase, BoundaryFluxOperatorBase):
    """
    .. note::

        This operator is purely for internal use. It is inserted
        by :class:`hedge.optemplate.mappers.OperatorSpecializer`
        when a :class:`FluxOperator` is applied to a quadrature
        boundary field.
    """
    def __init__(self, flux, quadrature_tag, boundary_tag):
        FluxOperatorBase.__init__(self, flux, is_lift=False)
        self.quadrature_tag = quadrature_tag
        self.boundary_tag = boundary_tag

    def __getinitargs__(self):
        return (self.flux, self.quadrature_tag, self.boundary_tag)

    mapper_method = intern("map_quad_bdry_flux")


class VectorFluxOperator(object):
    """Note that this isn't an actual operator. It's just a placeholder that pops
    out a vector of FluxOperators when applied to an operand.
    """
    def __init__(self, fluxes):
        self.fluxes = fluxes

    def __call__(self, arg):
        if isinstance(arg, int) and arg == 0:
            return 0
        from pytools.obj_array import make_obj_array
        from hedge.optemplate.primitives import OperatorBinding

        return make_obj_array(
                [OperatorBinding(FluxOperator(f), arg)
                    for f in self.fluxes])

    def __mul__(self, arg):
        from warnings import warn
        warn("Multiplying by a vector flux operator is deprecated. "
                "Use the less ambiguous parenthesized syntax instead.",
                DeprecationWarning, stacklevel=2)
        return self.__call__(arg)


class WholeDomainFluxOperator(pymbolic.primitives.AlgebraicLeaf):
    """Used by the CUDA backend to represent a flux computation on the
    whole domain--interior and boundary.

    Unlike other operators, :class:`WholeDomainFluxOperator` instances
    are not bound.
    """

    class FluxInfo(Record):
        __slots__ = []

        def __repr__(self):
            # override because we want flux_expr in infix
            return "%s(%s)" % (
                    self.__class__.__name__,
                    ", ".join("%s=%s" % (fld, getattr(self, fld))
                        for fld in self.__class__.fields
                        if hasattr(self, fld)))

    class InteriorInfo(FluxInfo):
        # attributes: flux_expr, field_expr,

        @property
        @memoize_method
        def dependencies(self):
            from hedge.optemplate.tools import get_flux_dependencies
            return set(get_flux_dependencies(
                self.flux_expr, self.field_expr))

    class BoundaryInfo(FluxInfo):
        # attributes: flux_expr, bpair

        @property
        @memoize_method
        def int_dependencies(self):
            from hedge.optemplate.tools import get_flux_dependencies
            return set(get_flux_dependencies(
                    self.flux_expr, self.bpair, bdry="int"))

        @property
        @memoize_method
        def ext_dependencies(self):
            from hedge.optemplate.tools import get_flux_dependencies
            return set(get_flux_dependencies(
                    self.flux_expr, self.bpair, bdry="ext"))

    def __init__(self, is_lift, interiors, boundaries,
            quadrature_tag):
        from hedge.optemplate.tools import get_flux_dependencies

        self.is_lift = is_lift

        self.interiors = tuple(interiors)
        self.boundaries = tuple(boundaries)
        self.quadrature_tag = quadrature_tag

        from pytools import set_sum
        interior_deps = set_sum(iflux.dependencies
                for iflux in interiors)
        boundary_int_deps = set_sum(bflux.int_dependencies
                for bflux in boundaries)
        boundary_ext_deps = set_sum(bflux.ext_dependencies
                for bflux in boundaries)

        self.interior_deps = list(interior_deps)
        self.boundary_int_deps = list(boundary_int_deps)
        self.boundary_ext_deps = list(boundary_ext_deps)
        self.boundary_deps = list(boundary_int_deps | boundary_ext_deps)

        self.dep_to_tag = {}
        for bflux in boundaries:
            for dep in get_flux_dependencies(
                    bflux.flux_expr, bflux.bpair, bdry="ext"):
                self.dep_to_tag[dep] = bflux.bpair.tag

    def stringifier(self):
        from hedge.optemplate import StringifyMapper
        return StringifyMapper

    def repr_op(self):
        return type(self)(False, [], [], self.quadrature_tag)

    @memoize_method
    def rebuild_optemplate(self):
        def generate_summands():
            for i in self.interiors:
                if self.quadrature_tag is None:
                    yield FluxOperator(
                            i.flux_expr, self.is_lift)(i.field_expr)
                else:
                    yield QuadratureFluxOperator(
                            i.flux_expr, self.quadrature_tag)(i.field_expr)
            for b in self.boundaries:
                if self.quadrature_tag is None:
                    yield BoundaryFluxOperator(
                            b.flux_expr, b.bpair.tag, self.is_lift)(b.bpair)
                else:
                    yield QuadratureBoundaryFluxOperator(
                            b.flux_expr, self.quadrature_tag,
                            b.bpair.tag)(b.bpair)

        from pymbolic.primitives import flattened_sum
        return flattened_sum(generate_summands())

    # infrastructure interaction
    def get_hash(self):
        return hash((self.__class__, self.rebuild_optemplate()))

    def is_equal(self, other):
        return (other.__class__ == WholeDomainFluxOperator
                and self.rebuild_optemplate() == other.rebuild_optemplate())

    def __getinitargs__(self):
        return (self.is_lift, self.interiors, self.boundaries,
                self.quadrature_tag)

    mapper_method = intern("map_whole_domain_flux")

# }}}


# vim: foldmethod=marker
