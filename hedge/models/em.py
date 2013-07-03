# -*- coding: utf8 -*-
"""Hedge operators modelling electromagnetic phenomena."""

from __future__ import division

__copyright__ = "Copyright (C) 2007, 2010 Andreas Kloeckner, David Powell"

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

from pytools import memoize_method

import hedge.mesh
from hedge.models import HyperbolicOperator
from hedge.optemplate.primitives import make_common_subexpression as cse
from hedge.tools import make_obj_array

# TODO: Check PML


class MaxwellOperator(HyperbolicOperator):
    """A 3D Maxwell operator which supports fixed or variable
    isotropic, non-dispersive, positive epsilon and mu.

    Field order is [Ex Ey Ez Hx Hy Hz].
    """

    _default_dimensions = 3

    def __init__(self, epsilon, mu,
            flux_type,
            bdry_flux_type=None,
            pec_tag=hedge.mesh.TAG_ALL,
            pmc_tag=hedge.mesh.TAG_NONE,
            absorb_tag=hedge.mesh.TAG_NONE,
            incident_tag=hedge.mesh.TAG_NONE,
            incident_bc=lambda maxwell_op, e, h: 0, current=0, dimensions=None):
        """
        :arg flux_type: can be in [0,1] for anything between central and upwind,
          or "lf" for Lax-Friedrichs
        :arg epsilon: can be a number, for fixed material throughout the
            computation domain, or a TimeConstantGivenFunction for spatially
            variable material coefficients
        :arg mu: can be a number, for fixed material throughout the computation
            domain, or a TimeConstantGivenFunction for spatially variable material
            coefficients
        :arg incident_bc_getter: a function of signature *(maxwell_op, e, h)* that
            accepts *e* and *h* as a symbolic object arrays
            returns a symbolic expression for the incident
            boundary condition
        """

        self.dimensions = dimensions or self._default_dimensions

        space_subset = [True]*self.dimensions + [False]*(3-self.dimensions)

        e_subset = self.get_eh_subset()[0:3]
        h_subset = self.get_eh_subset()[3:6]

        from hedge.tools import SubsettableCrossProduct
        self.space_cross_e = SubsettableCrossProduct(
                op1_subset=space_subset,
                op2_subset=e_subset,
                result_subset=h_subset)
        self.space_cross_h = SubsettableCrossProduct(
                op1_subset=space_subset,
                op2_subset=h_subset,
                result_subset=e_subset)

        self.epsilon = epsilon
        self.mu = mu

        from pymbolic.primitives import is_constant
        self.fixed_material = is_constant(epsilon) and is_constant(mu)

        self.flux_type = flux_type
        if bdry_flux_type is None:
            self.bdry_flux_type = flux_type
        else:
            self.bdry_flux_type = bdry_flux_type

        self.pec_tag = pec_tag
        self.pmc_tag = pmc_tag
        self.absorb_tag = absorb_tag
        self.incident_tag = incident_tag

        self.current = current
        self.incident_bc_data = incident_bc

    @property
    def c(self):
        from warnings import warn
        warn("MaxwellOperator.c is deprecated", DeprecationWarning)
        if not self.fixed_material:
            raise RuntimeError("Cannot compute speed of light "
                    "for non-constant material")

        return 1/(self.mu*self.epsilon)**0.5

    def flux(self, flux_type):
        """The template for the numerical flux for variable coefficients.

        :param flux_type: can be in [0,1] for anything between central and upwind,
          or "lf" for Lax-Friedrichs.

        As per Hesthaven and Warburton page 433.
        """
        from hedge.flux import (make_normal, FluxVectorPlaceholder,
                FluxConstantPlaceholder)
        from hedge.tools import join_fields

        normal = make_normal(self.dimensions)

        if self.fixed_material:
            from hedge.tools import count_subset
            w = FluxVectorPlaceholder(count_subset(self.get_eh_subset()))

            e, h = self.split_eh(w)
            epsilon = FluxConstantPlaceholder(self.epsilon)
            mu = FluxConstantPlaceholder(self.mu)

        else:
            from hedge.tools import count_subset
            w = FluxVectorPlaceholder(count_subset(self.get_eh_subset())+2)

            epsilon, mu, e, h = self.split_eps_mu_eh(w)

        Z_int = (mu.int/epsilon.int)**0.5
        Y_int = 1/Z_int
        Z_ext = (mu.ext/epsilon.ext)**0.5
        Y_ext = 1/Z_ext

        if flux_type == "lf":
            if self.fixed_material:
                max_c = (self.epsilon*self.mu)**(-0.5)
            else:
                from hedge.flux import Max
                c_int = (epsilon.int*mu.int)**(-0.5)
                c_ext = (epsilon.ext*mu.ext)**(-0.5)
                max_c = Max(c_int, c_ext)  # noqa

            return join_fields(
                    # flux e,
                    1/2*(
                        -self.space_cross_h(normal, h.int-h.ext)
                        # multiplication by epsilon undoes material divisor below
                        #-max_c*(epsilon.int*e.int - epsilon.ext*e.ext)
                    ),
                    # flux h
                    1/2*(
                        self.space_cross_e(normal, e.int-e.ext)
                        # multiplication by mu undoes material divisor below
                        #-max_c*(mu.int*h.int - mu.ext*h.ext)
                    ))
        elif isinstance(flux_type, (int, float)):
            # see doc/maxima/maxwell.mac
            return join_fields(
                    # flux e,
                    (
                        -1/(Z_int+Z_ext)*self.space_cross_h(normal,
                            Z_ext*(h.int-h.ext)
                            - flux_type*self.space_cross_e(normal, e.int-e.ext))
                        ),
                    # flux h
                    (
                        1/(Y_int + Y_ext)*self.space_cross_e(normal,
                            Y_ext*(e.int-e.ext)
                            + flux_type*self.space_cross_h(normal, h.int-h.ext))
                        ),
                    )
        else:
            raise ValueError("maxwell: invalid flux_type (%s)"
                    % self.flux_type)

    def local_derivatives(self, w=None):
        """Template for the spatial derivatives of the relevant components of
        :math:`E` and :math:`H`
        """

        e, h = self.split_eh(self.field_placeholder(w))

        def e_curl(field):
            return self.space_cross_e(nabla, field)

        def h_curl(field):
            return self.space_cross_h(nabla, field)

        from hedge.optemplate import make_nabla
        from hedge.tools import join_fields

        nabla = make_nabla(self.dimensions)

        # in conservation form: u_t + A u_x = 0
        return join_fields(
                (self.current - h_curl(h)),
                e_curl(e)
                )

    def field_placeholder(self, w=None):
        "A placeholder for E and H."
        from hedge.tools import count_subset
        fld_cnt = count_subset(self.get_eh_subset())
        if w is None:
            from hedge.optemplate import make_sym_vector
            w = make_sym_vector("w", fld_cnt)

        return w

    def pec_bc(self, w=None):
        "Construct part of the flux operator template for PEC boundary conditions"
        e, h = self.split_eh(self.field_placeholder(w))

        from hedge.tools import join_fields
        from hedge.optemplate import BoundarizeOperator
        pec_e = BoundarizeOperator(self.pec_tag)(e)
        pec_h = BoundarizeOperator(self.pec_tag)(h)

        return join_fields(-pec_e, pec_h)

    def pmc_bc(self, w=None):
        "Construct part of the flux operator template for PMC boundary conditions"
        e, h = self.split_eh(self.field_placeholder(w))

        from hedge.tools import join_fields
        from hedge.optemplate import BoundarizeOperator
        pmc_e = BoundarizeOperator(self.pmc_tag)(e)
        pmc_h = BoundarizeOperator(self.pmc_tag)(h)

        return join_fields(pmc_e, -pmc_h)

    def absorbing_bc(self, w=None):
        """Construct part of the flux operator template for 1st order
        absorbing boundary conditions.
        """

        from hedge.optemplate import normal
        absorb_normal = normal(self.absorb_tag, self.dimensions)

        from hedge.optemplate import BoundarizeOperator, Field
        from hedge.tools import join_fields

        e, h = self.split_eh(self.field_placeholder(w))

        if self.fixed_material:
            epsilon = self.epsilon
            mu = self.mu
        else:
            epsilon = cse(
                    BoundarizeOperator(self.absorb_tag)(Field("epsilon")))
            mu = cse(
                    BoundarizeOperator(self.absorb_tag)(Field("mu")))

        absorb_Z = (mu/epsilon)**0.5
        absorb_Y = 1/absorb_Z

        absorb_e = BoundarizeOperator(self.absorb_tag)(e)
        absorb_h = BoundarizeOperator(self.absorb_tag)(h)

        bc = join_fields(
                absorb_e + 1/2*(self.space_cross_h(absorb_normal, self.space_cross_e(
                    absorb_normal, absorb_e))
                    - absorb_Z*self.space_cross_h(absorb_normal, absorb_h)),
                absorb_h + 1/2*(
                    self.space_cross_e(absorb_normal, self.space_cross_h(
                        absorb_normal, absorb_h))
                    + absorb_Y*self.space_cross_e(absorb_normal, absorb_e)))

        return bc

    def incident_bc(self, w=None):
        "Flux terms for incident boundary conditions"
        # NOTE: Untested for inhomogeneous materials, but would usually be
        # physically meaningless anyway (are there exceptions to this?)

        e, h = self.split_eh(self.field_placeholder(w))
        if not self.fixed_material:
            from warnings import warn
            if self.incident_tag != hedge.mesh.TAG_NONE:
                warn("Incident boundary conditions assume homogeneous"
                     " background material, results may be unphysical")

        from hedge.tools import count_subset
        fld_cnt = count_subset(self.get_eh_subset())

        from hedge.tools import is_zero
        incident_bc_data = self.incident_bc_data(self, e, h)
        if is_zero(incident_bc_data):
            return make_obj_array([0]*fld_cnt)
        else:
            return cse(-incident_bc_data)

    def op_template(self, w=None):
        """The full operator template - the high level description of
        the Maxwell operator.

        Combines the relevant operator templates for spatial
        derivatives, flux, boundary conditions etc.
        """
        from hedge.tools import join_fields
        w = self.field_placeholder(w)

        if self.fixed_material:
            flux_w = w
        else:
            epsilon = self.epsilon
            mu = self.mu

            flux_w = join_fields(epsilon, mu, w)

        from hedge.optemplate import BoundaryPair, \
                InverseMassOperator, get_flux_operator

        flux_op = get_flux_operator(self.flux(self.flux_type))
        bdry_flux_op = get_flux_operator(self.flux(self.bdry_flux_type))

        from hedge.tools.indexing import count_subset
        elec_components = count_subset(self.get_eh_subset()[0:3])
        mag_components = count_subset(self.get_eh_subset()[3:6])

        if self.fixed_material:
            # need to check this
            material_divisor = (
                    [self.epsilon]*elec_components+[self.mu]*mag_components)
        else:
            material_divisor = join_fields(
                    [epsilon]*elec_components,
                    [mu]*mag_components)

        tags_and_bcs = [
                (self.pec_tag, self.pec_bc(w)),
                (self.pmc_tag, self.pmc_bc(w)),
                (self.absorb_tag, self.absorbing_bc(w)),
                (self.incident_tag, self.incident_bc(w)),
                ]

        def make_flux_bc_vector(tag, bc):
            if self.fixed_material:
                return bc
            else:
                from hedge.optemplate import BoundarizeOperator
                return join_fields(
                        cse(BoundarizeOperator(tag)(epsilon)),
                        cse(BoundarizeOperator(tag)(mu)),
                        bc)

        return (
                - self.local_derivatives(w)
                + InverseMassOperator()(
                    flux_op(flux_w)
                    + sum(
                        bdry_flux_op(BoundaryPair(
                            flux_w, make_flux_bc_vector(tag, bc), tag))
                        for tag, bc in tags_and_bcs))
                    ) / material_divisor

    def bind(self, discr):
        "Convert the abstract operator template into compiled code."
        from hedge.mesh import check_bc_coverage
        check_bc_coverage(discr.mesh, [
            self.pec_tag, self.absorb_tag, self.incident_tag])

        compiled_op_template = discr.compile(self.op_template())

        def rhs(t, w, **extra_context):
            kwargs = {}
            kwargs.update(extra_context)

            return compiled_op_template(w=w, t=t, **kwargs)

        return rhs

    def assemble_eh(self, e=None, h=None, discr=None):
        "Combines separate E and H vectors into a single array."
        if discr is None:
            def zero():
                return 0
        else:
            def zero():
                return discr.volume_zeros()

        from hedge.tools import count_subset
        e_components = count_subset(self.get_eh_subset()[0:3])
        h_components = count_subset(self.get_eh_subset()[3:6])

        def default_fld(fld, comp):
            if fld is None:
                return [zero() for i in xrange(comp)]
            else:
                return fld

        e = default_fld(e, e_components)
        h = default_fld(h, h_components)

        from hedge.tools import join_fields
        return join_fields(e, h)

    assemble_fields = assemble_eh

    @memoize_method
    def partial_to_eh_subsets(self):
        """Helps find the indices of the E and H components, which can vary
        depending on number of dimensions and whether we have a full/TE/TM
        operator.
        """

        e_subset = self.get_eh_subset()[0:3]
        h_subset = self.get_eh_subset()[3:6]

        from hedge.tools import partial_to_all_subset_indices
        return tuple(partial_to_all_subset_indices(
            [e_subset, h_subset]))

    def split_eps_mu_eh(self, w):
        """Splits an array into epsilon, mu, E and H components.

        Only used for fluxes.
        """
        e_idx, h_idx = self.partial_to_eh_subsets()
        epsilon, mu, e, h = w[[0]], w[[1]], w[e_idx+2], w[h_idx+2]

        from hedge.flux import FluxVectorPlaceholder as FVP
        if isinstance(w, FVP):
            return (
                    FVP(scalars=epsilon),
                    FVP(scalars=mu),
                    FVP(scalars=e),
                    FVP(scalars=h))
        else:
            return epsilon, mu, make_obj_array(e), make_obj_array(h)

    def split_eh(self, w):
        "Splits an array into E and H components"
        e_idx, h_idx = self.partial_to_eh_subsets()
        e, h = w[e_idx], w[h_idx]

        from hedge.flux import FluxVectorPlaceholder as FVP
        if isinstance(w, FVP):
            return FVP(scalars=e), FVP(scalars=h)
        else:
            return make_obj_array(e), make_obj_array(h)

    def get_eh_subset(self):
        """Return a 6-tuple of :class:`bool` objects indicating whether field
        components are to be computed. The fields are numbered in the order
        specified in the class documentation.
        """
        return 6*(True,)

    def max_eigenvalue_expr(self):
        """Return the largest eigenvalue of Maxwell's equations as a hyperbolic
        system.
        """
        from math import sqrt
        if self.fixed_material:
            return 1/sqrt(self.epsilon*self.mu)  # a number
        else:
            import hedge.optemplate as sym
            return sym.NodalMax()(1/sym.CFunction("sqrt")(self.epsilon*self.mu))

    def max_eigenvalue(self, t, fields=None, discr=None, context={}):
        if self.fixed_material:
            return self.max_eigenvalue_expr()
        else:
            raise ValueError("max_eigenvalue is no longer supported for "
                    "variable-coefficient problems--use max_eigenvalue_expr")


class TMMaxwellOperator(MaxwellOperator):
    """A 2D TM Maxwell operator with PEC boundaries.

    Field order is [Ez Hx Hy].
    """

    _default_dimensions = 2

    def get_eh_subset(self):
        return (
                (False, False, True)  # only ez
                +
                (True, True, False)  # hx and hy
                )


class TEMaxwellOperator(MaxwellOperator):
    """A 2D TE Maxwell operator.

    Field order is [Ex Ey Hz].
    """

    _default_dimensions = 2

    def get_eh_subset(self):
        return (
                (True, True, False)  # ex and ey
                +
                (False, False, True)  # only hz
                )


class TE1DMaxwellOperator(MaxwellOperator):
    """A 1D TE Maxwell operator.

    Field order is [Ex Ey Hz].
    """

    _default_dimensions = 1

    def get_eh_subset(self):
        return (
                (True, True, False)
                +
                (False, False, True)
                )


class SourceFree1DMaxwellOperator(MaxwellOperator):
    """A 1D TE Maxwell operator.

    Field order is [Ey Hz].
    """

    _default_dimensions = 1

    def get_eh_subset(self):
        return (
                (False, True, False)
                +
                (False, False, True)
                )
