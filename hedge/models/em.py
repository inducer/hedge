# -*- coding: utf8 -*-
"""Hedge operators modelling electromagnetic phenomena."""

from __future__ import division

__copyright__ = "Copyright (C) 2007, 2010 Andreas Kloeckner, David Powell"

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

from pytools import memoize_method

import hedge.mesh
from hedge.models import HyperbolicOperator

# TODO: Fix logging
# QUESTION: What's broken about it?

# TODO: Check PML
# TODO: Revive LF (see comment below)




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
            incident_bc=None, current=None, dimensions=None):
        """
        :param flux_type: can be in [0,1] for anything between central and upwind,
          or "lf" for Lax-Friedrichs (currently disabled).  !!!!!!!!!!!!!!!
        :param epsilon: can be a number, for fixed material throughout the computation
          domain, or a TimeConstantGivenFunction for spatially variable material coefficients
        :param mu: can be a number, for fixed material throughout the computation
          domain, or a TimeConstantGivenFunction for spatially variable material coefficients
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

        from math import sqrt

        self.epsilon = epsilon
        self.mu = mu

        from hedge.data import TimeConstantGivenFunction
        from pymbolic.primitives import is_constant

        self.fixed_material = is_constant(epsilon)

        if self.fixed_material != is_constant(mu):
            raise RuntimeError("mu and epsilon must both be "
                    "either hedge.data quantities or constants")

        if self.fixed_material:
            self.fixed_material = True
            self.Z = sqrt(mu/epsilon)
            self.Y = 1/self.Z

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
        warn("MaxwellOperator.c is deprecated")
        if not self.fixed_material:
            raise RuntimeError("Cannot compute speed of light "
                    "for non-constant material")

        return 1/(self.mu*self.epsilon)**0.5

    def flux(self, flux_type):
        """The template for the numerical flux for variable coefficients.

        :param flux_type: can be in [0,1] for anything between central and upwind,
          or "lf" for Lax-Friedrichs (currently disabled).

        As per Hesthaven and Warburton page 433
        """
        from hedge.flux import (make_normal, FluxVectorPlaceholder,
                FluxConstantPlaceholder)
        from hedge.tools import join_fields

        normal = make_normal(self.dimensions)

        if self.fixed_material:
            from hedge.tools import count_subset
            w = FluxVectorPlaceholder(count_subset(self.get_eh_subset()))

            e, h = self.split_eh(w)
            Y = FluxConstantPlaceholder(self.Y)
            Z = FluxConstantPlaceholder(self.Z)
        else:
            from hedge.tools import count_subset
            w = FluxVectorPlaceholder(count_subset(self.get_eh_subset())+2)

            Y, Z, e, h = self.split_yzeh(w)

        if flux_type == "lf":
            if not self.fixed_material:
                raise RuntimeError("L-F flux doesn't support variable "
                        "materials yet")

            # PROBLEM: We cannot do L-F properly if we don't have both
            # epsilon and mu. I think we should be passing that around
            # instead of Y and Z.
            return join_fields(
                    # flux e,
                    1/2*(
                        -1/self.epsilon*self.space_cross_h(normal, h.int-h.ext)
                        -self.c/2*(e.int-e.ext)
                    ),
                    # flux h
                    1/2*(
                        1/self.mu*self.space_cross_e(normal, e.int-e.ext)
                        -self.c/2*(h.int-h.ext))
                    )
        elif isinstance(flux_type, (int, float)):
            # see doc/maxima/maxwell.mac
            return join_fields(
                    # flux e,
                    (
                        -1/(Z.int+Z.ext)*self.space_cross_h(normal,
                            Z.ext*(h.int-h.ext)
                            -flux_type*self.space_cross_e(normal, e.int-e.ext))
                        ),
                    # flux h
                    (
                        1/(Y.int + Y.ext)*self.space_cross_e(normal,
                            Y.ext*(e.int-e.ext)
                            +flux_type*self.space_cross_h(normal, h.int-h.ext))
                        ),
                    )
        else:
            raise ValueError("maxwell: invalid flux_type (%s)"
                    % self.flux_type)

    def local_derivatives(self, w=None):
        """Template for the spatial derivatives of the relevant components of E and H"""
        e, h = self.split_eh(self.field_placeholder(w))

        def e_curl(field):
            return self.space_cross_e(nabla, field)

        def h_curl(field):
            return self.space_cross_h(nabla, field)

        from hedge.optemplate import make_nabla
        from hedge.tools import join_fields, count_subset

        nabla = make_nabla(self.dimensions)

        if self.current is not None:
            from hedge.optemplate import make_vector_field
            j = make_vector_field("j",
                    count_subset(self.get_eh_subset()[:3]))
        else:
            j = 0

        # in conservation form: u_t + A u_x = 0
        return join_fields(
                (j - h_curl(h)),
                e_curl(e)
                )

    def field_placeholder(self, w=None):
        "A placeholder for E and H."
        from hedge.tools import count_subset
        fld_cnt = count_subset(self.get_eh_subset())
        if w is None:
            from hedge.optemplate import make_vector_field
            w = make_vector_field("w", fld_cnt)

        return w

    def yz_field_placeholder(self, w=None):
        "A placeholder for Y, Z, E and H."

        from hedge.tools import count_subset
        fld_cnt = count_subset(self.get_eh_subset())
        if w is None:
            from hedge.optemplate import make_vector_field
            w = make_vector_field("w", fld_cnt+2)

        return w

    def pec_bc(self, w=None):
        "Construct part of the flux operator template for PEC boundary conditions"
        if self.fixed_material:
            e, h = self.split_eh(self.field_placeholder(w))
        else:
            y, z, e, h = self.split_yzeh(self.yz_field_placeholder(w))

        from hedge.tools import join_fields
        from hedge.optemplate import BoundarizeOperator
        pec_e = BoundarizeOperator(self.pec_tag)(e)
        pec_h = BoundarizeOperator(self.pec_tag)(h)

        if self.fixed_material:
            return join_fields(-pec_e, pec_h)
        else:
            pec_y = BoundarizeOperator(self.pec_tag)(y)
            pec_z = BoundarizeOperator(self.pec_tag)(z)
            return join_fields(pec_y, pec_z, -pec_e, pec_h)

    def pmc_bc(self, w=None):
        "Construct part of the flux operator template for PMC boundary conditions"
        if self.fixed_material:
            e, h = self.split_eh(self.field_placeholder(w))
        else:
            y, z, e, h = self.split_yzeh(self.yz_field_placeholder(w))

        from hedge.tools import join_fields
        from hedge.optemplate import BoundarizeOperator
        pmc_e = BoundarizeOperator(self.pmc_tag)(e)
        pmc_h = BoundarizeOperator(self.pmc_tag)(h)

        if self.fixed_material:
            return join_fields(pmc_e, -pmc_h)
        else:
            pmc_y = BoundarizeOperator(self.pmc_tag)(y)
            pmc_z = BoundarizeOperator(self.pmc_tag)(z)
            return join_fields(pmc_y, pmc_z, pmc_e, -pmc_h)

    def absorbing_bc(self, w=None):
        """Construct part of the flux operator template for 1st order
        absorbing boundary conditions.
        """

        from hedge.optemplate import make_normal
        absorb_normal = make_normal(self.absorb_tag, self.dimensions)

        from hedge.optemplate import BoundarizeOperator
        from hedge.tools import join_fields

        if self.fixed_material:
            e, h = self.split_eh(self.field_placeholder(w))

            absorb_Y = self.Y
            absorb_Z = self.Z
        else:
            Y, Z, e, h = self.split_yzeh(self.yz_field_placeholder(w))

            absorb_Y = BoundarizeOperator(self.absorb_tag)(Y)
            absorb_Z = BoundarizeOperator(self.absorb_tag)(Z)

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

        if self.fixed_material:
            return bc
        else:
            return join_fields(absorb_Y, absorb_Z, bc)

    def incident_bc(self, w=None):
        "Flux terms for incident boundary conditions, currently untested"
        if self.fixed_material:
            e, h = self.split_eh(self.field_placeholder(w))
        else:
            from hedge.optemplate import BoundarizeOperator
            Y, Z, e, h = self.split_yzeh(self.yz_field_placeholder(w))
            incident_Y = BoundarizeOperator(self.incident_tag)(Y)
            incident_Z = BoundarizeOperator(self.incident_tag)(Z)

        from hedge.tools import count_subset
        from hedge.tools import join_fields
        fld_cnt = count_subset(self.get_eh_subset())

        if self.incident_bc_data is not None:
            from hedge.tools.symbolic import make_common_subexpression
            from hedge.optemplate import make_vector_field
            inc_field = make_common_subexpression(
                   -make_vector_field("incident_bc", fld_cnt))

            if self.fixed_material:
                return inc_field
            else:
                return join_fields(incident_Y, incident_Z, inc_field)

        else:
            from hedge.tools import make_obj_array
            if self.fixed_material:
                return make_obj_array([0]*fld_cnt)
            else:
                return join_fields(incident_Y, incident_Z, make_obj_array([0]*fld_cnt))

    def op_template(self, w=None):
        """The full operator template - the high level description of 
        the Maxwell operator.

        Combines the relevant operator templates for spatial 
        derivatives, flux, boundary conditions etc.
        """
        from hedge.optemplate import Field
        from hedge.tools import join_fields
        w = self.field_placeholder(w)

        if self.fixed_material:
            flux_w = w
        else:
            epsilon = Field("epsilon")
            mu = Field("mu")

            from hedge.tools.symbolic import make_common_subexpression as cse
            Y = cse((epsilon/mu)**0.5, "Y")
            Z = cse(1/Y, "Z")

            # The above amounts to a volume calculation of Y and Z
            # --far more computation and mem bandwidth waste than
            # getting Y and Z from epsilon and mu within the fluxes.

            flux_w = join_fields(Y, Z, w)

        from hedge.optemplate import BoundaryPair, \
                InverseMassOperator, get_flux_operator

        flux_op = get_flux_operator(self.flux(self.flux_type))
        bdry_flux_op = get_flux_operator(self.flux(self.bdry_flux_type))

        from hedge.tools.indexing import count_subset
        elec_components = count_subset(self.get_eh_subset()[0:3])
        mag_components = count_subset(self.get_eh_subset()[3:6])

        if self.fixed_material:
            # need to check this
            material_divisor = [self.epsilon]*elec_components+[self.mu]*mag_components
        else:
            material_divisor = join_fields([epsilon]*elec_components, [mu]*mag_components)

        return (- self.local_derivatives(w) \
                + InverseMassOperator()*(
                    flux_op(flux_w)
                    +bdry_flux_op(BoundaryPair(
                        flux_w, self.pec_bc(flux_w), self.pec_tag))
                    +bdry_flux_op(BoundaryPair(
                        flux_w, self.pmc_bc(flux_w), self.pmc_tag))
                    +bdry_flux_op(BoundaryPair(
                        flux_w, self.absorbing_bc(flux_w), self.absorb_tag))
                    +bdry_flux_op(BoundaryPair(
                        flux_w, self.incident_bc(flux_w), self.incident_tag))
                    )) / material_divisor

    def bind(self, discr, **extra_context):
        "Convert the abstract operator template into compiled code."
        from hedge.mesh import check_bc_coverage
        check_bc_coverage(discr.mesh, [
            self.pec_tag, self.absorb_tag, self.incident_tag])

        compiled_op_template = discr.compile(self.op_template())

        from hedge.tools import full_to_subset_indices
        e_indices = full_to_subset_indices(self.get_eh_subset()[0:3])
        all_indices = full_to_subset_indices(self.get_eh_subset())

        def rhs(t, w):
            if self.current is not None:
                j = self.current.volume_interpolant(t, discr)[e_indices]
            else:
                j = 0

            if self.incident_bc_data is not None:
                incident_bc_data = self.incident_bc_data.boundary_interpolant(
                        t, discr, self.incident_tag)[all_indices]
            else:
                incident_bc_data = 0

            kwargs = {}
            kwargs.update(extra_context)
            if not self.fixed_material:
                kwargs["epsilon"] = self.epsilon.volume_interpolant(t, discr)
                kwargs["mu"] = self.mu.volume_interpolant(t, discr)

            return compiled_op_template(
                    w=w, j=j, incident_bc=incident_bc_data, 
                    **kwargs)

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

    def split_yzeh(self, w):
        "Splits an array into Y, Z, E and H components"
        e_idx, h_idx = self.partial_to_eh_subsets()
        y, z, e, h = w[[0]], w[[1]], w[e_idx+2], w[h_idx+2]

        from hedge.flux import FluxVectorPlaceholder as FVP
        if isinstance(w, FVP):
            return FVP(scalars=y), FVP(scalars=z), FVP(scalars=e), FVP(scalars=h)
        else:
            from hedge.tools import make_obj_array as moa
            return moa(y), moa(z), moa(e), moa(h)

    def split_eh(self, w):
        "Splits an array into E and H components"
        e_idx, h_idx = self.partial_to_eh_subsets()
        e, h = w[e_idx], w[h_idx]

        from hedge.flux import FluxVectorPlaceholder as FVP
        if isinstance(w, FVP):
            return FVP(scalars=e), FVP(scalars=h)
        else:
            from hedge.tools import make_obj_array as moa
            return moa(e), moa(h)

    def get_eh_subset(self):
        """Return a 6-tuple of :class:`bool` objects indicating whether field components
        are to be computed. The fields are numbered in the order specified
        in the class documentation.
        """
        return 6*(True,)

    def max_eigenvalue(self, t, fields=None, discr=None):
        """Return the largest eigenvalue of Maxwell's equations as a hyperbolic system."""
        from math import sqrt
        if self.fixed_material:
            return 1/sqrt(self.epsilon*self.mu)
        else:
            return discr.nodewise_max(
                    (self.epsilon.volume_interpolant(t, discr)
                        *self.mu.volume_interpolant(t, discr))**(-0.5))




class TMMaxwellOperator(MaxwellOperator):
    """A 2D TM Maxwell operator with PEC boundaries.

    Field order is [Ez Hx Hy].
    """

    _default_dimensions = 2

    def get_eh_subset(self):
        return (
                (False,False,True) # only ez
                +
                (True,True,False) # hx and hy
                )




class TEMaxwellOperator(MaxwellOperator):
    """A 2D TE Maxwell operator.

    Field order is [Ex Ey Hz].
    """

    _default_dimensions = 2

    def get_eh_subset(self):
        return (
                (True,True,False) # ex and ey
                +
                (False,False,True) # only hz
                )

class TE1DMaxwellOperator(MaxwellOperator):
    """A 1D TE Maxwell operator.

    Field order is [Ex Ey Hz].
    """

    _default_dimensions = 1

    def get_eh_subset(self):
        return (
                (True,True,False)
                +
                (False,False,True)
                )


class SourceFree1DMaxwellOperator(MaxwellOperator):
    """A 1D TE Maxwell operator.

    Field order is [Ey Hz].
    """

    _default_dimensions = 1

    def get_eh_subset(self):
        return (
                (False,True,False)
                +
                (False,False,True)
                )
