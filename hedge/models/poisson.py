# -*- coding: utf8 -*-
"""Operators for Poisson problems."""

from __future__ import division

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
import numpy.linalg as la

from hedge.models import Operator
import hedge.data
import hedge.iterative
from pytools import memoize_method




class WeakPoissonOperator(Operator):
    """Implements the Local Discontinuous Galerkin (LDG) Method for elliptic
    operators.

    See P. Castillo et al.,
    Local discontinuous Galerkin methods for elliptic problems",
    Communications in Numerical Methods in Engineering 18, no. 1 (2002): 69-75.
    """

    def __init__(self, dimensions, diffusion_tensor=None,
            dirichlet_bc=hedge.data.ConstantGivenFunction(), dirichlet_tag="dirichlet",
            neumann_bc=hedge.data.ConstantGivenFunction(), neumann_tag="neumann",
            flux="ip"):
        """Initialize the weak Poisson operator.

        :param flux: Either *"ip"* or *"ldg"* to indicate which type of flux is
            to be used. IP tends to be faster, and is therefore the default.
        """
        self.dimensions = dimensions
        assert isinstance(dimensions, int)

        self.flux_type = flux

        # treat diffusion tensor
        if diffusion_tensor is None:
            diffusion_tensor = hedge.data.ConstantGivenFunction(
                    numpy.eye(dimensions))

        self.diffusion_tensor = diffusion_tensor

        self.dirichlet_bc = dirichlet_bc
        self.dirichlet_tag = dirichlet_tag
        self.neumann_bc = neumann_bc
        self.neumann_tag = neumann_tag

    # fluxes ------------------------------------------------------------------
    def get_weak_flux_set(self, flux):
        class FluxSet: pass
        fs = FluxSet()

        if flux == "ldg":
            ldg_terms = True
        elif flux == "ip":
            ldg_terms = False
        else:
            raise "Invalid flux type '%s'" % flux

        from hedge.flux import FluxVectorPlaceholder, \
                make_normal, PenaltyTerm
        from numpy import dot

        dim = self.dimensions
        vec = FluxVectorPlaceholder(1+dim)
        fs.u = u = vec[0]
        fs.v = v = vec[1:]
        normal = make_normal(dim)

        # central flux
        fs.flux_u = u.avg*normal
        fs.flux_v = dot(v.avg, normal)

        if ldg_terms:
            # ldg terms
            ldg_beta = numpy.array([1]*dim)

            fs.flux_u = fs.flux_u - (u.int-u.ext)*0.5*ldg_beta
            fs.flux_v = fs.flux_v + dot((v.int-v.ext)*0.5, ldg_beta)

        # penalty term
        stab_term = 10 * PenaltyTerm() * (u.int - u.ext)
        fs.flux_v -= stab_term

        # boundary fluxes
        fs.flux_u_dbdry = normal * u.ext
        fs.flux_v_dbdry = dot(v.int, normal) - stab_term

        fs.flux_u_nbdry = normal * u.int
        fs.flux_v_nbdry = dot(normal, v.ext)

        return fs

    # operator application, rhs prep ------------------------------------------
    def grad_op_template(self):
        from hedge.optemplate import Field, BoundaryPair, get_flux_operator, \
                make_stiffness_t, InverseMassOperator

        stiff_t = make_stiffness_t(self.dimensions)
        m_inv = InverseMassOperator()

        u = Field("u")

        fs = self.get_weak_flux_set(self.flux_type)

        flux_u = get_flux_operator(fs.flux_u)
        flux_u_dbdry = get_flux_operator(fs.flux_u_dbdry)
        flux_u_nbdry = get_flux_operator(fs.flux_u_nbdry)

        return m_inv * (
                - (stiff_t * u)
                + flux_u*u
                + flux_u_dbdry*BoundaryPair(u, 0, self.dirichlet_tag)
                + flux_u_nbdry*BoundaryPair(u, 0, self.neumann_tag)
                )

    def div_op_template(self, apply_minv):
        from hedge.optemplate import make_vector_field, BoundaryPair, \
                make_stiffness_t, InverseMassOperator, get_flux_operator

        d = self.dimensions
        w = make_vector_field("w", 1+d)
        v = w[1:]
        dir_bc_w = make_vector_field("dir_bc_w", 1+d)
        neu_bc_w = make_vector_field("neu_bc_w", 1+d)

        stiff_t = make_stiffness_t(d)
        m_inv = InverseMassOperator()

        fs = self.get_weak_flux_set(self.flux_type)

        flux_v = get_flux_operator(fs.flux_v)
        flux_v_dbdry = get_flux_operator(fs.flux_v_dbdry)
        flux_v_nbdry = get_flux_operator(fs.flux_v_nbdry)

        result = (
                -numpy.dot(stiff_t, v)
                + flux_v * w
                + flux_v_dbdry * BoundaryPair(w, dir_bc_w, self.dirichlet_tag)
                + flux_v_nbdry * BoundaryPair(w, neu_bc_w, self.neumann_tag)
                )

        if apply_minv:
            return InverseMassOperator() * result
        else:
            return result

    @memoize_method
    def grad_bc_op_template(self):
        from hedge.optemplate import Field, BoundaryPair, \
                InverseMassOperator, get_flux_operator

        flux_u_dbdry = get_flux_operator(
                self.get_weak_flux_set(self.flux_type).flux_u_dbdry)

        return InverseMassOperator() * (
                flux_u_dbdry*BoundaryPair(0, Field("dir_bc_u"),
                    self.dirichlet_tag))

    # bound operator ----------------------------------------------------------
    def bind(self, discr):
        """Return a :class:`BoundPoissonOperator`."""

        assert self.dimensions == discr.dimensions

        from hedge.mesh import check_bc_coverage
        check_bc_coverage(discr.mesh, [self.dirichlet_tag, self.neumann_tag])

        return self.BoundPoissonOperator(self, discr)




class BoundPoissonOperator(hedge.iterative.OperatorBase):
    """Returned by :meth:`WeakPoissonOperator.bind`."""

    def __init__(self, poisson_op, discr):
        hedge.iterative.OperatorBase.__init__(self)
        self.discr = discr

        pop = self.poisson_op = poisson_op

        self.grad_c = discr.compile(pop.grad_op_template())
        self.div_c = discr.compile(pop.div_op_template(False))
        self.minv_div_c = discr.compile(pop.div_op_template(True))
        self.grad_bc_c = discr.compile(pop.grad_bc_op_template())

        self.neumann_normals = discr.boundary_normals(poisson_op.neumann_tag)

        if isinstance(pop.diffusion_tensor, hedge.data.ConstantGivenFunction):
            self.diffusion = self.neu_diff = pop.diffusion_tensor.value
        else:
            self.diffusion = pop.diffusion_tensor.volume_interpolant(discr)
            self.neu_diff = pop.diffusion_tensor.boundary_interpolant(discr,
                    poisson_op.neumann_tag)

        # Check whether use of Poincaré mean-value method is required.
        # This only is requested for periodic BC's over the entire domain.
        # Partial periodic BC mixed with other BC's does not need the
        # special treatment.

        from hedge.mesh import TAG_ALL
        self.poincare_mean_value_hack = (
                len(self.discr.get_boundary(TAG_ALL).nodes)
                == len(self.discr.get_boundary(poisson_op.neumann_tag).nodes))

    @property
    def dtype(self):
        return self.discr.default_scalar_type

    @property
    def shape(self):
        nodes = len(self.discr)
        return nodes, nodes

    # actual functionality
    def grad(self, u):
        return self.grad_c(u=u)

    def div(self, v, u=None, apply_minv=True):
        """Compute the divergence of v using an LDG operator.

        The divergence computation is unaffected by the scaling
        effected by the diffusion tensor.

        :param apply_minv: :class:`bool` specifying whether to compute a complete
          divergence operator. If False, the final application of the inverse
          mass operator is skipped. This is used in :meth:`op` in order to reduce
          the scheme :math:`M^{-1} S u = f` to :math:`S u = M f`, so that the mass operator
          only needs to be applied once, when preparing the right hand side
          in :meth:`prepare_rhs`.
        """
        from hedge.tools import join_fields

        dim = self.discr.dimensions

        if u is None:
            u = self.discr.volume_zeros()
        w = join_fields(u, v)

        dir_bc_w = join_fields(0, [0]*dim)
        neu_bc_w = join_fields(0, [0]*dim)

        if apply_minv:
            div_tpl = self.minv_div_c
        else:
            div_tpl = self.div_c

        return div_tpl(w=w, dir_bc_w=dir_bc_w, neu_bc_w=neu_bc_w)

    def op(self, u, apply_minv=False):
        from hedge.tools import ptwise_dot

        # Check if poincare mean value method has to be applied.
        if self.poincare_mean_value_hack:
            # ∫(Ω) u dΩ
            state_int = self.discr.integral(u)
            # calculate mean value:  (1/|Ω|) * ∫(Ω) u dΩ
            mean_state = state_int / self.discr.mesh_volume()
            m_mean_state = mean_state * self.discr._mass_ones()
            #m_mean_state = mean_state * m
        else:
            m_mean_state = 0

        return self.div(
                ptwise_dot(2, 1, self.diffusion, self.grad(u)),
                u, apply_minv=apply_minv) \
                        - m_mean_state

    __call__ = op

    def prepare_rhs(self, rhs):
        """Prepare the right-hand side for the linear system op(u)=rhs(f).

        In matrix form, LDG looks like this:

        .. math::
            Mv = Cu + g
            Mf = Av + Bu + h

        where v is the auxiliary vector, u is the argument of the operator, f
        is the result of the grad operator, g and h are inhom boundary data, and
        A,B,C are some operator+lifting matrices.

        .. math::

            M f = A M^{-1}(Cu + g) + Bu + h

        so the linear system looks like

        .. math::

            M f = A M^{-1} Cu + A M^{-1} g + Bu + h
            M f - A M^{-1} g - h = (A M^{-1} C + B)u (*)

        So the right hand side we're putting together here is really

        .. math::

            M f - A M^{-1} g - h

        Finally, note that the operator application above implements
        the equation (*) left-multiplied by Minv, so that the
        right-hand-side becomes

        .. math::
            \\text{rhs} = f - M^{-1}( A M^{-1} g + h)
        """
        dim = self.discr.dimensions

        pop = self.poisson_op

        dtag = pop.dirichlet_tag
        ntag = pop.neumann_tag

        dir_bc_u = pop.dirichlet_bc.boundary_interpolant(self.discr, dtag)
        vpart = self.grad_bc_c(dir_bc_u=dir_bc_u)

        from hedge.tools import ptwise_dot
        diff_v = ptwise_dot(2, 1, self.diffusion, vpart)

        def neu_bc_v():
            return ptwise_dot(2, 1, self.neu_diff,
                    self.neumann_normals*
                        pop.neumann_bc.boundary_interpolant(self.discr, ntag))

        from hedge.tools import join_fields
        w = join_fields(0, diff_v)
        dir_bc_w = join_fields(dir_bc_u, [0]*dim)
        neu_bc_w = join_fields(0, neu_bc_v())

        from hedge.optemplate import MassOperator

        return (MassOperator().apply(self.discr,
            rhs.volume_interpolant(self.discr))
            - self.div_c(w=w, dir_bc_w=dir_bc_w, neu_bc_w=neu_bc_w))



