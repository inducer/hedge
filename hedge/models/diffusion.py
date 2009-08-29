# -*- coding: utf8 -*-
"""Operators modeling diffusive phenomena."""

from __future__ import division

__copyright__ = "Copyright (C) 2009 Andreas Kloeckner"

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

import hedge.data
from hedge.models import TimeDependentOperator




class StrongHeatOperator(TimeDependentOperator):
    def __init__(self, dimensions, coeff=hedge.data.ConstantGivenFunction(1),
            dirichlet_bc=hedge.data.ConstantGivenFunction(), dirichlet_tag="dirichlet",
            neumann_bc=hedge.data.ConstantGivenFunction(), neumann_tag="neumann",
            ldg=True):
        self.dimensions = dimensions
        assert isinstance(dimensions, int)

        self.coeff = coeff
        self.ldg = ldg

        self.dirichlet_bc = dirichlet_bc
        self.dirichlet_tag = dirichlet_tag
        self.neumann_bc = neumann_bc
        self.neumann_tag = neumann_tag

    # fluxes ------------------------------------------------------------------
    def get_weak_flux_set(self, ldg):
        class FluxSet: pass
        fs = FluxSet()

        from hedge.flux import FluxVectorPlaceholder, make_normal

        # note here:

        # local DG is unlike the other kids in that the computation of the flux
        # of u depends *only* on u, whereas the computation of the flux of v
        # (yielding the final right hand side) may also depend on u. That's why
        # we use the layout [u,v], where v is simply omitted for the u flux
        # computation.

        dim = self.dimensions
        vec = FluxVectorPlaceholder(1+dim)
        fs.u = u = vec[0]
        fs.v = v = vec[1:]
        normal = fs.normal = make_normal(dim)

        # central
        fs.flux_u = u.avg*normal
        fs.flux_v = numpy.dot(v.avg, normal)

        # dbdry is "dirichlet boundary"
        # nbdry is "neumann boundary"
        fs.flux_u_dbdry = fs.flux_u
        fs.flux_u_nbdry = fs.flux_u

        fs.flux_v_dbdry = fs.flux_v
        fs.flux_v_nbdry = fs.flux_v

        if ldg:
            ldg_beta = numpy.ones((dim,))

            fs.flux_u = fs.flux_u - (u.int-u.ext)*0.5*ldg_beta
            fs.flux_v = fs.flux_v + numpy.dot((v.int-v.ext)*0.5, ldg_beta)

        return fs

    def get_strong_flux_set(self, ldg):
        fs = self.get_weak_flux_set(ldg)

        u = fs.u
        v = fs.v
        normal = fs.normal

        fs.flux_u = u.int*normal - fs.flux_u
        fs.flux_v = numpy.dot(v.int, normal) - fs.flux_v
        fs.flux_u_dbdry = u.int*normal - fs.flux_u_dbdry
        fs.flux_v_dbdry = numpy.dot(v.int, normal) - fs.flux_v_dbdry
        fs.flux_u_nbdry = u.int*normal - fs.flux_u_nbdry
        fs.flux_v_nbdry = numpy.dot(v.int, normal) - fs.flux_v_nbdry

        return fs

    # right-hand side ---------------------------------------------------------
    def grad_op_template(self):
        from hedge.optemplate import Field, pair_with_boundary, \
                InverseMassOperator, make_stiffness, get_flux_operator

        stiff = make_stiffness(self.dimensions)

        u = Field("u")
        sqrt_coeff_u = Field("sqrt_coeff_u")
        dir_bc_u = Field("dir_bc_u")
        neu_bc_u = Field("neu_bc_u")

        fs = self.get_strong_flux_set(self.ldg)
        flux_u = get_flux_operator(fs.flux_u)
        flux_u_dbdry = get_flux_operator(fs.flux_u_dbdry)
        flux_u_nbdry = get_flux_operator(fs.flux_u_nbdry)

        return InverseMassOperator() * (
                stiff * u
                - flux_u*sqrt_coeff_u
                - flux_u_dbdry*pair_with_boundary(sqrt_coeff_u, dir_bc_u, self.dirichlet_tag)
                - flux_u_nbdry*pair_with_boundary(sqrt_coeff_u, neu_bc_u, self.neumann_tag)
                )

    def div_op_template(self):
        from hedge.optemplate import make_vector_field, pair_with_boundary, \
                InverseMassOperator, get_flux_operator, make_stiffness

        stiff = make_stiffness(self.dimensions)

        d = self.dimensions
        w = make_vector_field("w", 1+d)
        v = w[1:]

        dir_bc_w = make_vector_field("dir_bc_w", 1+d)
        neu_bc_w = make_vector_field("neu_bc_w", 1+d)

        fs = self.get_strong_flux_set(self.ldg)
        flux_v = get_flux_operator(fs.flux_v)
        flux_v_dbdry = get_flux_operator(fs.flux_v_dbdry)
        flux_v_nbdry = get_flux_operator(fs.flux_v_nbdry)

        return InverseMassOperator() * (
                numpy.dot(stiff, v)
                - flux_v * w
                - flux_v_dbdry * pair_with_boundary(w, dir_bc_w, self.dirichlet_tag)
                - flux_v_nbdry * pair_with_boundary(w, neu_bc_w, self.neumann_tag)
                )

    # boundary conditions -----------------------------------------------------
    def bind(self, discr):
        from hedge.mesh import check_bc_coverage
        check_bc_coverage(discr.mesh, [self.dirichlet_tag, self.neumann_tag])

        return self.BoundHeatOperator(self, discr)

    class BoundHeatOperator:
        def __init__(self, heat_op, discr):
            hop = self.heat_op = heat_op
            self.discr = discr

            self.sqrt_coeff = numpy.sqrt(
                    hop.coeff.volume_interpolant(discr))
            self.dir_sqrt_coeff = numpy.sqrt(
                    hop.coeff.boundary_interpolant(discr, hop.dirichlet_tag))
            self.neu_sqrt_coeff = numpy.sqrt(
                    hop.coeff.boundary_interpolant(discr, hop.neumann_tag))

            self.neumann_normals = discr.boundary_normals(hop.neumann_tag)

            self.grad_c = discr.compile(hop.grad_op_template())
            self.div_c = discr.compile(hop.div_op_template())

        def dirichlet_bc_u(self, t, sqrt_coeff_u):
            hop = self.heat_op

            return (
                    -self.discr.boundarize_volume_field(sqrt_coeff_u, hop.dirichlet_tag)
                    +2*self.dir_sqrt_coeff*hop.dirichlet_bc.boundary_interpolant(
                        t, self.discr, hop.dirichlet_tag)
                    )

        def dirichlet_bc_v(self, t, sqrt_coeff_v):
            hop = self.heat_op

            return self.discr.boundarize_volume_field(sqrt_coeff_v, hop.dirichlet_tag)

        def neumann_bc_u(self, t, sqrt_coeff_u):
            hop = self.heat_op

            return self.discr.boundarize_volume_field(sqrt_coeff_u, hop.neumann_tag)

        def neumann_bc_v(self, t, sqrt_coeff_v):
            hop = self.heat_op

            from hedge.tools import to_obj_array
            return (
                    -self.discr.boundarize_volume_field(sqrt_coeff_v, hop.neumann_tag)
                    +
                    2*self.neumann_normals*
                        hop.neumann_bc.boundary_interpolant(
                            t, self.discr, hop.neumann_tag))

        def __call__(self, t, u):
            from hedge.tools import join_fields

            hop = self.heat_op

            dtag = hop.dirichlet_tag
            ntag = hop.neumann_tag

            sqrt_coeff_u = self.sqrt_coeff * u

            dir_bc_u = self.dirichlet_bc_u(t, sqrt_coeff_u)
            neu_bc_u = self.neumann_bc_u(t, sqrt_coeff_u)

            v = self.grad_c(
                    u=u, sqrt_coeff_u=sqrt_coeff_u,
                    dir_bc_u=dir_bc_u, neu_bc_u=neu_bc_u)

            from hedge.tools import ptwise_mul
            sqrt_coeff_v = ptwise_mul(self.sqrt_coeff, v)

            dir_bc_v = self.dirichlet_bc_v(t, sqrt_coeff_v)
            neu_bc_v = self.neumann_bc_v(t, sqrt_coeff_v)

            w = join_fields(sqrt_coeff_u, sqrt_coeff_v)
            dir_bc_w = join_fields(dir_bc_u, dir_bc_v)
            neu_bc_w = join_fields(neu_bc_u, neu_bc_v)

            return self.div_c(w=w, dir_bc_w=dir_bc_w, neu_bc_w=neu_bc_w)




