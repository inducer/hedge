# Hedge - the Hybrid'n'Easy DG Environment
# Copyright (C) 2007 Andreas Kloeckner
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.




from __future__ import division




class MaxwellOperator:
    """A 3D Maxwell operator with PEC boundaries.

    Field order is [Ex Ey Ez Hx Hy Hz].
    """

    def __init__(self, discr, epsilon, mu, upwind_alpha=1, pec_tag=None,
            direct_flux=True):
        from hedge.flux import make_normal, FluxVectorPlaceholder
        from hedge.discretization import pair_with_boundary
        from math import sqrt
        from pytools.arithmetic_container import join_fields
        from hedge.tools import cross

        self.discr = discr

        self.epsilon = epsilon
        self.mu = mu
        self.alpha = upwind_alpha

        self.pec_tag = pec_tag

        dim = discr.dimensions
        normal = make_normal(dim)
        w = FluxVectorPlaceholder(6)
        e = w[0:3]
        h = w[3:6]

        Z = sqrt(mu/epsilon)
        Y = 1/Z

        self.flux = discr.get_flux_operator(join_fields(
                # flux e
                1/epsilon*(
                    1/2*cross(normal, h.int-h.ext)
                    -upwind_alpha/(2*Z)*cross(normal, cross(normal, e.int-e.ext))
                    ),
                # flux h
                1/mu*(
                    -1/2*cross(normal, e.int-e.ext)
                    -upwind_alpha/(2*Y)*cross(normal, cross(normal, h.int-h.ext))
                    ),
                ), 
                direct=direct_flux)

        self.nabla = discr.nabla
        self.m_inv = discr.inverse_mass_operator


    def rhs(self, t, w):
        from hedge.tools import cross
        from hedge.discretization import pair_with_boundary
        from pytools.arithmetic_container import join_fields

        e = w[0:3]
        h = w[3:6]

        def curl(field):
            return cross(self.nabla, field)

        bc = join_fields(
                -self.discr.boundarize_volume_field(e, self.pec_tag),
                self.discr.boundarize_volume_field(h, self.pec_tag)
                )

        bpair = pair_with_boundary(w, bc, self.pec_tag)

        return (
                join_fields(
                    1/self.epsilon * curl(h),
                    - 1/self.mu * curl(e),
                    )
                - self.m_inv*(
                    self.flux * w
                    +self.flux * pair_with_boundary(w, bc, self.pec_tag)
                    )
                )
