# -*- coding: utf8 -*-
"""Wave equation operators."""

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
import hedge.mesh
from hedge.models import Operator, HyperbolicOperator




class StrongWaveOperator(HyperbolicOperator):
    """This operator discretizes the wave equation
    :math:`\\partial_t^2 u = c^2 \\Delta u`.

    To be precise, we discretize the hyperbolic system

    .. math::

        \partial_t u - c \\nabla \\cdot v = 0

        \partial_t v - c \\nabla u = 0

    The sign of :math:`v` determines whether we discretize the forward or the
    backward wave equation.

    :math:`c` is assumed to be constant across all space.
    """

    def __init__(self, c, dimensions, source_f=None,
            flux_type="upwind",
            dirichlet_tag=hedge.mesh.TAG_ALL,
            dirichlet_bc_f=None,
            neumann_tag=hedge.mesh.TAG_NONE,
            radiation_tag=hedge.mesh.TAG_NONE):
        assert isinstance(dimensions, int)

        self.c = c
        self.dimensions = dimensions
        self.source_f = source_f

        if self.c > 0:
            self.sign = 1
        else:
            self.sign = -1

        self.dirichlet_tag = dirichlet_tag
        self.neumann_tag = neumann_tag
        self.radiation_tag = radiation_tag

        self.dirichlet_bc_f = dirichlet_bc_f

        self.flux_type = flux_type

    def flux(self):
        from hedge.flux import FluxVectorPlaceholder, make_normal

        dim = self.dimensions
        w = FluxVectorPlaceholder(1+dim)
        u = w[0]
        v = w[1:]
        normal = make_normal(dim)

        from hedge.tools import join_fields
        flux_weak = join_fields(
                numpy.dot(v.avg, normal),
                u.avg * normal)

        if self.flux_type == "central":
            pass
        elif self.flux_type == "upwind":
            # see doc/notes/hedge-notes.tm
            flux_weak -= self.sign*join_fields(
                    0.5*(u.int-u.ext),
                    0.5*(normal * numpy.dot(normal, v.int-v.ext)))
        else:
            raise ValueError("invalid flux type '%s'" % self.flux_type)

        flux_strong = join_fields(
                numpy.dot(v.int, normal),
                u.int * normal) - flux_weak

        return -self.c*flux_strong

    def op_template(self):
        from hedge.optemplate import \
                make_vector_field, \
                BoundaryPair, \
                get_flux_operator, \
                make_nabla, \
                InverseMassOperator, \
                BoundarizeOperator

        d = self.dimensions

        w = make_vector_field("w", d+1)
        u = w[0]
        v = w[1:]

        # boundary conditions -------------------------------------------------
        from hedge.tools import join_fields


        # dirichlet BCs -------------------------------------------------------
        from hedge.optemplate import make_normal, Field

        dir_normal = make_normal(self.dirichlet_tag, d)

        dir_u = BoundarizeOperator(self.dirichlet_tag) * u
        dir_v = BoundarizeOperator(self.dirichlet_tag) * v
        if self.dirichlet_bc_f:
            # FIXME
            from warnings import warn
            warn("Inhomogeneous Dirichlet conditions on the wave equation "
                    "are still having issues.")

            dir_g = Field("dir_bc_u")
            dir_bc = join_fields(2*dir_g - dir_u, dir_v)
        else:
            dir_bc = join_fields(-dir_u, dir_v)

        # neumann BCs ---------------------------------------------------------
        neu_u = BoundarizeOperator(self.neumann_tag) * u
        neu_v = BoundarizeOperator(self.neumann_tag) * v
        neu_bc = join_fields(neu_u, -neu_v)

        # radiation BCs -------------------------------------------------------
        from hedge.optemplate import make_normal
        rad_normal = make_normal(self.radiation_tag, d)

        rad_u = BoundarizeOperator(self.radiation_tag) * u
        rad_v = BoundarizeOperator(self.radiation_tag) * v

        rad_bc = join_fields(
                0.5*(rad_u - self.sign*numpy.dot(rad_normal, rad_v)),
                0.5*rad_normal*(numpy.dot(rad_normal, rad_v) - self.sign*rad_u)
                )

        # entire operator -----------------------------------------------------
        nabla = make_nabla(d)
        flux_op = get_flux_operator(self.flux())

        from hedge.tools import join_fields
        result = (
                - join_fields(
                    -self.c*numpy.dot(nabla, v),
                    -self.c*(nabla*u)
                    )
                +
                InverseMassOperator() * (
                    flux_op(w)
                    + flux_op(BoundaryPair(w, dir_bc, self.dirichlet_tag))
                    + flux_op(BoundaryPair(w, neu_bc, self.neumann_tag))
                    + flux_op(BoundaryPair(w, rad_bc, self.radiation_tag))
                    ))

        if self.source_f is not None:
            result[0] += Field("source_u")

        return result

    def bind(self, discr):
        from hedge.mesh import check_bc_coverage
        check_bc_coverage(discr.mesh, [
            self.dirichlet_tag,
            self.neumann_tag,
            self.radiation_tag])

        compiled_op_template = discr.compile(self.op_template())

        def rhs(t, w):
            from hedge.tools import join_fields, ptwise_dot

            kwargs = {"w": w}
            if self.dirichlet_bc_f:
                kwargs["dir_bc_u"] = self.dirichlet_bc_f.boundary_interpolant(
                        t, discr, self.dirichlet_tag)

            if self.source_f is not None:
                kwargs["source_u"] = self.source_f.volume_interpolant(t, discr)

            return compiled_op_template(**kwargs)

        return rhs

    def max_eigenvalue(self, t, fields=None, discr=None):
        return abs(self.c)




class VariableVelocityStrongWaveOperator(HyperbolicOperator):
    """This operator discretizes the wave equation
    :math:`\\partial_t^2 u = c^2 \\Delta u`.

    To be precise, we discretize the hyperbolic system

    .. math::

        \partial_t u - c \\nabla \\cdot v = 0

        \partial_t v - c \\nabla u = 0
    """

    def __init__(self, c, dimensions, source=None,
            flux_type="upwind",
            dirichlet_tag=hedge.mesh.TAG_ALL,
            neumann_tag=hedge.mesh.TAG_NONE,
            radiation_tag=hedge.mesh.TAG_NONE,
            time_sign=1):
        """`c` is assumed to be positive and conforms to the
        :class:`hedge.data.ITimeDependentGivenFunction` interface.

        `source` also conforms to the
        :class:`hedge.data.ITimeDependentGivenFunction` interface.
        """
        assert isinstance(dimensions, int)

        self.c = c
        self.time_sign = time_sign
        self.dimensions = dimensions
        self.source = source

        self.dirichlet_tag = dirichlet_tag
        self.neumann_tag = neumann_tag
        self.radiation_tag = radiation_tag

        self.flux_type = flux_type

    def flux(self):
        from hedge.flux import FluxVectorPlaceholder, make_normal

        dim = self.dimensions
        w = FluxVectorPlaceholder(2+dim)
        c = w[0]
        u = w[1]
        v = w[2:]
        normal = make_normal(dim)

        from hedge.tools import join_fields
        flux = self.time_sign*1/2*join_fields(
                c.ext * numpy.dot(v.ext, normal)
                - c.int * numpy.dot(v.int, normal),
                normal*(c.ext*u.ext - c.int*u.int))

        if self.flux_type == "central":
            pass
        elif self.flux_type == "upwind":
            flux += join_fields(
                    c.ext*u.ext - c.int*u.int,
                    c.ext*normal*numpy.dot(normal, v.ext)
                    - c.int*normal*numpy.dot(normal, v.int)
                    )
        else:
            raise ValueError("invalid flux type '%s'" % self.flux_type)

        return flux

    def op_template(self):
        from hedge.optemplate import \
                Field, \
                make_vector_field, \
                BoundaryPair, \
                get_flux_operator, \
                make_nabla, \
                InverseMassOperator, \
                BoundarizeOperator

        d = self.dimensions

        w = make_vector_field("w", d+1)
        u = w[0]
        v = w[1:]

        from hedge.tools import join_fields
        c = Field("c")
        flux_w = join_fields(c, w)

        # boundary conditions -------------------------------------------------
        from hedge.flux import make_normal
        normal = make_normal(d)

        from hedge.tools import join_fields
        # dirichlet BC's ------------------------------------------------------
        dir_c = BoundarizeOperator(self.dirichlet_tag) * c
        dir_u = BoundarizeOperator(self.dirichlet_tag) * u
        dir_v = BoundarizeOperator(self.dirichlet_tag) * v

        dir_bc = join_fields(dir_c, -dir_u, dir_v)

        # neumann BC's --------------------------------------------------------
        neu_c = BoundarizeOperator(self.neumann_tag) * c
        neu_u = BoundarizeOperator(self.neumann_tag) * u
        neu_v = BoundarizeOperator(self.neumann_tag) * v

        neu_bc = join_fields(neu_c, neu_u, -neu_v)

        # radiation BC's ------------------------------------------------------
        from hedge.optemplate import make_normal
        rad_normal = make_normal(self.radiation_tag, d)

        rad_c = BoundarizeOperator(self.radiation_tag) * c
        rad_u = BoundarizeOperator(self.radiation_tag) * u
        rad_v = BoundarizeOperator(self.radiation_tag) * v

        rad_bc = join_fields(
                rad_c,
                0.5*(rad_u - self.time_sign*numpy.dot(rad_normal, rad_v)),
                0.5*rad_normal*(numpy.dot(rad_normal, rad_v) - self.time_sign*rad_u)
                )

        # entire operator -----------------------------------------------------
        nabla = make_nabla(d)
        flux_op = get_flux_operator(self.flux())

        return (
                - join_fields(
                    -self.time_sign*c*numpy.dot(nabla, v),
                    -self.time_sign*c*(nabla*u)
                    )
                +
                InverseMassOperator() * (
                    flux_op(flux_w)
                    + flux_op(BoundaryPair(flux_w, dir_bc, self.dirichlet_tag))
                    + flux_op(BoundaryPair(flux_w, neu_bc, self.neumann_tag))
                    + flux_op(BoundaryPair(flux_w, rad_bc, self.radiation_tag))
                    ))


    def bind(self, discr):
        from hedge.mesh import check_bc_coverage
        check_bc_coverage(discr.mesh, [
            self.dirichlet_tag,
            self.neumann_tag,
            self.radiation_tag])

        compiled_op_template = discr.compile(self.op_template())

        def rhs(t, w):
            rhs = compiled_op_template(w=w,
                    c=self.c.volume_interpolant(t, discr))

            if self.source is not None:
                rhs[0] += self.source.volume_interpolant(t, discr)

            return rhs

        return rhs

    def max_eigenvalue(self, t, fields=None, discr=None):
        # Gives the max eigenvalue of a vector of eigenvalues.
        # As the velocities of each node is stored in the velocity-vector-field
        # a pointwise dot product of this vector has to be taken to get the
        # magnitude of the velocity at each node. From this vector the maximum
        # values limits the timestep.

        c = self.c.volume_interpolant(t, discr)
        return discr.nodewise_max(c)
