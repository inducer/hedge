# -*- coding: utf8 -*-
"""Operators modeling advective phenomena."""

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

from hedge.models import TimeDependentOperator
import hedge.data




class AdvectionOperatorBase(TimeDependentOperator):
    flux_types = [
            "central",
            "upwind",
            "lf"
            ]

    def __init__(self, v,
            inflow_tag="inflow",
            inflow_u=hedge.data.make_tdep_constant(0),
            outflow_tag="outflow",
            flux_type="central"
            ):
        self.dimensions = len(v)
        self.v = v
        self.inflow_tag = inflow_tag
        self.inflow_u = inflow_u
        self.outflow_tag = outflow_tag
        self.flux_type = flux_type

    def weak_flux(self):
        from hedge.flux import make_normal, FluxScalarPlaceholder, IfPositive

        u = FluxScalarPlaceholder(0)
        normal = make_normal(self.dimensions)

        if self.flux_type == "central":
            return u.avg*numpy.dot(normal, self.v)
        elif self.flux_type == "lf":
            return u.avg*numpy.dot(normal, self.v) \
                    + 0.5*la.norm(self.v)*(u.int - u.ext)
        elif self.flux_type == "upwind":
            print IfPositive(numpy.dot(normal, self.v),u.int, u.ext)
            return (numpy.dot(normal, self.v)*
                    IfPositive(numpy.dot(normal, self.v),
                        u.int, # outflow
                        u.ext, # inflow
                        ))
        else:
            raise ValueError, "invalid flux type"

    def max_eigenvalue(self):
        return la.norm(self.v)

    def bind(self, discr):
        compiled_op_template = discr.compile(self.op_template())

        from hedge.mesh import check_bc_coverage
        check_bc_coverage(discr.mesh, [self.inflow_tag, self.outflow_tag])

        def rhs(t, u):
            bc_in = self.inflow_u.boundary_interpolant(t, discr, self.inflow_tag)
            return compiled_op_template(u=u, bc_in=bc_in)

        return rhs

    def bind_interdomain(self,
            my_discr, my_part_data,
            nb_discr, nb_part_data):
        from hedge.partition import compile_interdomain_flux
        compiled_op_template, from_nb_indices = compile_interdomain_flux(
                self.op_template(), "u", "nb_bdry_u",
                my_discr, my_part_data, nb_discr, nb_part_data,
                use_stupid_substitution=True)

        from hedge.tools import with_object_array_or_scalar, is_zero

        def nb_bdry_permute(fld):
            if is_zero(fld):
                return 0
            else:
                return fld[from_nb_indices]

        def rhs(t, u, u_neighbor):
            return compiled_op_template(u=u,
                    nb_bdry_u=with_object_array_or_scalar(nb_bdry_permute, u_neighbor))

        return rhs




class StrongAdvectionOperator(AdvectionOperatorBase):
    def flux(self):
        from hedge.flux import make_normal, FluxScalarPlaceholder

        u = FluxScalarPlaceholder(0)
        normal = make_normal(self.dimensions)

        return u.int * numpy.dot(normal, self.v) - self.weak_flux()

    def op_template(self):
        from hedge.optemplate import Field, pair_with_boundary, \
                get_flux_operator, make_nabla, InverseMassOperator

        u = Field("u")
        bc_in = Field("bc_in")

        nabla = make_nabla(self.dimensions)
        m_inv = InverseMassOperator()

        flux_op = get_flux_operator(self.flux())

        return (
                -numpy.dot(self.v, nabla*u)
                + m_inv*(
                flux_op * u
                + flux_op * pair_with_boundary(u, bc_in, self.inflow_tag)
                )
                )




class WeakAdvectionOperator(AdvectionOperatorBase):
    def flux(self):
        return self.weak_flux()

    def op_template(self):
        from hedge.optemplate import \
                Field, \
                pair_with_boundary, \
                get_flux_operator, \
                make_minv_stiffness_t, \
                InverseMassOperator, \
                BoundarizeOperator

        u = Field("u")

        # boundary conditions -------------------------------------------------
        from hedge.optemplate import BoundarizeOperator
        bc_in = Field("bc_in")
        bc_out = BoundarizeOperator(self.outflow_tag)*u

        minv_st = make_minv_stiffness_t(self.dimensions)
        m_inv = InverseMassOperator()

        flux_op = get_flux_operator(self.flux())

        return numpy.dot(self.v, minv_st*u) - m_inv*(
                    flux_op*u
                    + flux_op * pair_with_boundary(u, bc_in, self.inflow_tag)
                    + flux_op * pair_with_boundary(u, bc_out, self.outflow_tag)
                    )





class VariableCoefficientAdvectionOperator(TimeDependentOperator):
    """A class for space- and time-dependent DG-advection operators.

    `advec_v` is a callable expecting two arguments `(x, t)` representing space and time,
    and returning an n-dimensional vector representing the velocity at x.
    `bc_u_f` is a callable expecting `(x, t)` representing space and time,
    and returning an 1-dimensional vector representing the state on the boundary.
    Both `advec_v` and `bc_u_f` conform to the
    `hedge.data.ITimeDependentGivenFunction` interface.
    """

    flux_types = [
            "central",
            "upwind",
            "lf"
            ]

    def __init__(self,
            dimensions,
            advec_v,
            bc_u_f="None",
            flux_type="central"
            ):
        self.dimensions = dimensions
        self.advec_v = advec_v
        self.bc_u_f = bc_u_f
        self.flux_type = flux_type

    def flux(self, ):
        from hedge.flux import \
                make_normal, \
                FluxScalarPlaceholder, \
                FluxVectorPlaceholder, \
                IfPositive, flux_max, norm

        d = self.dimensions

        w = FluxVectorPlaceholder((1+d)+1)
        u = w[0]
        v = w[1:d+1]
        c = w[1+d]

        normal = make_normal(self.dimensions)

        if self.flux_type == "central":
            return (u.int*numpy.dot(v.int, normal )
                    + u.ext*numpy.dot(v.ext, normal)) * 0.5
        elif self.flux_type == "lf":
            n_vint = numpy.dot(normal, v.int)
            n_vext = numpy.dot(normal, v.ext)
            return 0.5 * (n_vint * u.int + n_vext * u.ext) \
                   - 0.5 * (u.ext - u.int) \
                   * flux_max(c.int, c.ext)

        elif self.flux_type == "upwind":
            return (
                    IfPositive(numpy.dot(normal, v.avg),
                        numpy.dot(normal, v.int) * u.int, # outflow
                        numpy.dot(normal, v.ext) * u.ext, # inflow
                        ))
            return f
        else:
            raise ValueError, "invalid flux type"

    def op_template(self):
        from hedge.optemplate import \
                Field, \
                pair_with_boundary, \
                get_flux_operator, \
                make_minv_stiffness_t, \
                InverseMassOperator,\
                make_vector_field, \
                ElementwiseMaxOperator, \
                BoundarizeOperator


        from hedge.tools import join_fields, \
                                ptwise_dot

        u = Field("u")
        v = make_vector_field("v", self.dimensions)
        c = ElementwiseMaxOperator()*ptwise_dot(1, 1, v, v)
        w = join_fields(u, v, c)

        # boundary conditions -------------------------------------------------
        from hedge.mesh import TAG_ALL
        bc_c = BoundarizeOperator(TAG_ALL) * c
        bc_u = Field("bc_u")
        bc_v = BoundarizeOperator(TAG_ALL) * v
        if self.bc_u_f is "None":
            bc_w = join_fields(0, bc_v, bc_c)
        else:
            bc_w = join_fields(bc_u, bc_v, bc_c)

        minv_st = make_minv_stiffness_t(self.dimensions)
        m_inv = InverseMassOperator()

        flux_op = get_flux_operator(self.flux())

        result = numpy.dot(minv_st, v*u) - m_inv*(
                    flux_op * w
                    + flux_op * pair_with_boundary(w, bc_w, TAG_ALL)
                    )
        return result

    def bind(self, discr):
        compiled_op_template = discr.compile(self.op_template())

        from hedge.mesh import check_bc_coverage, TAG_ALL
        check_bc_coverage(discr.mesh, [TAG_ALL])

        def rhs(t, u):
            v = self.advec_v.volume_interpolant(t, discr)

            if self.bc_u_f is not "None":
                bc_u = self.bc_u_f.boundary_interpolant(t, discr, tag=TAG_ALL)
                return compiled_op_template(u=u, v=v, bc_u=bc_u)
            else:
                return compiled_op_template(u=u, v=v)

        return rhs

    def max_eigenvalue(self, t, discr):
        # Gives the max eigenvalue of a vector of eigenvalues.
        # As the velocities of each node is stored in the velocity-vector-field
        # a pointwise dot product of this vector has to be taken to get the
        # magnitude of the velocity at each node. From this vector the maximum
        # values limits the timestep.

        from hedge.tools import ptwise_dot
        v = self.advec_v.volume_interpolant(t, discr)
        return (ptwise_dot(1, 1, v, v)**0.5).max()
