# -*- coding: utf8 -*-
"""Operators modeling advective phenomena."""

from __future__ import division

__copyright__ = "Copyright (C) 2009 Andreas Kloeckner"

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
import numpy.linalg as la

import hedge.data
from hedge.models import HyperbolicOperator
from hedge.second_order import CentralSecondDerivative




# {{{ constant-coefficient advection ------------------------------------------
class AdvectionOperatorBase(HyperbolicOperator):
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
        from hedge.flux import make_normal, FluxScalarPlaceholder
        from pymbolic.primitives import IfPositive

        u = FluxScalarPlaceholder(0)
        normal = make_normal(self.dimensions)

        if self.flux_type == "central":
            return u.avg*numpy.dot(normal, self.v)
        elif self.flux_type == "lf":
            return u.avg*numpy.dot(normal, self.v) \
                    + 0.5*la.norm(self.v)*(u.int - u.ext)
        elif self.flux_type == "upwind":
            return (numpy.dot(normal, self.v)*
                    IfPositive(numpy.dot(normal, self.v),
                        u.int, # outflow
                        u.ext, # inflow
                        ))
        else:
            raise ValueError, "invalid flux type"

    def max_eigenvalue(self, t=None, fields=None, discr=None):
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
        from hedge.optemplate import Field, BoundaryPair, \
                get_flux_operator, make_nabla, InverseMassOperator

        u = Field("u")
        bc_in = Field("bc_in")

        nabla = make_nabla(self.dimensions)
        m_inv = InverseMassOperator()

        flux_op = get_flux_operator(self.flux())

        return (
                -numpy.dot(self.v, nabla*u)
                + m_inv(
                flux_op(u)
                + flux_op(BoundaryPair(u, bc_in, self.inflow_tag))))




class WeakAdvectionOperator(AdvectionOperatorBase):
    def flux(self):
        return self.weak_flux()

    def op_template(self):
        from hedge.optemplate import (
                Field,
                BoundaryPair,
                get_flux_operator,
                make_stiffness_t,
                InverseMassOperator,
                BoundarizeOperator,
                QuadratureGridUpsampler,
                QuadratureInteriorFacesGridUpsampler)

        u = Field("u")

        to_quad = QuadratureGridUpsampler("quad")
        to_int_face_quad = QuadratureInteriorFacesGridUpsampler("quad")

        # boundary conditions -------------------------------------------------
        bc_in = Field("bc_in")
        bc_out = BoundarizeOperator(self.outflow_tag)*u

        stiff_t = make_stiffness_t(self.dimensions)
        m_inv = InverseMassOperator()

        flux_op = get_flux_operator(self.flux())

        return m_inv(numpy.dot(self.v, stiff_t*u) - (
                    flux_op(u)
                    + flux_op(BoundaryPair(u, bc_in, self.inflow_tag))
                    + flux_op(BoundaryPair(u, bc_out, self.outflow_tag))
                    ))

# }}}




# {{{ variable-coefficient advection ------------------------------------------
class VariableCoefficientAdvectionOperator(HyperbolicOperator):
    """A class for space- and time-dependent DG advection operators.

    :param advec_v: Adheres to the :class:`hedge.data.ITimeDependentGivenFunction`
      interfacer and is an n-dimensional vector representing the velocity.
    :param bc_u_f: Adheres to the :class:`hedge.data.ITimeDependentGivenFunction`
      interface and is a scalar representing the boundary condition at all
      boundary faces.

    Optionally allows diffusion.
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
            flux_type="central",
            diffusion_coeff=None,
            diffusion_scheme=CentralSecondDerivative()):
        self.dimensions = dimensions
        self.advec_v = advec_v
        self.bc_u_f = bc_u_f
        self.flux_type = flux_type
        self.diffusion_coeff = diffusion_coeff
        self.diffusion_scheme = diffusion_scheme

    # {{{ flux ----------------------------------------------------------------
    def flux(self):
        from hedge.flux import (
                make_normal,
                FluxVectorPlaceholder,
                flux_max)
        from pymbolic.primitives import IfPositive

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
        else:
            raise ValueError, "invalid flux type"
    # }}}

    def bind_characteristic_velocity(self, discr):
        from hedge.optemplate.operators import (
                ElementwiseMaxOperator)
        from hedge.optemplate import make_sym_vector
        velocity_vec = make_sym_vector("v", self.dimensions)
        velocity = ElementwiseMaxOperator()(
                numpy.dot(velocity_vec, velocity_vec)**0.5)

        compiled = discr.compile(velocity)

        def do(t, u):
            return compiled(v=self.advec_v.volume_interpolant(t, discr))

        return do

    def op_template(self, with_sensor=False):
        # {{{ operator preliminaries ------------------------------------------
        from hedge.optemplate import (Field, BoundaryPair, get_flux_operator,
                make_stiffness_t, InverseMassOperator, make_sym_vector,
                ElementwiseMaxOperator, BoundarizeOperator)

        from hedge.optemplate.primitives import make_common_subexpression as cse

        from hedge.optemplate.operators import (
                QuadratureGridUpsampler,
                QuadratureInteriorFacesGridUpsampler)

        to_quad = QuadratureGridUpsampler("quad")
        to_if_quad = QuadratureInteriorFacesGridUpsampler("quad")

        from hedge.tools import join_fields, \
                                ptwise_dot

        u = Field("u")
        v = make_sym_vector("v", self.dimensions)
        c = ElementwiseMaxOperator()(ptwise_dot(1, 1, v, v))

        quad_u = cse(to_quad(u))
        quad_v = cse(to_quad(v))

        w = join_fields(u, v, c)
        quad_face_w = to_if_quad(w)
        # }}}

        # {{{ boundary conditions ---------------------------------------------

        from hedge.mesh import TAG_ALL
        bc_c = to_quad(BoundarizeOperator(TAG_ALL)(c))
        bc_u = to_quad(Field("bc_u"))
        bc_v = to_quad(BoundarizeOperator(TAG_ALL)(v))

        if self.bc_u_f is "None":
            bc_w = join_fields(0, bc_v, bc_c)
        else:
            bc_w = join_fields(bc_u, bc_v, bc_c)

        minv_st = make_stiffness_t(self.dimensions)
        m_inv = InverseMassOperator()

        flux_op = get_flux_operator(self.flux())
        # }}}

        # {{{ diffusion -------------------------------------------------------
        if with_sensor or (
                self.diffusion_coeff is not None and self.diffusion_coeff != 0):
            if self.diffusion_coeff is None:
                diffusion_coeff = 0
            else:
                diffusion_coeff = self.diffusion_coeff

            if with_sensor:
                diffusion_coeff += Field("sensor")

            from hedge.second_order import SecondDerivativeTarget

            # strong_form here allows IPDG to reuse the value of grad u.
            grad_tgt = SecondDerivativeTarget(
                    self.dimensions, strong_form=True,
                    operand=u)

            self.diffusion_scheme.grad(grad_tgt, bc_getter=None,
                    dirichlet_tags=[], neumann_tags=[])

            div_tgt = SecondDerivativeTarget(
                    self.dimensions, strong_form=False,
                    operand=diffusion_coeff*grad_tgt.minv_all)

            self.diffusion_scheme.div(div_tgt,
                    bc_getter=None,
                    dirichlet_tags=[], neumann_tags=[])

            diffusion_part = div_tgt.minv_all
        else:
            diffusion_part = 0

        # }}}

        to_quad = QuadratureGridUpsampler("quad")
        quad_u = cse(to_quad(u))
        quad_v = cse(to_quad(v))

        return m_inv(numpy.dot(minv_st, cse(quad_v*quad_u))
                - (flux_op(quad_face_w)
                    + flux_op(BoundaryPair(quad_face_w, bc_w, TAG_ALL)))) \
                            + diffusion_part

    def bind(self, discr, sensor=None):
        compiled_op_template = discr.compile(
                self.op_template(with_sensor=sensor is not None))

        from hedge.mesh import check_bc_coverage, TAG_ALL
        check_bc_coverage(discr.mesh, [TAG_ALL])

        def rhs(t, u):
            kwargs = {}
            if sensor is not None:
                kwargs["sensor"] = sensor(t, u)

            if self.bc_u_f is not "None":
                kwargs["bc_u"] = \
                        self.bc_u_f.boundary_interpolant(t, discr, tag=TAG_ALL)

            return compiled_op_template(
                    u=u,
                    v=self.advec_v.volume_interpolant(t, discr),
                    **kwargs)

        return rhs

    def max_eigenvalue(self, t, fields=None, discr=None):
        # Gives the max eigenvalue of a vector of eigenvalues.
        # As the velocities of each node is stored in the velocity-vector-field
        # a pointwise dot product of this vector has to be taken to get the
        # magnitude of the velocity at each node. From this vector the maximum
        # values limits the timestep.

        from hedge.tools import ptwise_dot
        v = self.advec_v.volume_interpolant(t, discr)
        return discr.nodewise_max(ptwise_dot(1, 1, v, v)**0.5)

# }}}




# vim: foldmethod=marker
