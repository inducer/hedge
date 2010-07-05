# -*- coding: utf8 -*-
"""Operator for compressible Navier-Stokes and Euler equations."""

from __future__ import division

__copyright__ = "Copyright (C) 2007 Hendrik Riedmann, Andreas Kloeckner"

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
import hedge.tools
import hedge.mesh
import hedge.data
from hedge.models import TimeDependentOperator
from pytools import Record
from hedge.tools import is_zero
from hedge.second_order import (
        StabilizedCentralSecondDerivative,
        CentralSecondDerivative,
        IPDGSecondDerivative)




class GasDynamicsOperator(TimeDependentOperator):
    """An nD Navier-Stokes and Euler operator.

    see JSH, TW: Nodal Discontinuous Galerkin Methods p.320 and p.206

    dq/dt = d/dx * (-F + tau_:1) + d/dy * (-G + tau_:2)

    where e.g. in 2D

    q = (rho, rho_u_x, rho_u_y, E)
    F = (rho_u_x, rho_u_x^2 + p, rho_u_x * rho_u_y / rho, u_x * (E + p))
    G = (rho_u_y, rho_u_x * rho_u_y / rho, rho_u_y^2 + p, u_y * (E + p))

    tau_11 = mu * (2 * du/dx - 2/3 * (du/dx + dv/dy))
    tau_12 = mu * (du/dy + dv/dx)
    tau_21 = tau_12
    tau_22 = mu * (2 * dv/dy - 2/3 * (du/dx + dv/dy))
    tau_31 = u * tau_11 + v * tau_12
    tau_32 = u * tau_21 + v * tau_22

    For the heat flux:

    q = -k * nabla * T
    k = c_p * mu / Pr

    Field order is [rho E rho_u_x rho_u_y ...].
    """

    # {{{ initialization ------------------------------------------------------
    def __init__(self, dimensions,
            gamma, mu, 
            bc_inflow=None,
            bc_outflow=None,
            bc_noslip=None,
            bc_supersonic_inflow=None,
            prandtl=None, spec_gas_const=1.0,
            inflow_tag="inflow",
            outflow_tag="outflow",
            noslip_tag="noslip",
            wall_tag="wall",
            supersonic_inflow_tag="supersonic_inflow",
            supersonic_outflow_tag="supersonic_inflow",
            source=None,
            second_order_scheme=CentralSecondDerivative(),
            ):
        """
        :param source: should implement
          :class:`hedge.data.IFieldDependentGivenFunction`
          or be None.
        """
        from hedge.data import (
                TimeConstantGivenFunction,
                ConstantGivenFunction)

        from pytools.obj_array import make_obj_array
        dull_bc = TimeConstantGivenFunction(
                ConstantGivenFunction(make_obj_array(
                    [1, 1] + [0]*dimensions)))
        if bc_inflow is None:
            bc_inflow = dull_bc
        if bc_outflow is None:
            bc_outflow = dull_bc
        if bc_noslip is None:
            bc_noslip = dull_bc
        if bc_supersonic_inflow is None:
            bc_supersonic_inflow = dull_bc

        self.dimensions = dimensions

        self.gamma = gamma
        self.prandtl = prandtl
        self.spec_gas_const = spec_gas_const
        self.mu = mu

        self.bc_inflow = bc_inflow
        self.bc_outflow = bc_outflow
        self.bc_noslip = bc_noslip
        self.bc_supersonic_inflow = bc_supersonic_inflow

        self.inflow_tag = inflow_tag
        self.outflow_tag = outflow_tag
        self.noslip_tag = noslip_tag
        self.wall_tag = wall_tag
        self.supersonic_inflow_tag = supersonic_inflow_tag
        self.supersonic_outflow_tag = supersonic_outflow_tag

        self.source = source

        self.second_order_scheme = second_order_scheme

    # }}}

    # {{{ conversions ---------------------------------------------------------
    def rho(self, q):
        return q[0]

    def e(self, q):
        return q[1]

    def rho_u(self, q):
        return q[2:2+self.dimensions]

    def u(self, q):
        from hedge.tools import make_obj_array
        return make_obj_array([
                rho_u_i/self.rho(q)
                for rho_u_i in self.rho_u(q)])

    def p(self, q):
        return (self.gamma-1)*(
                self.e(q) - 0.5*numpy.dot(self.rho_u(q), self.u(q)))

    def cse_u(self, q):
        from hedge.tools.symbolic import make_common_subexpression as cse
        return cse(self.u(q), "u")

    def cse_rho(self, q):
        from hedge.tools.symbolic import make_common_subexpression as cse
        return cse(self.rho(q), "rho")

    def cse_rho_u(self, q):
        from hedge.tools.symbolic import make_common_subexpression as cse
        return cse(self.rho_u(q), "rho_u")

    def cse_p(self, q):
        from hedge.tools.symbolic import make_common_subexpression as cse
        return cse(self.p(q), "p")

    def temperature(self, q):
        c_v = 1 / (self.gamma - 1) *self.spec_gas_const
        return (self.e(q)/self.rho(q) - 0.5 * numpy.dot(self.u(q), self.u(q))) / c_v

    def primitive_to_conservative(self, prims, use_cses=True):
        if use_cses:
            from hedge.tools.symbolic import make_common_subexpression as cse
        else:
            def cse(x, name): return x

        rho = prims[0]
        p = prims[1]
        u = prims[2:]

        from hedge.tools import join_fields
        return join_fields(
               rho,
               cse(p / (self.gamma - 1) + rho / 2 * numpy.dot(u, u), "e"),
               cse(rho * u, "rho_u"))

    def conservative_to_primitive(self, q, use_cses=True):
        if use_cses:
            from hedge.tools.symbolic import make_common_subexpression as cse
        else:
            def cse(x, name): return x

        from hedge.tools import join_fields
        return join_fields(
               self.rho(q),
               self.p(q),
               self.u(q))

    def characteristic_velocity_optemplate(self, state):
        from hedge.optemplate.operators import ElementwiseMaxOperator
        from hedge.tools.symbolic import make_common_subexpression as cse

        from hedge.optemplate.primitives import CFunction
        sqrt = CFunction("sqrt")

        sound_speed = cse(sqrt(
            self.gamma*self.cse_p(state)/self.cse_rho(state)),
            "sound_speed")
        u = self.cse_u(state)
        speed = cse(sqrt(numpy.dot(u, u)), "norm_u") + sound_speed
        return ElementwiseMaxOperator()(speed)

    def bind_characteristic_velocity(self, discr):
        from hedge.optemplate.tools import make_vector_field
        state = make_vector_field("q", self.dimensions+2)

        from hedge.optemplate import Field
        compiled = discr.compile(
                self.characteristic_velocity_optemplate(state))

        def do(q):
            return compiled(q=q)

        return do

    # }}}

    # {{{ operator template ---------------------------------------------------
    def op_template(self, viscosity_mode=None, sensor_scaling=None,
            viscosity_only=False):
        if viscosity_mode not in ["cns", "diffusion", "blended", None]:
            raise ValueError("viscosity_mode has an invalid value")

        from hedge.optemplate.tools import make_vector_field
        from pytools.obj_array import make_obj_array, join_fields
        from hedge.tools.symbolic import make_common_subexpression as cse

        AXES = ["x", "y", "z", "w"]

        # {{{ cse'd conversions and helpers
        u = self.cse_u
        rho = self.cse_rho
        rho_u = self.rho_u
        p = self.p
        e = self.e

        def temperature(q):
            return cse(self.temperature(q), "temperature")

        def get_mu(q, to_quad_op):
            """
            :param to_quad_op: If not *None*, represents an operator which transforms
              nodal values onto a quadrature grid on which the returned :math:`\mu` 
              needs to be represented. In that case, *q* is assumed to already be on the
              same quadrature grid.
            """

            if to_quad_op is None:
                def to_quad_op(x):
                    return x

            if self.mu == "sutherland":
                # Sutherland's law: !!!not tested!!!
                t_s = 110.4
                mu_inf = 1.735e-5
                result = cse(
                        mu_inf * temperature(q) ** 1.5 * (1 + t_s) 
                        / (temperature(q) + t_s),
                        "sutherland_mu")
            else:
                result = self.mu

            if viscosity_mode == "cns":
                mapped_sensor = sensor
            elif viscosity_mode == "blended":
                exp = CFunction("exp")
                mapped_sensor = cse(
                        sensor_scaling
                        * unit_sensor
                        * exp(-unit_sensor**2), 
                        "mapped_sensor_cns")
            else:
                mapped_sensor = None

            if mapped_sensor is not None:
                result = result + cse(to_quad_op(mapped_sensor), "quad_sensor")

            return cse(result, "mu")

        # }}}

        # {{{ viscous stress tensor

        # {{{ compute gradient of state ---------------------------------------
        def grad_of_state():
            dimensions = self.dimensions

            dq = numpy.zeros((dimensions+2, dimensions), dtype=object)

            from hedge.second_order import SecondDerivativeTarget
            for i in range(self.dimensions+2):
                grad_tgt = SecondDerivativeTarget(
                        self.dimensions, strong_form=False,
                        operand=state[i],
                        bdry_flux_int_operand=faceq_state[i])

                dir_bcs = dict((tag, bc[i])
                        for tag, bc in all_tags_and_conservative_bcs)

                def grad_bc_getter(tag, expr):
                    return dir_bcs[tag]

                self.second_order_scheme.grad(grad_tgt,
                        bc_getter=grad_bc_getter,
                        dirichlet_tags=dir_bcs.keys(),
                        neumann_tags=[])

                dq[i,:] = grad_tgt.minv_all

            return dq

        # }}}

        def tau(to_quad_op):
            dimensions = self.dimensions

            # {{{ compute gradient of u ---------------------------------------
            # Use the product rule to compute the gradient of
            # u from the gradient of (rho u). This ensures we don't
            # compute the derivatives twice.

            from pytools.obj_array import with_object_array_or_scalar
            dq = with_object_array_or_scalar(
                    to_quad_op,
                    grad_of_state())

            q = cse(to_quad_op(state))

            du = numpy.zeros((dimensions, dimensions), dtype=object)
            for i in range(dimensions):
                for j in range(dimensions):
                    du[i,j] = cse(
                            (dq[i+2,j] - u(q)[i] * dq[0,j]) / self.rho(q),
                            "du%d_d%s" % (i, AXES[j]))

            # }}}

            # {{{ put together viscous stress tau -----------------------------
            from pytools import delta

            mu = get_mu(q, to_quad_op)

            tau = numpy.zeros((dimensions, dimensions), dtype=object)
            for i in range(dimensions):
                for j in range(dimensions):
                    tau[i,j] = cse(mu * (du[i,j] + du[j,i] -
                               2/3 * delta(i,j) * numpy.trace(du)),
                               "tau_%d%d" % (i, j))

            return tau
            # }}}

        # }}}

        # {{{ second order part
        def div(vol_operand, int_face_operand):
            from hedge.second_order import SecondDerivativeTarget
            div_tgt = SecondDerivativeTarget(
                    self.dimensions, strong_form=False,
                    operand=vol_operand,
                    int_flux_operand=int_face_operand)

            def unwrap_cse(expr):
                from pymbolic.primitives import CommonSubexpression
                if isinstance(expr, CommonSubexpression):
                    return expr.child
                else:
                    return expr

            dir_bcs = dict(((tag, unwrap_cse(faceq_state[i])), bc[i])
                    for tag, bc in all_tags_and_conservative_bcs
                    for i in range(len(faceq_state)))

            # supply BC for sensor, if necessary
            if viscosity_mode is not None:
                for tag, bc in all_tags_and_conservative_bcs:
                    dir_bcs[tag, to_int_face_quad(sensor)] = \
                            cse(to_bdry_quad(BoundarizeOperator(tag)(sensor)),
                                    "bdry_sensor")

            def div_bc_getter(tag, expr):
                try:
                    return dir_bcs[tag, expr]
                except KeyError:
                    print expr
                    raise NotImplementedError

            self.second_order_scheme.div(div_tgt,
                    bc_getter=div_bc_getter,
                    dirichlet_tags=
                    [tag for tag, bc in all_tags_and_conservative_bcs],
                    neumann_tags=[])

            return div_tgt.minv_all

        def make_second_order_part():

            volq_tau_mat = tau(to_vol_quad)
            faceq_tau_mat = tau(to_int_face_quad)

            return join_fields(
                    0, 
                    div(
                        numpy.sum(volq_tau_mat*u(volq_state), axis=1),
                        numpy.sum(faceq_tau_mat*u(faceq_state), axis=1)
                        ),
                    [div(volq_tau_mat[i], faceq_tau_mat[i])
                        for i in range(self.dimensions)]) 

        # }}}

        # {{{ artificial diffusion
        def make_artificial_diffusion():
            if viscosity_mode not in ["diffusion", "blended"]:
                return 0

            if viscosity_mode == "blended":
                exp = CFunction("exp")
                mapped_sensor = cse(
                        sensor_scaling
                        *unit_sensor*(1-exp(-unit_sensor**2)), 
                        "mapped_sensor_diff")
            else:
                mapped_sensor = sensor

            dq = grad_of_state()

            from pytools.obj_array import make_obj_array
            return make_obj_array([
                div(
                    to_vol_quad(mapped_sensor)*to_vol_quad(dq[i]),
                    to_int_face_quad(mapped_sensor)*to_int_face_quad(dq[i])) 
                for i in range(dq.shape[0])])
        # }}}

        # {{{ volume and boundary flux
        def flux_func(q):
            from pytools import delta

            return [ # one entry for each flux direction
                    cse(join_fields(
                        # flux rho
                        self.rho_u(q)[i],

                        # flux E
                        cse(self.e(q)+p(q))*u(q)[i],

                        # flux rho_u
                        make_obj_array([
                            self.rho_u(q)[i]*self.u(q)[j] + delta(i,j) * p(q)
                            for j in range(self.dimensions)
                            ])
                        ), "%s_flux" % AXES[i])
                    for i in range(self.dimensions)]

        def bdry_flux_func(q_bdry, q_vol, tag):
            from pytools import delta
            return [ # one entry for each flux direction
                    cse(join_fields(
                        # flux rho
                        self.rho_u(q_bdry)[i],

                        # flux E
                        cse(self.e(q_bdry)+p(q_bdry))*u(q_bdry)[i],

                        # flux rho_u
                        make_obj_array([
                            self.rho_u(q_bdry)[i]*self.u(q_bdry)[j] +
                            delta(i,j) * p(q_bdry)
                            for j in range(self.dimensions)
                            ])
                        ), "%s_bflux" % AXES[i])
                    for i in range(self.dimensions)]

        # }}}

        # {{{ state setup

        state = make_vector_field("q", self.dimensions+2)

        if viscosity_mode is not None:
            from hedge.optemplate.primitives import Field
            sensor = Field("sensor")

            if sensor_scaling is not None:
                assert viscosity_mode == "blended"
                unit_sensor = cse(sensor/sensor_scaling, "unit_sensor")

        from hedge.optemplate.operators import (
                QuadratureGridUpsampler,
                QuadratureInteriorFacesGridUpsampler)

        to_vol_quad = QuadratureGridUpsampler("gasdyn_vol")
        to_int_face_quad = QuadratureInteriorFacesGridUpsampler("gasdyn_face")
        to_bdry_quad = QuadratureGridUpsampler("gasdyn_face")

        volq_state = cse(to_vol_quad(state), "vol_quad_state")
        faceq_state = cse(to_int_face_quad(state), "face_quad_state")

        volq_flux = flux_func(volq_state)
        faceq_flux = flux_func(faceq_state)

        from hedge.optemplate.primitives import CFunction
        sqrt = CFunction("sqrt")

        speed = self.characteristic_velocity_optemplate(state)

        has_viscosity = not is_zero(get_mu(state, to_quad_op=None))
        # }}}

        # {{{ boundary conditions ---------------------------------------------
        from hedge.optemplate import BoundarizeOperator

        class BCInfo(Record):
            pass

        def make_bc_info(bc_name, tag, state):
            state0 = cse(to_bdry_quad(
                make_vector_field(bc_name, self.dimensions+2)
                ))

            from hedge.optemplate import make_normal

            rho0 = rho(state0)
            p0 = p(state0)
            u0 = u(state0)

            c0 = (self.gamma * p0 / rho0)**0.5

            bdrize_op = BoundarizeOperator(tag)
            return BCInfo(
                rho0=rho0, p0=p0, u0=u0, c0=c0,

                # notation: suffix "m" for "minus", i.e. "interior"
                drhom=cse(rho(cse(to_bdry_quad(bdrize_op(state)))) - rho0, "drhom"),
                dumvec=cse(u(cse(to_bdry_quad(bdrize_op(state)))) - u0, "dumvec"),
                dpm=cse(p(cse(to_bdry_quad(bdrize_op(state)))) - p0, "dpm"))

        def outflow_state(state):
            from hedge.optemplate import make_normal
            normal = make_normal(self.outflow_tag, self.dimensions)
            bc = make_bc_info("bc_q_out", self.outflow_tag, state)

            # see hedge/doc/maxima/euler.mac
            return join_fields(
                # bc rho
                cse(bc.rho0
                + bc.drhom + numpy.dot(normal, bc.dumvec)*bc.rho0/(2*bc.c0)
                - bc.dpm/(2*bc.c0*bc.c0), "bc_rho_outflow"),

                # bc p
                cse(bc.p0
                + bc.c0*bc.rho0*numpy.dot(normal, bc.dumvec)/2 + bc.dpm/2, "bc_p_outflow"),

                # bc u
                cse(bc.u0
                + bc.dumvec - normal*numpy.dot(normal, bc.dumvec)/2
                + bc.dpm*normal/(2*bc.c0*bc.rho0), "bc_u_outflow"))

        def inflow_state_inner(normal, bc, name):
            # see hedge/doc/maxima/euler.mac
            return join_fields(
                # bc rho
                cse(bc.rho0
                + numpy.dot(normal, bc.dumvec)*bc.rho0/(2*bc.c0) + bc.dpm/(2*bc.c0*bc.c0), "bc_rho_"+name),

                # bc p
                cse(bc.p0
                + bc.c0*bc.rho0*numpy.dot(normal, bc.dumvec)/2 + bc.dpm/2, "bc_p_"+name),

                # bc u
                cse(bc.u0
                + normal*numpy.dot(normal, bc.dumvec)/2 + bc.dpm*normal/(2*bc.c0*bc.rho0), "bc_u_"+name))

        def inflow_state(state):
            from hedge.optemplate import make_normal
            normal = make_normal(self.inflow_tag, self.dimensions)
            bc = make_bc_info("bc_q_in", self.inflow_tag, state)
            return inflow_state_inner(normal, bc, "inflow")

        def noslip_state(state):
            from hedge.optemplate import make_normal
            normal = make_normal(self.noslip_tag, self.dimensions)
            bc = make_bc_info("bc_q_noslip", self.noslip_tag, state)
            return inflow_state_inner(normal, bc, "noslip")

        def wall_state(state):
            bc = BoundarizeOperator(self.wall_tag)(state)
            wall_rho = rho(bc)
            wall_e = e(bc) # <3 eve
            wall_rho_u = rho_u(bc)

            from hedge.optemplate import make_normal
            normal = make_normal(self.wall_tag, self.dimensions)

            return to_bdry_quad(join_fields(
                    wall_rho,
                    wall_e,
                    wall_rho_u - 2*numpy.dot(wall_rho_u, normal) * normal))

        primitive_tags_and_bcs = [
                (self.outflow_tag, outflow_state(state)),
                (self.inflow_tag, inflow_state(state)),
                (self.noslip_tag, noslip_state(state))
                    ]
        conservative_tags_and_bcs = [
                (self.supersonic_inflow_tag, 
                    to_bdry_quad(make_vector_field(
                        "bc_q_supersonic_in", self.dimensions+2))),
                (self.supersonic_outflow_tag,
                    to_bdry_quad(
                        BoundarizeOperator(self.supersonic_outflow_tag)(
                            (state)))),
                (self.wall_tag, wall_state(state))
                ]

        all_tags_and_primitive_bcs = primitive_tags_and_bcs \
                + [(tag, self.conservative_to_primitive(bc))
                    for tag, bc in conservative_tags_and_bcs]

        all_tags_and_conservative_bcs = [
                (tag, self.primitive_to_conservative(bc))
                for tag, bc in primitive_tags_and_bcs
                ] + conservative_tags_and_bcs

        # }}}

        # {{{ operator assembly -----------------------------------------------
        from hedge.flux.tools import make_lax_friedrichs_flux
        from hedge.optemplate.operators import (InverseMassOperator,
                ElementwiseMaxOperator)

        from hedge.optemplate.tools import make_stiffness_t

        first_order_part = InverseMassOperator()(
                numpy.dot(make_stiffness_t(self.dimensions), volq_flux)
                - make_lax_friedrichs_flux(

                    # This is not quite the right order, but this is
                    # not really easy to fix. The process to calculate
                    # 'speed' is nonlinear, but we can't compute that
                    # on a quadrature grid, because we need to form both
                    # the elementwise max *and* then interpolate up.
                    # The latter requires that we know a basis.
                    # (Maybe the facewise max is enough?)
                    wave_speed=cse(to_int_face_quad(
                        ElementwiseMaxOperator()(speed)), "emax_c"),

                    state=faceq_state, fluxes=faceq_flux,
                    bdry_tags_states_and_fluxes=[
                        (tag, bc, bdry_flux_func(bc, faceq_state, tag))
                        for tag, bc in all_tags_and_conservative_bcs
                        ],
                    strong=False))

        if viscosity_only:
            first_order_part = 0*first_order_part

        result = join_fields(
                first_order_part 
                + make_second_order_part()
                + make_artificial_diffusion(),
                 speed)

        if self.source is not None:
            result = result + join_fields(
                    make_vector_field("source_vect", self.dimensions+2),
                    # extra field for speed
                    0)

        return result

        # }}}

    # }}}

    # {{{ operator binding ----------------------------------------------------
    def bind(self, discr, sensor=None, viscosity_mode="diffusion", sensor_scaling=None,
            viscosity_only=False):
        if sensor is None:
            viscosity_mode = None

        bound_op = discr.compile(self.op_template(
            viscosity_mode=viscosity_mode,
            sensor_scaling=sensor_scaling,
            viscosity_only=False))

        from hedge.mesh import check_bc_coverage
        check_bc_coverage(discr.mesh, [
            self.inflow_tag,
            self.outflow_tag,
            self.noslip_tag,
            self.wall_tag,
            self.supersonic_inflow_tag,
            self.supersonic_outflow_tag,
            ])

        from hedge.mesh import TAG_NONE
        if self.mu == 0 and not discr.get_boundary(self.noslip_tag).is_empty():
            raise RuntimeError("no-slip BCs only make sense for "
                    "viscous problems")

        def rhs(t, q):
            extra_kwargs = {}
            if self.source is not None:
                extra_kwargs["source_vect"] = self.source.volume_interpolant(
                        t, q, discr)

            if sensor is not None:
                extra_kwargs["sensor"] = sensor(q)

            opt_result = bound_op(q=q,
                    bc_q_in=self.bc_inflow.boundary_interpolant(
                        t, discr, self.inflow_tag),
                    bc_q_out=self.bc_outflow.boundary_interpolant(
                        t, discr, self.outflow_tag),
                    bc_q_noslip=self.bc_noslip.boundary_interpolant(
                        t, discr, self.noslip_tag),
                    bc_q_supersonic_in=self.bc_supersonic_inflow
                    .boundary_interpolant(t, discr, 
                        self.supersonic_inflow_tag),
                    **extra_kwargs
                    )

            max_speed = opt_result[-1]
            ode_rhs = opt_result[:-1]
            return ode_rhs, discr.nodewise_max(max_speed)

        return rhs

    # }}}

    # {{{ timestep estimation -------------------------------------------------

    def estimate_timestep(self, discr, 
            stepper=None, stepper_class=None, stepper_args=None,
            t=None, max_eigenvalue=None):
        u"""Estimate the largest stable timestep, given a time stepper
        `stepper_class`. If none is given, RK4 is assumed.
        """

        dg_factor = (discr.dt_non_geometric_factor()
                * discr.dt_geometric_factor())

        # see JSH/TW, eq. (7.32)
        rk4_dt = dg_factor / (max_eigenvalue + self.mu / dg_factor)

        from hedge.timestep.stability import \
                approximate_rk4_relative_imag_stability_region
        return rk4_dt * approximate_rk4_relative_imag_stability_region(
                stepper, stepper_class, stepper_args)

    # }}}




# {{{ limiter (unfinished, deprecated)
class SlopeLimiter1NEuler:
    def __init__(self, discr,gamma,dimensions,op):
        """Construct a limiter from Jan's book page 225
        """
        self.discr = discr
        self.gamma=gamma
        self.dimensions=dimensions
        self.op=op

        #AVE*colVect=average of colVect
        self.AVE_map = {}

        for eg in discr.element_groups:
            ldis = eg.local_discretization
            node_count = ldis.node_count()


            # build AVE matrix
            massMatrix = ldis.mass_matrix()
            #compute area of the element
            self.standard_el_vol= numpy.sum(numpy.dot(massMatrix,numpy.ones(massMatrix.shape[0])))

            from numpy import size, zeros, sum
            AVEt = sum(massMatrix,0)
            AVEt = AVEt/self.standard_el_vol
            AVE = zeros((size(AVEt),size(AVEt)))
            for ii in range(0,size(AVEt)):
                AVE[ii]=AVEt
            self.AVE_map[eg] = AVE

    def get_average(self,vec):
        from hedge.tools import log_shape
        from pytools import indices_in_shape
        from hedge._internal import perform_elwise_operator

        ls = log_shape(vec)
        result = self.discr.volume_zeros(ls)

        for i in indices_in_shape(ls):
            for eg in self.discr.element_groups:
                perform_elwise_operator(eg.ranges, eg.ranges,
                        self.AVE_map[eg], vec[i], result[i])

                return result

    def __call__(self, fields):
        from hedge.tools import join_fields

        #get conserved fields
        rho=self.op.rho(fields)
        e=self.op.e(fields)
        rho_velocity=self.op.rho_u(fields)

        #get primitive fields
        #to do

        #reset field values to cell average
        rhoLim=self.get_average(rho)
        eLim=self.get_average(e)
        temp=join_fields([self.get_average(rho_vel)
                for rho_vel in rho_velocity])

        #should do for primitive fields too

        return join_fields(rhoLim, eLim, temp)

# }}}




# vim: foldmethod=marker
