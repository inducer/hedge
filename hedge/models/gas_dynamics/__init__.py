"""Operator for compressible Navier-Stokes and Euler equations."""

from __future__ import division

__copyright__ = "Copyright (C) 2007 Hendrik Riedmann, Andreas Kloeckner"

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
from hedge.optemplate.primitives import make_common_subexpression as cse
from pytools import memoize_method
from hedge.optemplate.tools import make_sym_vector
from pytools.obj_array import make_obj_array, join_fields

AXES = ["x", "y", "z", "w"]

from hedge.optemplate.operators import (
        QuadratureGridUpsampler,
        QuadratureInteriorFacesGridUpsampler)

to_vol_quad = QuadratureGridUpsampler("gasdyn_vol")

# It is recommended (though not required) that these two
# remain the same so that they can be computed together
# by the CUDA backend

to_int_face_quad = QuadratureInteriorFacesGridUpsampler("gasdyn_face")
to_bdry_quad = QuadratureGridUpsampler("gasdyn_face")




# {{{ equations of state
class EquationOfState(object):
    def q_to_p(self, op, q):
        raise NotImplementedError

    def p_to_e(self, p, rho, u):
        raise NotImplementedError

class GammaLawEOS(EquationOfState):
    # FIXME Shouldn't gamma only occur in the equation of state?
    # I.e. shouldn't all uses of gamma go through the EOS?

    def __init__(self, gamma):
        self.gamma = gamma

    def q_to_p(self, op, q):
        return (self.gamma-1)*(op.e(q)-0.5*numpy.dot(op.rho_u(q), op.u(q)))

    def p_to_e(self, p, rho, u):
        return p / (self.gamma - 1) + rho / 2 * numpy.dot(u, u)

class PolytropeEOS(GammaLawEOS):
    # inverse is same as superclass

    def q_to_p(self, op, q):
        return  op.rho(q)**self.gamma

# }}}





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
            gamma=None, mu=0,
            bc_inflow=None,
            bc_outflow=None,
            bc_noslip=None,
            bc_supersonic_inflow=None,
            prandtl=None, spec_gas_const=1.0,
            equation_of_state=None,
            inflow_tag="inflow",
            outflow_tag="outflow",
            noslip_tag="noslip",
            wall_tag="wall",
            supersonic_inflow_tag="supersonic_inflow",
            supersonic_outflow_tag="supersonic_outflow",
            source=None,
            second_order_scheme=CentralSecondDerivative(),
            artificial_viscosity_mode=None,
            ):
        """
        :param source: should implement
          :class:`hedge.data.IFieldDependentGivenFunction`
          or be None.

        :param artificial_viscosity_mode:
        """
        from hedge.data import (
                TimeConstantGivenFunction,
                ConstantGivenFunction)

        if gamma is not None:
            if equation_of_state is not None:
                raise ValueError("can only specify one of gamma and equation_of_state")

            from warnings import warn
            warn("argument gamma is deprecated in favor of equation_of_state",
                    DeprecationWarning, stacklevel=2)

            equation_of_state = GammaLawEOS(gamma)

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
        self.equation_of_state = equation_of_state

        self.second_order_scheme = second_order_scheme

        if artificial_viscosity_mode not in [
                "cns", "diffusion", "blended", None]:
            raise ValueError("artificial_viscosity_mode has an invalid value")

        self.artificial_viscosity_mode = artificial_viscosity_mode



    # }}}

    # {{{ conversions ---------------------------------------------------------
    def state(self):
        return make_sym_vector("q", self.dimensions+2)

    @memoize_method
    def volq_state(self):
        return cse(to_vol_quad(self.state()), "vol_quad_state")

    @memoize_method
    def faceq_state(self):
        return cse(to_int_face_quad(self.state()), "face_quad_state")

    @memoize_method
    def sensor(self):
        from hedge.optemplate.primitives import Field
        sensor = Field("sensor")

    def rho(self, q):
        return q[0]

    def e(self, q):
        return q[1]

    def rho_u(self, q):
        return q[2:2+self.dimensions]

    def u(self, q):
        return make_obj_array([
                rho_u_i/self.rho(q)
                for rho_u_i in self.rho_u(q)])

    def p(self,q):
        return self.equation_of_state.q_to_p(self, q)

    def cse_u(self, q):
        return cse(self.u(q), "u")

    def cse_rho(self, q):
        return cse(self.rho(q), "rho")

    def cse_rho_u(self, q):
        return cse(self.rho_u(q), "rho_u")

    def cse_p(self, q):
        return cse(self.p(q), "p")

    def temperature(self, q):
        c_v = 1 / (self.equation_of_state.gamma - 1) * self.spec_gas_const
        return (self.e(q)/self.rho(q) - 0.5 * numpy.dot(self.u(q), self.u(q))) / c_v

    def cse_temperature(self, q):
        return cse(self.temperature(q), "temperature")

    def get_mu(self, q, to_quad_op):
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
                    mu_inf * self.cse_temperature(q) ** 1.5 * (1 + t_s) 
                    / (self.cse_temperature(q) + t_s),
                    "sutherland_mu")
        else:
            result = self.mu

        if self.artificial_viscosity_mode == "cns":
            mapped_sensor = self.sensor()
        else:
            mapped_sensor = None

        if mapped_sensor is not None:
            result = result + cse(to_quad_op(mapped_sensor), "quad_sensor")

        return cse(result, "mu")

    def primitive_to_conservative(self, prims, use_cses=True):
        if not use_cses:
            from hedge.optemplate.primitives import make_common_subexpression as cse
        else:
            def cse(x, name): return x

        rho = prims[0]
        p = prims[1]
        u = prims[2:]
        e = self.equation_of_state.p_to_e(p, rho, u)

        return join_fields(
               rho,
               cse(e, "e"),
               cse(rho * u, "rho_u"))

    def conservative_to_primitive(self, q, use_cses=True):
        if use_cses:
            from hedge.optemplate.primitives import make_common_subexpression as cse
        else:
            def cse(x, name): return x

        return join_fields(
               self.rho(q),
               self.p(q),
               self.u(q))

    def characteristic_velocity_optemplate(self, state):
        from hedge.optemplate.operators import ElementwiseMaxOperator

        from hedge.optemplate.primitives import CFunction
        sqrt = CFunction("sqrt")

        sound_speed = cse(sqrt(
            self.equation_of_state.gamma*self.cse_p(state)/self.cse_rho(state)),
            "sound_speed")
        u = self.cse_u(state)
        speed = cse(sqrt(numpy.dot(u, u)), "norm_u") + sound_speed
        return ElementwiseMaxOperator()(speed)

    def bind_characteristic_velocity(self, discr):
        state = make_sym_vector("q", self.dimensions+2)

        compiled = discr.compile(
                self.characteristic_velocity_optemplate(state))

        def do(q):
            return compiled(q=q)

        return do

    # }}}

    # {{{ helpers for second-order part ---------------------------------------

    # {{{ compute gradient of state ---------------------------------------
    def grad_of(self, var, faceq_var):
        from hedge.second_order import SecondDerivativeTarget
        grad_tgt = SecondDerivativeTarget(
                self.dimensions, strong_form=False,
                operand=var,
                int_flux_operand=faceq_var,
                bdry_flux_int_operand=faceq_var)

        self.second_order_scheme.grad(grad_tgt,
                bc_getter=self.get_boundary_condition_for,
                dirichlet_tags=self.get_boundary_tags(),
                neumann_tags=[])

        return grad_tgt.minv_all

    def grad_of_state(self):
        dimensions = self.dimensions

        state = self.state()

        dq = numpy.zeros((len(state), dimensions), dtype=object)

        for i in range(len(state)):
            dq[i,:] = self.grad_of(
                    state[i], self.faceq_state()[i])

        return dq

    def grad_of_state_func(self, func, of_what_descr):
        return cse(self.grad_of(
            func(self.volq_state()),
            func(self.faceq_state())),
            "grad_"+of_what_descr)

    # }}}

    # {{{ viscous stress tensor

    def tau(self, to_quad_op, state, mu=None):
        faceq_state = self.faceq_state()

        dimensions = self.dimensions

        # {{{ compute gradient of u ---------------------------------------
        # Use the product rule to compute the gradient of
        # u from the gradient of (rho u). This ensures we don't
        # compute the derivatives twice.

        from pytools.obj_array import with_object_array_or_scalar
        dq = with_object_array_or_scalar(
                to_quad_op, self.grad_of_state())

        q = cse(to_quad_op(state))

        du = numpy.zeros((dimensions, dimensions), dtype=object)
        for i in range(dimensions):
            for j in range(dimensions):
                du[i,j] = cse(
                        (dq[i+2,j] - self.cse_u(q)[i] * dq[0,j]) / self.rho(q),
                        "du%d_d%s" % (i, AXES[j]))

        # }}}

        # {{{ put together viscous stress tau -----------------------------
        from pytools import delta

        if mu is None:
            mu = self.get_mu(q, to_quad_op)

        tau = numpy.zeros((dimensions, dimensions), dtype=object)
        for i in range(dimensions):
            for j in range(dimensions):
                tau[i,j] = cse(mu * cse(du[i,j] + du[j,i] -
                           2/self.dimensions * delta(i,j) * numpy.trace(du)),
                           "tau_%d%d" % (i, j))

        return tau

        # }}}

    # }}}

    # }}}

    # {{{ heat conduction

    def heat_conduction_coefficient(self, to_quad_op):
        mu = self.get_mu(self.state(), to_quad_op)
        if self.prandtl is None or numpy.isinf(self.prandtl):
            return 0

        eos = self.equation_of_state
        return (mu / self.prandtl) * (eos.gamma / (eos.gamma-1))

    def heat_conduction_grad(self, to_quad_op):
        grad_p_over_rho = self.grad_of_state_func(
                lambda state: self.p(state)/self.rho(state),
                "p_over_rho")

        return (self.heat_conduction_coefficient(to_quad_op)
                * to_quad_op(grad_p_over_rho))

    # }}}

    # {{{ flux

    def flux(self, q):
        from pytools import delta

        return [ # one entry for each flux direction
                cse(join_fields(
                    # flux rho
                    self.rho_u(q)[i],

                    # flux E
                    cse(self.e(q)+self.cse_p(q))*self.cse_u(q)[i],

                    # flux rho_u
                    make_obj_array([
                        self.rho_u(q)[i]*self.cse_u(q)[j] 
                        + delta(i,j) * self.cse_p(q)
                        for j in range(self.dimensions)
                        ])
                    ), "%s_flux" % AXES[i])
                for i in range(self.dimensions)]

    # }}}

    # {{{ boundary conditions ---------------------------------------------

    def make_bc_info(self, bc_name, tag, state, state0=None):
        """
        :param state0: The boundary 'free-stream' state around which the
          BC is linearized.
        """
        if state0 is None:
            state0 = make_sym_vector(bc_name, self.dimensions+2)

        state0 = cse(to_bdry_quad(state0))

        rho0 = self.rho(state0)
        p0 = self.cse_p(state0)
        u0 = self.cse_u(state0)

        c0 = (self.equation_of_state.gamma * p0 / rho0)**0.5

        from hedge.optemplate import BoundarizeOperator
        bdrize_op = BoundarizeOperator(tag)

        class SingleBCInfo(Record):
            pass

        return SingleBCInfo(
            rho0=rho0, p0=p0, u0=u0, c0=c0,

            # notation: suffix "m" for "minus", i.e. "interior"
            drhom=cse(self.rho(cse(to_bdry_quad(bdrize_op(state)))) 
                - rho0, "drhom"),
            dumvec=cse(self.cse_u(cse(to_bdry_quad(bdrize_op(state)))) 
                - u0, "dumvec"),
            dpm=cse(self.cse_p(cse(to_bdry_quad(bdrize_op(state)))) 
                - p0, "dpm"))

    def outflow_state(self, state):
        from hedge.optemplate import make_normal
        normal = make_normal(self.outflow_tag, self.dimensions)
        bc = self.make_bc_info("bc_q_out", self.outflow_tag, state)

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

    def inflow_state_inner(self, normal, bc, name):
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

    def inflow_state(self, state):
        from hedge.optemplate import make_normal
        normal = make_normal(self.inflow_tag, self.dimensions)
        bc = self.make_bc_info("bc_q_in", self.inflow_tag, state)
        return self.inflow_state_inner(normal, bc, "inflow")

    def noslip_state(self, state):
        from hedge.optemplate import make_normal
        state0 = join_fields(
            make_sym_vector("bc_q_noslip", 2),
            [0]*self.dimensions)
        normal = make_normal(self.noslip_tag, self.dimensions)
        bc = self.make_bc_info("bc_q_noslip", self.noslip_tag, state, state0)
        return self.inflow_state_inner(normal, bc, "noslip")

    def wall_state(self, state):
        from hedge.optemplate import BoundarizeOperator
        bc = BoundarizeOperator(self.wall_tag)(state)
        wall_rho = self.rho(bc)
        wall_e = self.e(bc) # <3 eve
        wall_rho_u = self.rho_u(bc)

        from hedge.optemplate import make_normal
        normal = make_normal(self.wall_tag, self.dimensions)

        return join_fields(
                wall_rho,
                wall_e,
                wall_rho_u - 2*numpy.dot(wall_rho_u, normal) * normal)

    @memoize_method
    def get_primitive_boundary_conditions(self):
        state = self.state()

        return {
                self.outflow_tag: self.outflow_state(state),
                self.inflow_tag: self.inflow_state(state),
                self.noslip_tag: self.noslip_state(state)
                }


    @memoize_method
    def get_conservative_boundary_conditions(self):
        state = self.state()

        from hedge.optemplate import BoundarizeOperator
        return {
                self.supersonic_inflow_tag:
                make_sym_vector("bc_q_supersonic_in", self.dimensions+2),
                self.supersonic_outflow_tag:
                BoundarizeOperator(self.supersonic_outflow_tag)(
                            (state)),
                self.wall_tag: self.wall_state(state),
                }

    @memoize_method
    def get_boundary_tags(self):
        return (set(self.get_primitive_boundary_conditions().keys())
                | set(self.get_conservative_boundary_conditions().keys()))

    @memoize_method
    def _normalize_expr(self, expr):
        """Normalize expressions for use as hash keys."""
        from hedge.optemplate.mappers import (
                QuadratureUpsamplerRemover,
                CSERemover)

        return CSERemover()(
                QuadratureUpsamplerRemover({}, do_warn=False)(expr))

    @memoize_method
    def _get_norm_primitive_exprs(self):
        return [
                self._normalize_expr(expr) for expr in
                self.conservative_to_primitive(self.state())
                ]

    @memoize_method
    def get_boundary_condition_for(self, tag, expr):
        prim_bcs = self.get_primitive_boundary_conditions()
        cons_bcs = self.get_conservative_boundary_conditions()

        if tag in prim_bcs:
            # BC is given in primitive variables, avoid converting
            # to conservative and back.
            try:
                norm_expr = self._normalize_expr(expr)
                prim_idx = self._get_norm_primitive_exprs().index(norm_expr)
            except ValueError:
                cbstate = self.primitive_to_conservative(
                        prim_bcs[tag])
            else:
                return prim_bcs[tag][prim_idx]
        else:
            # BC is given in conservative variables, no potential
            # for optimization.

            cbstate = to_bdry_quad(cons_bcs[tag])

        # 'cbstate' is the boundary state in conservative variables.

        from hedge.optemplate.mappers import QuadratureUpsamplerRemover
        expr = QuadratureUpsamplerRemover({}, do_warn=False)(expr)

        def subst_func(expr):
            from pymbolic.primitives import Subscript, Variable

            if isinstance(expr, Subscript):
                assert (isinstance(expr.aggregate, Variable) 
                        and expr.aggregate.name == "q")

                return cbstate[expr.index]
            elif isinstance(expr, Variable) and expr.name =="sensor":
                from hedge.optemplate import BoundarizeOperator
                result = BoundarizeOperator(tag)(self.sensor())
                return cse(to_bdry_quad(result), "bdry_sensor")

        from hedge.optemplate import SubstitutionMapper
        return SubstitutionMapper(subst_func)(expr)

    # }}}

    # {{{ second order part
    def div(self, vol_operand, int_face_operand):
        from hedge.second_order import SecondDerivativeTarget
        div_tgt = SecondDerivativeTarget(
                self.dimensions, strong_form=False,
                operand=vol_operand,
                int_flux_operand=int_face_operand)

        self.second_order_scheme.div(div_tgt,
                bc_getter=self.get_boundary_condition_for,
                dirichlet_tags=list(self.get_boundary_tags()),
                neumann_tags=[])

        return div_tgt.minv_all

    def make_second_order_part(self):
        state = self.state()
        faceq_state = self.faceq_state()
        volq_state = self.volq_state()

        volq_tau_mat = self.tau(to_vol_quad, state)
        faceq_tau_mat = self.tau(to_int_face_quad, state)

        return join_fields(
                0, 
                self.div(
                    numpy.sum(volq_tau_mat*self.cse_u(volq_state), axis=1)
                    + self.heat_conduction_grad(to_vol_quad)
                    ,
                    numpy.sum(faceq_tau_mat*self.cse_u(faceq_state), axis=1)
                    + self.heat_conduction_grad(to_int_face_quad)
                    ,
                    ),
                [
                    self.div(volq_tau_mat[i], faceq_tau_mat[i])
                    for i in range(self.dimensions)]
                )

    # }}}

    # {{{ operator template ---------------------------------------------------
    def make_extra_terms(self):
        return 0

    def op_template(self, sensor_scaling=None, viscosity_only=False):
        u = self.cse_u
        rho = self.cse_rho
        rho_u = self.rho_u
        p = self.p
        e = self.e

        # {{{ artificial diffusion
        def make_artificial_diffusion():
            if self.artificial_viscosity_mode not in ["diffusion"]:
                return 0

            dq = self.grad_of_state()

            return make_obj_array([
                self.div(
                    to_vol_quad(self.sensor())*to_vol_quad(dq[i]),
                    to_int_face_quad(self.sensor())*to_int_face_quad(dq[i])) 
                for i in range(dq.shape[0])])
        # }}}

        # {{{ state setup

        volq_flux = self.flux(self.volq_state())
        faceq_flux = self.flux(self.faceq_state())

        from hedge.optemplate.primitives import CFunction
        sqrt = CFunction("sqrt")

        speed = self.characteristic_velocity_optemplate(self.state())

        has_viscosity = not is_zero(self.get_mu(self.state(), to_quad_op=None))

        # }}}

        # {{{ operator assembly -----------------------------------------------
        from hedge.flux.tools import make_lax_friedrichs_flux
        from hedge.optemplate.operators import InverseMassOperator

        from hedge.optemplate.tools import make_stiffness_t

        primitive_bcs_as_quad_conservative = dict(
                (tag, self.primitive_to_conservative(to_bdry_quad(bc)))
                for tag, bc in 
                self.get_primitive_boundary_conditions().iteritems())

        def get_bc_tuple(tag):
            state = self.state()
            bc = make_obj_array([
                self.get_boundary_condition_for(tag, s_i) for s_i in state])
            return tag, bc, self.flux(bc)

        first_order_part = InverseMassOperator()(
                numpy.dot(make_stiffness_t(self.dimensions), volq_flux)
                - make_lax_friedrichs_flux(
                    wave_speed=cse(to_int_face_quad(speed), "emax_c"),

                    state=self.faceq_state(), fluxes=faceq_flux,
                    bdry_tags_states_and_fluxes=[
                        get_bc_tuple(tag) for tag in self.get_boundary_tags()],
                    strong=False))

        if viscosity_only:
            first_order_part = 0*first_order_part

        result = join_fields(
                first_order_part 
                + self.make_second_order_part()
                + make_artificial_diffusion()
                + self.make_extra_terms(),
                 speed)

        if self.source is not None:
            result = result + join_fields(
                    make_sym_vector("source_vect", len(self.state())),
                    # extra field for speed
                    0)

        return result

        # }}}

    # }}}

    # {{{ operator binding ----------------------------------------------------
    def bind(self, discr, sensor=None, sensor_scaling=None, viscosity_only=False):
        if (sensor is None and 
                self.artificial_viscosity_mode is not None):
            raise ValueError("must specify a sensor if using "
                    "artificial viscosity")

        bound_op = discr.compile(self.op_template(
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
    def __init__(self, discr, gamma, dimensions, op):
        """Construct a limiter from Jan's book page 225
        """
        self.discr = discr
        self.gamma=gamma
        self.dimensions=dimensions
        self.op=op

        from hedge.optemplate.operators import AveragingOperator
        self.get_average = AveragingOperator().bind(discr)

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
