"""Kennedy-Carpenter implicit/explicit RK."""

from __future__ import division

__copyright__ = "Copyright (C) 2010 Andreas Kloeckner"

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




from hedge.timestep.runge_kutta import EmbeddedRungeKuttaTimeStepperBase
import numpy




class KennedyCarpenterIMEXRungeKuttaBase(EmbeddedRungeKuttaTimeStepperBase):
    """
    Christopher A. Kennedy, Mark H. Carpenter. Additive Runge-Kutta
    schemes for convection-diffusion-reaction equations.
    Applied Numerical Mathematics
    Volume 44, Issues 1-2, January 2003, Pages 139-181 
    http://dx.doi.org/10.1016/S0168-9274(02)00138-1
    """

    def __call__(self, y, t, dt, rhs_expl, rhs_impl, reject_hook=None):
        r"""
        :arg rhs_impl: for a signature of (t, y0, alpha), returns
          a value of *k* satisfying

          .. math::

              k = f(t, y_0 + \alpha*k)

          where *f* is the right-hand side function.

          .. note::
            For a linear *f*, the relationship above may be
            rewritten as

            .. math::

                (Id-\alpha A)k = A y_0.
        """
        from hedge.tools import count_dofs

        # {{{ preparation, linear combiners
        try:
            self.last_rhs_expl
        except AttributeError:
            self.last_rhs_expl = rhs_expl(t, y)
            self.last_rhs_impl = rhs_impl(t, y, 0)

            self.dof_count = count_dofs(self.last_rhs_expl)

            if self.adaptive:
                self.ip = self.vector_primitive_factory \
                        .make_inner_product(self.last_rhs)
            else:
                self.ip = None

        ip = self.ip

        # }}}

        flop_count = [0]

        while True:
            explicit_rhss = []
            implicit_rhss = []

            # {{{ stage loop

            for c, coeffs_expl, coeffs_impl,  in zip(
                    self.c, self.a_explicit, self.a_implicit):
                if len(coeffs_expl) == 0:
                    assert c == 0
                    assert len(coeffs_impl) == 0
                    this_rhs_expl = self.last_rhs_expl
                    this_rhs_impl = self.last_rhs_impl
                else:
                    sub_timer = self.timer.start_sub_timer()

                    assert len(coeffs_expl) == len(explicit_rhss)
                    assert len(coeffs_impl) == len(implicit_rhss)

                    args = [(1, y)] + [
                            (dt*coeff, erhs)
                            for coeff, erhs in zip(
                                coeffs_expl + coeffs_impl, 
                                explicit_rhss + implicit_rhss)
                            if coeff]
                    flop_count[0] += len(args)*2 - 1
                    sub_y = self.get_linear_combiner(
                            len(args), self.last_rhs_expl)(*args)
                    sub_timer.stop().submit()

                    this_rhs_expl = rhs_expl(t + c*dt, sub_y)
                    this_rhs_impl = rhs_impl(t + c*dt, sub_y, 
                            self.gamma*dt)

                explicit_rhss.append(this_rhs_expl)
                implicit_rhss.append(this_rhs_impl)

            # }}}

            def finish_solution(coeffs):
                assert (len(coeffs) == len(explicit_rhss)
                        == len(implicit_rhss))

                args = [(1, y)] + [
                        (dt*coeff, erhs) 
                        for coeff, erhs in zip(
                                coeffs + coeffs, 
                                explicit_rhss + implicit_rhss)
                        if coeff]
                flop_count[0] += len(args)*2 - 1
                return self.get_linear_combiner(
                        len(args), self.last_rhs_expl)(*args)

            if not self.adaptive:
                if self.use_high_order:
                    y = finish_solution(self.high_order_coeffs)
                else:
                    y = finish_solution(self.low_order_coeffs)

                self.last_rhs_expl = this_rhs_expl
                self.last_rhs_impl = this_rhs_impl
                self.flop_counter.add(self.dof_count*flop_count[0])
                return y
            else:
                # {{{ step size adaptation
                high_order_end_y = finish_solution(self.high_order_coeffs)
                low_order_end_y = finish_solution(self.low_order_coeffs)

                flop_count[0] += 3+1 # one two-lincomb, one norm

                from hedge.timestep.runge_kutta import adapt_step_size
                accept_step, next_dt, rel_err = adapt_step_size(
                        t, dt, y, high_order_end_y, low_order_end_y,
                        self, self.get_linear_combiner(2, self.last_rhs), ip)

                if not accept_step:
                    if reject_hook:
                        y = reject_hook(dt, rel_err, t, y)

                    dt = next_dt
                    # ... and go back to top of loop
                else:
                    # finish up
                    self.last_rhs_expl = this_rhs_expl
                    self.last_rhs_impl = this_rhs_impl
                    self.flop_counter.add(self.dof_count*flop_count[0])

                    return high_order_end_y, t+dt, dt, next_dt
                # }}}




class KennedyCarpenterIMEXARK4(KennedyCarpenterIMEXRungeKuttaBase):
    gamma = 1./4.

    c = [0, 1/2, 83/250, 31/50, 17/20, 1]
    low_order = 3
    high_order = 4

    high_order_coeffs = [
            82889/524892,
            0,
            15625/83664,
            69875/102672,
            -2260/8211,
            1/4]

    low_order_coeffs = [
            4586570599/29645900160,
            0,
            178811875/945068544,
            814220225/1159782912,
            -3700637/11593932,
            61727/225920
            ]

    # ARK4(3)6L[2]SA-ERK (explicit)
    a_explicit = [[],
            [1/2],
            [13861/62500, 6889/62500],
            [-116923316275/2393684061468,
                -2731218467317/15368042101831,
                9408046702089/11113171139209],
            [-451086348788/2902428689909,
                -2682348792572/7519795681897,
                12662868775082/11960479115383,
                3355817975965/11060851509271],
            [647845179188/3216320057751,
                73281519250/8382639484533,
                552539513391/3454668386233,
                3354512671639/8306763924573,
                4040/17871]]

    # ARK4(3)6L[2]SA-ESDIRK (implicit)
    a_implicit = [[],
            [1/4],
            [8611/62500, -1743/31250],
            [5012029/34652500, -654441/2922500, 174375/388108],
            [15267082809/155376265600, -71443401/120774400,
                730878875/902184768, 2285395/8070912],
            [82889/524892, 0, 15625/83664, 69875/102672,
                -2260/8211]]


    assert (len(a_explicit) == len(a_implicit)
            == len(low_order_coeffs)
            == len(high_order_coeffs)
            == len(c))
