"""Runge-Kutta ODE timestepper."""

from __future__ import division

__copyright__ = "Copyright (C) 2007 Andreas Kloeckner"

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
from hedge.timestep.base import TimeStepper


# {{{ Carpenter/Kennedy low-storage fourth-order Runge-Kutta

class LSRK4TimeStepper(TimeStepper):
    """A low storage fourth-order Runge-Kutta method

    See JSH, TW: Nodal Discontinuous Galerkin Methods p.64
    or
    Carpenter, M.H., and Kennedy, C.A., Fourth-order-2N-storage
    Runge-Kutta schemes, NASA Langley Tech Report TM 109112, 1994
    """

    _RK4A = [
            0.0,
            -567301805773 / 1357537059087,
            -2404267990393 / 2016746695238,
            -3550918686646 / 2091501179385,
            -1275806237668 / 842570457699,
            ]

    _RK4B = [
            1432997174477 / 9575080441755,
            5161836677717 / 13612068292357,
            1720146321549 / 2090206949498,
            3134564353537 / 4481467310338,
            2277821191437 / 14882151754819,
            ]

    _RK4C = [
            0.0,
            1432997174477/9575080441755,
            2526269341429/6820363962896,
            2006345519317/3224310063776,
            2802321613138/2924317926251,
            #1,
            ]

    dt_fudge_factor = 1

    adaptive = False

    def __init__(self, dtype=numpy.float64, rcon=None,
            vector_primitive_factory=None):
        if vector_primitive_factory is None:
            from hedge.vector_primitives import VectorPrimitiveFactory
            self.vector_primitive_factory = VectorPrimitiveFactory()
        else:
            self.vector_primitive_factory = vector_primitive_factory

        from pytools.log import IntervalTimer, EventCounter
        timer_factory = IntervalTimer
        if rcon is not None:
            timer_factory = rcon.make_timer

        self.timer = timer_factory(
                "t_rk4", "Time spent doing algebra in RK4")
        self.flop_counter = EventCounter(
                "n_flops_rk4", "Floating point operations performed in RK4")

        from pytools import match_precision
        self.dtype = numpy.dtype(dtype)
        self.scalar_dtype = match_precision(
                numpy.dtype(numpy.float64), self.dtype)
        self.coeffs = numpy.array([self._RK4A, self._RK4B, self._RK4C],
                dtype=self.scalar_dtype).T

    def get_stability_relevant_init_args(self):
        return ()

    def add_instrumentation(self, logmgr):
        logmgr.add_quantity(self.timer)
        logmgr.add_quantity(self.flop_counter)

    def __call__(self, y, t, dt, rhs):
        try:
            self.residual
        except AttributeError:
            self.residual = 0*rhs(t, y)
            from hedge.tools import count_dofs
            self.dof_count = count_dofs(self.residual)

            self.linear_combiner = self.vector_primitive_factory\
                    .make_linear_combiner(self.dtype, self.scalar_dtype,
                            y, arg_count=2)

        lc = self.linear_combiner

        for a, b, c in self.coeffs:
            this_rhs = rhs(t + c*dt, y)

            sub_timer = self.timer.start_sub_timer()
            self.residual = lc((a, self.residual), (dt, this_rhs))
            del this_rhs
            y = lc((1, y), (b, self.residual))
            sub_timer.stop().submit()

        # 5 is the number of flops above, *NOT* the number of stages,
        # which is already captured in len(self.coeffs)
        self.flop_counter.add(len(self.coeffs)*self.dof_count*5)

        return y


class RK4TimeStepper(LSRK4TimeStepper):
    def __init__(self, *args, **kwargs):
        from warnings import warn
        warn("RK4TimeStepper is a deprecated name of LSRK4TimeStepper",
                DeprecationWarning, stacklevel=2)

        LSRK4TimeStepper.__init__(self, *args, **kwargs)

# }}}


# {{{ Embedded Runge-Kutta schemes base class

def adapt_step_size(t, dt,
        start_y, high_order_end_y, low_order_end_y, stepper, lc2, norm):
    normalization = stepper.atol + stepper.rtol*max(
                norm(low_order_end_y), norm(start_y))

    error = lc2(
        (1/normalization, high_order_end_y),
        (-1/normalization, low_order_end_y)
        )

    from hedge.tools import count_dofs
    rel_err = norm(error)/count_dofs(error)**0.5
    if rel_err == 0:
        rel_err = 1e-14

    if rel_err > 1 or numpy.isnan(rel_err):
        # reject step

        if not numpy.isnan(rel_err):
            dt = max(
                    0.9 * dt * rel_err**(-1/stepper.low_order),
                    stepper.min_dt_shrinkage * dt)
        else:
            dt = stepper.min_dt_shrinkage*dt

        if t + dt == t:
            from hedge.timestep import TimeStepUnderflow
            raise TimeStepUnderflow()

        return False, dt, rel_err
    else:
        # accept step

        next_dt = min(
                0.9 * dt * rel_err**(-1/stepper.high_order),
                stepper.max_dt_growth*dt)

        return True, next_dt, rel_err


class EmbeddedRungeKuttaTimeStepperBase(TimeStepper):
    def __init__(self, use_high_order=True, dtype=numpy.float64, rcon=None,
            vector_primitive_factory=None, atol=0, rtol=0,
            max_dt_growth=5, min_dt_shrinkage=0.1,
            limiter=None):
        if vector_primitive_factory is None:
            from hedge.vector_primitives import VectorPrimitiveFactory
            self.vector_primitive_factory = VectorPrimitiveFactory()
        else:
            self.vector_primitive_factory = vector_primitive_factory

        from pytools.log import IntervalTimer, EventCounter
        timer_factory = IntervalTimer
        if rcon is not None:
            timer_factory = rcon.make_timer

        if limiter is None:
            self.limiter = lambda x: x
        else:
            self.limiter = limiter

        self.timer = timer_factory(
                "t_rk", "Time spent doing algebra in Runge-Kutta")
        self.flop_counter = EventCounter(
                "n_flops_rk", "Floating point operations performed in Runge-Kutta")

        self.use_high_order = use_high_order

        self.dtype = numpy.dtype(dtype)

        self.adaptive = bool(atol or rtol)
        self.atol = atol
        self.rtol = rtol

        from pytools import match_precision
        self.scalar_dtype = match_precision(
                numpy.dtype(numpy.float64), self.dtype)

        self.max_dt_growth = max_dt_growth
        self.min_dt_shrinkage = min_dt_shrinkage

        self.linear_combiner_cache = {}

    def get_stability_relevant_init_args(self):
        return (self.use_high_order,)

    def add_instrumentation(self, logmgr):
        logmgr.add_quantity(self.timer)
        logmgr.add_quantity(self.flop_counter)

    def get_linear_combiner(self, arg_count, sample_vec):
        try:
            return self.linear_combiner_cache[arg_count]
        except KeyError:
            lc = self.vector_primitive_factory \
                    .make_linear_combiner(
                            self.dtype, self.scalar_dtype, sample_vec,
                            arg_count=arg_count)
            self.linear_combiner_cache[arg_count] = lc
            return lc


class EmbeddedButcherTableauTimeStepperBase(EmbeddedRungeKuttaTimeStepperBase):
    def __call__(self, y, t, dt, rhs, reject_hook=None):
        from hedge.tools import count_dofs

        # {{{ preparation
        try:
            self.last_rhs
        except AttributeError:
            self.last_rhs = rhs(t, y)
            self.dof_count = count_dofs(self.last_rhs)

            if self.adaptive:
                self.norm = self.vector_primitive_factory \
                        .make_maximum_norm(self.last_rhs)
            else:
                self.norm = None

        # }}}

        flop_count = [0]

        while True:
            rhss = []

            # {{{ stage loop

            for i, (c, coeffs) in enumerate(self.butcher_tableau):
                if len(coeffs) == 0:
                    assert c == 0
                    this_rhs = self.last_rhs
                else:
                    sub_timer = self.timer.start_sub_timer()
                    args = [(1, y)] + [
                            (dt*coeff, rhss[j]) for j, coeff in enumerate(coeffs)
                            if coeff]
                    flop_count[0] += len(args)*2 - 1
                    sub_y = self.limiter(self.get_linear_combiner(
                            len(args), self.last_rhs)(*args))
                    sub_timer.stop().submit()

                    this_rhs = rhs(t + c*dt, sub_y)

                rhss.append(this_rhs)

            # }}}

            def finish_solution(coeffs):
                args = [(1, y)] + [
                        (dt*coeff, rhss[i]) for i, coeff in enumerate(coeffs)
                        if coeff]
                flop_count[0] += len(args)*2 - 1
                return self.get_linear_combiner(
                        len(args), self.last_rhs)(*args)

            if not self.adaptive:
                if self.use_high_order:
                    y = self.limiter(finish_solution(self.high_order_coeffs))
                else:
                    y = self.limiter(finish_solution(self.low_order_coeffs))

                self.last_rhs = this_rhs
                self.flop_counter.add(self.dof_count*flop_count[0])
                return y
            else:
                # {{{ step size adaptation
                high_order_end_y = finish_solution(self.high_order_coeffs)
                low_order_end_y = finish_solution(self.low_order_coeffs)

                flop_count[0] += 3+1  # one two-lincomb, one norm

                # Perform error estimation based on un-limited solutions.
                accept_step, next_dt, rel_err = adapt_step_size(
                        t, dt, y, high_order_end_y, low_order_end_y,
                        self, self.get_linear_combiner(2, high_order_end_y),
                        self.norm)

                if not accept_step:
                    if reject_hook:
                        y = reject_hook(dt, rel_err, t, y)

                    dt = next_dt
                    # ... and go back to top of loop
                else:
                    # finish up
                    self.last_rhs = this_rhs
                    self.flop_counter.add(self.dof_count*flop_count[0])

                    return self.limiter(high_order_end_y), t+dt, dt, next_dt
                # }}}

# }}}


# {{{ Bogacki-Shampine second/third-order Runge-Kutta

class ODE23TimeStepper(EmbeddedButcherTableauTimeStepperBase):
    """Bogacki-Shampine second/third-order Runge-Kutta.

    (same as Matlab's ode23)

    Bogacki, Przemyslaw; Shampine, Lawrence F. (1989), "A 3(2) pair of
    Runge-Kutta formulas", Applied Mathematics Letters 2 (4): 321-325,
    doi:10.1016/0893-9659(89)90079-7
    """

    dt_fudge_factor = 1

    butcher_tableau = [
            (0, []),
            (1/2, [1/2]),
            (3/4, [0, 3/4]),
            (1, [2/9, 1/3, 4/9])
            ]

    low_order = 2
    low_order_coeffs = [7/24, 1/4, 1/3, 1/8]
    high_order = 3
    high_order_coeffs = [2/9, 1/3, 4/9, 0]

# }}}


# {{{ Dormand-Prince fourth/fifth-order Runge-Kutta

class ODE45TimeStepper(EmbeddedButcherTableauTimeStepperBase):
    """Dormand-Prince fourth/fifth-order Runge-Kutta.

    (same as Matlab's ode45)

    Dormand, J. R.; Prince, P. J. (1980), "A family of embedded Runge-Kutta
    formulae", Journal of Computational and Applied Mathematics 6 (1): 19-26,
    doi:10.1016/0771-050X(80)90013-3.
    """

    dt_fudge_factor = 1

    butcher_tableau = [
            (0, []),
            (1/5, [1/5]),
            (3/10, [3/40, 9/40]),
            (4/5, [44/45, -56/15, 32/9]),
            (8/9, [19372/6561, -25360/2187, 64448/6561, -212/729]),
            (1, [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656]),
            (1, [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84])
            ]

    low_order = 4
    low_order_coeffs = [5179/57600, 0, 7571/16695, 393/640, -92097/339200,
            187/2100, 1/40]
    high_order = 5
    high_order_coeffs = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]

# }}}


# {{{ Shu-Osher-form SSP RK

class EmbeddedShuOsherFormTimeStepperBase(EmbeddedRungeKuttaTimeStepperBase):
    r"""
    The attribute *shu_osher_tableau* is defined by and consists of a tuple of
    lists of :math:`\alpha` and :math:`\beta` as given in (2.10) of [1]. Each
    list entry contains a coefficient and an index. Within the list of
    :math:`\alpha`, these index into values of :math:`u^{(i)}`, where the
    initial condition has index 0, the first row of the tableau index 1, and so
    on.  Within the list of :math:`beta`, the index is into function
    evaluations at :math:`u^{(i)}`.

    *low_order_index* and *high_order_index* give the result of the embedded
    high- and low-order methods.

    [1] S. Gottlieb, D. Ketcheson, and C.-W. Shu, Strong Stability Preserving
    Time Discretizations. World Scientific, 2011.
    """

    def __call__(self, y, t, dt, rhs, reject_hook=None):

        flop_count = 0

        def get_rhs(i):
            try:
                return rhss[i]
            except KeyError:
                result = rhs(t + time_fractions[i]*dt, row_values[i])
                rhss[i] = result

                try:
                    self.dof_count
                except AttributeError:
                    from hedge.tools import count_dofs
                    self.dof_count = count_dofs(result)

                return result

        while True:
            time_fractions = [0]
            row_values = [y]
            rhss = {}

            # {{{ row loop

            for alpha_list, beta_list in self.shu_osher_tableau:
                sub_timer = self.timer.start_sub_timer()
                args = ([(alpha, row_values[i]) for alpha, i in alpha_list]
                        + [(dt*beta, get_rhs(i)) for beta, i in beta_list])
                flop_count += len(args)*2 - 1

                some_rhs = iter(rhss.itervalues()).next()
                row_values.append(
                        self.limiter(
                            self.get_linear_combiner(len(args), some_rhs)(*args)))
                sub_timer.stop().submit()

                time_fractions.append(
                        sum(alpha * time_fractions[i] for alpha, i in alpha_list)
                        + sum(beta for beta, i in beta_list))

            # }}}

            if not self.adaptive:
                self.flop_counter.add(self.dof_count*flop_count)

                if self.use_high_order:
                    assert abs(time_fractions[self.high_order_index] - 1) < 1e-15
                    return row_values[self.high_order_index]
                else:
                    assert abs(time_fractions[self.low_order_index] - 1) < 1e-15
                    return row_values[self.low_order_index]
            else:
                # {{{ step size adaptation

                assert abs(time_fractions[self.high_order_index] - 1) < 1e-15
                assert abs(time_fractions[self.low_order_index] - 1) < 1e-15

                high_order_end_y = row_values[self.high_order_index]
                low_order_end_y = row_values[self.low_order_index]

                some_rhs = iter(rhss.itervalues()).next()

                try:
                    norm = self.norm
                except AttributeError:
                    norm = self.norm = self.vector_primitive_factory \
                            .make_maximum_norm(some_rhs)

                flop_count += 3+1  # one two-lincomb, one norm
                accept_step, next_dt, rel_err = adapt_step_size(
                        t, dt, y, high_order_end_y, low_order_end_y,
                        self, self.get_linear_combiner(2, some_rhs), norm)

                if not accept_step:
                    if reject_hook:
                        y = reject_hook(dt, rel_err, t, y)

                    dt = next_dt
                    # ... and go back to top of loop
                else:
                    # finish up
                    self.flop_counter.add(self.dof_count*flop_count)

                    return high_order_end_y, t+dt, dt, next_dt

                # }}}


class SSP2TimeStepper(EmbeddedShuOsherFormTimeStepperBase):
    """
    See Theorem 2.2 of Section 2.4.1 in [1].

    [1] S. Gottlieb, D. Ketcheson, and C.-W. Shu, Strong Stability Preserving
    Time Discretizations. World Scientific, 2011.
    """

    dt_fudge_factor = 1

    shu_osher_tableau = [
            ([(1, 0)], [(1, 0)]),
            ([(1/2, 0), (1/2, 1)], [(1/2, 1)]),
            ]

    # no low-order
    high_order = 2
    high_order_index = 2


class SSP3TimeStepper(EmbeddedShuOsherFormTimeStepperBase):
    """
    See Theorem 2.2 of Section 2.4.1 in [1].

    [1] S. Gottlieb, D. Ketcheson, and C.-W. Shu, Strong Stability Preserving
    Time Discretizations. World Scientific, 2011.
    """

    dt_fudge_factor = 1

    shu_osher_tableau = [
            ([(1, 0)], [(1, 0)]),
            ([(3/4, 0), (1/4, 1)], [(1/4, 1)]),
            ([(1/3, 0), (2/3, 2)], [(2/3, 2)]),
            ]

    # no low-order
    high_order = 3
    high_order_index = 3


class SSP23FewStageTimeStepper(EmbeddedShuOsherFormTimeStepperBase):
    """
    See Example 6.1 of Section 6.3 in [1].

    [1] S. Gottlieb, D. Ketcheson, and C.-W. Shu, Strong Stability Preserving
    Time Discretizations. World Scientific, 2011.
    """

    dt_fudge_factor = 1

    shu_osher_tableau = [
            ([(1, 0)], [(1/2, 0)]),
            ([(1, 1)], [(1/2, 1)]),
            ([(1/3, 0), (2/3, 2)], [(1/2*2/3, 2)]),
            ([(2/3, 0), (1/3, 2)], [(1/2*1/3, 2)]),
            ([(1, 4)], [(1/2, 4)]),
            ]

    low_order = 2
    low_order_index = 3
    high_order = 3
    high_order_index = 5


class SSP23ManyStageTimeStepper(EmbeddedShuOsherFormTimeStepperBase):
    """
    See Example 6.2 of Section 6.3 in [1].

    [1] S. Gottlieb, D. Ketcheson, and C.-W. Shu, Strong Stability Preserving
    Time Discretizations. World Scientific, 2011.
    """

    dt_fudge_factor = 1

    # This has a known bug--perhaps an issue with reference [1]?
    # Entry 7 is not consistent as a second-order approximation.

    shu_osher_tableau = [
                ([(1, i)], [(1/6, i)]) for i in range(6)
            ]+[
                ([(1/7, 0), (6*1/7, 6)], [(1/6*1/7, 6)]),  # 7
                ([(3/5, 1), (2/5, 6)], []),  # 8: u^{(6)\ast}
            ]+[
                ([(1, i-1)], [(1/6, i-1)]) for i in range(9, 9+2+1)
            ]

    low_order = 2
    low_order_index = 7
    high_order = 3
    high_order_index = -1

# }}}




# vim: foldmethod=marker
