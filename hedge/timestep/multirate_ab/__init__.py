# -*- coding: utf8 -*-

"""Multirate-AB ODE solver."""

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
from pytools import memoize, memoize_method
from hedge.timestep.base import TimeStepper
from hedge.timestep.rk4 import RK4TimeStepper
from hedge.timestep.ab import \
        make_generic_ab_coefficients, \
        make_ab_coefficients
from hedge.timestep.multirate_ab.methods import \
        HIST_NAMES
from hedge.timestep.multirate_ab.processors import \
        MRABProcessor




# helper functions ------------------------------------------------------------

def _linear_comb(coefficients, vectors):
    from operator import add
    return reduce(add,
            (coeff * v for coeff, v in
                zip(coefficients, vectors)))





class TwoRateAdamsBashforthTimeStepper(TimeStepper):
    """Simultaneously timesteps two parts of an ODE system,
    the first with a small timestep, the second with a large timestep.

    [1] C.W. Gear and D.R. Wells, "Multirate linear multistep methods," BIT
    Numerical Mathematics,  vol. 24, Dec. 1984, pg. 484-502.
    """

    def __init__(self, method, large_dt, substep_count, order,
            order_f2f=None, order_s2f=None,
            order_f2s=None, order_s2s=None,
            startup_stepper=None):

        if isinstance(method, str):
            from hedge.timestep.multirate_ab.methods import methods
            method = methods[method]
        self.method = method

        self.large_dt = large_dt
        self.small_dt = large_dt/substep_count
        self.substep_count = substep_count

        from hedge.timestep.multirate_ab.methods import \
                HIST_F2F, HIST_F2S, HIST_S2F, HIST_S2S

        self.orders = {
                HIST_F2F: order_f2f,
                HIST_S2F: order_s2f,
                HIST_F2S: order_f2s,
                HIST_S2S: order_s2s,
                }

        for hn in HIST_NAMES:
            if self.orders[hn] is None:
                self.orders[hn] = order

        self.max_order = max(self.orders.values())

        # histories of rhs evaluations
        self.histories = dict((hn, []) for hn in HIST_NAMES)

        if startup_stepper is not None:
            self.startup_stepper = startup_stepper
        else:
            self.startup_stepper = RK4TimeStepper()

        self.startup_history = []

        self.hist_is_fast = {
                HIST_F2F: True,
                HIST_S2F: self.method.s2f_hist_is_fast,
                HIST_S2S: False,
                HIST_F2S: False
                }

    def __call__(self, ys, t, rhss):
        """
        @arg rhss: Matrix of right-hand sides, stored in row-major order, i.e.
        C{[f2f, s2f, f2s, s2s]}.
        """
        from hedge.tools import make_obj_array

        def finish_startup():
            # we're done starting up, pack data into split histories
            for hn, hist in zip(HIST_NAMES, self.startup_history):
                if not self.hist_is_fast[hn]:
                    hist = hist[::self.substep_count]

                self.histories[hn] = list(hist[:self.orders[hn]])

            # here's some memory we won't need any more
            self.startup_stepper = None
            del self.startup_history

        def combined_rhs(t, y):
            y_fast, y_slow = y
            return make_obj_array([
                rhs(t, lambda: y_fast, lambda: y_slow)
                for rhs in rhss])

        def combined_summed_rhs(t, y):
            return numpy.sum(combined_rhs(t, y).reshape((2,2), order="C"), axis=1)

        if self.startup_stepper is not None:
            ys = make_obj_array(ys)

            if self.max_order == 1:
                # we're running forward Euler, no need for the startup stepper

                assert not self.startup_history
                self.startup_history.append(combined_rhs(t, ys))
                finish_startup()
                return self.run_ab(ys, t, rhss)

            for i in range(self.substep_count):
                ys = self.startup_stepper(ys, t+i*self.small_dt, self.small_dt,
                        combined_summed_rhs)
                self.startup_history.insert(0, combined_rhs(t+(i+1)*self.small_dt, ys))

            if len(self.startup_history) == self.max_order*self.substep_count:
                finish_startup()

            return ys
        else:
            return self.run_ab(ys, t, rhss)

    def run_ab(self, ys, t, rhss):
        step_evaluator = _MRABEvaluator(self, ys, t, rhss)
        step_evaluator.run()
        return step_evaluator.get_result()

    @memoize_method
    def get_coefficients(self, for_fast_history, t_start, t_end, order):
        history_times = numpy.arange(0, -order, -1,
                dtype=numpy.float64)

        if for_fast_history:
            history_times /= self.substep_count

        return make_generic_ab_coefficients(history_times, t_start, t_end)




class _MRABEvaluator(MRABProcessor):
    def __init__(self, stepper, y, t, rhss):
        MRABProcessor.__init__(self, stepper.method, stepper.substep_count)

        self.stepper = stepper

        self.t_start = t

        self.context = {}
        self.var_time_level = {}

        self.rhss = rhss

        y_fast, y_slow = y
        from hedge.timestep.multirate_ab.methods import CO_FAST, CO_SLOW
        self.last_y = {CO_FAST: y_fast, CO_SLOW: y_slow}

    def integrate_in_time(self, insn):
        from hedge.timestep.multirate_ab.methods import CO_FAST, CO_SLOW
        from hedge.timestep.multirate_ab.methods import \
                HIST_F2F, HIST_F2S, HIST_S2F, HIST_S2S

        if insn.component == CO_FAST:
            self_hn, cross_hn = HIST_F2F, HIST_S2F
        else:
            self_hn, cross_hn = HIST_S2S, HIST_F2S

        start_time_level = self.eval_expr(insn.start)
        end_time_level = self.eval_expr(insn.end)

        self_coefficients = self.stepper.get_coefficients(
                self.stepper.hist_is_fast[self_hn],
                start_time_level/self.stepper.substep_count,
                end_time_level/self.stepper.substep_count,
                self.stepper.orders[self_hn])
        cross_coefficients = self.stepper.get_coefficients(
                self.stepper.hist_is_fast[cross_hn],
                start_time_level/self.stepper.substep_count,
                end_time_level/self.stepper.substep_count,
                self.stepper.orders[cross_hn])

        if start_time_level == 0 or (insn.result_name not in self.context):
            my_y = self.last_y[insn.component]
            assert start_time_level == 0
        else:
            my_y = self.context[insn.result_name]()
            assert start_time_level == \
                    self.var_time_level[insn.result_name]

        hists = self.stepper.histories
        my_integrated_y = memoize(
                lambda: my_y + self.stepper.large_dt * (
                _linear_comb(self_coefficients, hists[self_hn])
                + _linear_comb(cross_coefficients, hists[cross_hn])))

        self.context[insn.result_name] = my_integrated_y
        self.var_time_level[insn.result_name] = end_time_level

        MRABProcessor.integrate_in_time(self, insn)

    def history_update(self, insn):
        time_slow = self.var_time_level[insn.slow_arg]
        time_fast = self.var_time_level[insn.fast_arg]

        assert time_slow == time_fast

        t = self.t_start + time_slow/self.stepper.substep_count

        rhs = self.rhss[HIST_NAMES.index(insn.which)]

        hist = self.stepper.histories[insn.which]
        hist.pop()
        hist.insert(0,
                rhs(t,
                    self.context[insn.slow_arg],
                    self.context[insn.fast_arg]))

        MRABProcessor.history_update(self, insn)

    def get_result(self):
        meth = self.stepper.method

        assert self.var_time_level[meth.result_slow] == self.stepper.substep_count
        assert self.var_time_level[meth.result_fast] == self.stepper.substep_count

        return (self.context[meth.result_fast](),
                self.context[meth.result_slow]())



class TwoRateAdamsBashforthTimeStepperBase:
    pass

class TwoRateAdamsBashforthTimeStepperFastestFirstMethod(TwoRateAdamsBashforthTimeStepperBase):
    def __init__(self, large_dt, step_ratio, order, startup_stepper=None,
            eval_s2f_dt_small=False):
        TwoRateAdamsBashforthTimeStepperBase.__init__(self, large_dt, step_ratio, order,
                startup_stepper, eval_s2f_dt_small)

    def run_ab(self, ys, t, rhss):
        y_fast, y_slow = ys
        rhs_f2f, rhs_s2f, rhs_f2s, rhs_s2s = rhss
        hist_f2f, hist_s2f, hist_f2s, hist_s2s = self.rhs_histories

        coeff = self.coefficients

        y_slow_substep = y_slow

        # substep the faster component - y_fast -
        # first and extrapolate the the slow component -y_slow:
        for i in range(self.step_ratio):
            # Extrapolation of coupling RHS, l2s, on substep level.
            sub_ex_coeff = self.ab_extrapol_substep_coefficients[i]
            sub_ex_side_free_coeffs = self.ab_extrapol_side_effect_free_substep_coefficients[i]
#
            if self.eval_s2f_dt_small:
                y_fast = y_fast + (
                        self.small_dt * _linear_comb(coeff, hist_f2f)
                        + self.small_dt * _linear_comb(coeff, hist_s2f))
            else:
                y_fast = y_fast + (
                        self.small_dt * _linear_comb(coeff, hist_f2f)
                        + self.large_dt * _linear_comb(sub_ex_coeff, hist_s2f))

            if i == self.step_ratio-1:
                break

            def y_slow_substep_func():
                return y_slow + (
                        self.large_dt * _linear_comb(sub_ex_side_free_coeffs, hist_s2s)
                        + self.large_dt * _linear_comb(sub_ex_side_free_coeffs, hist_f2s))

            # calculate the RHS's on substep level:
            _rotate_insert(hist_f2f, rhs_f2f(t+(i+1)*self.small_dt,
                          y_fast, y_slow_substep_func()))

            if self.eval_s2f_dt_small:
                _rotate_insert(hist_s2f, rhs_s2f(t+(i+1)*self.small_dt,
                    y_fast, y_slow_substep_func()))

        # step the 'large' part
        y_slow = y_slow + self.large_dt * (
                _linear_comb(coeff, hist_s2s)
                + _linear_comb(coeff, hist_f2s))

        # calculate all RHS:
        _rotate_insert(hist_f2s, rhs_f2s(t+self.large_dt, y_fast, y_slow))
        _rotate_insert(hist_s2s, rhs_s2s(t+self.large_dt, y_fast, y_slow))
        _rotate_insert(hist_s2f, rhs_s2f(t+self.large_dt, y_fast, y_slow))
        _rotate_insert(hist_f2f, rhs_f2f(t+self.large_dt, y_fast, y_slow))

        from hedge.tools import make_obj_array
        return make_obj_array([y_fast, y_slow])






class TwoRateAdamsBashforthTimeStepperSlowestFirstMethodType_2(TwoRateAdamsBashforthTimeStepperBase):
    def __init__(self, large_dt, step_ratio, order, startup_stepper=None,
            eval_s2f_dt_small=False):
        TwoRateAdamsBashforthTimeStepperBase.__init__(self, large_dt, step_ratio, order,
                startup_stepper, eval_s2f_dt_small)

        # get the AB interpolation coefficients on
        # small_dt substep level between two large_dt values:
        self.ab_interp_substep_coefficients = [
                make_generic_ab_coefficients(
                    numpy.arange(0, -order, -1),
                    (i-1)/step_ratio-1,
                    i/step_ratio-1)
                for i in range(1, step_ratio+1)]

        # get the AB interpolation coefficients on
        # small_dt substep level between two large_dt values:
        self.ab_interp_side_free_substep_coefficients = [
                make_generic_ab_coefficients(
                    numpy.arange(0, -order, -1),
                    -1,
                    i/step_ratio-1)
                for i in range(1, step_ratio+1)]

        # illegal extrapolation coeffs:
        self.ab_illegal_extrap_coefficients = make_generic_ab_coefficients(
                numpy.arange(0, -order, -1),
                0,
                step_ratio)

    def run_ab(self, ys, t, rhss):
        y_fast, y_slow = ys
        rhs_f2f, rhs_s2f, rhs_f2s, rhs_s2s = rhss
        hist_f2f, hist_s2f, hist_f2s, hist_s2s = self.rhs_histories

        coeff = self.coefficients

        y_slow_start = y_slow

        # extrapolate y_slow from t=0 to t=1
        y_slow = y_slow + (
                self.large_dt * _linear_comb(coeff, hist_s2s)
                + self.large_dt * _linear_comb(coeff, hist_f2s))

        # illegal extrapolation of y_fast from t=0 to t=1
        illegal_extrap_coeff = self.ab_illegal_extrap_coefficients
        y_fast_tilde = y_fast + (
                        self.small_dt * _linear_comb(illegal_extrap_coeff, hist_f2f)
                        + self.large_dt * _linear_comb(coeff, hist_s2f)
                        )
        # update slow RHS:
        _rotate_insert(hist_s2s, rhs_s2s(t+self.large_dt, y_fast_tilde, y_slow))

        for i in range(self.step_ratio):
            # substep fast component - y_small - and interpolate
            # the values of the coupling s2f-RHS:
            if self.eval_s2f_dt_small:
                # extrapolate y_fast from t=0 to t=h on substeplevel:
                y_fast = y_fast +  (
                        self.small_dt * _linear_comb(coeff, hist_f2f)
                        + self.small_dt * _linear_comb(coeff, hist_s2f)
                        )
            else:
                sub_ex_coeff = self.ab_extrapol_substep_coefficients[i]
                #sub_int_coeff = self.ab_interp_substep_coefficients[i]
                y_fast = y_fast + (
                        self.small_dt * _linear_comb(coeff, hist_f2f)
                        + self.large_dt * _linear_comb(sub_ex_coeff, hist_s2f)
                        )

            if i == self.step_ratio-1:
                break

            # define function to ensure "lazy evaluation". This ensures, that only if
            # required y_slow gets extrapolated on substep level.
            def y_slow_substep_func():
                sub_ex_side_free_coeff = self.ab_extrapol_side_effect_free_substep_coefficients[i]
                sub_int_side_free_coeff = self.ab_interp_side_free_substep_coefficients[i]
                return y_slow_start + (
                        self.large_dt * _linear_comb(sub_int_side_free_coeff, hist_s2s)
                        + self.large_dt * _linear_comb(sub_ex_side_free_coeff, hist_f2s))

            # compute f2f-RHS of fast component:
            _rotate_insert(hist_f2f, rhs_f2f(t+(i+1)*self.small_dt,
                          y_fast, y_slow_substep_func()))

            if self.eval_s2f_dt_small:
                _rotate_insert(hist_s2f, rhs_s2f(t+(i+1)*self.small_dt,
                          y_fast, y_slow_substep_func()))

        # compute RHS's:
        _rotate_insert(hist_f2f, rhs_f2f(t+self.large_dt, y_fast, y_slow_substep_func()))
        _rotate_insert(hist_f2s, rhs_f2s(t+self.large_dt, y_fast, y_slow_substep_func()))
        _rotate_insert(hist_s2f, rhs_s2f(t+self.large_dt, y_fast, y_slow_substep_func()))

        from hedge.tools import make_obj_array
        return make_obj_array([y_fast, y_slow])





class TwoRateAdamsBashforthTimeStepperSlowestFirstMethodType_1(TwoRateAdamsBashforthTimeStepperBase):
    def __init__(self, large_dt, step_ratio, order, startup_stepper=None,
            eval_s2f_dt_small=False):
        TwoRateAdamsBashforthTimeStepperBase.__init__(self, large_dt, step_ratio, order,
                startup_stepper, eval_s2f_dt_small)

        # get the AB interpolation coefficients on
        # small_dt substep level between two large_dt values:
        self.ab_interp_substep_coefficients = [
                make_generic_ab_coefficients(
                    numpy.arange(0, -order, -1),
                    (i-1)/step_ratio-1,
                    i/step_ratio-1)
                for i in range(1, step_ratio+1)]

        # get the AB interpolation coefficients on
        # small_dt substep level between two large_dt values:
        self.ab_interp_side_free_substep_coefficients = [
                make_generic_ab_coefficients(
                    numpy.arange(0, -order, -1),
                    -1,
                    i/step_ratio-1)
                for i in range(1, step_ratio+1)]

        # illegal extrapolation coeffs:
        self.ab_illegal_extrap_coefficients = make_generic_ab_coefficients(
                numpy.arange(0, -order, -1),
                0,
                step_ratio)

    def run_ab(self, ys, t, rhss):
        y_fast, y_slow = ys
        rhs_f2f, rhs_s2f, rhs_f2s, rhs_s2s = rhss
        hist_f2f, hist_s2f, hist_f2s, hist_s2s = self.rhs_histories

        coeff = self.coefficients

        y_slow_start = y_slow

        # extrapolate y_slow from t=0 to t=1
        y_slow = y_slow + (
                self.large_dt * _linear_comb(coeff, hist_s2s)
                + self.large_dt * _linear_comb(coeff, hist_f2s))

        # illegal extrapolation of y_fast from t=0 to t=1
        illegal_extrap_coeff = self.ab_illegal_extrap_coefficients
        y_fast_tilde = y_fast + (
                        self.small_dt * _linear_comb(illegal_extrap_coeff, hist_f2f)
                        + self.large_dt * _linear_comb(coeff, hist_s2f)
                        )
        # update slow RHS:
        _rotate_insert(hist_s2s, rhs_s2s(t+self.large_dt, y_fast_tilde, y_slow))
        _rotate_insert(hist_s2f, rhs_s2f(t+self.large_dt, y_fast_tilde, y_slow))

        for i in range(self.step_ratio):
            # step y_fast from t=0 to t=h
            sub_ex_coeff = self.ab_extrapol_substep_coefficients[i]
            sub_int_coeff = self.ab_interp_substep_coefficients[i]
            y_fast = y_fast + (
                    self.small_dt * _linear_comb(coeff, hist_f2f)
                    + self.large_dt * _linear_comb(sub_int_coeff, hist_s2f)
                    )

            if i == self.step_ratio-1:
                break

            # define function to ensure "lazy evaluation". This ensures, that only if
            # required y_slow gets extrapolated on substep level.
            def y_slow_substep_func():
                sub_ex_side_free_coeff = self.ab_extrapol_side_effect_free_substep_coefficients[i]
                sub_int_side_free_coeff = self.ab_interp_side_free_substep_coefficients[i]
                return y_slow_start + (
                        self.large_dt * _linear_comb(sub_int_side_free_coeff, hist_s2s)
                        + self.large_dt * _linear_comb(sub_ex_side_free_coeff, hist_f2s))

            # compute f2f-RHS of fast component:
            _rotate_insert(hist_f2f, rhs_f2f(t+(i+1)*self.small_dt,
                          y_fast, y_slow_substep_func()))

        # compute RHS's:
        _rotate_insert(hist_f2f, rhs_f2f(t+self.large_dt, y_fast, y_slow_substep_func()))
        _rotate_insert(hist_f2s, rhs_f2s(t+self.large_dt, y_fast, y_slow_substep_func()))

        from hedge.tools import make_obj_array
        return make_obj_array([y_fast, y_slow])




class TwoRateAdamsBashforthTimeStepperSlowestFirstMethodType_4(TwoRateAdamsBashforthTimeStepperBase):
    def __init__(self, large_dt, step_ratio, order, startup_stepper=None,
            eval_s2f_dt_small=False):
        TwoRateAdamsBashforthTimeStepperBase.__init__(self, large_dt, step_ratio, order,
                startup_stepper, eval_s2f_dt_small)

        # get the AB interpolation coefficients on
        # small_dt substep level between two large_dt values:
        self.ab_interp_substep_coefficients = [
                make_generic_ab_coefficients(
                    numpy.arange(0, -order, -1),
                    (i-1)/step_ratio-1,
                    i/step_ratio-1)
                for i in range(1, step_ratio+1)]

        # get the AB interpolation coefficients on
        # small_dt substep level between two large_dt values:
        self.ab_interp_side_free_substep_coefficients = [
                make_generic_ab_coefficients(
                    numpy.arange(0, -order, -1),
                    -1,
                    i/step_ratio-1)
                for i in range(1, step_ratio+1)]

        # illegal extrapolation coeffs:
        self.ab_illegal_extrap_coefficients = make_generic_ab_coefficients(
                numpy.arange(0, -order, -1),
                0,
                step_ratio)

    def run_ab(self, ys, t, rhss):
        y_fast, y_slow = ys
        rhs_f2f, rhs_s2f, rhs_f2s, rhs_s2s = rhss
        hist_f2f, hist_s2f, hist_f2s, hist_s2s = self.rhs_histories

        coeff = self.coefficients

        y_slow_start = y_slow

        # extrapolate y_slow from t=0 to t=1
        y_slow = y_slow + (
                self.large_dt * _linear_comb(coeff, hist_s2s)
                + self.large_dt * _linear_comb(coeff, hist_f2s))

        # illegal extrapolation of y_fast from t=0 to t=1
        illegal_extrap_coeff = self.ab_illegal_extrap_coefficients
        y_fast_tilde = y_fast + (
                        self.small_dt * _linear_comb(illegal_extrap_coeff, hist_f2f)
                        + self.large_dt * _linear_comb(coeff, hist_s2f)
                        )
        # update slow RHS:
        _rotate_insert(hist_s2s, rhs_s2s(t+self.large_dt, y_fast_tilde, y_slow))
        _rotate_insert(hist_s2f, rhs_s2f(t+self.large_dt, y_fast_tilde, y_slow))
        _rotate_insert(hist_f2s, rhs_f2s(t+self.large_dt, y_fast_tilde, y_slow))

        for i in range(self.step_ratio):
            # step y_fast from t=0 to t=h
            sub_ex_coeff = self.ab_extrapol_substep_coefficients[i]
            sub_int_coeff = self.ab_interp_substep_coefficients[i]
            y_fast = y_fast + (
                    self.small_dt * _linear_comb(coeff, hist_f2f)
                    + self.large_dt * _linear_comb(sub_int_coeff, hist_s2f)
                    )

            if i == self.step_ratio-1:
                break

            # define function to ensure "lazy evaluation". This ensures, that only if
            # required y_slow gets extrapolated on substep level.
            def y_slow_substep_func():
                sub_ex_side_free_coeff = self.ab_extrapol_side_effect_free_substep_coefficients[i]
                sub_int_side_free_coeff = self.ab_interp_side_free_substep_coefficients[i]
                return y_slow_start + (
                        self.large_dt * _linear_comb(sub_int_side_free_coeff, hist_s2s)
                        + self.large_dt * _linear_comb(sub_int_side_free_coeff, hist_f2s))

            # compute f2f-RHS of fast component:
            _rotate_insert(hist_f2f, rhs_f2f(t+(i+1)*self.small_dt,
                          y_fast, y_slow_substep_func()))

        # compute RHS's:
        _rotate_insert(hist_f2f, rhs_f2f(t+self.large_dt, y_fast, y_slow_substep_func()))

        from hedge.tools import make_obj_array
        return make_obj_array([y_fast, y_slow])




class TwoRateAdamsBashforthTimeStepperSlowestFirstMethodType_3(TwoRateAdamsBashforthTimeStepperBase):
    def __init__(self, large_dt, step_ratio, order, startup_stepper=None,
            eval_s2f_dt_small=False):
        TwoRateAdamsBashforthTimeStepperBase.__init__(self, large_dt, step_ratio, order,
                startup_stepper, eval_s2f_dt_small)

        # get the AB interpolation coefficients on
        # small_dt substep level between two large_dt values:
        self.ab_interp_substep_coefficients = [
                make_generic_ab_coefficients(
                    numpy.arange(0, -order, -1),
                    (i-1)/step_ratio-1,
                    i/step_ratio-1)
                for i in range(1, step_ratio+1)]

        # get the AB interpolation coefficients on
        # small_dt substep level between two large_dt values:
        self.ab_interp_side_free_substep_coefficients = [
                make_generic_ab_coefficients(
                    numpy.arange(0, -order, -1),
                    -1,
                    i/step_ratio-1)
                for i in range(1, step_ratio+1)]

        # illegal extrapolation coeffs:
        self.ab_illegal_extrap_coefficients = make_generic_ab_coefficients(
                numpy.arange(0, -order, -1),
                0,
                step_ratio)

    def run_ab(self, ys, t, rhss):
        y_fast, y_slow = ys
        rhs_f2f, rhs_s2f, rhs_f2s, rhs_s2s = rhss
        hist_f2f, hist_s2f, hist_f2s, hist_s2s = self.rhs_histories

        coeff = self.coefficients

        y_slow_start = y_slow

        # extrapolate y_slow from t=0 to t=1
        y_slow = y_slow + (
                self.large_dt * _linear_comb(coeff, hist_s2s)
                + self.large_dt * _linear_comb(coeff, hist_f2s))

        # illegal extrapolation of y_fast from t=0 to t=1
        illegal_extrap_coeff = self.ab_illegal_extrap_coefficients
        y_fast_tilde = y_fast + (
                        self.small_dt * _linear_comb(illegal_extrap_coeff, hist_f2f)
                        + self.large_dt * _linear_comb(coeff, hist_s2f)
                        )
        # update slow RHS:
        _rotate_insert(hist_s2s, rhs_s2s(t+self.large_dt, y_fast_tilde, y_slow))
        _rotate_insert(hist_f2s, rhs_f2s(t+self.large_dt, y_fast_tilde, y_slow))

        for i in range(self.step_ratio):
            # step y_fast from t=0 to t=h
            sub_ex_coeff = self.ab_extrapol_substep_coefficients[i]
            sub_int_coeff = self.ab_interp_substep_coefficients[i]
            y_fast = y_fast + (
                    self.small_dt * _linear_comb(coeff, hist_f2f)
                    + self.large_dt * _linear_comb(sub_ex_coeff, hist_s2f)
                    )

            if i == self.step_ratio-1:
                break

            # define function to ensure "lazy evaluation". This ensures, that only if
            # required y_slow gets extrapolated on substep level.
            def y_slow_substep_func():
                sub_ex_side_free_coeff = self.ab_extrapol_side_effect_free_substep_coefficients[i]
                sub_int_side_free_coeff = self.ab_interp_side_free_substep_coefficients[i]
                return y_slow_start + (
                        self.large_dt * _linear_comb(sub_int_side_free_coeff, hist_s2s)
                        + self.large_dt * _linear_comb(sub_int_side_free_coeff, hist_f2s))

            # compute f2f-RHS of fast component:
            _rotate_insert(hist_f2f, rhs_f2f(t+(i+1)*self.small_dt,
                          y_fast, y_slow_substep_func()))

        # compute RHS's:
        _rotate_insert(hist_f2f, rhs_f2f(t+self.large_dt, y_fast, y_slow_substep_func()))
        _rotate_insert(hist_s2f, rhs_s2f(t+self.large_dt, y_fast, y_slow_substep_func()))

        from hedge.tools import make_obj_array
        return make_obj_array([y_fast, y_slow])
