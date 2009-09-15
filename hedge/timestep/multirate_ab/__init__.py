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
                HIST_F2F, HIST_S2F, HIST_F2S, HIST_S2S

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
        :param rhss: Matrix of right-hand sides, stored in row-major order, 
            i.e. *[f2f, s2f, f2s, s2s]*.
        """
        from hedge.tools import make_obj_array

        def finish_startup():
            # we're done starting up, pack data into split histories
            for i, hn in enumerate(HIST_NAMES):
                hist = self.startup_history
                if not self.hist_is_fast[hn]:
                    hist = hist[::self.substep_count]

                hist = hist[:self.orders[hn]]

                self.histories[hn] = [
                        hist_entry[i] for hist_entry in hist]

                assert len(self.histories[hn]) == self.orders[hn]

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
    def get_coefficients(self, 
            for_fast_history, hist_head_time_level, 
            start_level, end_level, order):

        history_times = numpy.arange(0, -order, -1,
                dtype=numpy.float64)

        if for_fast_history:
            history_times /= self.substep_count

        history_times += hist_head_time_level/self.substep_count

        t_start = start_level/self.substep_count
        t_end = end_level/self.substep_count

        return make_generic_ab_coefficients(
                history_times, t_start, t_end)




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

        self.hist_head_time_level = dict((hn, 0) for hn in HIST_NAMES)

    def integrate_in_time(self, insn):
        from hedge.timestep.multirate_ab.methods import CO_FAST, CO_SLOW
        from hedge.timestep.multirate_ab.methods import \
                HIST_F2F, HIST_S2F, HIST_F2S, HIST_S2S

        if insn.component == CO_FAST:
            self_hn, cross_hn = HIST_F2F, HIST_S2F
        else:
            self_hn, cross_hn = HIST_S2S, HIST_F2S

        start_time_level = self.eval_expr(insn.start)
        end_time_level = self.eval_expr(insn.end)

        self_coefficients = self.stepper.get_coefficients(
                self.stepper.hist_is_fast[self_hn],
                self.hist_head_time_level[self_hn],
                start_time_level, end_time_level,
                self.stepper.orders[self_hn])
        cross_coefficients = self.stepper.get_coefficients(
                self.stepper.hist_is_fast[cross_hn],
                self.hist_head_time_level[cross_hn],
                start_time_level, end_time_level,
                self.stepper.orders[cross_hn])

        if start_time_level == 0 or (insn.result_name not in self.context):
            my_y = self.last_y[insn.component]
            assert start_time_level == 0
        else:
            my_y = self.context[insn.result_name]()
            assert start_time_level == \
                    self.var_time_level[insn.result_name]

        hists = self.stepper.histories
        self_history = hists[self_hn][:]
        cross_history = hists[cross_hn][:]
        if False:
            my_integrated_y = memoize(
                    lambda: my_y + self.stepper.large_dt * (
                    _linear_comb(self_coefficients, self_history)
                    + _linear_comb(cross_coefficients, cross_history)))
        else:
            my_new_y = my_y + self.stepper.large_dt * (
                    _linear_comb(self_coefficients, self_history)
                    + _linear_comb(cross_coefficients, cross_history))
            my_integrated_y = lambda: my_new_y

        self.context[insn.result_name] = my_integrated_y
        self.var_time_level[insn.result_name] = end_time_level

        MRABProcessor.integrate_in_time(self, insn)

    def history_update(self, insn):
        time_slow = self.var_time_level[insn.slow_arg]
        time_fast = self.var_time_level[insn.fast_arg]

        assert time_slow == time_fast

        t = (self.t_start 
                + self.stepper.large_dt*time_slow/self.stepper.substep_count)

        rhs = self.rhss[HIST_NAMES.index(insn.which)]

        hist = self.stepper.histories[insn.which]
        hist.pop()
        hist.insert(0,
                rhs(t,
                    self.context[insn.fast_arg],
                    self.context[insn.slow_arg]))

        if self.stepper.hist_is_fast[insn.which]:
            self.hist_head_time_level[insn.which] += 1
        else:
            self.hist_head_time_level[insn.which] += self.stepper.substep_count

        MRABProcessor.history_update(self, insn)

    def get_result(self):
        meth = self.stepper.method

        assert self.var_time_level[meth.result_slow] == self.stepper.substep_count
        assert self.var_time_level[meth.result_fast] == self.stepper.substep_count

        for hn in HIST_NAMES:
            assert self.hist_head_time_level[hn] == self.stepper.substep_count

        return (self.context[meth.result_fast](),
                self.context[meth.result_slow]())
