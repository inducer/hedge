from __future__ import division
from pytools import Record
from pymbolic import var

# symbols ---------------------------------------------------------------------
# components
class co_fast: pass
class co_slow: pass

# histories:
class hist_s2s: pass
class hist_s2f: pass
class hist_f2s: pass
class hist_f2f: pass




# actual method descriptions --------------------------------------------------
n = substep_count = var("substep_count")
i = substep_index = var("substep_index")




# method building blocks ------------------------------------------------------
class IntegrateInTime(Record):
    __slots__ = ["start", "end", "component", "result_name"]

    def visit(self, processor):
        processor.integrate_in_time(self)

class HistoryUpdate(Record):
    __slots__ = ["which", "slow_arg", "fast_arg"]

    def visit(self, processor):
        processor.history_update(self)

class StartSubstepLoop(Record):
    # everything from this point gets executed substep_count times
    def visit(self, processor):
        processor.start_substep_loop(self)

class EndSubstepLoop(Record):
    __slots__ = ["loop_end"]
    # everything up to this point gets executed substep_count times

    def __init__(self, loop_end=n):
        Record.__init__(self, loop_end=loop_end)

    def visit(self, processor):
        processor.end_substep_loop(self)




# actual method descriptions --------------------------------------------------
class MRABMethod(Record):
    __slots__ = ["steps", "s2f_hist_is_fast"]

methods = {
        "fastest_first_1a": MRABMethod(s2f_hist_is_fast=False,
            steps=[
            IntegrateInTime(start=0, end=(i+1)/n, component=co_slow,
                result_name="y_s"),
            IntegrateInTime(start=i/n, end=(i+1)/n, component=co_fast,
                result_name="y_f"),
            HistoryUpdate(slow_arg="y_s", fast_arg="y_f",
                which=hist_f2f),
            EndSubstepLoop(),
            HistoryUpdate(slow_arg="y_s", fast_arg="y_f",
                which=hist_s2f),
            HistoryUpdate(slow_arg="y_s", fast_arg="y_f",
                which=hist_f2s),
            HistoryUpdate(slow_arg="y_s", fast_arg="y_f",
                which=hist_s2s),
            ]),

        "fastest_first_1b": MRABMethod(s2f_hist_is_fast=True,
            steps=[
            IntegrateInTime(start=0, end=(i+1)/n, component=co_slow,
                result_name="y_s"),
            IntegrateInTime(start=i/n, end=(i+1)/n, component=co_fast,
                result_name="y_f"),
            HistoryUpdate(slow_arg="y_s", fast_arg="y_f",
                which=hist_f2f),
            HistoryUpdate(slow_arg="y_s", fast_arg="y_f",
                which=hist_s2f),
            EndSubstepLoop(),
            HistoryUpdate(slow_arg="y_s", fast_arg="y_f",
                which=hist_f2s),
            HistoryUpdate(slow_arg="y_s", fast_arg="y_f",
                which=hist_s2s),
            ]),
        "slowest_first_1": MRABMethod(s2f_hist_is_fast=False,
            steps=[
                IntegrateInTime(start=0, end=1, component=co_slow,
                result_name="\\tilde y_s"),
            IntegrateInTime(start=0, end=1, component=co_fast,
                result_name="\\tilde y_f"),
            HistoryUpdate(slow_arg="\\tilde y_s", fast_arg="\\tilde y_f",
                which=hist_s2s),
            HistoryUpdate(slow_arg="\\tilde y_s", fast_arg="\\tilde y_f",
                which=hist_s2f),
            StartSubstepLoop(),
            IntegrateInTime(start=0, end=(i+1)/n, component=co_slow,
                result_name="y_s"),
            IntegrateInTime(start=i/n, end=(i+1)/n, component=co_fast,
                result_name="y_f"),
            HistoryUpdate(slow_arg="y_s", fast_arg="y_f",
                which=hist_f2f),
            EndSubstepLoop(),
            HistoryUpdate(slow_arg="y_s", fast_arg="y_f",
                which=hist_f2s),
            ]),
        "slowest_first_2a": MRABMethod(s2f_hist_is_fast=False,
                steps=[
            IntegrateInTime(start=0, end=1, component=co_slow,
                result_name="\\tilde y_s"),
            IntegrateInTime(start=0, end=1, component=co_fast,
                result_name="\\tilde y_f"),
            HistoryUpdate(slow_arg="\\tilde y_s", fast_arg="\\tilde y_f",
                which=hist_s2s),
            StartSubstepLoop(),
            IntegrateInTime(start=0, end=(i+1)/n, component=co_slow,
                result_name="y_s"),
            IntegrateInTime(start=i/n, end=(i+1)/n, component=co_fast,
                result_name="y_f"),
            HistoryUpdate(slow_arg="y_s", fast_arg="y_f",
                which=hist_f2f),
            EndSubstepLoop(),
            HistoryUpdate(slow_arg="y_s", fast_arg="y_f",
                which=hist_f2s),
            HistoryUpdate(slow_arg="y_s", fast_arg="y_f",
                which=hist_s2f),
            ]),
        "slowest_first_2b": MRABMethod(s2f_hist_is_fast=True,
                steps=[
            IntegrateInTime(start=0, end=1, component=co_slow,
                result_name="\\tilde y_s"),
            IntegrateInTime(start=0, end=1, component=co_fast,
                result_name="\\tilde y_f"),
            HistoryUpdate(slow_arg="\\tilde y_s", fast_arg="\\tilde y_f",
                which=hist_s2s),
            StartSubstepLoop(),
            IntegrateInTime(start=0, end=(i+1)/n, component=co_slow,
                result_name="y_s"),
            IntegrateInTime(start=i/n, end=(i+1)/n, component=co_fast,
                result_name="y_f"),
            HistoryUpdate(slow_arg="y_s", fast_arg="y_f",
                which=hist_f2f),
            HistoryUpdate(slow_arg="y_s", fast_arg="y_f",
                which=hist_s2f),
            EndSubstepLoop(),
            HistoryUpdate(slow_arg="y_s", fast_arg="y_f",
                which=hist_f2s),
            ]),
        "slowest_first_3a": MRABMethod(s2f_hist_is_fast=False,
                steps=[
            IntegrateInTime(start=0, end=1, component=co_slow,
                result_name="\\tilde y_s"),
            IntegrateInTime(start=0, end=1, component=co_fast,
                result_name="\\tilde y_f"),
            HistoryUpdate(slow_arg="\\tilde y_s", fast_arg="\\tilde y_f",
                which=hist_s2s),
            HistoryUpdate(slow_arg="\\tilde y_s", fast_arg="\\tilde y_f",
                which=hist_f2s),
            StartSubstepLoop(),
            IntegrateInTime(start=0, end=(i+1)/n, component=co_slow,
                result_name="y_s"),
            IntegrateInTime(start=i/n, end=(i+1)/n, component=co_fast,
                result_name="y_f"),
            HistoryUpdate(slow_arg="y_s", fast_arg="y_f",
                which=hist_f2f),
            EndSubstepLoop(),
            HistoryUpdate(slow_arg="y_s", fast_arg="y_f",
                which=hist_s2f),
            ]),
        "slowest_first_3b": MRABMethod(s2f_hist_is_fast=True,
                steps=[
            IntegrateInTime(start=0, end=1, component=co_slow,
                result_name="\\tilde y_s"),
            IntegrateInTime(start=0, end=1, component=co_fast,
                result_name="\\tilde y_f"),
            HistoryUpdate(slow_arg="\\tilde y_s", fast_arg="\\tilde y_f",
                which=hist_s2s),
            HistoryUpdate(slow_arg="\\tilde y_s", fast_arg="\\tilde y_f",
                which=hist_f2s),
            StartSubstepLoop(),
            IntegrateInTime(start=0, end=(i+1)/n, component=co_slow,
                result_name="y_s"),
            IntegrateInTime(start=i/n, end=(i+1)/n, component=co_fast,
                result_name="y_f"),
            HistoryUpdate(slow_arg="y_s", fast_arg="y_f",
                which=hist_f2f),
            HistoryUpdate(slow_arg="y_s", fast_arg="y_f",
                which=hist_s2f),
            EndSubstepLoop(),
            ]),
        "slowest_first_4": MRABMethod(s2f_hist_is_fast=False,
                steps=[
            IntegrateInTime(start=0, end=1, component=co_slow,
                result_name="\\tilde y_s"),
            IntegrateInTime(start=0, end=1, component=co_fast,
                result_name="\\tilde y_f"),
            HistoryUpdate(slow_arg="\\tilde y_s", fast_arg="\\tilde y_f",
                which=hist_s2s),
            HistoryUpdate(slow_arg="\\tilde y_s", fast_arg="\\tilde y_f",
                which=hist_f2s),
            HistoryUpdate(slow_arg="\\tilde y_s", fast_arg="\\tilde y_f",
                which=hist_s2f),
            StartSubstepLoop(),
            IntegrateInTime(start=0, end=(i+1)/n, component=co_slow,
                result_name="y_s"),
            IntegrateInTime(start=i/n, end=(i+1)/n, component=co_fast,
                result_name="y_f"),
            HistoryUpdate(slow_arg="y_s", fast_arg="y_f",
                which=hist_f2f),
            EndSubstepLoop(),
            ])
        }




def _remove_last_yslow_evaluation_from_slowest_first(method):
    m_steps = method.steps
    try:
        loop_start_index = m_steps.index(StartSubstepLoop())
    except ValueError:
        loop_start_index = 0

    loop_end_index = m_steps.index(EndSubstepLoop())

    before_loop = m_steps[:loop_start_index]
    loop_body = m_steps[loop_start_index+1:loop_end_index]
    after_loop = m_steps[loop_end_index+1:]

    new_steps = (before_loop
            + [StartSubstepLoop()]
            + loop_body
            + [EndSubstepLoop(m_steps[loop_end_index].loop_end-1)]
            )

    for lb_entry in loop_body+after_loop:
        if isinstance(lb_entry, IntegrateInTime):
            if lb_entry.result_name != "y_s":
                new_steps.append(lb_entry)
        elif isinstance(lb_entry, HistoryUpdate):
            assert lb_entry.slow_arg == "y_s"
            new_steps.append(lb_entry.copy(
                slow_arg="\\tilde y_s"))
        else:
            raise NotImplementedError

    return method.copy(steps=new_steps)



def _add_slowest_first_variants(methods):
    result = {}
    for name, method in methods.iteritems():
        result[name] = method
        if name.startswith("slowest_first"):
            result[name+"_no_yslow_reeval"] = \
                    _remove_last_yslow_evaluation_from_slowest_first(method)

    return result




methods = _add_slowest_first_variants(methods)

