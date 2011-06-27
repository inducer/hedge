"""Multirate-AB ODE solver."""

from __future__ import division

__copyright__ = "Copyright (C) 2009 Andreas Stock, Andreas Kloeckner"

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




from pytools import Record
from pymbolic import var




# symbols ---------------------------------------------------------------------
# components
class CO_FAST:
    pass
class CO_SLOW:
    pass

# histories:
class HIST_F2F:
    pass
class HIST_F2S:
    pass
class HIST_S2F:
    pass
class HIST_S2S:
    pass

HIST_NAMES = [HIST_F2F, HIST_S2F, HIST_F2S, HIST_S2S]




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
    """
    What does the letters at the end of the scheme name mean?
       -  w: weak coupling => s2f_hist_is_fast = False
       -  s: strong coupling => s2f_hist_is_fast = True
    """

    __slots__ = ["steps", "s2f_hist_is_fast", "result_slow", "result_fast"]

methods = {
        # If you are wondering--Stock's nomenclature appears to *not* be
        # a direct transcription of the old, long scheme names.

        "Fqsr": # old: -none-
        MRABMethod(s2f_hist_is_fast=True,
            steps=[
                IntegrateInTime(start=0, end=n, component=CO_SLOW,
                    result_name="\\tilde y_s"),
                IntegrateInTime(start=0, end=n, component=CO_FAST,
                    result_name="\\tilde y_f"),
                HistoryUpdate(slow_arg="\\tilde y_s", fast_arg="\\tilde y_f",
                    which=HIST_F2S),
                StartSubstepLoop(),
                IntegrateInTime(start=0, end=i+1, component=CO_SLOW,
                    result_name="y_s"),
                IntegrateInTime(start=i, end=i+1, component=CO_FAST,
                    result_name="y_f"),
                HistoryUpdate(slow_arg="y_s", fast_arg="y_f",
                    which=HIST_F2F),
                HistoryUpdate(slow_arg="y_s", fast_arg="y_f",
                    which=HIST_S2F),
                EndSubstepLoop(),
                HistoryUpdate(slow_arg="y_s", fast_arg="y_f",
                    which=HIST_S2S),
                ],
            result_slow="y_s",
            result_fast="y_f"),
        "Fq": # old: fastest_first_s
        MRABMethod(s2f_hist_is_fast=True,
            steps=[
                IntegrateInTime(start=0, end=i+1, component=CO_SLOW,
                    result_name="y_s"),
                IntegrateInTime(start=i, end=i+1, component=CO_FAST,
                    result_name="y_f"),
                HistoryUpdate(slow_arg="y_s", fast_arg="y_f",
                    which=HIST_F2F),
                HistoryUpdate(slow_arg="y_s", fast_arg="y_f",
                    which=HIST_S2F),
                EndSubstepLoop(),
                HistoryUpdate(slow_arg="y_s", fast_arg="y_f",
                    which=HIST_F2S),
                HistoryUpdate(slow_arg="y_s", fast_arg="y_f",
                    which=HIST_S2S),
                ],
            result_slow="y_s",
            result_fast="y_f"),
        "Fsfr": # old: -none-
        MRABMethod(s2f_hist_is_fast=False,
            steps=[
                IntegrateInTime(start=0, end=n, component=CO_SLOW,
                    result_name="\\tilde y_s"),
                IntegrateInTime(start=0, end=n, component=CO_FAST,
                    result_name="\\tilde y_f"),
                HistoryUpdate(slow_arg="\\tilde y_s", fast_arg="\\tilde y_f",
                    which=HIST_F2S),
                HistoryUpdate(slow_arg="\\tilde y_s", fast_arg="\\tilde y_f",
                    which=HIST_S2F),
                StartSubstepLoop(),
                IntegrateInTime(start=0, end=i+1, component=CO_SLOW,
                    result_name="y_s"),
                IntegrateInTime(start=i, end=i+1, component=CO_FAST,
                    result_name="y_f"),
                HistoryUpdate(slow_arg="y_s", fast_arg="y_f",
                    which=HIST_F2F),
                EndSubstepLoop(),
                HistoryUpdate(slow_arg="y_s", fast_arg="y_f",
                    which=HIST_S2S),
                ],
            result_slow="y_s",
            result_fast="y_f"),
        "Fsr": # old: -none-
        MRABMethod(s2f_hist_is_fast=False,
            steps=[
                IntegrateInTime(start=0, end=n, component=CO_SLOW,
                    result_name="\\tilde y_s"),
                IntegrateInTime(start=0, end=n, component=CO_FAST,
                    result_name="\\tilde y_f"),
                HistoryUpdate(slow_arg="\\tilde y_s", fast_arg="\\tilde y_f",
                    which=HIST_F2S),
                StartSubstepLoop(),
                IntegrateInTime(start=0, end=i+1, component=CO_SLOW,
                    result_name="y_s"),
                IntegrateInTime(start=i, end=i+1, component=CO_FAST,
                    result_name="y_f"),
                HistoryUpdate(slow_arg="y_s", fast_arg="y_f",
                    which=HIST_F2F),
                EndSubstepLoop(),
                HistoryUpdate(slow_arg="y_s", fast_arg="y_f",
                    which=HIST_S2F),
                HistoryUpdate(slow_arg="y_s", fast_arg="y_f",
                    which=HIST_S2S),
                ],
            result_slow="y_s",
            result_fast="y_f"),
        "Ffr": # old: -none-
        MRABMethod(s2f_hist_is_fast=False,
            steps=[
                IntegrateInTime(start=0, end=n, component=CO_SLOW,
                    result_name="\\tilde y_s"),
                IntegrateInTime(start=0, end=n, component=CO_FAST,
                    result_name="\\tilde y_f"),
                HistoryUpdate(slow_arg="\\tilde y_s", fast_arg="\\tilde y_f",
                    which=HIST_S2F),
                StartSubstepLoop(),
                IntegrateInTime(start=0, end=i+1, component=CO_SLOW,
                    result_name="y_s"),
                IntegrateInTime(start=i, end=i+1, component=CO_FAST,
                    result_name="y_f"),
                HistoryUpdate(slow_arg="y_s", fast_arg="y_f",
                    which=HIST_F2F),
                EndSubstepLoop(),
                HistoryUpdate(slow_arg="y_s", fast_arg="y_f",
                    which=HIST_F2S),
                HistoryUpdate(slow_arg="y_s", fast_arg="y_f",
                    which=HIST_S2S),
                ],
            result_slow="y_s",
            result_fast="y_f"),
        "F": # old: fastest_first_w
        MRABMethod(s2f_hist_is_fast=False,
            steps=[
                IntegrateInTime(start=0, end=i+1, component=CO_SLOW,
                    result_name="y_s"),
                IntegrateInTime(start=i, end=i+1, component=CO_FAST,
                    result_name="y_f"),
                HistoryUpdate(slow_arg="y_s", fast_arg="y_f",
                    which=HIST_F2F),
                EndSubstepLoop(),
                HistoryUpdate(slow_arg="y_s", fast_arg="y_f",
                    which=HIST_S2F),
                HistoryUpdate(slow_arg="y_s", fast_arg="y_f",
                    which=HIST_F2S),
                HistoryUpdate(slow_arg="y_s", fast_arg="y_f",
                    which=HIST_S2S),
                ],
            result_slow="y_s",
            result_fast="y_f"),
        # ---------------------------------------------------------------------
        "Srf": # old: slowest_first_1
        MRABMethod(s2f_hist_is_fast=False,
            steps=[
                IntegrateInTime(start=0, end=n, component=CO_SLOW,
                    result_name="\\tilde y_s"),
                IntegrateInTime(start=0, end=n, component=CO_FAST,
                    result_name="\\tilde y_f"),
                HistoryUpdate(slow_arg="\\tilde y_s", fast_arg="\\tilde y_f",
                    which=HIST_S2S),
                HistoryUpdate(slow_arg="\\tilde y_s", fast_arg="\\tilde y_f",
                    which=HIST_S2F),
                StartSubstepLoop(),
                IntegrateInTime(start=0, end=i+1, component=CO_SLOW,
                    result_name="y_s"),
                IntegrateInTime(start=i, end=i+1, component=CO_FAST,
                    result_name="y_f"),
                HistoryUpdate(slow_arg="y_s", fast_arg="y_f",
                    which=HIST_F2F),
                EndSubstepLoop(),
                HistoryUpdate(slow_arg="y_s", fast_arg="y_f",
                    which=HIST_F2S),
                ],
            result_slow="y_s",
            result_fast="y_f"),
        "Sr": # old: slowest_first_2w
            MRABMethod(s2f_hist_is_fast=False,
                steps=[
                    IntegrateInTime(start=0, end=n, component=CO_SLOW,
                        result_name="\\tilde y_s"),
                    IntegrateInTime(start=0, end=n, component=CO_FAST,
                        result_name="\\tilde y_f"),
                    HistoryUpdate(slow_arg="\\tilde y_s", fast_arg="\\tilde y_f",
                        which=HIST_S2S),
                    StartSubstepLoop(),
                    IntegrateInTime(start=0, end=i+1, component=CO_SLOW,
                        result_name="y_s"),
                    IntegrateInTime(start=i, end=i+1, component=CO_FAST,
                        result_name="y_f"),
                    HistoryUpdate(slow_arg="y_s", fast_arg="y_f",
                        which=HIST_F2F),
                    EndSubstepLoop(),
                    HistoryUpdate(slow_arg="y_s", fast_arg="y_f",
                        which=HIST_F2S),
                    HistoryUpdate(slow_arg="y_s", fast_arg="y_f",
                        which=HIST_S2F),
                    ],
                result_slow="y_s",
                result_fast="y_f"),
        "Sqr": # old: slowest_first_2s
            MRABMethod(s2f_hist_is_fast=True,
                steps=[
                    IntegrateInTime(start=0, end=n, component=CO_SLOW,
                        result_name="\\tilde y_s"),
                    IntegrateInTime(start=0, end=n, component=CO_FAST,
                        result_name="\\tilde y_f"),
                    HistoryUpdate(slow_arg="\\tilde y_s", fast_arg="\\tilde y_f",
                        which=HIST_S2S),
                    StartSubstepLoop(),
                    IntegrateInTime(start=0, end=i+1, component=CO_SLOW,
                        result_name="y_s"),
                    IntegrateInTime(start=i, end=i+1, component=CO_FAST,
                        result_name="y_f"),
                    HistoryUpdate(slow_arg="y_s", fast_arg="y_f",
                        which=HIST_F2F),
                    HistoryUpdate(slow_arg="y_s", fast_arg="y_f",
                        which=HIST_S2F),
                    EndSubstepLoop(),
                    HistoryUpdate(slow_arg="y_s", fast_arg="y_f",
                        which=HIST_F2S),
            ],
            result_slow="y_s",
            result_fast="y_f"),
        "Srs": # old: slowest_first_3w
            MRABMethod(s2f_hist_is_fast=False,
                steps=[
                    IntegrateInTime(start=0, end=n, component=CO_SLOW,
                        result_name="\\tilde y_s"),
                    IntegrateInTime(start=0, end=n, component=CO_FAST,
                        result_name="\\tilde y_f"),
                    HistoryUpdate(slow_arg="\\tilde y_s", fast_arg="\\tilde y_f",
                        which=HIST_S2S),
                    HistoryUpdate(slow_arg="\\tilde y_s", fast_arg="\\tilde y_f",
                        which=HIST_F2S),
                    StartSubstepLoop(),
                    IntegrateInTime(start=0, end=i+1, component=CO_SLOW,
                        result_name="y_s"),
                    IntegrateInTime(start=i, end=i+1, component=CO_FAST,
                        result_name="y_f"),
                    HistoryUpdate(slow_arg="y_s", fast_arg="y_f",
                        which=HIST_F2F),
                    EndSubstepLoop(),
                    HistoryUpdate(slow_arg="y_s", fast_arg="y_f",
                        which=HIST_S2F),
                    ],
                result_slow="y_s",
                result_fast="y_f"),
        "Sqrs": # old: slowest_first_3s
            MRABMethod(s2f_hist_is_fast=True,
                steps=[
                    IntegrateInTime(start=0, end=n, component=CO_SLOW,
                        result_name="\\tilde y_s"),
                    IntegrateInTime(start=0, end=n, component=CO_FAST,
                        result_name="\\tilde y_f"),
                    HistoryUpdate(slow_arg="\\tilde y_s", fast_arg="\\tilde y_f",
                        which=HIST_S2S),
                    HistoryUpdate(slow_arg="\\tilde y_s", fast_arg="\\tilde y_f",
                        which=HIST_F2S),
                    StartSubstepLoop(),
                    IntegrateInTime(start=0, end=i+1, component=CO_SLOW,
                        result_name="y_s"),
                    IntegrateInTime(start=i, end=i+1, component=CO_FAST,
                        result_name="y_f"),
                    HistoryUpdate(slow_arg="y_s", fast_arg="y_f",
                        which=HIST_F2F),
                    HistoryUpdate(slow_arg="y_s", fast_arg="y_f",
                        which=HIST_S2F),
                    EndSubstepLoop(),
                    ],
                result_slow="y_s",
                result_fast="y_f"),
        "Srsf": # old: slowest_first_4
            MRABMethod(s2f_hist_is_fast=False,
                steps=[
                    IntegrateInTime(start=0, end=n, component=CO_SLOW,
                        result_name="\\tilde y_s"),
                    IntegrateInTime(start=0, end=n, component=CO_FAST,
                        result_name="\\tilde y_f"),
                    HistoryUpdate(slow_arg="\\tilde y_s", fast_arg="\\tilde y_f",
                        which=HIST_S2S),
                    HistoryUpdate(slow_arg="\\tilde y_s", fast_arg="\\tilde y_f",
                        which=HIST_F2S),
                    HistoryUpdate(slow_arg="\\tilde y_s", fast_arg="\\tilde y_f",
                        which=HIST_S2F),
                    StartSubstepLoop(),
                    IntegrateInTime(start=0, end=i+1, component=CO_SLOW,
                        result_name="y_s"),
                    IntegrateInTime(start=i, end=i+1, component=CO_FAST,
                        result_name="y_f"),
                    HistoryUpdate(slow_arg="y_s", fast_arg="y_f",
                        which=HIST_F2F),
                    EndSubstepLoop(),
                    ],
                result_slow="y_s",
                result_fast="y_f")
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

    return method.copy(steps=new_steps, result_slow="\\tilde y_s")



def _add_slowest_first_variants(methods):
    result = {}
    for name, method in methods.iteritems():
        result[name] = method
        if "r" in name:
            result[name.replace("r", "")] = \
                    _remove_last_yslow_evaluation_from_slowest_first(method)

    return result




methods = _add_slowest_first_variants(methods)
