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




class MRABProcessor:
    def __init__(self, method, substep_count):
        self.method = method
        self.substep_count = substep_count
        self.substep_loop_start = 0

    def integrate_in_time(self, insn):
        self.insn_counter += 1

    def history_update(self, insn):
        self.insn_counter += 1

    def start_substep_loop(self, insn):
        self.insn_counter += 1
        self.substep_loop_start = self.insn_counter

    def eval_expr(self, expr):
        from pymbolic import evaluate_kw
        return evaluate_kw(expr,
                substep_index=self.substep_index,
                substep_count=self.substep_count)

    def end_substep_loop(self, insn):
        self.substep_index += 1
        if self.substep_index >= self.eval_expr(insn.loop_end):
            self.insn_counter += 1
        else:
            self.insn_counter = self.substep_loop_start

    def run(self):
        self.insn_counter = 0
        self.substep_index = 0

        while not self.insn_counter >= len(self.method.steps):
            self.method.steps[self.insn_counter].visit(self)




class MRABToTeXProcessor(MRABProcessor):
    def __init__(self, method, substep_count, no_mixing):
        MRABProcessor.__init__(self, method, substep_count)

        self.no_mixing = no_mixing

        self.result = []

        self.s2s_hist_head = 0
        self.s2f_hist_head = 0
        self.f2s_hist_head = 0
        self.f2f_hist_head = 0

        # maps var name to list of last assignments
        self.last_assigned_at = {}
        # maps var name to use count
        self.use_count = {}

    def remove_if_unused(self, var_name):
        if var_name in self.use_count and self.use_count[var_name] == 0:
            assert self.no_mixing

            del_idx = self.last_assigned_at[var_name].pop(-1)
            del self.result[del_idx]
            # we assume the prior (surviving) assignment got at least one use
            self.use_count[var_name] = 1

            for other_var_name in self.last_assigned_at:
                la_list = self.last_assigned_at[other_var_name]
                for i in range(len(la_list)):
                    if la_list[i] > del_idx:
                        la_list[i] -= 1

    def integrate_in_time(self, insn):
        from hedge.timestep.multirate_ab.methods import CO_FAST

        if insn.component == CO_FAST:
            self_name = "fast"
            src_self_speed = r"\mrabsmallstep"
            src_self_where = self.f2f_hist_head

            if self.method.s2f_hist_is_fast:
                src_other_speed = r"\mrabsmallstep"
            else:
                src_other_speed = r"\mrabbigstep"
            src_other_where = self.s2f_hist_head
        else:
            self_name = "slow"
            src_self_speed = r"\mrabbigstep"
            src_self_where = self.s2s_hist_head

            src_other_speed = r"\mrabbigstep"
            src_other_where = self.f2s_hist_head

        self.remove_if_unused(insn.result_name)

        self.last_assigned_at.setdefault(insn.result_name, []) \
                .append(len(self.result))
        self.use_count[insn.result_name] = 0

        self.result.append(
                "\mrabintegrate {%s}{%f}{%f}{%s} {%f}{%s} {%f}{%s}"
                % (insn.result_name.replace("y_", ""), 
                    self.eval_expr(insn.start)/self.substep_count, 
                    self.eval_expr(insn.end)/self.substep_count,
                    self_name,
                    src_self_where, src_self_speed,
                    src_other_where, src_other_speed))

        MRABProcessor.integrate_in_time(self, insn)

    def history_update(self, insn):
        from hedge.timestep.multirate_ab.methods import \
                HIST_F2F, HIST_S2F, \
                HIST_F2S, HIST_S2S

        if insn.which == HIST_F2F:
            step_size = 1/self.substep_count
            name = "f2f"
            args = [insn.fast_arg]
        elif insn.which == HIST_S2F:
            if self.method.s2f_hist_is_fast:
                step_size = 1/self.substep_count
            else:
                step_size = 1
            name = "s2f"
            args = [insn.slow_arg]
        elif insn.which == HIST_F2S:
            step_size = 1
            name = "f2s"
            args = [insn.fast_arg]
        elif insn.which == HIST_S2S:
            step_size = 1
            name = "s2s"
            args = [insn.slow_arg]

        if not self.no_mixing:
            args = [insn.fast_arg, insn.slow_arg]

        for var in args:
            self.use_count[var] += 1

        args = [arg.replace("y_", "") for arg in args]

        hist_head_name = name+"_hist_head"
        where = getattr(self, hist_head_name)
        where += step_size
        setattr(self, hist_head_name, where)

        self.result.append("\mrabhistupdate {%s}{%s} {%f} {%s}"
                % (name.replace("2", "t"), name.replace("2", "")[::-1], where,
                    ",".join(args)))

        MRABProcessor.history_update(self, insn)

    def get_result(self):
        self.use_count[self.method.result_fast] += 1
        self.use_count[self.method.result_slow] += 1

        for var_name in self.last_assigned_at.keys():
            self.remove_if_unused(var_name)

        if self.method.s2f_hist_is_fast:
            which_hist = "mrabfaststfhist"
        else:
            which_hist = "mrabslowstfhist"

        result_generators = [
                self.last_assigned_at[self.method.result_fast][-1],
                self.last_assigned_at[self.method.result_slow][-1],
                ]

        texed_instructions = []
        for i, insn_tex in enumerate(self.result):
            if insn_tex.startswith("\mrabintegrate"):
                if i in result_generators:
                    insn_tex += " {isresult}"
                else:
                    insn_tex += " {}"

            texed_instructions.append(insn_tex)

        return "\n".join(
                [
                    "{"
                    r"\setcounter{mrabstep}{0}",
                    r"\def\mrabsmallstepcount{%d}" % self.substep_count,
                    r"\def\mrabsmallstep{%f}" % (1/self.substep_count),
                    r"\begin{tikzpicture}[mrabpic]"
                    r"\%s{0}" % which_hist,
                    ]+texed_instructions+[
                    r"\%s{1}" % which_hist,
                    r"\mrabcolorlegend",
                    r"\mrabmakeaxis",
                    r"\end{tikzpicture}",
                    "}"])
