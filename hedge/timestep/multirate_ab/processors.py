"""Multirate-AB ODE solver."""

from __future__ import division

__copyright__ = "Copyright (C) 2009 Andreas Stock, Andreas Kloeckner"

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
    def __init__(self, method, substep_count):
        MRABProcessor.__init__(self, method, substep_count)

        self.result = []

        self.s2s_hist_head = 0
        self.s2f_hist_head = 0
        self.f2s_hist_head = 0
        self.f2f_hist_head = 0

        self.last_assigned_at = {}

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

        self.last_assigned_at[insn.result_name] = len(self.result)

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
        elif insn.which == HIST_S2F:
            if self.method.s2f_hist_is_fast:
                step_size = 1/self.substep_count
            else:
                step_size = 1
            name = "s2f"
        elif insn.which == HIST_F2S:
            step_size = 1
            name = "f2s"
        elif insn.which == HIST_S2S:
            step_size = 1
            name = "s2s"

        hist_head_name = name+"_hist_head"
        where = getattr(self, hist_head_name)
        where += step_size
        setattr(self, hist_head_name, where)

        self.result.append("\mrabhistupdate {%s}{%s} {%f} {%s,%s}"
                % (name.replace("2", "t"), name.replace("2", "")[::-1], where,
                    insn.slow_arg.replace("y_", ""), 
                    insn.fast_arg.replace("y_", "")))

        MRABProcessor.history_update(self, insn)

    def get_result(self):
        if self.method.s2f_hist_is_fast:
            which_hist = "mrabfaststfhist"
        else:
            which_hist = "mrabslowstfhist"

        result_generators = [
                self.last_assigned_at[self.method.result_fast],
                self.last_assigned_at[self.method.result_slow],
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
