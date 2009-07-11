from __future__ import division




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

    def integrate_in_time(self, insn):
        from mrab_description import co_fast

        if insn.component == co_fast:
            src_self = "ftf"
            src_self_speed = r"\smallstep"
            src_self_where = self.f2f_hist_head

            src_other = "stf"
            if self.method.s2f_hist_is_fast:
                src_other_speed = r"\smallstep"
            else:
                src_other_speed = r"\bigstep"
            src_other_where = self.s2f_hist_head
        else:
            src_self = "sts"
            src_self_speed = r"\bigstep"
            src_self_where = self.s2s_hist_head

            src_other = "fts"
            src_other_speed = r"\bigstep"
            src_other_where = self.f2s_hist_head

        self.result.append(
                "\integrate {%s}{%f}{%f} {%s}{%f}{%s} {%s}{%f}{%s}"
                % (insn.result_name, 
                    self.eval_expr(insn.start), 
                    self.eval_expr(insn.end),
                    src_self, src_self_where, src_self_speed,
                    src_other, src_other_where, src_other_speed))

        MRABProcessor.integrate_in_time(self, insn)

    def history_update(self, insn):
        from mrab_description import \
                hist_f2f, hist_s2f, \
                hist_f2s, hist_s2s

        if insn.which == hist_f2f:
            step_size = 1/self.substep_count
            name = "f2f"
        elif insn.which == hist_s2f:
            if self.method.s2f_hist_is_fast:
                step_size = 1/self.substep_count
            else:
                step_size = 1
            name = "s2f"
        elif insn.which == hist_f2s:
            step_size = 1
            name = "f2s"
        elif insn.which == hist_s2s:
            step_size = 1
            name = "s2s"

        hist_head_name = name+"_hist_head"
        where = getattr(self, hist_head_name)
        where += step_size
        setattr(self, hist_head_name, where)

        self.result.append("\histupdate {%s}{%s} {%f} {%s,%s}"
                % (name.replace("2", "t"), name, where,
                    insn.slow_arg, insn.fast_arg))

        MRABProcessor.history_update(self, insn)

    def get_result(self):
        if self.method.s2f_hist_is_fast:
            which_hist = "faststfhist"
        else:
            which_hist = "slowstfhist"

        return "\n".join(
                [
                    "{"
                    r"\setcounter{step}{0}",
                    r"\def\smallstepcount{%d}" % self.substep_count,
                    r"\def\smallstep{%f}" % (1/self.substep_count),
                    r"\begin{tikzpicture}[mrabpic]"
                    r"\%s{0}" % which_hist,
                    ]+self.result+[
                    r"\%s{1}" % which_hist,
                    r"\colorlegend",
                    r"\makeaxis",
                    r"\end{tikzpicture}",
                    "}"])




if __name__ == "__main__":
    from mrab_description import methods
    for name, method in methods.iteritems():
        mrab2tex = MRABToTeXProcessor(method, 2)
        mrab2tex.run()
        open("out/%s.tex" % name, "w").write(
                "\\verb|%s|\n\n" % name+
                mrab2tex.get_result())

