from __future__ import division

__copyright__ = "Copyright (C) 2008 Andreas Kloeckner"

__license__ = """
Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.
"""




from pytools import Record
from hedge.optemplate import OpTemplateIdentityMapper




class Instruction(Record):
    __slots__ = []

class Discard(Instruction): 
    __slots__ = ["name"]

class Assign(Instruction): 
    __slots__ = ["name", "expr"]

class FluxBatchAssign(Instruction):
    __slots__ = ["names", "fluxes"]

class DiffBatchAssign(Instruction):
    __slots__ = ["names", "op_class", "operators", "field"]





class OperatorCompilerBase(OpTemplateIdentityMapper):
    class FluxRecord(Record):
        __slots__ = ["flux_expr", "dependencies"]

    def __init__(self, prefix="_expr"):
        OpTemplateIdentityMapper.__init__(self)
        self.prefix = prefix
        self.code = []
        self.assigned_var_count = 0
        self.expr_to_var = {}

    def get_contained_fluxes(self, expr):
        """Recursively enumerate all flux expressions in the expression tree
        `expr`. The returned list consists of `ExecutionPlanner.FluxRecord`
        instances with fields `flux_expr` and `dependencies`.
        """

        # overridden by subclasses
        raise NotImplementedError

    def __call__(self, expr):
        # Fluxes can be evaluated faster in batches. Here, we find flux batches
        # that we can evaluate together.

        # For each FluxRecord, find the other fluxes its flux depends on.
        flux_queue = self.get_contained_fluxes(expr)
        for fr in flux_queue:
            fr.dependencies = set()
            for d in fr.dependencies:
                fr.dependencies |= set(sf.flux_expr 
                        for sf in self.get_contained_fluxes(d))

        # Then figure out batches of fluxes to evaluate
        self.flux_batches = []
        admissible_deps = set()
        while flux_queue:
            present_batch = set()
            i = 0
            while i < len(flux_queue):
                fr = flux_queue[i]
                if fr.dependencies <= admissible_deps:
                    present_batch.add(fr.flux_expr)
                    flux_queue.pop(0)
                else:
                    i += 1

            if present_batch:
                self.flux_batches.append(present_batch)

                admissible_deps |= present_batch
            else:
                raise RuntimeError, "cannot resolve flux evaluation order"

        # Once flux batching is figured out, we also need to know which
        # derivatives are going to be needed, because once the 
        # rst-derivatives are available, it's best to calculate the 
        # xyz ones and throw the rst ones out. It's naturally good if
        # we can avoid computing (or storing) some of the xyz ones.
        # So figure out which XYZ derivatives of what are needed.

        from hedge.optemplate import DiffOpCollector
        self.diff_ops = DiffOpCollector()(expr)

        # Then walk the expression to build up the code
        result = OpTemplateIdentityMapper.__call__(self, expr)

        # Then, put the toplevel expressions into variables as well.
        from hedge.tools import is_obj_array, make_obj_array
        if is_obj_array(result):
            result = make_obj_array([self.make_assign(i) for i in result])
        else:
            result = self.make_assign(result)

        return self.code, result

    def get_var_name(self):
        new_name = self.prefix+str(self.assigned_var_count)
        self.assigned_var_count += 1
        return new_name

    def make_assign(self, expr):
        from pymbolic.primitives import Variable
        if isinstance(expr, Variable):
            return expr
            
        new_name = self.get_var_name()
        self.code.append(Assign(
            name=new_name, expr=expr))

        return Variable(new_name)

    def map_common_subexpression(self, expr):
        try:
            return self.expr_to_var[expr.child]
        except KeyError:
            cse_var = self.make_assign(self.rec(expr.child))
            self.expr_to_var[expr.child] = cse_var
            return cse_var

    def map_operator_binding(self, expr):
        from hedge.optemplate import DiffOperatorBase
        if not isinstance(expr.op, DiffOperatorBase):
            return OpTemplateIdentityMapper.map_operator_binding(self, expr)

        try:
            return self.expr_to_var[expr]
        except KeyError:
            all_diffs = [diff
                    for diff in self.diff_ops
                    if diff.op.__class__ == expr.op.__class__
                    and diff.field == expr.field]

            from pytools import single_valued
            names = [self.get_var_name() for d in all_diffs]
            self.code.append(
                    DiffBatchAssign(
                        names=names,
                        op_class=single_valued(
                            d.op.__class__ for d in all_diffs),
                        operators=[d.op for d in all_diffs],
                        field=single_valued(d.field for d in all_diffs)))

            from pymbolic import var
            for n, d in zip(names, all_diffs):
                self.expr_to_var[d] = var(n)

            return self.expr_to_var[expr]

    def map_planned_flux(self, expr):
        try:
            return self.expr_to_var[expr]
        except KeyError:
            for fb in self.flux_batches:
                if expr in fb:
                    fluxes = [self.internal_map_flux(f) for f in fb]

                    names = [self.get_var_name() for f in fluxes]
                    self.code.append(
                            FluxBatchAssign(
                                names=names,
                                fluxes=fluxes))

                    from pymbolic import var
                    for n, f in zip(names, fluxes):
                        self.expr_to_var[f] = var(n)

                    return self.expr_to_var[expr]

            raise RuntimeError("flux '%s' not in any flux batch" % expr)

    @classmethod
    def stringify_instruction(self, insn):
        if isinstance(insn, Assign):
            yield "%s <- %s" % (insn.name, insn.expr)
        elif isinstance(insn, FluxBatchAssign):
            for n, f in zip(insn.names, insn.fluxes):
                yield "%s <- %s" % (n, f)
        elif isinstance(insn, DiffBatchAssign):
            for n, d in zip(insn.names, insn.operators):
                yield "%s <- %s * %s" % (n, d, insn.field)
        else:
            raise TypeError, "invalid plan statement"

    @classmethod
    def stringify_code(cls, code, result):
        lines = []
        for insn in code:
            lines.extend(cls.stringify_instruction(insn))
        lines.append(str(result))

        return "\n".join(lines)
                
