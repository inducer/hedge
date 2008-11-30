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




class ExecutionPlannerBase(OpTemplateIdentityMapper):
    class Assignment(Record): 
        __slots__ = ["name", "expr"]

    class FluxBatchAssignment(Record):
        __slots__ = ["names", "fluxes", "dependencies"]

    class FluxBatch(Record): 
        __slots__ = ["fluxes", "dependencies"]

    class FluxRecord(Record):
        __slots__ = ["flux_expr", "dependencies"]

    def __init__(self, prefix="_expr"):
        OpTemplateIdentityMapper.__init__(self)
        self.prefix = prefix
        self.assignments = []
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
        flux_batches = []
        flux_queue = self.get_contained_fluxes(expr)
        for fr in flux_queue:
            fr.flux_dependencies = set()
            for d in fr.dependencies:
                fr.flux_dependencies |= set(sf.flux_expr 
                        for sf in self.get_contained_fluxes(d))

        # Then figure out batches of fluxes to evaluate
        self.flux_batches = []
        admissible_deps = set()
        while flux_queue:
            present_batch = set()
            present_deps = set()
            i = 0
            while i < len(flux_queue):
                fr = flux_queue[i]
                if fr.flux_dependencies <= admissible_deps:
                    present_batch.add(fr.flux_expr)
                    present_deps |= fr.dependencies
                    flux_queue.pop(0)
                else:
                    i += 1

            if present_batch:
                self.flux_batches.append(self.FluxBatch(
                    fluxes=present_batch,
                    dependencies=present_deps))

                admissible_deps |= present_batch
            else:
                raise RuntimeError, "cannot resolve flux evaluation order"

        # Then do the remainder of the planning
        result = OpTemplateIdentityMapper.__call__(self, expr)
        return self.assignments, result

    def get_var_name(self):
        new_name = self.prefix+str(self.assigned_var_count)
        self.assigned_var_count += 1
        return new_name

    def map_common_subexpression(self, expr):
        try:
            return self.expr_to_var[expr.child]
        except KeyError:
            processed_child = self.rec(expr.child)

            new_name = self.get_var_name()
            self.assignments.append(self.Assignment(
                name=new_name, expr=processed_child))

            from pymbolic import var
            cse_var = var(new_name)
            self.expr_to_var[expr.child] = cse_var
            return cse_var

    def map_planned_flux(self, expr):
        try:
            return self.expr_to_var[expr]
        except KeyError:
            for fb in self.flux_batches:
                if expr in fb.fluxes:
                    # fix an order
                    ordered_fluxes = list(fb.fluxes)

                    names = [self.get_var_name() for f in ordered_fluxes]
                    self.assignments.append(
                            self.FluxBatchAssignment(
                                names=names,
                                fluxes=ordered_fluxes,
                                dependencies=fb.dependencies))

                    from pymbolic import var
                    for n, f in zip(names, ordered_fluxes):
                        self.expr_to_var[f] = var(n)

                    return self.expr_to_var[expr]

            raise RuntimeError("flux '%s' not in any flux batch" % expr)

    @classmethod
    def stringify_plan(cls, assignments, result):
        lines = []
        for a in assignments:
            if isinstance(a, cls.Assignment):
                lines.append("%s <- %s" % (a.name, a.expr))
            elif isinstance(a, cls.FluxBatchAssignment):
                for n, f in zip(a.names, a.fluxes):
                    lines.append("%s <- %s" % (n, f))
        lines.append(str(result))

        return "\n".join(lines)
                
