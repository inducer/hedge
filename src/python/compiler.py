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
from hedge.optemplate import IdentityMapper




# instructions ----------------------------------------------------------------
class Instruction(Record):
    __slots__ = []

    def get_assignees(self):
        raise NotImplementedError("no get_assignees in %s" % self.__class__)

    def get_dependencies(self):
        raise NotImplementedError("no get_dependencies in %s" % self.__class__)

    def __str__(self):
        raise NotImplementedError

    def get_executor_method(self, executor):
        raise NotImplementedError

class Discard(Instruction): 
    __slots__ = ["name"]

    def get_assignees(self):
        return set([self.name])

    def get_dependencies(self):
        return set()

    def __str__(self):
        return "discard %s" % self.name

    def get_executor_method(self, executor):
        return executor.exec_discard

class Assign(Instruction): 
    __slots__ = ["name", "expr", "dep_mapper_class"]

    def get_assignees(self):
        return set([self.name])

    def get_dependencies(self):
        return self.dep_mapper_class(include_calls="descend_args")(self.expr)

    def __str__(self):
        return "%s <- %s" % (self.name, self.expr)

    def get_executor_method(self, executor):
        return executor.exec_assign

class FluxBatchAssign(Instruction):
    __slots__ = ["names", "fluxes", "kind"]

    def get_assignees(self):
        return set(self.names)

    def __str__(self):
        lines = []
        lines.append("{ /* %s */" % self.kind)
        for n, f in zip(self.names, self.fluxes):
            lines.append("  %s <- %s" % (n, f))
        lines.append("}")
        return "\n".join(lines)

    def get_executor_method(self, executor):
        return executor.exec_flux_batch_assign

class DiffBatchAssign(Instruction):
    __slots__ = ["names", "op_class", "operators", "field"]

    def get_assignees(self):
        return set(self.names)

    def get_dependencies(self):
        return set([self.field])

    def __str__(self):
        lines = []

        if len(self.names) > 1:
            lines.append("{")
            for n, d in zip(self.names, self.operators):
                lines.append("  %s <- %s * %s" % (n, d, self.field))
            lines.append("}")
        else:
            for n, d in zip(self.names, self.operators):
                lines.append("%s <- %s * %s" % (n, d, self.field))

        return "\n".join(lines)

    def get_executor_method(self, executor):
        return executor.exec_diff_batch_assign

class MassAssign(Instruction):
    __slots__ = ["name", "op_class", "field"]

    def get_assignees(self):
        return set([self.name])

    def get_dependencies(self):
        return set([self.field])

    def __str__(self):
        return "%s <- Mass * %s" % (self.name, self.field)

    def get_executor_method(self, executor):
        return executor.exec_mass_assign


class FluxReceiveBatchAssign(Instruction):
    __slots__ = ["names", "indices_and_ranks", "field"]

    def get_assignees(self):
        return set(self.names)

    def get_dependencies(self):
        return set([self.field])

    def __str__(self):
        lines = []

        lines.append("{")
        for n, (index, rank) in zip(self.names, self.indices_and_ranks):
            lines.append("  %s <- receive index %d from rank %d [%s]" % (
                n, index, rank, self.field))
        lines.append("}")

        return "\n".join(lines)

    def get_executor_method(self, executor):
        return executor.exec_flux_receive_batch_assign




# code ------------------------------------------------------------------------
class Code(object):
    def __init__(self, code, result):
        self.code = code
        self.result = result

    def __str__(self):
        lines = []
        for insn in self.code:
            lines.extend(str(insn).split("\n"))
        lines.append(str(self.result))

        return "\n".join(lines)

    def execute(self, exec_mapper):
        for insn in self.code:
            insn.get_executor_method(exec_mapper)(insn)

        from hedge.tools import with_object_array_or_scalar
        return with_object_array_or_scalar(exec_mapper, self.result)




# compiler --------------------------------------------------------------------
class OperatorCompilerBase(IdentityMapper):
    from hedge.optemplate import DependencyMapper as dep_mapper_class
    from hedge.optemplate import BoundOperatorCollector \
            as bound_op_collector_class

    class FluxRecord(Record):
        __slots__ = ["flux_expr", "dependencies", "kind"]

    class FluxBatch(Record):
        __slots__ = ["flux_exprs", "kind"]

    def __init__(self, prefix="_expr"):
        IdentityMapper.__init__(self)
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

    def collect_diff_ops(self, expr):
        from hedge.optemplate import DiffOperatorBase
        return self.bound_op_collector_class(DiffOperatorBase)(expr)

    def collect_flux_receive_ops(self, expr):
        from hedge.optemplate import FluxReceiveOperator
        return self.bound_op_collector_class(FluxReceiveOperator)(expr)

    def insert_discards(self, code, result_expr):
        rev_code_and_used_vars = []
        from hedge.tools import setify_field as setify
        used_vars = set(v.name for v in setify(result_expr))

        from pymbolic.primitives import Variable

        for line in code[::-1]:
            used_vars.update(
                v.name for v in line.get_dependencies()
                if isinstance(v, Variable) and v.name.startswith(self.prefix))
            rev_code_and_used_vars.append((line, used_vars.copy()))
            used_vars = used_vars - line.get_assignees()

        result = []
        last_used_vars = set()
        for line, used_vars in rev_code_and_used_vars[::-1]:
            for var_name in last_used_vars - used_vars:
                result.append(Discard(name=var_name))

            result.append(line)

            last_used_vars = used_vars

        return result

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
                    present_batch.add(fr)
                    flux_queue.pop(i)
                else:
                    i += 1

            if present_batch:

                batches_by_kind = {}
                for fr in present_batch:
                    batches_by_kind[fr.kind] = \
                            batches_by_kind.get(fr.kind, set()) | set([fr.flux_expr])

                for kind, batch in batches_by_kind.iteritems():
                    self.flux_batches.append(
                            self.FluxBatch(kind=kind, flux_exprs=list(batch)))

                admissible_deps |= present_batch
            else:
                raise RuntimeError, "cannot resolve flux evaluation order"

        # Once flux batching is figured out, we also need to know which
        # derivatives are going to be needed, because once the 
        # rst-derivatives are available, it's best to calculate the 
        # xyz ones and throw the rst ones out. It's naturally good if
        # we can avoid computing (or storing) some of the xyz ones.
        # So figure out which XYZ derivatives of what are needed.

        self.diff_ops = self.collect_diff_ops(expr)

        # Flux receiving also works better when batched.
        self.flux_receive_ops = self.collect_flux_receive_ops(expr)

        # Finally, walk the expression and build the code.
        result = IdentityMapper.__call__(self, expr)

        # Then, put the toplevel expressions into variables as well.
        from hedge.tools import with_object_array_or_scalar
        result = with_object_array_or_scalar(self.assign_to_new_var, result)
        return Code(self.insert_discards(self.code, result), result)

    def get_var_name(self):
        new_name = self.prefix+str(self.assigned_var_count)
        self.assigned_var_count += 1
        return new_name

    def map_common_subexpression(self, expr):
        try:
            return self.expr_to_var[expr.child]
        except KeyError:
            cse_var = self.assign_to_new_var(self.rec(expr.child))
            self.expr_to_var[expr.child] = cse_var
            return cse_var

    def map_operator_binding(self, expr):
        from hedge.optemplate import \
                DiffOperatorBase, \
                MassOperatorBase, \
                FluxReceiveOperator
        if isinstance(expr.op, DiffOperatorBase):
            return self.map_diff_op_binding(expr)
        elif isinstance(expr.op, MassOperatorBase):
            return self.map_mass_op_binding(expr)
        elif isinstance(expr.op, FluxReceiveOperator):
            return self.map_flux_receive_op_binding(expr)
        else:
            return IdentityMapper.map_operator_binding(self, expr)

    def map_diff_op_binding(self, expr):
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
                        field=self.rec(single_valued(d.field for d in all_diffs))))

            from pymbolic import var
            for n, d in zip(names, all_diffs):
                self.expr_to_var[d] = var(n)

            return self.expr_to_var[expr]

    def map_mass_op_binding(self, expr):
        try:
            return self.expr_to_var[expr]
        except KeyError:
            ma = MassAssign(
                    name=self.get_var_name,
                    op_class=expr.op.__class__,
                    field=self.rec(expr.field))
            self.code.append(ma)

            from pymbolic import var
            v = var(ma.name)
            self.expr_to_var[expr] = v
            return v

    def map_flux_receive_op_binding(self, expr):
        try:
            return self.expr_to_var[expr]
        except KeyError:
            all_frs = [fr
                    for fr in self.flux_receive_ops
                    if fr.field == expr.field]

            assert len(all_frs) > 0

            from pytools import single_valued
            names = [self.get_var_name() for d in all_frs]
            self.code.append(
                    FluxReceiveBatchAssign(
                        names=names,
                        indices_and_ranks=[
                            (fr.op.index, fr.op.rank) for fr in all_frs],
                        field=self.rec(single_valued(fr.field for fr in all_frs))))

            from pymbolic import var
            for n, d in zip(names, all_frs):
                self.expr_to_var[d] = var(n)

            return self.expr_to_var[expr]

    def map_planned_flux(self, expr):
        try:
            return self.expr_to_var[expr]
        except KeyError:
            for fb in self.flux_batches:
                try:
                    idx = fb.flux_exprs.index(expr)
                except ValueError:
                    pass
                else:
                    # found at idx
                    mapped_fluxes = [self.internal_map_flux(f) for f in fb.flux_exprs]

                    names = [self.get_var_name() for f in mapped_fluxes]
                    self.code.append(
                            self.make_flux_batch_assign(names, mapped_fluxes, fb.kind))

                    from pymbolic import var
                    for n, f in zip(names, fb.flux_exprs):
                        self.expr_to_var[f] = var(n)

                    return var(names[idx])

            raise RuntimeError("flux '%s' not in any flux batch" % expr)

    def assign_to_new_var(self, expr):
        from pymbolic.primitives import Variable
        if isinstance(expr, Variable):
            return expr
            
        new_name = self.get_var_name()
        self.code.append(self.make_assign(new_name, expr))

        return Variable(new_name)

    # instruction producers ---------------------------------------------------
    def make_assign(self, name, expr):
        return Assign(name=name, expr=expr, dep_mapper_class=self.dep_mapper_class)

    def make_flux_batch_assign(self, names, fluxes, kind):
        return FluxBatchAssign(names=names, fluxes=fluxes, kind=kind)
