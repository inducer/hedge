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




from pytools import Record, memoize_method
from hedge.optemplate import IdentityMapper




# instructions ----------------------------------------------------------------
class Instruction(Record):
    __slots__ = ["dep_mapper_factory"]
    priority = 0

    def get_assignees(self):
        raise NotImplementedError("no get_assignees in %s" % self.__class__)

    def get_dependencies(self):
        raise NotImplementedError("no get_dependencies in %s" % self.__class__)

    def __str__(self):
        raise NotImplementedError

    def get_executor_method(self, executor):
        raise NotImplementedError

class Assign(Instruction): 
    # attributes: name, expr, priority, flop_count

    def __init__(self, name, expr, **kwargs):
        Instruction.__init__(self, name=name, expr=expr, **kwargs)

        from pymbolic.mapper.flop_counter import FlopCounter
        self.flop_count = FlopCounter()(expr)

    def get_assignees(self):
        return set([self.name])

    @memoize_method
    def get_dependencies(self):
        return self.dep_mapper_factory()(self.expr)

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
    # attributes: names, op_class, operators, field

    def get_assignees(self):
        return set(self.names)

    @memoize_method
    def get_dependencies(self):
        return self.dep_mapper_factory()(self.field)

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
        return "%s <- %s * %s" % (
                self.name, 
                str(self.op_class()), 
                self.field)

    def get_executor_method(self, executor):
        return executor.exec_mass_assign


class FluxExchangeBatchAssign(Instruction):
    __slots__ = ["names", "indices_and_ranks", "rank_to_index_and_name", "field"]
    priority = 1

    def __init__(self, names, indices_and_ranks, field, dep_mapper_factory):
        rank_to_index_and_name = {}
        for name, (index, rank) in zip(
                names, indices_and_ranks):
            rank_to_index_and_name.setdefault(rank, []).append(
                (index, name))

        Instruction.__init__(self, 
                names=names,
                indices_and_ranks=indices_and_ranks,
                rank_to_index_and_name=rank_to_index_and_name,
                field=field,
                dep_mapper_factory=dep_mapper_factory)

    def get_assignees(self):
        return set(self.names)

    def get_dependencies(self):
        return self.dep_mapper_factory()(self.field)

    def __str__(self):
        lines = []

        lines.append("{")
        for n, (index, rank) in zip(self.names, self.indices_and_ranks):
            lines.append("  %s <- receive index %s from rank %d [%s]" % (
                n, index, rank, self.field))
        lines.append("}")

        return "\n".join(lines)

    def get_executor_method(self, executor):
        return executor.exec_flux_exchange_batch_assign




def dot_dataflow_graph(code, max_node_label_length=30):
    origins = {}
    node_names = {}

    result = [
            "initial [label=\"initial\"]"
            "result [label=\"result\"]"
            ]

    for num, insn in enumerate(code.instructions):
        node_name = "node%d" % num
        node_names[insn] = node_name
        node_label = repr(str(insn))[1:-1][:max_node_label_length]
        result.append("%s [ label=\"p%d: %s\" shape=box ];" % (
            node_name, insn.priority, node_label))

        for assignee in insn.get_assignees():
            origins[assignee] = node_name

    def get_orig_node(expr):
        from pymbolic.primitives import Variable
        if isinstance(expr, Variable):
            return origins.get(expr.name, "initial")
        else:
            return "initial"

    def gen_expr_arrow(expr, target_node):
        result.append("%s -> %s [label=\"%s\"];"
                % (get_orig_node(expr), target_node, expr))

    for insn in code.instructions:
        for dep in insn.get_dependencies():
            gen_expr_arrow(dep, node_names[insn])

    from hedge.tools import is_obj_array
    
    if is_obj_array(code.result):
        for subexp in code.result:
            gen_expr_arrow(subexp, "result")
    else:
        gen_expr_arrow(code.result, "result")

    return "digraph dataflow {\n%s\n}\n" % "\n".join(result)


            


# code ------------------------------------------------------------------------
class Code(object):
    def __init__(self, instructions, result):
        self.instructions = instructions
        self.result = result

    class NoInstructionAvailable(Exception):
        pass

    @memoize_method
    def get_next_step(self, available_names, done_insns):
        from pytools import all, argmax2
        available_insns = [
                (insn, insn.priority) for insn in self.instructions
                if insn not in done_insns 
                and all(dep.name in available_names 
                    for dep in insn.get_dependencies())]

        if not available_insns:
            raise self.NoInstructionAvailable
        
        from pytools import flatten
        discardable_vars = set(available_names) - set(flatten(
            [dep.name for dep in insn.get_dependencies()]
            for insn in self.instructions
            if insn not in done_insns ))

        from hedge.tools import with_object_array_or_scalar
        with_object_array_or_scalar(
                lambda var: discardable_vars.discard(var.name),
                self.result)

        return argmax2(available_insns), discardable_vars

    def __str__(self):
        lines = []
        for insn in self.instructions:
            lines.extend(str(insn).split("\n"))
        lines.append(str(self.result))

        return "\n".join(lines)

    def execute(self, exec_mapper):
        context = exec_mapper.context

        futures = []
        done_insns = set()

        quit_flag = False
        force_future = False
        while not quit_flag:
            # check futures for completion
            i = 0
            while i < len(futures):
                future = futures[i]
                if force_future or future.is_ready():
                    assignments, new_futures = future()
                    for target, value in assignments:
                        context[target] = value
                    futures.extend(new_futures)
                    futures.pop(i)
                    force_future = False
                else:
                    i += 1

                del future

            # pick the next insn 
            try:
                insn, discardable_vars = self.get_next_step(
                        frozenset(context.keys()),
                        frozenset(done_insns))
            except self.NoInstructionAvailable:
                if futures:
                    # no insn ready: we need a future to complete to continue
                    force_future = True
                else:
                    # no futures, no available instructions: we're done
                    quit_flag = True
            else:
                for name in discardable_vars:
                    del context[name]

                done_insns.add(insn)
                assignments, new_futures = \
                        insn.get_executor_method(exec_mapper)(insn)
                for target, value in assignments:
                    context[target] = value
                futures.extend(new_futures)

        from hedge.tools import with_object_array_or_scalar
        return with_object_array_or_scalar(exec_mapper, self.result)




# compiler --------------------------------------------------------------------
class OperatorCompilerBase(IdentityMapper):
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

    def dep_mapper_factory(self):
        from hedge.optemplate import DependencyMapper
        return DependencyMapper(
                include_operator_bindings=False,
                include_subscripts=False,
                include_calls="descend_args")

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

    def collect_flux_exchange_ops(self, expr):
        from hedge.optemplate import FluxExchangeOperator
        return self.bound_op_collector_class(FluxExchangeOperator)(expr)

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

        # Flux exchange also works better when batched.
        self.flux_exchange_ops = self.collect_flux_exchange_ops(expr)

        # Finally, walk the expression and build the code.
        result = IdentityMapper.__call__(self, expr)

        # Then, put the toplevel expressions into variables as well.
        from hedge.tools import with_object_array_or_scalar
        result = with_object_array_or_scalar(self.assign_to_new_var, result)
        return Code(self.code, result)

    def get_var_name(self):
        new_name = self.prefix+str(self.assigned_var_count)
        self.assigned_var_count += 1
        return new_name

    def map_common_subexpression(self, expr):
        try:
            return self.expr_to_var[expr.child]
        except KeyError:
            priority = getattr(expr, "priority", 0)
            cse_var = self.assign_to_new_var(self.rec(expr.child),
                    priority=priority)
            self.expr_to_var[expr.child] = cse_var
            return cse_var

    def map_operator_binding(self, expr):
        from hedge.optemplate import \
                DiffOperatorBase, \
                MassOperatorBase, \
                FluxExchangeOperator
        if isinstance(expr.op, DiffOperatorBase):
            return self.map_diff_op_binding(expr)
        elif isinstance(expr.op, MassOperatorBase):
            return self.map_mass_op_binding(expr)
        elif isinstance(expr.op, FluxExchangeOperator):
            return self.map_flux_exchange_op_binding(expr)
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
                        field=self.rec(single_valued(d.field for d in all_diffs)),
                        dep_mapper_factory=self.dep_mapper_factory))

            from pymbolic import var
            for n, d in zip(names, all_diffs):
                self.expr_to_var[d] = var(n)

            return self.expr_to_var[expr]

    def map_mass_op_binding(self, expr):
        try:
            return self.expr_to_var[expr]
        except KeyError:
            ma = MassAssign(
                    name=self.get_var_name(),
                    op_class=expr.op.__class__,
                    field=self.rec(expr.field))
            self.code.append(ma)

            from pymbolic import var
            v = var(ma.name)
            self.expr_to_var[expr] = v
            return v

    def map_flux_exchange_op_binding(self, expr):
        try:
            return self.expr_to_var[expr]
        except KeyError:
            from hedge.tools import is_field_equal
            all_flux_xchgs = [fe
                    for fe in self.flux_exchange_ops
                    if is_field_equal(fe.field, expr.field)]

            assert len(all_flux_xchgs) > 0

            from pytools import single_valued
            names = [self.get_var_name() for d in all_flux_xchgs]
            self.code.append(
                    FluxExchangeBatchAssign(
                        names=names,
                        indices_and_ranks=[
                            (fe.op.index, fe.op.rank) for fe in all_flux_xchgs],
                        field=self.rec(
                            single_valued(
                                (fe.field for fe in all_flux_xchgs),
                                equality_pred=is_field_equal)),
                        dep_mapper_factory=self.dep_mapper_factory))

            from pymbolic import var
            for n, d in zip(names, all_flux_xchgs):
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

    def assign_to_new_var(self, expr, priority=0):
        from pymbolic.primitives import Variable
        if isinstance(expr, Variable):
            return expr
            
        new_name = self.get_var_name()
        self.code.append(self.make_assign(new_name, expr, priority))

        return Variable(new_name)

    # instruction producers ---------------------------------------------------
    def make_assign(self, name, expr, priority):
        return Assign(name=name, expr=expr, dep_mapper_factory=self.dep_mapper_factory,
                priority=priority)

    def make_flux_batch_assign(self, names, fluxes, kind):
        return FluxBatchAssign(names=names, fluxes=fluxes, kind=kind)
