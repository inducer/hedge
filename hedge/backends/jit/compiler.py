"""Just-in-time compiling backend."""

from __future__ import division

__copyright__ = "Copyright (C) 2008 Andreas Kloeckner"

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




import hedge.discretization
import hedge.optemplate
from pytools import memoize_method
from hedge.compiler import OperatorCompilerBase, FluxBatchAssign, \
        Assign




# compiler stuff --------------------------------------------------------------
class VectorExprAssign(Assign):
    __slots__ = ["compiled"]

    def get_executor_method(self, executor):
        return executor.exec_vector_expr_assign

    def __str__(self):
        return "%s <- (compiled) %s" % (self.name, self.expr)




class CompiledFluxBatchAssign(FluxBatchAssign):
    # members: compiled_func, arg_specs, is_boundary

    @memoize_method
    def get_dependencies(self):
        deps = set()

        from hedge.tools import setify_field as setify
        from hedge.optemplate import OperatorBinding, BoundaryPair
        for f in self.fluxes:
            assert isinstance(f, OperatorBinding)
            if isinstance(f.field, BoundaryPair):
                deps |= setify(f.field.field) |  setify(f.field.bfield)
            else:
                deps |= setify(f.field)

        dep_mapper = self.dep_mapper_factory()

        from pytools import flatten
        return set(flatten(dep_mapper(dep) for dep in deps))

    @memoize_method
    def get_module(self, discr, dtype):
        from hedge.backends.jit.flux import \
                get_interior_flux_mod, \
                get_boundary_flux_mod

        if not self.is_boundary:
            mod = get_interior_flux_mod(
                    self.fluxes, self.flux_var_info, discr.toolchain, dtype)

            if discr.instrumented:
                from hedge.tools import time_count_flop, gather_flops
                mod.gather_flux = \
                        time_count_flop(
                                mod.gather_flux,
                                discr.gather_timer,
                                discr.gather_counter,
                                discr.gather_flop_counter,
                                len(self.fluxes)
                                * gather_flops(discr)
                                * len(self.flux_var_info.arg_names))

        else:
            mod = get_boundary_flux_mod(
                    self.fluxes, self.flux_var_info, discr.toolchain, dtype)

            if discr.instrumented:
                from pytools.log import time_and_count_function
                mod.gather_flux = time_and_count_function(
                        mod.gather_flux, discr.gather_timer)

        return mod





# flux kinds ------------------------------------------------------------------
class InteriorFluxKind(object):
    def __hash__(self):
        return hash(self.__class__)

    def __str__(self):
        return "interior"

    def __eq__(self, other):
        return (other.__class__ == self.__class__)

class BoundaryFluxKind(object):
    def __init__(self, tag):
        self.tag = tag

    def __str__(self):
        return "boundary(%s)" % self.tag

    def __hash__(self):
        return hash((self.__class__, self.tag))

    def __eq__(self, other):
        return (other.__class__ == self.__class__
                and other.tag == self.tag)




class OperatorCompiler(OperatorCompilerBase):
    def __init__(self, discr):
        OperatorCompilerBase.__init__(self)
        self.discr = discr

    def get_contained_fluxes(self, expr):
        from hedge.optemplate import FluxCollector, BoundaryPair
        from hedge.tools import is_obj_array

        def get_deps(field):
            if is_obj_array(field):
                return set(field)
            else:
                return set([field])

        def get_flux_deps(op_binding):
            if isinstance(op_binding.field, BoundaryPair):
                bpair = op_binding.field
                return get_deps(bpair.field) | get_deps(bpair.bfield)
            else:
                return get_deps(op_binding.field)

        def get_flux_kind(op_binding):
            if isinstance(op_binding.field, BoundaryPair):
                return BoundaryFluxKind(op_binding.field.tag)
            else:
                return InteriorFluxKind()

        return [self.FluxRecord(
            flux_expr=flux_binding,
            kind=get_flux_kind(flux_binding),
            dependencies=get_flux_deps(flux_binding))
            for flux_binding in FluxCollector()(expr)]

    def internal_map_flux(self, flux_bind):
        from hedge.optemplate import IdentityMapper
        return IdentityMapper.map_operator_binding(self, flux_bind)

    def map_operator_binding(self, expr):
        from hedge.optemplate import FluxOperatorBase
        if isinstance(expr.op, FluxOperatorBase):
            return self.map_planned_flux(expr)
        else:
            return OperatorCompilerBase.map_operator_binding(self, expr)

    # flux compilation --------------------------------------------------------
    def make_flux_batch_assign(self, names, fluxes, kind):
        if isinstance(kind, BoundaryFluxKind):
            return self.make_boundary_flux_batch_assign(names, fluxes, kind)
        elif isinstance(kind, InteriorFluxKind):
            return self.make_interior_flux_batch_assign(names, fluxes, kind)
        else:
            raise ValueError("invalid flux batch kind: %s" % kind)

    def make_interior_flux_batch_assign(self, names, fluxes, kind):
        from hedge.backends.jit.flux import get_flux_var_info
        return CompiledFluxBatchAssign(is_boundary=False,
                names=names, fluxes=fluxes, kind=kind,
                flux_var_info=get_flux_var_info(fluxes),
                dep_mapper_factory=self.dep_mapper_factory)

    def make_boundary_flux_batch_assign(self, names, fluxes, kind):
        from hedge.backends.jit.flux import get_flux_var_info
        return CompiledFluxBatchAssign(is_boundary=True,
                names=names, fluxes=fluxes, kind=kind,
                flux_var_info=get_flux_var_info(fluxes),
                dep_mapper_factory=self.dep_mapper_factory)

    def make_assign(self, name, expr, priority):
        def result_dtype_getter(vector_dtype_map, scalar_dtype_map, const_dtypes):
            from pytools import common_dtype
            return common_dtype(
                    vector_dtype_map.values()
                    + scalar_dtype_map.values()
                    + const_dtypes)

        from hedge.backends.jit.vector_expr import CompiledVectorExpression
        return VectorExprAssign(
                name=name,
                expr=expr,
                dep_mapper_factory=self.dep_mapper_factory,
                compiled=CompiledVectorExpression(
                    expr,
                    is_vector_func=lambda expr: True,
                    result_dtype_getter=result_dtype_getter,
                    toolchain=self.discr.toolchain),
                priority=priority)
