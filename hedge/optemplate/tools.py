"""Operator templates: extra bits of functionality."""

from __future__ import division

__copyright__ = "Copyright (C) 2008 Andreas Kloeckner"

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


import numpy as np
import pymbolic.primitives  # noqa
from pytools import MovedFunctionDeprecationWrapper
from decorator import decorator


# {{{ convenience functions for optemplate creation

make_vector_field = \
        MovedFunctionDeprecationWrapper(pymbolic.primitives.make_sym_vector)


def get_flux_operator(flux):
    """Return a flux operator that can be multiplied with
    a volume field to obtain the interior fluxes
    or with a :class:`BoundaryPair` to obtain the lifted boundary
    flux.
    """
    from hedge.tools import is_obj_array
    from hedge.optemplate import VectorFluxOperator, FluxOperator

    if is_obj_array(flux):
        return VectorFluxOperator(flux)
    else:
        return FluxOperator(flux)


def make_nabla(dim):
    from hedge.tools import make_obj_array
    from hedge.optemplate import DifferentiationOperator
    return make_obj_array(
            [DifferentiationOperator(i) for i in range(dim)])


def make_minv_stiffness_t(dim):
    from hedge.tools import make_obj_array
    from hedge.optemplate import MInvSTOperator
    return make_obj_array(
        [MInvSTOperator(i) for i in range(dim)])


def make_stiffness(dim):
    from hedge.tools import make_obj_array
    from hedge.optemplate import StiffnessOperator
    return make_obj_array(
        [StiffnessOperator(i) for i in range(dim)])


def make_stiffness_t(dim):
    from hedge.tools import make_obj_array
    from hedge.optemplate import StiffnessTOperator
    return make_obj_array(
        [StiffnessTOperator(i) for i in range(dim)])


def integral(arg):
    import hedge.optemplate as sym
    return sym.NodalSum()(sym.MassOperator()(sym.Ones())*arg)


def norm(p, arg):
    """
    :arg arg: is assumed to be a vector, i.e. have shape ``(n,)``.
    """
    import hedge.optemplate as sym

    if p == 2:
        comp_norm_squared = sym.NodalSum()(
                sym.CFunction("fabs")(
                    arg * sym.MassOperator()(arg)))
        return sym.CFunction("sqrt")(sum(comp_norm_squared))

    elif p == np.Inf:
        comp_norm = sym.NodalMax()(sym.CFunction("fabs")(arg))
        from pymbolic.primitives import Max
        return reduce(Max, comp_norm)

    else:
        return sum(sym.NodalSum()(sym.CFunction("fabs")(arg)**p))**(1/p)


def flat_end_sin(x):
    from hedge.optemplate.primitives import CFunction
    from pymbolic.primitives import IfPositive
    from math import pi
    return IfPositive(-pi/2-x,
            -1, IfPositive(x-pi/2, 1, CFunction("sin")(x)))


def smooth_ifpos(crit, right, left, width):
    from math import pi
    return 0.5*((left+right)
            + (right-left)*flat_end_sin(
                pi/2/width * crit))
# }}}


# {{{ optemplate tools

def is_scalar(expr):
    from hedge.optemplate import ScalarParameter
    return isinstance(expr, (int, float, complex, ScalarParameter))


def get_flux_dependencies(flux, field, bdry="all"):
    from hedge.flux import FluxDependencyMapper, FieldComponent
    in_fields = list(FluxDependencyMapper(
        include_calls=False)(flux))

    # check that all in_fields are FieldComponent instances
    assert not [in_field
        for in_field in in_fields
        if not isinstance(in_field, FieldComponent)]

    def maybe_index(fld, index):
        from hedge.tools import is_obj_array
        if is_obj_array(fld):
            return fld[inf.index]
        else:
            return fld

    from hedge.tools import is_zero
    from hedge.optemplate import BoundaryPair
    if isinstance(field, BoundaryPair):
        for inf in in_fields:
            if inf.is_interior:
                if bdry in ["all", "int"]:
                    value = maybe_index(field.field, inf.index)

                    if not is_zero(value):
                        yield value
            else:
                if bdry in ["all", "ext"]:
                    value = maybe_index(field.bfield, inf.index)

                    if not is_zero(value):
                        yield value
    else:
        for inf in in_fields:
            value = maybe_index(field, inf.index)
            if not is_zero(value):
                yield value


def split_optemplate_for_multirate(state_vector, op_template,
        index_groups):
    class IndexGroupKillerSubstMap:
        def __init__(self, kill_set):
            self.kill_set = kill_set

        def __call__(self, expr):
            if expr in kill_set:
                return 0
            else:
                return None

    # make IndexGroupKillerSubstMap that kill everything
    # *except* what's in that index group
    killers = []
    for i in range(len(index_groups)):
        kill_set = set()
        for j in range(len(index_groups)):
            if i != j:
                kill_set |= set(index_groups[j])

        killers.append(IndexGroupKillerSubstMap(kill_set))

    from hedge.optemplate import \
            SubstitutionMapper, \
            CommutativeConstantFoldingMapper

    return [
            CommutativeConstantFoldingMapper()(
                SubstitutionMapper(killer)(
                    op_template[ig]))
            for ig in index_groups
            for killer in killers]


def ptwise_mul(a, b):
    from pytools.obj_array import log_shape
    a_log_shape = log_shape(a)
    b_log_shape = log_shape(b)

    from pytools import indices_in_shape

    if a_log_shape == ():
        result = np.empty(b_log_shape, dtype=object)
        for i in indices_in_shape(b_log_shape):
            result[i] = a*b[i]
    elif b_log_shape == ():
        result = np.empty(a_log_shape, dtype=object)
        for i in indices_in_shape(a_log_shape):
            result[i] = a[i]*b
    else:
        raise ValueError("ptwise_mul can't handle two non-scalars")

    return result


def ptwise_dot(logdims1, logdims2, a1, a2):
    a1_log_shape = a1.shape[:logdims1]
    a2_log_shape = a1.shape[:logdims2]

    assert a1_log_shape[-1] == a2_log_shape[0]
    len_k = a2_log_shape[0]

    result = np.empty(a1_log_shape[:-1]+a2_log_shape[1:], dtype=object)

    from pytools import indices_in_shape
    for a1_i in indices_in_shape(a1_log_shape[:-1]):
        for a2_i in indices_in_shape(a2_log_shape[1:]):
            result[a1_i+a2_i] = sum(
                    a1[a1_i+(k,)] * a2[(k,)+a2_i]
                    for k in xrange(len_k)
                    )

    if result.shape == ():
        return result[()]
    else:
        return result

# }}}


# {{{ process_optemplate function

def process_optemplate(optemplate, post_bind_mapper=None,
        dumper=lambda name, optemplate: None, mesh=None,
        type_hints={}):

    from hedge.optemplate.mappers import (
            OperatorBinder, CommutativeConstantFoldingMapper,
            EmptyFluxKiller, InverseMassContractor, DerivativeJoiner,
            ErrorChecker, OperatorSpecializer, GlobalToReferenceMapper)
    from hedge.optemplate.mappers.bc_to_flux import BCToFluxRewriter
    from hedge.optemplate.mappers.type_inference import TypeInferrer

    dumper("before-bind", optemplate)
    optemplate = OperatorBinder()(optemplate)

    ErrorChecker(mesh)(optemplate)

    if post_bind_mapper is not None:
        dumper("before-postbind", optemplate)
        optemplate = post_bind_mapper(optemplate)

    if mesh is not None:
        dumper("before-empty-flux-killer", optemplate)
        optemplate = EmptyFluxKiller(mesh)(optemplate)

    dumper("before-cfold", optemplate)
    optemplate = CommutativeConstantFoldingMapper()(optemplate)

    dumper("before-bc2flux", optemplate)
    optemplate = BCToFluxRewriter()(optemplate)

    # Ordering restriction:
    #
    # - Must run constant fold before first type inference pass, because zeros,
    # while allowed, violate typing constraints (because they can't be assigned
    # a unique type), and need to be killed before the type inferrer sees them.
    #
    # - Must run BC-to-flux before first type inferrer run so that zeros in
    # flux arguments can be removed.

    dumper("before-specializer", optemplate)
    optemplate = OperatorSpecializer(
            TypeInferrer()(optemplate, type_hints)
            )(optemplate)

    # Ordering restriction:
    #
    # - Must run OperatorSpecializer before performing the GlobalToReferenceMapper,
    # because otherwise it won't differentiate the type of grids (node or quadrature
    # grids) that the operators will apply on.

    assert mesh is not None
    dumper("before-global-to-reference", optemplate)
    optemplate = GlobalToReferenceMapper(mesh.dimensions)(optemplate)

    # Ordering restriction:
    #
    # - Must specialize quadrature operators before performing inverse mass
    # contraction, because there are no inverse-mass-contracted variants of the
    # quadrature operators.

    dumper("before-imass", optemplate)
    optemplate = InverseMassContractor()(optemplate)

    dumper("before-cfold-2", optemplate)
    optemplate = CommutativeConstantFoldingMapper()(optemplate)

    dumper("before-derivative-join", optemplate)
    optemplate = DerivativeJoiner()(optemplate)

    dumper("process-optemplate-finished", optemplate)

    return optemplate

# }}}


# {{{ pretty printing

def pretty(optemplate):
    from hedge.optemplate.mappers import PrettyStringifyMapper

    stringify_mapper = PrettyStringifyMapper()
    from pymbolic.mapper.stringifier import PREC_NONE
    result = stringify_mapper(optemplate, PREC_NONE)

    splitter = "="*75 + "\n"

    bc_strs = stringify_mapper.get_bc_strings()
    if bc_strs:
        result = "\n".join(bc_strs)+"\n"+splitter+result

    cse_strs = stringify_mapper.get_cse_strings()
    if cse_strs:
        result = "\n".join(cse_strs)+"\n"+splitter+result

    flux_strs = stringify_mapper.get_flux_strings()
    if flux_strs:
        result = "\n".join(flux_strs)+"\n"+splitter+result

    flux_cses = stringify_mapper.flux_stringify_mapper.get_cse_strings()
    if flux_cses:
        result = "\n".join("flux "+fs for fs in flux_cses)+"\n\n"+result

    return result

# }}}


@decorator
def memoize_method_with_obj_array_args(method, instance, *args):
    """This decorator manages to memoize functions that
    take object arrays (which are mutable, but are assumed
    to never change) as arguments.
    """
    dicname = "_memoize_dic_"+method.__name__

    new_args = []
    for arg in args:
        if isinstance(arg, np.ndarray) and arg.dtype == object:
            new_args.append(tuple(arg))
        else:
            new_args.append(arg)
    new_args = tuple(new_args)

    try:
        return getattr(instance, dicname)[new_args]
    except AttributeError:
        result = method(instance, *args)
        setattr(instance, dicname, {new_args: result})
        return result
    except KeyError:
        result = method(instance, *args)
        getattr(instance, dicname)[new_args] = result
        return result


# vim: foldmethod=marker
