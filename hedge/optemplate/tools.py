"""Operator templates: extra bits of functionality."""

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




import numpy
import pymbolic.primitives




# {{{ convenience functions for optemplate creation ---------------------------
def make_vector_field(name, components):
    """Return an object array of *components* subscripted
    :class:`Field` instances.

    :param components: The number of components in the vector.
    """
    if isinstance(components, int):
        components = range(components)

    from hedge.tools import join_fields
    vfld = pymbolic.primitives.Variable(name)
    return join_fields(*[vfld[i] for i in components])




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



# }}}

# {{{ optemplate tools --------------------------------------------------------
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
        result = numpy.empty(b_log_shape, dtype=object)
        for i in indices_in_shape(b_log_shape):
            result[i] = a*b[i]
    elif b_log_shape == ():
        result = numpy.empty(a_log_shape, dtype=object)
        for i in indices_in_shape(a_log_shape):
            result[i] = a[i]*b
    else:
        raise ValueError, "ptwise_mul can't handle two non-scalars"

    return result




def ptwise_dot(logdims1, logdims2, a1, a2):
    a1_log_shape = a1.shape[:logdims1]
    a2_log_shape = a1.shape[:logdims2]

    assert a1_log_shape[-1] == a2_log_shape[0]
    len_k = a2_log_shape[0]

    result = numpy.empty(a1_log_shape[:-1]+a2_log_shape[1:], dtype=object)

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

# {{{ process_optemplate function ---------------------------------------------
def process_optemplate(optemplate, post_bind_mapper=None,
        dumper=lambda name, optemplate: None, mesh=None):

    from hedge.optemplate.mappers import (
            OperatorBinder, BCToFluxRewriter, CommutativeConstantFoldingMapper,
            EmptyFluxKiller, InverseMassContractor, DerivativeJoiner,
            ErrorChecker, OperatorSpecializer, GlobalToReferenceMapper)
    from hedge.optemplate.mappers.type_inference import TypeInferrer

    dumper("before-bind", optemplate)
    optemplate = OperatorBinder()(optemplate)

    ErrorChecker(mesh)(optemplate)

    if post_bind_mapper is not None:
        dumper("before-postbind", optemplate)
        optemplate = post_bind_mapper(optemplate)

    dumper("before-cfold", optemplate)
    optemplate = CommutativeConstantFoldingMapper()(optemplate)

    assert mesh is not None
    dumper("before-global-to-reference", optemplate)
    optemplate = GlobalToReferenceMapper(mesh.dimensions)(optemplate)

    dumper("before-bc2flux", optemplate)
    optemplate = BCToFluxRewriter()(optemplate)

    # Ordering restriction:
    # - Must run constant fold before first type inference pass, because
    # zeros, while allowed, violate typing constraints, and need to be killed
    # before the type inferrer sees them.
    # - Must run BC-to-flux before first type inferrer run so that zeros in
    # flux arguments can be removed.

    dumper("before-specializer", optemplate)
    optemplate = OperatorSpecializer(
            TypeInferrer()(optemplate)
            )(optemplate)

    # Ordering restrictions: 
    # - Must specialize quadrature operators before performing inverse mass
    # contraction, because there are no inverse-mass-contracted variants of the
    # quadrature operators.

    if mesh is not None:
        dumper("before-empty-flux-killer", optemplate)
        optemplate = EmptyFluxKiller(mesh)(optemplate)

    dumper("before-imass", optemplate)
    optemplate = InverseMassContractor()(optemplate)

    dumper("before-cfold-2", optemplate)
    optemplate = CommutativeConstantFoldingMapper()(optemplate)

    dumper("before-derivative-join", optemplate)
    optemplate = DerivativeJoiner()(optemplate)

    dumper("process-optemplate-finished", optemplate)

    return optemplate



# }}}

# {{{ pretty printing ---------------------------------------------------------
def pretty_print_optemplate(optemplate):
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




# vim: foldmethod=marker
