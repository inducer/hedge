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




# convenience functions -------------------------------------------------------
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




# optemplate tools ------------------------------------------------------------
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



# process optemplate ----------------------------------------------------------
def process_optemplate(optemplate, post_bind_mapper=None,
        dumper=lambda name, optemplate: None, mesh=None):

    from hedge.optemplate import (
            OperatorBinder, BCToFluxRewriter, CommutativeConstantFoldingMapper,
            EmptyFluxKiller, InverseMassContractor, DerivativeJoiner)

    dumper("before-bind", optemplate)
    optemplate = OperatorBinder()(optemplate)

    if post_bind_mapper is not None:
        dumper("before-postbind", optemplate)
        optemplate = post_bind_mapper(optemplate)

    dumper("before-bc2flux", optemplate)
    optemplate = BCToFluxRewriter()(optemplate)
    dumper("before-cfold", optemplate)
    optemplate = CommutativeConstantFoldingMapper()(optemplate)
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
