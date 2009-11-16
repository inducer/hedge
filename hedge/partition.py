"""Mesh partitioning subsystem.

This is used by parallel execution (MPI) and local timestepping.
"""

from __future__ import division

__copyright__ = "Copyright (C) 2009 Andreas Kloeckner"

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
import numpy.linalg as la
import pytools
from pytools import memoize_method
import hedge.mesh
import hedge.optemplate




class PartitionData(pytools.Record):
    def __init__(self,
            part_nr,
            mesh,
            global2local_elements,
            global2local_vertex_indices,
            neighbor_parts,
            global_periodic_opposite_faces,
            part_boundary_tags
            ):
        pytools.Record.__init__(self, locals())




def partition_from_tags(mesh, tag_to_number):
    partition = numpy.zeros((len(mesh.elements),), dtype=numpy.int32)

    for tag, number in tag_to_number.iteritems():
        for el in mesh.tag_to_elements[tag]:
            partition[el.id] += number

    return partition




def partition_mesh(mesh, partition, part_bdry_tag_factory):
    """*partition* is a mapping that maps element id to
    integers that represent different pieces of the mesh.

    For historical reasons, the values in partition are called
    'parts'.
    """

    # Find parts to which we need to distribute.
    all_parts = list(set(
        partition[el.id] for el in mesh.elements))

    # Prepare a mapping of elements to tags to speed up
    # copy_el_tagger, below.
    el2tags = {}
    for tag, elements in mesh.tag_to_elements.iteritems():
        if tag == hedge.mesh.TAG_ALL:
            continue
        for el in elements:
            el2tags.setdefault(el, []).append(tag)

    # prepare a mapping of (el, face_nr) to boundary_tags
    # to speed up partition_bdry_tagger, below
    elface2tags = {}
    for tag, elfaces in mesh.tag_to_boundary.iteritems():
        if tag == hedge.mesh.TAG_ALL:
            continue
        for el, fn in elfaces:
            elface2tags.setdefault((el, fn), []).append(tag)

    # prepare a mapping from (el, face_nr) to the part
    # at the other end of the interface, if different from
    # current. concurrently, prepare a mapping
    #  part -> set([parts that border me])
    elface2part = {}
    neighboring_parts = {}

    for elface1, elface2 in mesh.interfaces:
        e1, f1 = elface1
        e2, f2 = elface2
        r1 = partition[e1.id]
        r2 = partition[e2.id]

        if r1 != r2:
            neighboring_parts.setdefault(r1, set()).add(r2)
            neighboring_parts.setdefault(r2, set()).add(r1)

            elface2part[elface1] = r2
            elface2part[elface2] = r1

    # prepare a new mesh for each part and send it
    from hedge.mesh import TAG_NO_BOUNDARY

    for part in all_parts:
        part_global_elements = [el
                for el in mesh.elements
                if partition[el.id] == part]

        # pick out this part's vertices
        from pytools import flatten
        part_global_vertex_indices = set(flatten(
                el.vertex_indices for el in part_global_elements))

        part_local_vertices = [mesh.points[vi]
                for vi in part_global_vertex_indices]

        # find global-to-local maps
        part_global2local_vertex_indices = dict(
                (gvi, lvi) for lvi, gvi in
                enumerate(part_global_vertex_indices))

        part_global2local_elements = dict(
                (el.id, i) for i, el in
                enumerate(part_global_elements))

        # find elements in local numbering
        part_local_elements = [
                [part_global2local_vertex_indices[vi]
                    for vi in el.vertex_indices]
                for el in part_global_elements]

        # make new local Mesh object, including
        # boundary and element tagging
        def partition_bdry_tagger(fvi, local_el, fn, all_vertices):
            el = part_global_elements[local_el.id]

            result = elface2tags.get((el, fn), [])
            try:
                opp_part = elface2part[el, fn]
                result.append(part_bdry_tag_factory(opp_part))

                # keeps this part of the boundary from falling
                # under TAG_ALL.
                result.append(TAG_NO_BOUNDARY)

            except KeyError:
                pass

            return result

        def copy_el_tagger(local_el, all_vertices):
            return el2tags.get(part_global_elements[local_el.id], [])

        def is_partbdry_face((local_el, face_nr)):
            return (part_global_elements[local_el.id], face_nr) in elface2part

        from hedge.mesh import make_conformal_mesh
        part_mesh = make_conformal_mesh(
                part_local_vertices,
                part_local_elements,
                partition_bdry_tagger, copy_el_tagger,
                mesh.periodicity,
                is_partbdry_face)

        # assemble per-part data

        my_nb_parts = neighboring_parts.get(part, [])
        yield PartitionData(
                part,
                part_mesh,
                part_global2local_elements,
                part_global2local_vertex_indices,
                my_nb_parts,
                mesh.periodic_opposite_faces,
                part_boundary_tags=dict(
                    (nb_part, part_bdry_tag_factory(nb_part))
                    for nb_part in my_nb_parts),
                )





def find_neighbor_vol_indices(
        my_discr, my_part_data,
        nb_discr, nb_part_data,
        debug=False):

    from pytools import reverse_dictionary
    l2g_vertex_indices = \
            reverse_dictionary(my_part_data.global2local_vertex_indices)
    nb_l2g_vertex_indices = \
            reverse_dictionary(nb_part_data.global2local_vertex_indices)

    my_bdry_tag = my_part_data.part_boundary_tags[nb_part_data.part_nr]
    nb_bdry_tag = nb_part_data.part_boundary_tags[my_part_data.part_nr]

    my_mesh_bdry = my_part_data.mesh.tag_to_boundary[my_bdry_tag]
    nb_mesh_bdry = nb_part_data.mesh.tag_to_boundary[nb_bdry_tag]

    my_discr_bdry = my_discr.get_boundary(my_bdry_tag)
    nb_discr_bdry = nb_discr.get_boundary(nb_bdry_tag)

    nb_vertices_to_face = dict(
            (frozenset(el.faces[face_nr]), (el, face_nr))
            for el, face_nr
            in nb_mesh_bdry)

    from_indices = []

    shuffled_indices_cache = {}

    def get_shuffled_indices(face_node_count, shuffle_op):
        try:
            return shuffled_indices_cache[shuffle_op]
        except KeyError:
            unshuffled_indices = range(face_node_count)
            result = shuffled_indices_cache[shuffle_op] = \
                    shuffle_op(unshuffled_indices)
            return result

    for my_el, my_face_nr in my_mesh_bdry:
        eslice, ldis = my_discr.find_el_data(my_el.id)

        my_vertices = my_el.faces[my_face_nr]
        my_global_vertices = tuple(l2g_vertex_indices[vi]
                for vi in my_vertices)

        face_node_count = ldis.face_node_count()
        try:
            nb_vertices = frozenset(
                    nb_part_data.global2local_vertex_indices[vi]
                    for vi in my_global_vertices)
            # continue below in else part
        except KeyError:
            # this happens if my_global_vertices is not a permutation
            # of the neighbor's face vertices. Periodicity is the only
            # reason why that would be so.
            my_global_vertices_there, axis = my_part_data.global_periodic_opposite_faces[
                    my_global_vertices]

            nb_vertices = frozenset(
                    nb_part_data.global2local_vertex_indices[vi]
                    for vi in my_global_vertices_there)

            nb_el, nb_face_nr = nb_vertices_to_face[nb_vertices]
            nb_global_vertices_there = tuple(
                    nb_l2g_vertex_indices[vi]
                    for vi in nb_el.faces[nb_face_nr])

            nb_global_vertices, axis2 = nb_part_data.global_periodic_opposite_faces[
                    nb_global_vertices_there]

            assert axis == axis2

            nb_face_start = nb_discr_bdry \
                    .find_facepair((nb_el, nb_face_nr)) \
                    .opp.el_base_index

            shuffle_op = \
                    ldis.get_face_index_shuffle_to_match(
                            my_global_vertices,
                            nb_global_vertices)

            shuffled_nb_node_indices = [nb_face_start+i
                    for i in get_shuffled_indices(face_node_count, shuffle_op)]

            from_indices.extend(shuffled_nb_node_indices)

            # check if the nodes really match up
            if debug and ldis.has_facial_nodes:
                my_node_indices = [eslice.start+i for i in ldis.face_indices()[my_face_nr]]

                for my_i, nb_i in zip(my_node_indices, shuffled_nb_node_indices):
                    dist = my_discr.nodes[my_i]-nb_discr_bdry.nodes[nb_i]
                    dist[axis] = 0
                    assert la.norm(dist) < 1e-14
        else:
            # continue handling of nonperiodic case
            nb_el, nb_face_nr = nb_vertices_to_face[nb_vertices]
            nb_global_vertices = tuple(
                    nb_l2g_vertex_indices[vi]
                    for vi in nb_el.faces[nb_face_nr])

            nb_face_start = nb_discr_bdry \
                    .find_facepair((nb_el, nb_face_nr)) \
                    .opp.el_base_index

            shuffle_op = \
                    ldis.get_face_index_shuffle_to_match(
                            my_global_vertices,
                            nb_global_vertices)

            shuffled_nb_node_indices = [nb_face_start+i
                    for i in get_shuffled_indices(face_node_count, shuffle_op)]

            from_indices.extend(shuffled_nb_node_indices)

            # Check if the nodes really match up
            if debug and ldis.has_facial_nodes:
                my_node_indices = [eslice.start+i
                        for i in ldis.face_indices()[my_face_nr]]

                for my_i, nb_i in zip(my_node_indices, shuffled_nb_node_indices):
                    dist = my_discr.nodes[my_i]-nb_discr_bdry.nodes[nb_i]
                    assert la.norm(dist) < 1e-14

        # Finally, unify FluxFace.h values across boundary.
        my_flux_face = my_discr_bdry.find_facepair_side((my_el, my_face_nr))
        nb_flux_face = nb_discr_bdry.find_facepair_side((nb_el, nb_face_nr))
        my_flux_face.h = nb_flux_face.h = max(my_flux_face.h, nb_flux_face.h)

    assert len(from_indices) \
            == len(my_discr_bdry.nodes) \
            == len(nb_discr_bdry.nodes)

    # Convert nb's boundary indices to nb's volume indices.
    return nb_discr_bdry.vol_indices[
            numpy.asarray(from_indices, dtype=numpy.intp)]




class StupidInterdomainFluxMapper(hedge.optemplate.IdentityMapper):
    """Attempts to map a regular optemplate into one that is
    suitable for inter-domain flux computation.

    Maps everything to zero that is not an interior flux or
    inverse mass operator. Interior fluxes on the other hand are
    mapped to boundary fluxes on the specified tag.
    """

    def __init__(self, bdry_tag, vol_var, bdry_val_var):
        self.bdry_tag = bdry_tag
        self.vol_var = vol_var
        self.bdry_val_var = bdry_val_var

    def map_operator_binding(self, expr):
        from hedge.optemplate import \
                FluxOperatorBase, \
                BoundaryPair, \
                OperatorBinding, \
                IdentityMapperMixin, \
                InverseMassOperator

        from pymbolic.mapper.substitutor import SubstitutionMapper

        class FieldIntoBdrySubstitutionMapper(
                SubstitutionMapper,
                IdentityMapperMixin):
            def map_normal(self, expr):
                return expr

        if isinstance(expr.op, FluxOperatorBase):
            if isinstance(expr.field, BoundaryPair):
                return 0
            else:
                # Finally, an interior flux. Rewrite it.

                def subst_func(expr):
                    if expr == self.vol_var:
                        return self.bdry_val_var
                    else:
                        return None

                return OperatorBinding(expr.op,
                        BoundaryPair(
                            expr.field,
                            SubstitutionMapper(subst_func)(expr.field),
                        self.bdry_tag))
        elif isinstance(expr.op, InverseMassOperator):
            return OperatorBinding(expr.op, self.rec(expr.field))
        else:
            return 0




def compile_interdomain_flux(optemplate, vol_var, bdry_var,
        my_discr, my_part_data,
        nb_discr, nb_part_data,
        use_stupid_substitution=False):
    """
    `use_stupid_substitution` uses `StupidInterdomainFluxMapper` to
    try to pare down a full optemplate to one that is suitable for
    interdomain flux computation. While technique is stupid, it
    will work for many common DG operators. See the description of
    `StupidInterdomainFluxMapper` to see what exactly is done.
    """

    from hedge.optemplate import make_field

    neighbor_indices = find_neighbor_vol_indices(
            my_discr, my_part_data,
            nb_discr, nb_part_data,
            debug="node_permutation" in my_discr.debug | nb_discr.debug)

    my_bdry_tag = my_part_data.part_boundary_tags[nb_part_data.part_nr]

    kwargs = {}
    if use_stupid_substitution:
        kwargs = {"post_bind_mapper": StupidInterdomainFluxMapper(
                my_bdry_tag, make_field(vol_var), make_field(bdry_var))}

    return my_discr.compile(optemplate, **kwargs), neighbor_indices




class Transformer:
    def __init__(self, whole_discr, parts_data, parts_discr):
        self.whole_discr = whole_discr
        self.parts_data = parts_data
        self.parts_discr = parts_discr

    @memoize_method
    def _embeddings(self):
        result = []
        for part_data, part_discr in zip(self.parts_data, self.parts_discr):
            part_emb = numpy.zeros((len(part_discr),), dtype=numpy.intp)
            result.append(part_emb)

            for g_el, l_el in part_data.global2local_elements.iteritems():
                g_slice = self.whole_discr.find_el_range(g_el)
                part_emb[part_discr.find_el_range(l_el)] = \
                        numpy.arange(g_slice.start, g_slice.stop)
        return result

    def reassemble(self, parts_vol_vectors):
        from pytools import single_valued, indices_in_shape
        from hedge.tools import log_shape
        ls = single_valued(log_shape(pvv) for pvv in parts_vol_vectors)

        def remap_scalar_field(idx):
            result = self.whole_discr.volume_zeros()
            for part_emb, part_vol_vector in zip(
                    self._embeddings(), parts_vol_vectors):
                result[part_emb] = part_vol_vector[idx]

            return result

        if ls != ():
            result = numpy.zeros(ls, dtype=object)
            for i in indices_in_shape(ls):
                result[i] = remap_scalar_field(i)
            return result
        else:
            return remap_scalar_field(())

    def split(self, whole_vol_vector):
        from pytools import indices_in_shape
        from hedge.tools import log_shape

        ls = log_shape(whole_vol_vector)

        if ls != ():
            result = [numpy.zeros(ls, dtype=object)
                    for part_emb in self._embeddings()]
            for p, part_emb in enumerate(self._embeddings()):
                for i in indices_in_shape(ls):
                    result[p][i] = whole_vol_vector[part_emb]
            return result
        else:
            return [whole_vol_vector[part_emb]
                    for part_emb in self._embeddings()]
