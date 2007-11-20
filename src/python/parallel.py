"""Parallelism support"""

__copyright__ = "Copyright (C) 2007 Andreas Kloeckner"

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





import pylinear.array as num
import pylinear.computation as comp
from pytools.arithmetic_container import work_with_arithmetic_containers
import hedge.discretization
import hedge.mesh




class ParallelizationContext(object):
    @property
    def rank(self):
        raise NotImplementedError

    @property
    def ranks(self):
        raise NotImplementedError

    @property
    def head_rank(self):
        raise NotImplementedError

    @property
    def is_head_rank(self):
        return self.rank == self.head_rank

    def distribute_mesh(self, mesh, partition=None):
        """Take the Mesh instance `mesh' and distribute it according to `partition'.

        If partition is an integer, invoke PyMetis to partition the mesh into this
        many parts, distributing over the first `partition' ranks.

        If partition is None, act as if partition was the integer corresponding
        to the current number of ranks on the job.

        If partition is not an integer, it must be a mapping from element number to 
        rank. (A list or tuple of rank numbers will do, for example, or so will
        a full-blown dict.)

        Returns a mesh chunk.

        We deliberately do not define the term `mesh chunk'. The return value
        of this function is to be treated as opaque by the user, only to be
        used as an argument to L{make_discretization}().

        This routine may only be invoked on the head rank.
        """
        raise NotImplementedError

    def receive_mesh(self):
        """Wait for a mesh chunk to be sent by the head rank.

        We deliberately do not define the term `mesh chunk'. The return value
        of this function is to be treated as opaque by the user, only to be
        used as an argument to L{make_discretization}().

        This routine should only be invoked on non-head ranks.
        """

        raise NotImplementedError

    def make_discretization(self, mesh_data, *args, **kwargs):
        """Construct a Discretization instance.

        `mesh_data' is whatever gets returned from distribute_mesh and
        receive_mesh(). Any extra arguments are directly forwarded to
        the respective Discretization constructor.
        """
        raise NotImplementedError





class SerialParallelizationContext(ParallelizationContext):
    @property
    def rank(self):
        return 0

    @property
    def ranks(self):
        return [0]

    @property
    def head_rank(self):
        return 0

    def distribute_mesh(self, mesh, partition=None):
        return mesh

    def make_discretization(self, mesh_data, *args, **kwargs):
        from hedge.discretization import Discretization

        return Discretization(mesh_data, *args, **kwargs)




class MPIParallelizationContext(ParallelizationContext):
    def __init__(self, communicator):
        self.communicator = communicator

    @property
    def rank(self):
        return self.communicator.rank

    @property
    def ranks(self):
        return range(0, self.communicator.size)

    @property
    def head_rank(self):
        return 0

    def distribute_mesh(self, mesh, partition=None):
        assert self.is_head_rank

        if partition is None:
            partition = len(self.ranks)

        # compute partition using Metis, if necessary
        if isinstance(partition, int):
            if partition == 1:
                return 

            from pymetis import part_graph
            cuts, partition = part_graph(partition, 
                    mesh.element_adjacency_graph)

        # find ranks to which we need to distribute
        target_ranks = set()
        for el in mesh.elements:
            target_ranks.add(partition[el.id])
        target_ranks = list(target_ranks)

        # prepare a mapping of elements to tags to speed up
        # copy_el_tagger, below
        el2tags = {}
        for tag, elements in mesh.tag_to_elements.iteritems():
            if tag == hedge.mesh.TAG_ALL:
                continue
            for el in elements:
                el2tags.setdefault(el, []).append(tag)

        # prepare a mapping of (el, face_nr) to boundary_tags
        # to speed up parallelizer_bdry_tagger, below
        elface2tags = {}
        for tag, elfaces in mesh.tag_to_boundary.iteritems():
            if tag == hedge.mesh.TAG_ALL:
                continue
            for el, fn in elfaces:
                elface2tags.setdefault((el, fn), []).append(tag)

        # prepare a mapping from (el, face_nr) to the rank
        # at the other end of the interface, if different from
        # current. concurrently, prepare a mapping 
        #  rank -> set([ranks that border me])
        elface2rank = {}
        neighboring_ranks = {}

        for elface1, elface2 in mesh.interfaces:
            e1, f1 = elface1
            e2, f2 = elface2
            r1 = partition[e1.id]
            r2 = partition[e2.id]

            if r1 != r2:
                neighboring_ranks.setdefault(r1, set()).add(r2)
                neighboring_ranks.setdefault(r2, set()).add(r1)

                elface2rank[elface1] = r2
                elface2rank[elface2] = r1

        # prepare a new mesh for each rank and send it
        import boost.mpi as mpi
        from hedge.mesh import TAG_NO_BOUNDARY

        for rank in target_ranks:
            rank_global_elements = [el 
                    for el in mesh.elements
                    if partition [el.id] == rank]

            # pick out this rank's vertices
            from pytools import flatten
            rank_global_vertex_indices = set(flatten(
                    el.vertex_indices for el in rank_global_elements))

            rank_local_vertices = [mesh.points[vi] 
                    for vi in rank_global_vertex_indices]

            # find global-to-local maps
            rank_global2local_vertex_indices = dict(
                    (gvi, lvi) for lvi, gvi in 
                    enumerate(rank_global_vertex_indices))

            rank_global2local_elements = dict(
                    (el.id, i) for i, el in 
                    enumerate(rank_global_elements))

            # find elements in local numbering
            rank_local_elements = [
                    [rank_global2local_vertex_indices[vi] 
                        for vi in el.vertex_indices]
                    for el in rank_global_elements]

            # make new local Mesh object, including 
            # boundary and element tagging
            def parallelizer_bdry_tagger(fvi, local_el, fn):
                el = rank_global_elements[local_el.id]

                result = elface2tags.get((el, fn), [])
                try:
                    opp_rank = elface2rank[el, fn]
                    result.append(hedge.mesh.TAG_RANK_BOUNDARY(opp_rank))

                    # keeps this part of the boundary from falling
                    # under the "None" tag.
                    result.append(TAG_NO_BOUNDARY)
                except KeyError:
                    pass

                return result

            def copy_el_tagger(local_el):
                return el2tags.get(rank_global_elements[local_el.id], [])

            def is_rankbdry_face((local_el, face_nr)):
                return (rank_global_elements[local_el.id], face_nr) in elface2rank

            from hedge.mesh import ConformalMesh
            rank_mesh = ConformalMesh(
                    rank_local_vertices,
                    rank_local_elements,
                    parallelizer_bdry_tagger, copy_el_tagger,
                    mesh.periodicity,
                    is_rankbdry_face)

            # assemble per-rank data
            rank_data = (
                    rank_mesh, 
                    rank_global2local_elements,
                    rank_global2local_vertex_indices,
                    neighboring_ranks[rank],
                    mesh.periodic_opposite_faces)

            if rank == self.head_rank:
                result = self, rank_data
            else:
                self.communicator.send(rank, 0, rank_data)

        return result

    def receive_mesh(self):
        return self, self.communicator.recv(self.head_rank, 0)

    def make_discretization(self, mesh_data, *args, **kwargs):
        return ParallelDiscretization(mesh_data, *args, **kwargs)




class ParallelDiscretization(hedge.discretization.Discretization):
    def __init__(self, mesh_data, local_discretization, 
            reorder=hedge.mesh.REORDER_CMK):
        self.received_bdrys = {}

        (self.context,
                (mesh, 
                    global2local_elements, 
                    self.global2local_vertex_indices, 
                    self.neighbor_ranks,
                    self.global_periodic_opposite_faces)) = \
                mesh_data

        from hedge.tools import reverse_lookup_table
        old_el_indices = mesh.reorder(reorder)

        if old_el_indices is None:
            self.global2local_elements = global2local_elements
        else:
            new_el_indices = reverse_lookup_table(old_el_indices)
            self.global2local_elements = dict(
                    (gi, new_el_indices[li])
                    for gi, li in global2local_elements.iteritems())

        hedge.discretization.Discretization.__init__(self,
                mesh, local_discretization, 
                reorder=hedge.mesh.REORDER_NONE)

        self._setup_from_neighbor_maps()

    def _setup_from_neighbor_maps(self):
        import boost.mpi as mpi

        comm = self.context.communicator

        if self.neighbor_ranks:
            # send interface information to neighboring ranks -----------------
            from pytools import reverse_dictionary
            local2global_vertex_indices = \
                    reverse_dictionary(self.global2local_vertex_indices)

            send_requests = mpi.RequestList()

            for rank in self.neighbor_ranks:
                rank_bdry = self.mesh.tag_to_boundary[
                        hedge.mesh.TAG_RANK_BOUNDARY(rank)]
                
                # a list of global vertex numbers for each face
                my_vertices_global = [
                        tuple(local2global_vertex_indices[vi]
                            for vi in el.faces[face_nr])
                        for el, face_nr in rank_bdry]

                # a list of node coordinates, indicating the order
                # in which nodal values will be sent, this is for
                # testing only and could (potentially) be omitted
                my_node_coords = []
                for el, face_nr in rank_bdry:
                    (estart, eend), ldis = self.find_el_data(el.id)
                    findices = ldis.face_indices()[face_nr]
                    my_node_coords.append(
                            [self.nodes[estart+i] for i in findices])
                
                packet = (my_vertices_global, my_node_coords)

                send_requests.append(comm.isend(rank, 0, packet))

            received_packets = {}
            while len(received_packets) < len(self.neighbor_ranks):
                received_packet, status = comm.recv(tag=0, return_status=True)
                received_packets[status.source] = received_packet

            mpi.wait_all(send_requests)

            # process received packages ---------------------------------------
            from pytools import flatten

            # nb_ stands for neighbor_

            self.from_neighbor_maps = {}

            for rank, (nb_all_facevertices_global, nb_node_coords) in \
                    received_packets.iteritems():
                rank_bdry = self.mesh.tag_to_boundary[
                        hedge.mesh.TAG_RANK_BOUNDARY(rank)]
                flat_nb_node_coords = list(flatten(nb_node_coords))

                # step 1: find start node indices for each 
                # of the neighbor's elements
                nb_face_starts = [0]
                for node_coords in nb_node_coords[:-1]:
                    nb_face_starts.append(
                            nb_face_starts[-1]+len(node_coords))

                # step 2: match faces by matching vertices
                nb_face_order = dict(
                        (frozenset(vertices), i)
                        for i, vertices in enumerate(nb_all_facevertices_global))

                # step 3: make a list of indices into the data we
                # receive from our neighbor that'll tell us how
                # to reshuffle them to match our node order
                from_indices = []

                for el, face_nr in rank_bdry:
                    (estart, eend), ldis = self.find_el_data(el.id)

                    my_vertices = el.faces[face_nr]
                    my_global_vertices = tuple(local2global_vertex_indices[vi]
                            for vi in my_vertices)

                    try:
                        nb_face_idx = nb_face_order[frozenset(my_global_vertices)]
                        # continue below in else part
                    except KeyError:
                        # ok, our face must be part of a periodic pair
                        my_vertices_there, axis = self.global_periodic_opposite_faces[
                                my_global_vertices]
                        nb_face_idx = nb_face_order[frozenset(my_vertices_there)]

                        his_vertices_here, axis2 = self.global_periodic_opposite_faces[
                                nb_all_facevertices_global[nb_face_idx]]

                        assert axis == axis2
                        
                        face_node_count = len(nb_node_coords[nb_face_idx])
                        nb_face_start = nb_face_starts[nb_face_idx]
                        nb_node_indices = range(
                                nb_face_start, nb_face_start+face_node_count)

                        shuffled_other_node_indices = \
                                ldis.shuffle_face_indices_to_match(
                                        my_global_vertices,
                                        his_vertices_here,
                                        nb_node_indices)

                        from_indices.extend(shuffled_other_node_indices)

                        # check if the nodes really match up
                        my_node_indices = [estart+i for i in ldis.face_indices()[face_nr]]

                        for my_i, other_i in zip(my_node_indices, shuffled_other_node_indices):
                            dist = self.nodes[my_i]-flat_nb_node_coords[other_i]
                            dist[axis] = 0
                            assert comp.norm_2(dist) < 1e-14
                    else:
                        # continue handling of nonperiodic case
                        nb_vertices = nb_all_facevertices_global[nb_face_idx]

                        face_node_count = len(nb_node_coords[nb_face_idx])
                        nb_face_start = nb_face_starts[nb_face_idx]
                        nb_node_indices = range(
                                nb_face_start, nb_face_start+face_node_count)

                        shuffled_other_node_indices = \
                                ldis.shuffle_face_indices_to_match(
                                        my_global_vertices,
                                        nb_vertices,
                                        nb_node_indices)

                        from_indices.extend(shuffled_other_node_indices)

                        # check if the nodes really match up
                        my_node_indices = [estart+i for i in ldis.face_indices()[face_nr]]

                        for my_i, other_i in zip(my_node_indices, shuffled_other_node_indices):
                            dist = self.nodes[my_i]-flat_nb_node_coords[other_i]
                            assert comp.norm_2(dist) < 1e-14

                assert len(from_indices) == len(flat_nb_node_coords)

                # turn from_indices into an IndexMap

                from hedge._internal import IndexMap
                self.from_neighbor_maps[rank] = IndexMap(
                        len(from_indices), len(from_indices), from_indices)

    def dt_non_geometric_factor(self):
        import boost.mpi as mpi
        from hedge.discretization import Discretization
        return mpi.all_reduce(self.context.communicator, 
                Discretization.dt_non_geometric_factor(self),
                min)

    def dt_geometric_factor(self):
        import boost.mpi as mpi
        from hedge.discretization import Discretization
        return mpi.all_reduce(self.context.communicator, 
                Discretization.dt_geometric_factor(self),
                min)

    def get_flux_operator(self, flux, direct=True):
        """Return a flux operator that can be multiplied with
        a volume field to obtain the lifted interior fluxes
        or with a boundary pair to obtain the lifted boundary
        flux.

        `direct' determines whether the operator is applied in a
        matrix-free fashion or uses precomputed matrices.
        """
        return _ParallelFluxOperator(self, flux, direct)




class _ParallelFluxOperator(object):
    def __init__(self, discr, flux, direct):
        self.discr = discr
        from hedge.discretization import Discretization
        self.serial_flux_op = Discretization.get_flux_operator(
                discr, flux, direct)

    def __mul__(self, field):
        from hedge.discretization import pair_with_boundary, BoundaryPair
        from pytools.arithmetic_container import ArithmeticList

        if isinstance(field, BoundaryPair):
            return self.serial_flux_op * field

        import boost.mpi as mpi
        from hedge._internal import irecv_vector, isend_vector

        comm = self.discr.context.communicator

        if isinstance(field, ArithmeticList):
            field_count = len(field)
            if field_count:
                if isinstance(field[0], BoundaryPair):
                    return self.serial_flux_op * field

        else:
            field_count = None

        def boundary_zeros(tag):
            if field_count is not None:
                return num.hstack([
                    self.discr.boundary_zeros(tag)
                    for field_i in range(field_count)])
            else:
                return self.discr.boundary_zeros(tag)

        def boundarize(field, tag):
            if field_count is not None:
                return num.hstack([
                    self.discr.boundarize_volume_field(component, tag)
                    for component in field])
            else:
                return self.discr.boundarize_volume_field(field, tag)

        def unstack(field, tag):
            if field_count is not None:
                bdry_len = len(self.discr._get_boundary(tag).nodes)
                return ArithmeticList(
                    field[i*bdry_len:(i+1)*bdry_len]
                    for i in range(field_count))
            else:
                return field

        # Subtlety here: The vectors for isend and irecv need to stay allocated
        # for as long as the request is not completed. The wrapper aids this
        # by making sure the vector outlives the request by using Boost.Python's
        # with_custodian_and_ward mechanism. However, this "life support" gets
        # eliminated if the only reference to the request is from inside a 
        # RequestList, so we need to provide our own life support for these vectors.

        neigh_recv_vecs = dict(
                (rank, boundary_zeros(hedge.mesh.TAG_RANK_BOUNDARY(rank)))
                for rank in self.discr.neighbor_ranks)

        recv_requests = mpi.RequestList(
                irecv_vector(comm, rank, 1, neigh_recv_vecs[rank])
                for rank in self.discr.neighbor_ranks)

        send_requests = [isend_vector(comm, rank, 1,
            boundarize(field, hedge.mesh.TAG_RANK_BOUNDARY(rank)))
            for rank in self.discr.neighbor_ranks]

        result = [self.serial_flux_op * field]

        def receive(_, status):
            from hedge.tools import apply_index_map

            tag = hedge.mesh.TAG_RANK_BOUNDARY(status.source)

            foreign_order_bfield = unstack(neigh_recv_vecs[status.source], tag)

            bfield = apply_index_map(
                    self.discr.from_neighbor_maps[status.source],
                    foreign_order_bfield)

            result[0] += self.serial_flux_op * \
                    pair_with_boundary(field, bfield, tag)

        mpi.wait_all(recv_requests, receive)
        mpi.wait_all(mpi.RequestList(send_requests))
        return result[0]





def guess_parallelization_context():
    try:
        import boost.mpi as mpi

        if mpi.size == 1:
            return SerialParallelizationContext()
        else:
            return MPIParallelizationContext(mpi.world)
    except ImportError:
        return SerialParallelizationContext()




@work_with_arithmetic_containers
def reassemble_volume_field(pcon, global_discr, local_discr, field):
    from pytools import reverse_dictionary
    local2global_element = reverse_dictionary(
            local_discr.global2local_elements)

    send_packet = {}
    for eg in local_discr.element_groups:
        for el, (estart, eend) in zip(eg.members, eg.ranges):
            send_packet[local2global_element[el.id]] = field[estart:eend]

    def reduction(a, b):
        a.update(b)
        return a

    import boost.mpi as mpi

    gfield_parts = mpi.reduce(
            pcon.communicator, send_packet, reduction, pcon.head_rank)

    if pcon.is_head_rank:
        result = global_discr.volume_zeros()
        for eg in global_discr.element_groups:
            for el, (estart, eend) in zip(eg.members, eg.ranges):
                my_part = gfield_parts[el.id]
                assert len(my_part) == eend-estart
                result[estart:eend] = my_part
        return result
    else:
        return None
