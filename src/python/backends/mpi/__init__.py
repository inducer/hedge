"""Parallelism support"""

from __future__ import division

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




import pytools
import numpy
import numpy.linalg as la
import hedge.discretization
import hedge.mesh
from hedge.optemplate import \
        IdentityMapper, \
        FluxOpReducerMixin
from hedge.backends import RunContext
from hedge.partition import PartitionData




class RankData(pytools.Record):
    def __init__(self, 
            mesh, 
            global2local_elements,
            global2local_vertex_indices,
            neighbor_ranks,
            global_periodic_opposite_faces,
            old_el_numbers=None
            ):
        pytools.Record.__init__(self, locals())

    def reordered_by(self, *args, **kwargs):
        old_el_numbers = self.mesh.get_reorder_oldnumbers(*args, **kwargs)
        mesh = self.mesh.reordered(old_el_numbers)
        return self.copy(
                mesh=mesh,
                old_el_numbers=old_el_numbers
                )




class MPIRunContext(RunContext):
    def __init__(self, communicator, discr_class):
        self.communicator = communicator
        self.discr_class = discr_class

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
            from pymetis import part_graph
            dummy, partition = part_graph(partition, 
                    mesh.element_adjacency_graph())

        from hedge.partition import partition_mesh
        from hedge.mesh import TAG_RANK_BOUNDARY
        for part_data in partition_mesh(
                mesh, partition, part_bdry_tag_factory=TAG_RANK_BOUNDARY):

            rank_data = RankData(
                    mesh=part_data.mesh,
                    global2local_elements=part_data.global2local_elements,
                    global2local_vertex_indices=part_data.global2local_vertex_indices,
                    neighbor_ranks=part_data.neighbor_parts,
                    global_periodic_opposite_faces=part_data.global_periodic_opposite_faces)
            
            rank = part_data.part_nr

            if rank == self.head_rank:
                result = rank_data
            else:
                print "send rank", rank
                self.communicator.send(rank, 0, rank_data)
                print "end send", rank

        return result

    def receive_mesh(self):
        return self.communicator.recv(self.head_rank, 0)
        print "receive end rank", self.rank

    def make_discretization(self, mesh_data, *args, **kwargs):
        return ParallelDiscretization(self, self.discr_class, mesh_data, *args, **kwargs)




def make_custom_exec_mapper_class(superclass):
    class ExecutionMapper(superclass):
        def __init__(self, context, executor):
            superclass.__init__(self, context, executor)
            self.discr = executor.discr

        # actual functionality ----------------------------------------------------
        def map_flux_send(self, op, field_expr):
            import boostmpi as mpi
            from hedge.tools import log_shape, is_obj_array

            pdiscr = self.discr.parallel_discr
            comm = pdiscr.context.communicator

            if self.discr.instrumented:
                self.discr.parallel_discr.flux_send_timer.start()
                self.discr.parallel_discr.comm_flux_counter.add(
                        len(pdiscr.neighbor_ranks))

            field = self.rec(field_expr)
            shp = log_shape(field)

            # Subtlety here: The vectors for isend and irecv need to stay allocated
            # for as long as the request is not completed. The wrapper aids this
            # by making sure the vector outlives the request by using Boost.Python's
            # with_custodian_and_ward mechanism. However, this "life support" gets
            # eliminated if the only reference to the request is from inside a 
            # RequestList, so we need to provide our own life support for these vectors.

            neigh_recv_vecs = dict(
                    (rank, pdiscr.boundary_empty(
                        hedge.mesh.TAG_RANK_BOUNDARY(rank),
                        shape=shp,
                        kind="numpy",
                        dtype=self.discr.default_scalar_type))
                    for rank in pdiscr.neighbor_ranks)

            from hedge._internal import irecv_buffer, isend_buffer
            recv_requests = mpi.RequestList(
                    irecv_buffer(comm, rank, tag=1, vector=neigh_recv_vecs[rank])
                    for rank in pdiscr.neighbor_ranks)

            def flatten_and_convert_array(ary):
                if is_obj_array(ary):
                    result = numpy.empty(shp+ary[0].shape,
                            dtype=self.discr.default_scalar_type)
                    for i in range(shp[0]):
                        result[i,:] = ary[i]
                    return result
                else:
                    return numpy.asarray(ary, 
                            dtype=self.discr.default_scalar_type)

            from hedge.mesh import TAG_RANK_BOUNDARY
            neigh_send_vecs = [
                    flatten_and_convert_array(pdiscr.convert_boundary(
                        pdiscr.boundarize_volume_field(
                            field, 
                            TAG_RANK_BOUNDARY(rank)),
                        TAG_RANK_BOUNDARY(rank),
                        kind="numpy"))
                    for rank in pdiscr.neighbor_ranks]

            send_requests = [isend_buffer(comm, rank, tag=1, vector=nsv)
                for rank, nsv in zip(
                    pdiscr.neighbor_ranks,
                    neigh_send_vecs)]

            class CommunicationRecord(pytools.Record):
                pass

            return CommunicationRecord(
                    neigh_recv_vecs=neigh_recv_vecs,
                    neigh_send_vecs=neigh_send_vecs,
                    recv_requests=recv_requests,
                    send_requests=send_requests,
                    shape=shp,
                    )

            if self.discr.instrumented:
                self.discr.parallel_discr.flux_send_timer.stop()

        def exec_flux_receive_batch_assign(self, efrba):
            if self.discr.instrumented:
                self.discr.parallel_discr.flux_recv_timer.start()

            import boostmpi as mpi

            pdiscr = self.discr.parallel_discr
            comm_record = self.rec(efrba.field)

            rank_to_index_and_name = {}
            for name, (index, rank) in zip(
                    efrba.names, efrba.indices_and_ranks):
                rank_to_index_and_name.setdefault(rank, []).append(
                    (index, name))

            recv_req = comm_record.recv_requests

            from hedge.mesh import TAG_RANK_BOUNDARY
            while len(recv_req):
                value, status, index = mpi.wait_any(recv_req)
                del recv_req[index]

                received_vec = comm_record.neigh_recv_vecs[status.source]

                fnm = pdiscr.from_neighbor_maps[status.source]
                for idx, name in rank_to_index_and_name[status.source]:
                    self.context[name] = self.discr.convert_boundary(
                            received_vec[idx, fnm],
                            TAG_RANK_BOUNDARY(status.source),
                            kind=self.discr.compute_kind)

            mpi.wait_all(mpi.RequestList(comm_record.send_requests))

            if self.discr.instrumented:
                self.discr.parallel_discr.flux_recv_timer.stop()

    return ExecutionMapper




class FluxCommunicationInserter(
        IdentityMapper, 
        FluxOpReducerMixin):
    def __init__(self, interacting_ranks):
        self.interacting_ranks = interacting_ranks

    def map_operator_binding(self, expr):
        from hedge.optemplate import \
                FluxOperatorBase, \
                FluxCoefficientOperatorBase, \
                BoundaryPair, OperatorBinding, \
                FluxSendOperator, FluxReceiveOperator

        if isinstance(expr, OperatorBinding):
            if isinstance(expr.op, FluxCoefficientOperatorBase):
                raise ValueError("flux coefficient operators not supported for MPI")

            if isinstance(expr.op, FluxOperatorBase):
                if isinstance(expr.field, BoundaryPair):
                    # we're only worried about internal fluxes
                    return IdentityMapper.map_operator_binding(self, expr)

                # by now we've narrowed it down to a bound interior flux
                from pymbolic.primitives import \
                        flattened_sum, \
                        CommonSubexpression

                def func_and_cse(func, formal_fields, arg_fields):
                    from hedge.tools import is_obj_array, make_obj_array
                    if is_obj_array(arg_fields):
                        return make_obj_array([
                            CommonSubexpression(
                                OperatorBinding(
                                    func(i),
                                    formal_fields))
                                for i in range(len(arg_fields))])
                    else:
                        return CommonSubexpression(
                                OperatorBinding(
                                    func(()),
                                    formal_fields))
                    
                from hedge.tools import with_object_array_or_scalar
                from hedge.mesh import TAG_RANK_BOUNDARY

                formal_sent_fields = CommonSubexpression(
                        OperatorBinding(
                            FluxSendOperator(), 
                            expr.field))

                def receive_and_cse(rank):
                    return func_and_cse(
                            lambda i: FluxReceiveOperator(i, rank),
                            formal_sent_fields,
                            expr.field)

                return flattened_sum([expr]
                    + [OperatorBinding(expr.op, BoundaryPair(
                        expr.field, 
                        # This boundary field has no true data dependency--
                        # it is received from a different rank.
                        # Instead of field data, the "formal sent fields" 
                        # contain communication handles.
                        receive_and_cse(rank),
                        TAG_RANK_BOUNDARY(rank)))
                        for rank in self.interacting_ranks])
            else:
                return IdentityMapper.map_operator_binding(self, expr)




class ParallelDiscretization(object):
    @classmethod
    def my_debug_flags(cls):
        return set([
            "parallel_setup", 
            ])

    @classmethod
    def all_debug_flags(cls, subcls):
        return cls.my_debug_flags() | subcls.all_debug_flags()

    def __init__(self, rcon, subdiscr_class, rank_data, *args, **kwargs):
        debug = set(kwargs.pop("debug", set()))
        self.debug = self.my_debug_flags() & debug
        kwargs["debug"] = debug - self.debug
        kwargs["run_context"] = rcon

        self.subdiscr = subdiscr_class(rank_data.mesh, *args, **kwargs)
        self.subdiscr.exec_mapper_class = make_custom_exec_mapper_class(
                self.subdiscr.exec_mapper_class)
        self.subdiscr.parallel_discr = self

        self.received_bdrys = {}
        self.context = rcon

        self.global2local_vertex_indices = rank_data.global2local_vertex_indices 
        self.neighbor_ranks = rank_data.neighbor_ranks
        self.global_periodic_opposite_faces = rank_data.global_periodic_opposite_faces

        if rank_data.old_el_numbers is not None:
            from hedge.tools import reverse_lookup_table
            new_el_numbers = reverse_lookup_table(rank_data.old_el_numbers)
            self.global2local_elements = dict(
                    (gi, new_el_numbers[li])
                    for gi, li in rank_data.global2local_elements.iteritems())
        else:
            self.global2local_elements = rank_data.global2local_elements 

        self._setup_neighbor_connections()

    def add_instrumentation(self, mgr):
        self.subdiscr.add_instrumentation(mgr)

        from pytools.log import IntervalTimer, EventCounter
        self.comm_flux_counter = EventCounter("n_comm_flux", 
                "Number of inner flux communication runs")
        self.flux_send_timer = IntervalTimer("t_flux_send", 
                "Time spent sending flux data")
        self.flux_recv_timer = IntervalTimer("t_flux_recv", 
                "Time spent waiting for and receiving flux data")

        mgr.add_quantity(self.comm_flux_counter)
        mgr.add_quantity(self.flux_send_timer)
        mgr.add_quantity(self.flux_recv_timer)

    # property forwards -------------------------------------------------------
    def __len__(self):
        return len(self.subdiscr)

    def __getattr__(self, name):
        if not name.startswith("_"):
            return getattr(self.subdiscr, name)
        else:
            raise AttributeError(name)

    # neighbor connectivity ---------------------------------------------------
    def _setup_neighbor_connections(self):
        import boostmpi as mpi

        comm = self.context.communicator

        if self.neighbor_ranks:
            # send interface information to neighboring ranks -----------------
            from pytools import reverse_dictionary
            local2global_vertex_indices = \
                    reverse_dictionary(self.global2local_vertex_indices)

            send_requests = mpi.RequestList()

            for rank in self.neighbor_ranks:
                bdry_tag = hedge.mesh.TAG_RANK_BOUNDARY(rank)
                rank_bdry = self.subdiscr.mesh.tag_to_boundary[bdry_tag]
                rank_discr_boundary = self.subdiscr.get_boundary(bdry_tag)

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
                    eslice, ldis = self.subdiscr.find_el_data(el.id)
                    findices = ldis.face_indices()[face_nr]

                    my_node_coords.append(
                            [self.nodes[eslice.start+i] for i in findices])

                # compile a list of FluxFace.h values for unification
                # across the rank boundary

                my_h_values = [rank_discr_boundary.find_facepair_side(el_face).h 
                        for el_face in rank_bdry]
                
                packet = (my_vertices_global, my_node_coords, my_h_values)

                send_requests.append(comm.isend(rank, 0, packet))

            received_packets = {}
            while len(received_packets) < len(self.neighbor_ranks):
                received_packet, status = comm.recv(tag=0, return_status=True)
                received_packets[status.source] = received_packet

            mpi.wait_all(send_requests)

            # process received packets ----------------------------------------
            from pytools import flatten

            # nb_ stands for neighbor_

            self.from_neighbor_maps = {}

            for rank, (nb_all_facevertices_global, nb_node_coords, nb_h_values) in \
                    received_packets.iteritems():
                bdry_tag = hedge.mesh.TAG_RANK_BOUNDARY(rank)
                rank_bdry = self.subdiscr.mesh.tag_to_boundary[bdry_tag]
                rank_discr_boundary = self.subdiscr.get_boundary(bdry_tag)

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

                shuffled_indices_cache = {}

                def get_shuffled_indices(face_node_count, shuffle_op):
                    try:
                        return shuffled_indices_cache[shuffle_op]
                    except KeyError:
                        unshuffled_indices = range(face_node_count)
                        result = shuffled_indices_cache[shuffle_op] = \
                                shuffle_op(unshuffled_indices)
                        return result

                for el, face_nr in rank_bdry:
                    eslice, ldis = self.subdiscr.find_el_data(el.id)

                    my_vertices = el.faces[face_nr]
                    my_global_vertices = tuple(local2global_vertex_indices[vi]
                            for vi in my_vertices)

                    face_node_count = ldis.face_node_count()
                    try:
                        nb_face_idx = nb_face_order[frozenset(my_global_vertices)]
                        # continue below in else part
                    except KeyError:
                        # this happens if my_global_vertices is not a permutation 
                        # of the neighbor's face vertices. Periodicity is the only 
                        # reason why that would be so.
                        my_vertices_there, axis = self.global_periodic_opposite_faces[
                                my_global_vertices]
                        nb_face_idx = nb_face_order[frozenset(my_vertices_there)]

                        his_vertices_here, axis2 = self.global_periodic_opposite_faces[
                                nb_all_facevertices_global[nb_face_idx]]

                        assert axis == axis2
                        
                        nb_face_start = nb_face_starts[nb_face_idx]

                        shuffle_op = \
                                ldis.get_face_index_shuffle_to_match(
                                        my_global_vertices,
                                        his_vertices_here) 

                        shuffled_other_node_indices = [nb_face_start+i 
                                for i in get_shuffled_indices(
                                    face_node_count, shuffle_op)]

                        from_indices.extend(shuffled_other_node_indices)

                        # check if the nodes really match up
                        if "parallel_setup" in self.debug:
                            my_node_indices = [eslice.start+i for i in ldis.face_indices()[face_nr]]

                            for my_i, other_i in zip(my_node_indices, shuffled_other_node_indices):
                                dist = self.nodes[my_i]-flat_nb_node_coords[other_i]
                                dist[axis] = 0
                                assert la.norm(dist) < 1e-14
                    else:
                        # continue handling of nonperiodic case
                        nb_global_vertices = nb_all_facevertices_global[nb_face_idx]

                        nb_face_start = nb_face_starts[nb_face_idx]

                        shuffle_op = \
                                ldis.get_face_index_shuffle_to_match(
                                        my_global_vertices,
                                        nb_global_vertices)

                        shuffled_other_node_indices = [nb_face_start+i 
                                for i in get_shuffled_indices(
                                    face_node_count, shuffle_op)]

                        from_indices.extend(shuffled_other_node_indices)

                        # check if the nodes really match up
                        if "parallel_setup" in self.debug:
                            my_node_indices = [eslice.start+i 
                                    for i in ldis.face_indices()[face_nr]]

                            for my_i, other_i in zip(my_node_indices, shuffled_other_node_indices):
                                dist = self.nodes[my_i]-flat_nb_node_coords[other_i]
                                assert la.norm(dist) < 1e-14

                    # finally, unify FluxFace.h values across boundary
                    nb_h = nb_h_values[nb_face_idx]
                    flux_face = rank_discr_boundary.find_facepair_side((el, face_nr))
                    flux_face.h = max(nb_h, flux_face.h)

                if "parallel_setup" in self.debug:
                    assert len(from_indices) == len(flat_nb_node_coords)

                # construct from_neighbor_map
                self.from_neighbor_maps[rank] = numpy.asarray(
                        from_indices, dtype=numpy.intp)

    # norm and integral -------------------------------------------------------
    def norm(self, volume_vector, p=2):
        import boostmpi as mpi

        def add_norms(x, y):
            return (x**p + y**p)**(1/p)

        return mpi.all_reduce(self.context.communicator, 
                self.subdiscr.norm(volume_vector, p),
                add_norms)

    def integral(self, volume_vector):
        import boostmpi as mpi
        from operator import add
        return mpi.all_reduce(self.context.communicator, 
                self.subdiscr.integral(volume_vector),
                add)

    # dt estimation -----------------------------------------------------------
    def dt_non_geometric_factor(self):
        import boostmpi as mpi
        return mpi.all_reduce(self.context.communicator, 
                self.subdiscr.dt_non_geometric_factor(),
                min)

    def dt_geometric_factor(self):
        import boostmpi as mpi
        return mpi.all_reduce(self.context.communicator, 
                self.subdiscr.dt_geometric_factor(),
                min)

    def dt_factor(self, max_system_ev):
        return 1/max_system_ev \
                * self.dt_non_geometric_factor() \
                * self.dt_geometric_factor()

    # compilation -------------------------------------------------------------
    def compile(self, optemplate, post_bind_mapper=lambda x:x ):
        fci = FluxCommunicationInserter(self.neighbor_ranks)
        return self.subdiscr.compile(
                optemplate,
                post_bind_mapper=lambda x: fci(post_bind_mapper(x)))
        




def reassemble_volume_field(rcon, global_discr, local_discr, field):
    from pytools import reverse_dictionary
    local2global_element = reverse_dictionary(
            local_discr.global2local_elements)

    send_packet = {}
    for eg in local_discr.element_groups:
        for el, eslice in zip(eg.members, eg.ranges):
            send_packet[local2global_element[el.id]] = field[eslice]

    def reduction(a, b):
        a.update(b)
        return a

    import boostmpi as mpi

    gfield_parts = mpi.reduce(
            rcon.communicator, send_packet, reduction, rcon.head_rank)

    if rcon.is_head_rank:
        result = global_discr.volume_zeros()
        for eg in global_discr.element_groups:
            for el, eslice in zip(eg.members, eg.ranges):
                my_part = gfield_parts[el.id]
                assert len(my_part) == eslice.stop-eslice.start
                result[eslice] = my_part
        return result
    else:
        return None
