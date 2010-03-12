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
import hedge.discretization
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pytools import memoize_method, Record
from hedge.tools.futures import Future




# {{{ gpu data organization ---------------------------------------------------
class GPUBlock(object):
    """Describes what data is local to each thread block on the GPU.

    @ivar number: The global number of this block.
    @ivar local_discretization: The L{hedge.discretization.local.LocalDiscretization} 
      instance used for elements in this block.
    @ivar cpu_slices: A list of slices describing the CPU-side
      storage locations for the block's elements.
    @ivar microblocks: A list of lists of L{hedge.mesh.Element} instances,
      each representing the elements in one block, and together representing
      one block.
    @ivar el_offsets_list: A lsit containing the offsets of the elements in
      this block, in order.
    @ivar el_number_map: A dictionary mapping L{hedge.mesh.Element} instances
      to their number within this block.
    """
    __slots__ = ["number", "local_discretization", "cpu_slices", "microblocks",
            "el_offsets_list", "el_number_map", "el_number_map"
            ]

    def __init__(self, number, local_discretization, cpu_slices, microblocks,
            el_offsets_list, el_number_map):
        self.number = number
        self.local_discretization = local_discretization
        self.cpu_slices = cpu_slices
        self.microblocks = microblocks
        self.el_offsets_list = el_offsets_list
        self.el_number_map = el_number_map





class GPUFaceStorage(object):
    """Describes where the dofs of an element face are stored.

    @ivar opposite: the L{GPUFaceStorage} instance for the face
      oposite to this one.
    """
    __slots__ = ["opposite"]

    def __init__(self):
        self.opposite = None

    def set_opposite(self, opp):
        if self.opposite is None:
            self.opposite = opp
        else:
            assert self.opposite is opp

class GPUInteriorFaceStorage(GPUFaceStorage):
    """Describes storage locations for a face local to an element in a block.

    :ivar el_face: a tuple *(element, face_number)*.
    :ivar cpu_slice: the base index of the element in CPU numbering.
    :ivar native_index_list_id:
    :ivar ext_write_index_list_id:
    :ivar native_block: block in which element is to be found.
    :ivar face_pair_side:
    """
    __slots__ = [
            "el_face", "cpu_slice",
            "native_index_list_id", "ext_write_index_list_id",
            "global_int_flux_index_list_id", "global_ext_flux_index_list_id",
            "native_block",
            "face_pair_side"]

    def __init__(self, el_face, cpu_slice, native_index_list_id,
            native_block, face_pair_side):
        GPUFaceStorage.__init__(self)
        self.el_face = el_face
        self.cpu_slice = cpu_slice
        self.native_index_list_id = native_index_list_id
        self.native_block = native_block
        self.face_pair_side = face_pair_side

class GPUBoundaryFaceStorage(GPUFaceStorage):
    """Describes storage locations for a boundary face.

    @ivar cpu_bdry_index_in_floats: this face's starting index
      in the CPU-based TAG_ALL boundary array [floats].
    @ivar gpu_bdry_index_in_floats: this face's starting index
      in the GPU-based TAG_ALL boundary array [floats].
    @ivar face_pair_side:
    """
    __slots__ = [
            "cpu_bdry_index_in_floats",
            "gpu_bdry_index_in_floats",
            "face_pair_side",
            ]

    def __init__(self,
            cpu_bdry_index_in_floats,
            gpu_bdry_index_in_floats,
            face_pair_side
            ):
        GPUFaceStorage.__init__(self)
        self.cpu_bdry_index_in_floats = cpu_bdry_index_in_floats
        self.gpu_bdry_index_in_floats = gpu_bdry_index_in_floats
        self.face_pair_side = face_pair_side

# }}}




# {{{ GPU mesh partition ------------------------------------------------------
def make_gpu_partition_greedy(adjgraph, max_block_size):

    def first(iterable):
        it = iter(iterable)
        try:
            return it.next()
        except StopIteration:
            return None

    def bfs(top_node):
        queue = [top_node]

        result = set()

        def num_connections_to_result(node):
            return sum(1 for rn in result if node in adjgraph[rn])

        from pytools import argmax2

        while True:
            curr_node_idx = argmax2((i, num_connections_to_result(qn))
                    for i, qn in enumerate(queue))

            curr_node = queue.pop(curr_node_idx)

            if curr_node in avail_nodes:
                avail_nodes.remove(curr_node)
                result.add(curr_node)
                if len(result) == max_block_size:
                    return result, first(node for node in queue if node in avail_nodes)

                queue.extend(adjgraph[curr_node])

            if not queue:
                # ran out of nodes in immediate vicinity -- add new ones from elsewhere
                if avail_nodes:
                    queue.append(iter(avail_nodes).next())
                else:
                    return result, None

    avail_nodes = set(adjgraph.iterkeys())
    next_node = None

    partition = [0]*len(adjgraph)

    blocks = []
    while avail_nodes:
        if next_node is None:
            from pytools import argmax2
            next_node = argmax2((node, len(adjgraph[node])) for node in avail_nodes)

        block, next_node = list(bfs(next_node))

        for el in block:
            partition[el] = len(blocks)
        blocks.append(block)

    return partition, blocks




def make_gpu_partition_metis(adjgraph, max_block_size):
    from pymetis import part_graph

    orig_part_count = part_count = len(adjgraph)//max_block_size+1

    attempt_count = 5
    for attempt in range(attempt_count):
        if part_count > 1:
            cuts, partition = part_graph(part_count,
                    adjgraph, vweights=[1000]*len(adjgraph))
        else:
            # metis bug workaround:
            # metis returns ones (instead of zeros) if part_count == 1
            partition = [0]*len(adjgraph)

        blocks = [[] for i in range(part_count)]
        for el_id, block in enumerate(partition):
            blocks[block].append(el_id)
        block_elements = max(len(block_els) for block_els in blocks)

        if block_elements <= max_block_size:
            return partition, blocks

        part_count += min(5, int(part_count*0.01))

    from warnings import warn

    warn("could not achieve Metis partition after %d attempts, falling back to greedy"
            % attempt_count)

    return make_gpu_partition_greedy(adjgraph, max_block_size)

# }}}




# GPU discretization ----------------------------------------------------------
class Discretization(hedge.discretization.Discretization):
    from hedge.backends.cuda.execute import ExecutionMapper \
            as exec_mapper_class
    from hedge.backends.cuda.execute import Executor \
            as executor_class

    # {{{ debug flags ---------------------------------------------------------
    @classmethod
    def all_debug_flags(cls):
        return hedge.discretization.Discretization.all_debug_flags() | set([
            "cuda_ilist_generation",
            "cuda_compare",
            "cuda_diff",
            "cuda_diff_plan",
            "cuda_flux",
            "cuda_lift",
            "cuda_gather_plan",
            "cuda_lift_plan",
            "cuda_debugbuf",
            "cuda_memory",
            "cuda_dump_kernels",
            "cuda_fastbench",
            "cuda_no_microblock",
            "cuda_no_smem_matrix",
            "cuda_no_plan",
            "cuda_no_plan_el_local",
            "cuda_keep_kernels",
            "cuda_try_no_microblock",
            "cuda_plan_log",
            "cuda_plan_no_progress",
            "cuda_no_metis",
            ])

    # }}}

    # {{{ partitioning --------------------------------------------------------
    class PartitionData(Record):
        pass

    def _get_partition_data(self, max_block_size):
        try:
            return self.partition_cache[max_block_size]
        except KeyError:
            pass

        try:
            import pymetis
            metis_available = True
        except ImportError:
            metis_available = False

        if "cuda_no_metis" in self.debug:
            metis_available = False

        if max_block_size >= 10 and metis_available:
            partition, blocks = make_gpu_partition_metis(
                    self.mesh.element_adjacency_graph(),
                    max_block_size)
        else:
            partition, blocks = make_gpu_partition_greedy(
                    self.mesh.element_adjacency_graph(),
                    max_block_size)

        # prepare a mapping:  block# -> # of external interfaces
        block2extifaces = dict((i, 0) for i in range(len(blocks)))

        for (e1, f1), (e2, f2) in self.mesh.both_interfaces():
            b1 = partition[e1.id]
            b2 = partition[e2.id]

            if b1 != b2:
                block2extifaces[b1] += 1

        for el, face_nbr in self.mesh.tag_to_boundary[hedge.mesh.TAG_REALLY_ALL]:
            b1 = partition[el.id]
            block2extifaces[b1] += 1

        eg, = self.element_groups

        max_facepairs = 0
        int_face_pair_count = 0
        face_pair_count = 0

        for b in range(len(blocks)):
            b_ext_faces = block2extifaces[b]
            b_int_faces = (len(blocks[b])*eg.local_discretization.face_count()
                    - b_ext_faces)
            assert b_int_faces % 2 == 0
            b_facepairs = b_int_faces//2 + b_ext_faces

            int_face_pair_count += b_int_faces//2
            max_facepairs = max(max_facepairs, b_facepairs)
            face_pair_count += b_facepairs

        from pytools import average

        result = self.PartitionData(
                partition=partition,
                blocks=blocks,
                max_face_pair_count=max_facepairs,
                ext_face_avg=average(
                    block2extifaces.itervalues()),
                int_face_pair_avg=int_face_pair_count/len(blocks),
                face_pair_avg=face_pair_count/len(blocks),
                )

        self.partition_cache[max_block_size] = result
        return result

    # }}}

    # {{{ construction and destruction ----------------------------------------
    def __init__(self, mesh, local_discretization=None,
            order=None, quad_min_degrees={},
            init_cuda=True, debug=set(),
            default_scalar_type=numpy.float32,
            tune_for=None, run_context=None,
            mpi_cuda_dev_filter=lambda dev: True):
        """

        :param tune_for: An optemplate for whose application this discretization's
        flux plan will be tuned.
        """

        if quad_min_degrees:
            raise NotImplementedError("CUDA backend doesn't support quadrature yet")

        if not isinstance(mesh, hedge.mesh.Mesh):
            raise TypeError("mesh must be of type hedge.mesh.Mesh")

        if tune_for is None:
            from warnings import warn
            warn("You can achieve better performance if you pass an optemplate "
                    "in the tune_for= kwarg.")

        # {{{ initialize superclass
        ldis = self.get_local_discretization(mesh, local_discretization, order)

        hedge.discretization.Discretization.__init__(self, mesh, ldis, debug=debug,
                default_scalar_type=default_scalar_type,
                run_context=run_context)
        # }}}

        # {{{ cuda init
        self.cleanup_context = None
        if init_cuda:
            cuda.init()

            if run_context is None or len(run_context.ranks) == 1:
                from pycuda.tools import make_default_context
                self.cleanup_context = make_default_context()
            else:
                from hedge.backends.cuda.tools import mpi_get_default_device
                device = mpi_get_default_device(run_context.communicator,
                        dev_filter=mpi_cuda_dev_filter)
                self.cleanup_context = device.make_context()

        # if not init_cuda, assume we're in an active context
        self.device = cuda.Context.get_device()

        from pycuda.tools import DeviceData

        # }}}

        # {{{ initialize memory pool
        if "cuda_memory" in self.debug:
            from pycuda.tools import DebugMemoryPool
            if run_context is not None and run_context.ranks > 1:
                self.pool = DebugMemoryPool(
                        interactive=False,
                        logfile=open("rank-%d-mem.log" % run_context.rank, "w")
                        )
            else:
                self.pool = DebugMemoryPool(
                        interactive=False,
                        logfile=open("mem.log", "w"))
        else:
            from pycuda.tools import DeviceMemoryPool
            self.pool = DeviceMemoryPool()

        from pycuda.tools import PageLockedMemoryPool
        self.pagelocked_pool = PageLockedMemoryPool()

        # }}}

        # {{{ generate flux plan
        self.partition_cache = {}

        from pytools import Record
        class OverallPlan(Record):
            pass

        def generate_overall_plans():
            if "cuda_no_microblock" in self.debug:
                allow_mb_values = [False]
            elif "cuda_try_no_microblock" in self.debug:
                allow_mb_values = [True, False]
            else:
                allow_mb_values = [True]

            for allow_mb in allow_mb_values:
                from hedge.backends.cuda.plan import PlanGivenData
                given = PlanGivenData(
                        DeviceData(self.device), ldis,
                        allow_microblocking=allow_mb,
                        float_type=default_scalar_type)

                import hedge.backends.cuda.fluxgather as fluxgather

                # copy from enclosing scope
                my_tune_for = tune_for
                if my_tune_for is not None:
                    from hedge.optemplate.mappers import QuadratureUpsamplerRemover
                    my_tune_for = QuadratureUpsamplerRemover(self.quad_min_degrees) \
                            (tune_for)

                flux_plan, flux_time = fluxgather.make_plan(self, given, my_tune_for)

                # partition mesh, obtain updated plan
                pdata = self._get_partition_data(
                        flux_plan.elements_per_block())
                given.post_decomposition(
                        block_count=len(pdata.blocks),
                        microblocks_per_block=flux_plan.microblocks_per_block())

                # plan local operations
                from hedge.backends.cuda.plan import \
                        make_diff_plan, \
                        make_element_local_plan

                diff_plan, diff_time = make_diff_plan(self, given)
                fluxlocal_plan, fluxlocal_time = make_element_local_plan(
                        self, given,
                        op_name="lift",
                        aligned_preimage_dofs_per_microblock=
                            given.aligned_face_dofs_per_microblock(),
                        preimage_dofs_per_el=given.face_dofs_per_el(),
                        with_index_check=False)

                sys_size = flux_plan.flux_count
                total_time = flux_time + sys_size*(diff_time+fluxlocal_time)

                yield OverallPlan(
                    given=given,
                    flux_plan=flux_plan,
                    partition=pdata.partition,
                    diff_plan=diff_plan,
                    fluxlocal_plan=fluxlocal_plan), total_time

        overall_plans = list(generate_overall_plans())

        for op, total_time in overall_plans:
            print "microblocking=%s -> time=%g" % (
                    op.given.microblock.elements != 1, total_time)

        from pytools import argmin2
        overall_plan = argmin2(overall_plans)

        self.given = overall_plan.given
        self.flux_plan = overall_plan.flux_plan
        self.partition = overall_plan.partition
        self.diff_plan = overall_plan.diff_plan
        self.fluxlocal_plan = overall_plan.fluxlocal_plan

        print "actual flux exec plan:", self.flux_plan
        print "actual diff op exec plan:", self.diff_plan
        print "actual flux local exec plan:", self.fluxlocal_plan

        # }}}

        # {{{ build data structures
        self.blocks = self._build_blocks()
        self.face_storage_map = self._build_face_storage_map()
        # }}}

        # {{{ make a CPU reference discretization
        if "cuda_compare" in self.debug:
            from hedge.backends.jit import Discretization
            self.test_discr = Discretization(mesh, ldis)
        # }}}

        self.stream_pool = []

    def close(self):
        del self.stream_pool

        self.pool.stop_holding()
        self.pagelocked_pool.stop_holding()
        if self.cleanup_context is not None:
            self.cleanup_context.pop()

    # }}}

    # {{{ setup ---------------------------------------------------------------
    def _build_blocks(self):
        block_el_numbers = {}
        for el_id, block in enumerate(self.partition):
            block_el_numbers.setdefault(block, []).append(el_id)

        block_count = len(block_el_numbers)

        def make_block(block_num):
            given = self.given

            microblocks = []
            current_microblock = []
            el_offsets_list = []
            el_number_map = {}
            elements = [self.mesh.elements[ben]
                    for ben in block_el_numbers.get(block_num, [])]
            for block_el_nr, el in enumerate(elements):
                el_offset = (
                        len(microblocks)*given.microblock.aligned_floats
                        + len(current_microblock)*given.dofs_per_el())
                el_number_map[el] = block_el_nr
                el_offsets_list.append(el_offset)

                current_microblock.append(el)
                if len(current_microblock) == given.microblock.elements:
                    microblocks.append(current_microblock)
                    current_microblock = []

            if current_microblock:
                microblocks.append(current_microblock)

            assert len(microblocks) <= self.flux_plan.microblocks_per_block()

            eg, = self.element_groups
            return GPUBlock(block_num,
                    local_discretization=eg.local_discretization,
                    cpu_slices=[self.find_el_range(el.id) for el in elements],
                    microblocks=microblocks,
                    el_offsets_list=el_offsets_list,
                    el_number_map=el_number_map)

        return [make_block(block_num) for block_num in range(block_count)]




    def _build_face_storage_map(self):
        # Side effects:
        # - fill in GPUBlock.extfaces
        # - set self.aligned_boundary_floats
        fsm = {}

        from hedge.tools import IndexListRegistry
        fil_registry = IndexListRegistry("cuda_ilist_generation" in self.debug)

        def make_int_face(face_pair_side):
            el = self.mesh.elements[face_pair_side.element_id]
            elface = (el, face_pair_side.face_id)

            block = self.blocks[self.partition[el.id]]
            iln = fil_registry.register(
                    (ldis, face_pair_side.face_id),
                    lambda: ldis.face_indices()[face_pair_side.face_id]
                    )
            result = GPUInteriorFaceStorage(
                elface,
                cpu_slice=self.find_el_range(el.id),
                native_index_list_id=iln,
                native_block=block,
                face_pair_side=face_pair_side
                )

            assert elface not in fsm
            fsm[elface] = result
            return result

        int_fg, = self.face_groups
        ldis = int_fg.ldis_loc
        assert ldis == int_fg.ldis_opp

        id_face_index_list_number = fil_registry.register(
                None,
                lambda: tuple(xrange(ldis.face_node_count()))
                )
        assert id_face_index_list_number == 0

        for fp in int_fg.face_pairs:
            face1 = make_int_face(fp.int_side)
            face2 = make_int_face(fp.ext_side)
            face1.opposite = face2
            face2.opposite = face1

            def apply_write_map(wmap, sequence):
                result = [None] * len(sequence)
                for wm_i, seq_i in zip(wmap, sequence):
                    result[wm_i] = seq_i
                assert None not in result
                return tuple(result)

            f_ind = ldis.face_indices()

            def face1_in_el_ilist():
                return tuple(int_fg.index_lists[
                    fp.int_side.face_index_list_number])

            def face2_in_el_ilist():
                return tuple(int_fg.index_lists[
                    fp.ext_side.face_index_list_number])

            def ext_write_map():
                return tuple(
                        int_fg.index_lists[fp.ext_native_write_map])

            face1.global_int_flux_index_list_id = fil_registry.register(
                    (int_fg, fp.int_side.face_index_list_number),
                    face1_in_el_ilist)
            face1.global_ext_flux_index_list_id = fil_registry.register(
                    (int_fg, fp.ext_side.face_index_list_number),
                    face2_in_el_ilist)

            face2.global_int_flux_index_list_id = fil_registry.register(
                    (int_fg, fp.ext_native_write_map,
                        fp.ext_side.face_index_list_number),
                    lambda: apply_write_map(
                        ext_write_map(), face2_in_el_ilist())
                    )
            face2.global_ext_flux_index_list_id = fil_registry.register(
                    (int_fg, fp.ext_native_write_map,
                        fp.int_side.face_index_list_number),
                    lambda: apply_write_map(
                        ext_write_map(), face1_in_el_ilist())
                    )

            from pytools import get_write_to_map_from_permutation as gwtm
            #assert gwtm(face2_in_el_ilist, f_ind[fp.ext_side.face_id]) == ext_write_map
            face1.ext_write_index_list_id = fil_registry.register(
                    (int_fg, "wtm", fp.ext_side.face_index_list_number,
                        fp.ext_side.face_id),
                    lambda: gwtm(face2_in_el_ilist(), f_ind[fp.ext_side.face_id])
                    )
            face2.ext_write_index_list_id = fil_registry.register(
                    (int_fg, "wtm",
                        fp.ext_native_write_map,
                        fp.int_side.face_index_list_number,
                        fp.int_side.face_id),
                    lambda: gwtm(
                        apply_write_map(ext_write_map(), face1_in_el_ilist()),
                        f_ind[fp.int_side.face_id])
                    )

        self.aligned_boundary_floats = 0
        from hedge.mesh import TAG_REALLY_ALL

        for bdry_fg in self.get_boundary(TAG_REALLY_ALL).face_groups:
            if bdry_fg.ldis_loc is None:
                assert len(bdry_fg.face_pairs) == 0
                continue

            assert ldis == bdry_fg.ldis_loc

            aligned_fnc = self.given.devdata.align_dtype(ldis.face_node_count(),
                    self.given.float_size())
            for fp in bdry_fg.face_pairs:
                assert fp.ext_side.element_id == hedge._internal.INVALID_ELEMENT
                #assert (tuple(bdry_fg.index_lists[fp.ext_side.face_index_list_number])
                        #== id_face_index_list)

                face1 = make_int_face(fp.int_side)
                face2 = GPUBoundaryFaceStorage(
                        fp.ext_side.el_base_index,
                        self.aligned_boundary_floats,
                        fp.ext_side
                        )
                self.aligned_boundary_floats += aligned_fnc
                face1.opposite = face2
                face2.opposite = face1

                face1.global_int_flux_index_list_id = fil_registry.register(
                        (bdry_fg,fp.int_side.face_index_list_number),
                        lambda: tuple(bdry_fg.index_lists[
                            fp.int_side.face_index_list_number])
                        )
                face1.global_ext_flux_index_list_id = fil_registry.register(
                        (bdry_fg, fp.ext_side.face_index_list_number),
                        lambda: tuple(bdry_fg.index_lists[
                            fp.ext_side.face_index_list_number])
                        )

        self.index_lists = fil_registry.index_lists
        return fsm

    # }}}

    # {{{ stream pooling ------------------------------------------------------
    # (stupid CUDA isn't smart enough to allocate streams without synchronizing.
    # sigh.)
    def _get_stream(self):
        if not self.stream_pool:
            return cuda.Stream()
        else:
            return self.stream_pool.pop()

    def _release_stream(self, s):
        self.stream_pool.append(s)

    # }}}

    # {{{ instrumentation -----------------------------------------------------
    def add_instrumentation(self, mgr):
        mgr.set_constant("flux_plan", str(self.flux_plan))
        mgr.set_constant("diff_plan", str(self.diff_plan))
        mgr.set_constant("fluxlocal_plan", str(self.fluxlocal_plan))

        from pytools.log import EventCounter

        self.gmem_bytes_gather = EventCounter("gmem_bytes_gather",
                "Bytes of gmem traffic during gather")
        self.gmem_bytes_el_local = EventCounter("gmem_bytes_el_local",
                "Bytes of gmem traffic during element-local matrix application")
        self.gmem_bytes_diff = EventCounter("gmem_bytes_diff",
                "Bytes of gmem traffic during lift")
        self.gmem_bytes_vector_math = EventCounter("gmem_bytes_vector_math",
                "Bytes of gmem traffic during vector math")
        self.gmem_bytes_rk4 = EventCounter("gmem_bytes_rk4",
                "Bytes of gmem traffic during RK4")

        mgr.add_quantity(self.gmem_bytes_gather)
        mgr.add_quantity(self.gmem_bytes_el_local)
        mgr.add_quantity(self.gmem_bytes_diff)
        mgr.add_quantity(self.gmem_bytes_vector_math)
        mgr.add_quantity(self.gmem_bytes_rk4)

        hedge.discretization.Discretization.add_instrumentation(self, mgr)

    def create_op_timers(self):
        self.flux_gather_timer = self.run_context.make_timer(
                "t_gather",
                "Time spent gathering fluxes")
        self.el_local_timer = self.run_context.make_timer(
                "t_el_local",
                "Time spent applying element-local matrices (lift, mass)")
        self.diff_op_timer = self.run_context.make_timer(
                "t_diff",
                "Time spent applying applying differentiation operators")
        self.vector_math_timer = self.run_context.make_timer(
                "t_vector_math", 
                "Time spent doing vector math")

        return [self.flux_gather_timer,
                self.el_local_timer,
                self.diff_op_timer,
                self.vector_math_timer ]

    # }}}

    # {{{ utilities -----------------------------------------------------------
    def find_el_gpu_index(self, el):
        given = self.given
        block = self.blocks[self.partition[el.id]]

        mb_nr, in_mb_nr = divmod(block.el_number_map[el], given.microblock.elements)

        return (block.number * self.flux_plan.dofs_per_block()
                + mb_nr*given.microblock.aligned_floats
                + in_mb_nr*given.dofs_per_el())

    def find_number_in_block(self, el):
        block = self.blocks[self.partition[el.id]]
        return block.el_number_map[el]

    @memoize_method
    def gpu_dof_count(self):
        from hedge.backends.cuda.tools import int_ceiling

        fplan = self.flux_plan
        return int_ceiling(
                int_ceiling(
                    fplan.dofs_per_block() * len(self.blocks),
                    self.diff_plan.dofs_per_macroblock()),
                self.fluxlocal_plan.dofs_per_macroblock())

    @memoize_method
    def _gpu_volume_embedding(self):
        result = numpy.zeros((len(self.nodes),), dtype=numpy.intp)
        block_offset = 0
        block_size = self.flux_plan.dofs_per_block()
        for block in self.blocks:
            el_length = block.local_discretization.node_count()

            for el_offset, cpu_slice in zip(
                    block.el_offsets_list, block.cpu_slices):
                result[cpu_slice] = \
                        block_offset+el_offset+numpy.arange(el_length)

            block_offset += block_size

        assert (result <= self.gpu_dof_count()).all()

        return result

    @memoize_method
    def _meaningful_volume_indices(self):
        return gpuarray.to_gpu(
                numpy.asarray(
                    numpy.sort(self._gpu_volume_embedding()),
                    dtype=numpy.uint32),
                allocator=self.pool.allocate)

    def _volume_to_gpu(self, field, dtype):
        def f(subfld):
            if dtype is not None:
                subfld = subfld.astype(dtype)

            cpu_transfer = self.pagelocked_pool.allocate(
                    (self.gpu_dof_count(),), dtype=subfld.dtype)

            cpu_transfer[self._gpu_volume_embedding()] = subfld
            return gpuarray.to_gpu(cpu_transfer, allocator=self.pool.allocate)

        from hedge.tools import with_object_array_or_scalar
        return with_object_array_or_scalar(f, field)

    def _volume_from_gpu(self, field, dtype=None):
        def f(subfld):
            if dtype is not None:
                subfld = subfld.astype(dtype)

            return subfld.get(pagelocked=True)[self._gpu_volume_embedding()]

        from hedge.tools import with_object_array_or_scalar
        return with_object_array_or_scalar(f, field)

    @memoize_method
    def _gpu_boundary_embedding(self, tag):
        """Return an array of indices embedding a CPU boundary
        field for *tag* into the GPU boundary field."""

        bdry = self.get_boundary(tag)
        result = numpy.empty(
                (len(bdry.nodes),),
                dtype=numpy.intp)
        result.fill(-1)

        cpu_base = 0
        for elface in self.mesh.tag_to_boundary.get(tag, []):
            face_stor = self.face_storage_map[elface]
            bdry_stor = face_stor.opposite
            assert isinstance(bdry_stor, GPUBoundaryFaceStorage)

            face_len = (bdry_stor.opposite.native_block
                    .local_discretization.face_node_count())
            gpu_base = bdry_stor.gpu_bdry_index_in_floats
            result[cpu_base:cpu_base+face_len] = \
                    numpy.arange(gpu_base, gpu_base+face_len)
            cpu_base += face_len

        assert (result>=0).all()
        return result

    @memoize_method
    def _gpu_boundary_embedding_on_gpu(self, tag):
        return gpuarray.to_gpu(
                numpy.asarray(
                    self._gpu_boundary_embedding(tag),
                    dtype=numpy.uint32))

    class _BoundaryToGPUFuture(Future):
        def __init__(self, discr, field, tag, read_map=None):
            self.discr = discr

            from hedge.tools import log_shape

            ls = log_shape(field)
            if ls == ():
                field_list = [field]
                n = 1
            else:
                field_list = field
                n, = ls

            one_field = field_list[0]
            one_field_size = len(one_field)

            if field.dtype == object:
                self.buf = buf = discr.pagelocked_pool.allocate(
                        ls+one_field.shape, dtype=self.discr.default_scalar_type)
                for i, subf in enumerate(field):
                    assert subf.flags.c_contiguous
                    buf[i, :] = subf
            else:
                assert field.flags.c_contiguous
                buf = field

            self.stream = discr._get_stream()

            buf.shape = buf.size,
            try:
                self.buf_gpu = buf_gpu = gpuarray.to_gpu_async(
                        buf, discr.pool.allocate, self.stream)
            except cuda.LogicError:
                # buf is not pagelocked
                self.buf_gpu = buf_gpu = gpuarray.to_gpu(buf, discr.pool.allocate)

            from hedge.tools import make_obj_array
            out = make_obj_array([
                    discr.boundary_empty(tag) for i in range(n)])

            if one_field_size:
                if read_map is None:
                    gpuarray.multi_put(
                            arrays=[
                                buf_gpu[i*one_field_size:(i+1)*one_field_size]
                                for i in range(n)],
                            dest_indices=discr._gpu_boundary_embedding_on_gpu(tag),
                            out=out, stream=self.stream)
                else:
                    gpuarray.multi_take_put(
                            arrays=[buf_gpu for i in range(n)],
                            dest_indices=discr._gpu_boundary_embedding_on_gpu(tag),
                            src_indices=read_map,
                            src_offsets=[i*one_field_size for i in range(n)],
                            out=out, stream=self.stream)

            if ls == ():
                self.result = out[0]
            else:
                self.result = out

        def is_ready(self):
            return self.stream.is_done()

        def __call__(self):
            self.stream.synchronize()
            self.discr._release_stream(self.stream)
            return self.result

    def _boundary_from_gpu(self, field, tag, dtype):
        def f(field):
            if dtype is not None:
                field = field.astype(dtype)

            return field.get()[self._gpu_boundary_embedding(tag)]

        from hedge.tools import with_object_array_or_scalar
        return with_object_array_or_scalar(f, field)

    def convert_volume(self, field, kind, dtype=None):
        orig_kind = self.get_kind(field)

        if kind == "numpy" and orig_kind == "gpu":
            return self._volume_from_gpu(field, dtype)
        elif kind == "gpu" and orig_kind == "numpy":
            return self._volume_to_gpu(field, dtype)
        else:
            return hedge.discretization.Discretization.convert_volume(
                    self, field, kind)

    def convert_boundary(self, field, tag, kind, dtype=None):
        orig_kind = self.get_kind(field)

        if kind == "numpy" and orig_kind == "gpu":
            return self._boundary_from_gpu(field, tag, dtype)
        elif kind == "gpu" and orig_kind == "numpy":
            if dtype is not None:
                from hedge.tools import cast_field
                field = cast_field(field, dtype)

            return self._BoundaryToGPUFuture(self, field, tag)()
        else:
            return hedge.discretization.Discretization.convert_boundary(
                    self, field, tag, kind)

    def convert_boundary_async(self, field, tag, kind, read_map=None):
        orig_kind = self.get_kind(field)

        if kind == "gpu" and orig_kind == "numpy":
            return self._BoundaryToGPUFuture(self, field, tag, read_map)
        else:
            return hedge.discretization.Discretization.convert_boundary_async(
                    self, field, tag, kind, read_map)

    # }}}

    # {{{ vector construction tools -------------------------------------------
    def _empty_gpuarray(self, shape, dtype):
        return gpuarray.empty(shape, dtype=dtype,
                allocator=self.pool.allocate)

    def _zeros_gpuarray(self, shape, dtype):
        result = gpuarray.empty(shape, dtype=dtype,
                allocator=self.pool.allocate)
        result.fill(0)
        return result

    def _new_vec(self, shape, create_func, dtype, base_size):
        if dtype is None:
            dtype = self.default_scalar_type

        if shape == ():
            return create_func((base_size,), dtype=dtype)

        result = numpy.empty(shape, dtype=object)
        from pytools import indices_in_shape
        for i in indices_in_shape(shape):
            result[i] = create_func((base_size,), dtype=dtype)
        return result

    # }}}

    # {{{ vector construction -------------------------------------------------
    compute_kind = "gpu"

    def get_kind(self, field):
        if isinstance(field, gpuarray.GPUArray):
            return "gpu"

        from hedge.tools import log_shape
        from pytools import indices_in_shape

        first_field = field[iter(indices_in_shape(log_shape(field))).next()]
        if isinstance(first_field, numpy.ndarray):
            return "numpy"
        elif isinstance(first_field, gpuarray.GPUArray):
            return "gpu"
        else:
            raise TypeError, "invalid field kind"

    def volume_empty(self, shape=(), dtype=None, kind="gpu"):
        if kind != "gpu":
            return hedge.discretization.Discretization.volume_empty(
                    self, shape, dtype, kind)

        return self._new_vec(shape, self._empty_gpuarray, dtype,
                self.gpu_dof_count())

    def volume_zeros(self, shape=(), dtype=None, kind="gpu"):
        if kind != "gpu":
            return hedge.discretization.Discretization.volume_zeros(
                    self, shape, dtype, kind)

        return self._new_vec(shape, self._zeros_gpuarray, dtype,
                self.gpu_dof_count())

    def boundary_empty(self, tag, shape=(), dtype=None, kind="gpu"):
        if kind == "gpu":
            return self._new_vec(shape, self._empty_gpuarray, dtype,
                    self.aligned_boundary_floats)
        elif kind == "numpy-mpi-recv":
            return self.pagelocked_pool.allocate(
                    shape+(len(self.get_boundary(tag).nodes),),
                    dtype)
        else:
            return hedge.discretization.Discretization.boundary_empty(
                    self, tag, shape, dtype, kind)


    def boundary_zeros(self, tag, shape=(), dtype=None, kind="gpu"):
        if kind == "gpu":
            return self._new_vec(shape, self._zeros_gpuarray, dtype,
                    self.aligned_boundary_floats)
        elif kind == "numpy-mpi-recv":
            result = self.pagelocked_pool.allocate(
                    shape+(len(self.get_boundary(tag).nodes),),
                    dtype)
            result.fill(0)
            return result
        else:
            return hedge.discretization.Discretization.boundary_zeros(
                    self, tag, shape, dtype, kind)

    def volumize_boundary_field(self, bfield, tag):
        if self.get_kind(bfield) != "gpu":
            return hedge.discretization.Discretization.volumize_boundary_field(
                    self, bfield, tag)

        raise NotImplementedError

    @memoize_method
    def _boundarize_info(self, tag):
        from_indices = []
        to_indices = []

        for elface in self.mesh.tag_to_boundary.get(tag, []):
            vol_face = self.face_storage_map[elface]
            bdry_face = vol_face.opposite
            assert isinstance(bdry_face, GPUBoundaryFaceStorage)

            vol_el_index = \
                    self.find_el_gpu_index(vol_face.el_face[0])
            native_ilist = self.index_lists[vol_face.native_index_list_id]
            from_indices.extend(vol_el_index+i for i in native_ilist)
            bdry_index = bdry_face.gpu_bdry_index_in_floats
            to_indices.extend(
                    xrange(bdry_index, bdry_index+len(native_ilist)))

        return from_indices, to_indices

    @memoize_method
    def _gpu_boundarize_info(self, tag):
        from_indices, to_indices = self._boundarize_info(tag)
        return (
                gpuarray.to_gpu(
                    numpy.array(from_indices, dtype=numpy.uint32)),
                gpuarray.to_gpu(
                    numpy.array(to_indices, dtype=numpy.uint32)),
                )

    @memoize_method
    def _numpy_boundarize_info(self, tag):
        from_indices, to_indices = self._boundarize_info(tag)

        temp_tgt = numpy.zeros((self.aligned_boundary_floats,), dtype=numpy.int32)
        temp_tgt[to_indices] = from_indices

        result = temp_tgt[self._gpu_boundary_embedding(tag)]

        return gpuarray.to_gpu(result)

    class _BoundarizeGPUToNumpyFuture(Future):
        def __init__(self, discr, field, log_shape, tag):
            self.discr = discr

            base_size = len(discr.get_boundary(tag).nodes)
            self.result = result = discr.pagelocked_pool.allocate(
                    log_shape+(base_size,),
                    dtype=discr.default_scalar_type)
            self.result_gpu = gpuarray.empty((result.size,), result.dtype,
                    allocator=discr.pool.allocate)

            # FIXME Technically, we're missing a sync primitive here.
            # We would need to make sure that 'field' is completely
            # written, but CUDA doesn't quite allow that without a
            # full cuda.Context.synchronize() as of version 2.2.
            # For now we're ok omitting the sync primitive since in a
            # typical DG operator, what we're using here is not
            # computed just before--it's data from the last
            # timestep, and that's guaranteed to be there.

            self.stream = discr._get_stream()
            gpuarray.multi_take(field, discr._numpy_boundarize_info(tag),
                    [self.result_gpu[i*base_size:(i+1)*base_size] for i in range(len(field))],
                    stream=self.stream)
            self.result_gpu.get_async(self.stream, result)

        def is_ready(self):
            return self.stream.is_done()

        def __call__(self):
            self.stream.synchronize()
            self.discr._release_stream(self.stream)
            return self.result

    def boundarize_volume_field(self, field, tag, kind=None):
        if self.get_kind(field) == self.compute_kind:
            from hedge.tools import log_shape
            ls = log_shape(field)

            if kind is None or kind == self.compute_kind:
                # GPU -> GPU boundarize

                from_indices, to_indices = \
                        self._gpu_boundarize_info(tag)
                kind = None

                from hedge.mesh import TAG_ALL
                if tag != TAG_ALL:
                    make_new = self.boundary_zeros
                else:
                    make_new = self.boundary_empty

                if ls != ():
                    from pytools import single_valued
                    out = result = make_new(tag, shape=ls,
                            dtype=single_valued(f.dtype for f in field))
                    src = field
                else:
                    result = make_new(tag, dtype=field.dtype)
                    out = [result]
                    src = [field]

                gpuarray.multi_take_put(src,
                        to_indices, from_indices, out=out)
                if kind is None:
                    return result
                else:
                    return self.convert_boundary(result, tag, kind)

            elif kind == "numpy":
                # GPU -> CPU boundarize, for MPI

                if ls == ():
                    field = [field]

                return self._BoundarizeGPUToNumpyFuture(self, field, ls, tag)()
            else:
                raise ValueError("invalid target boundary kind: %s" % kind)
        else:
            return hedge.discretization.Discretization.boundarize_volume_field(
                    self, field, tag, kind)

    def boundarize_volume_field_async(self, field, tag, kind=None):
        if self.get_kind(field) == self.compute_kind and kind == "numpy":
            from hedge.tools import log_shape
            ls = log_shape(field)
            if ls == ():
                field = [field]

            return self._BoundarizeGPUToNumpyFuture(self, field, ls, tag)
        else:
            return hedge.discretization.Discretization\
                    .boundarize_volume_field_async(self, field, tag, kind)

    def prepare_from_neighbor_map(self, indices):
        return gpuarray.to_gpu(numpy.asarray(indices, dtype=numpy.uint32))

    # }}}

    # {{{ ancillary kernel planning/construction ------------------------------
    @memoize_method
    def element_local_kernel(self):
        from hedge.backends.cuda.plan import make_element_local_plan
        el_local_plan, _ = make_element_local_plan(
                self, self.given,
                op_name="el_local",
                aligned_preimage_dofs_per_microblock=
                    self.given.microblock.aligned_floats,
                preimage_dofs_per_el=self.given.dofs_per_el(),
                with_index_check=True)
        return el_local_plan.make_kernel(self, with_index_check=True)
    # }}}

    # {{{ scalar reduction ----------------------------------------------------
    def nodewise_dot_product(self, a, b):
        return gpuarray.subset_dot(
                self._meaningful_volume_indices(),
                a, b, dtype=numpy.float64).get()

    def nodewise_max(self, a):
        return gpuarray.subset_max(self._meaningful_volume_indices(), a).get()

    def nodewise_min(self, a):
        return gpuarray.subset_min(self._meaningful_volume_indices(), a).get()

    # }}}

    # {{{ vector primitives ---------------------------------------------------
    def get_vector_primitive_factory(self):
        from hedge.vector_primitives import CUDAVectorPrimitiveFactory
        return CUDAVectorPrimitiveFactory(discr=self)

    # }}}

    # {{{ numbering tools -----------------------------------------------------
    @memoize_method
    def elgroup_microblock_indices(self, elgroup):
        """For a given :class:`hedge.discretization._ElementGroup` instance
        *elgroup*, return an index array (of dtype :class:`numpy.intp`) that,
        indexed by the block-microblock element number, gives the element
        number within *elgroup*.
        """

        def get_el_index_in_el_group(el):
            mygroup, idx = self.group_map[el.id]
            assert mygroup is elgroup
            return idx

        given = self.given

        el_count = len(self.blocks) * given.elements_per_block()
        elgroup_indices = numpy.zeros((el_count,), dtype=numpy.intp)

        for block in self.blocks:
            block_elgroup_indices = [get_el_index_in_el_group(el)
                    for mb in block.microblocks
                    for el in mb]
            offset = block.number * given.elements_per_block()
            elgroup_indices[offset:offset+len(block_elgroup_indices)] = \
                    block_elgroup_indices

        return elgroup_indices

    # }}}




def make_block_visualization(discr):
    result = discr.volume_zeros(kind="numpy")
    for block in discr.blocks:
        for cpu_slice in block.cpu_slices:
            result[cpu_slice] = block.number

    return result




# vim: foldmethod=marker
