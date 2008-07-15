"""Interface with Nvidia CUDA."""

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
import numpy.linalg as la
import hedge.discretization
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pytools import memoize_method, memoize




# gpu data organization -------------------------------------------------------
class GPUBlock(object):
    """Describes what data is local to each thread block on the GPU.

    @ivar number: The global number of this block.
    @ivar local_discretization: The L{hedge.element.Element} instance used
      for elements in this block.
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

    @ivar opposite: the L{GPUFacestorage} instance for the face
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

    @ivar el_face: a tuple C{(element, face_number)}.
    @ivar cpu_slice: the base index of the element in CPU numbering.
    @ivar native_index_list_id: 
    @ivar opp_write_index_list_id:
    @ivar native_block: block in which element is to be found.
    @ivar face_pair_side:
    """
    __slots__ = [
            "el_face", "cpu_slice", 
            "native_index_list_id", "opp_write_index_list_id",
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




@memoize
def _boundarize_kernel():
    mod = cuda.SourceModule("""
    texture<float, 1, cudaReadModeElementType> field_tex;
    __global__ void boundarize(float *bfield, 
      unsigned int *to_indices,
      unsigned int *from_indices,
      unsigned int n)
    {
      int tid = threadIdx.x;
      int total_threads = gridDim.x*blockDim.x;
      int cta_start = blockDim.x*blockIdx.x;
      int i;
            
      for (i = cta_start + tid; i < n; i += total_threads) 
      {
        bfield[to_indices[i]] = 
          tex1Dfetch(field_tex, from_indices[i]);
      }
    }
    """)

    return (mod.get_function("boundarize"), 
            mod.get_texref("field_tex"))




# GPU discretization ----------------------------------------------------------
class Discretization(hedge.discretization.Discretization):
    def _make_plan(self, ldis, mesh, float_type):
        from hedge.cuda.plan import \
                FluxExecutionPlan, \
                Parallelism, \
                optimize_plan

        def generate_valid_plans():
            for parallel_faces in range(1,32):
                for mbs_per_block in range(1,256):
                    flux_plan = FluxExecutionPlan(
                            self.devdata, ldis, parallel_faces,
                            mbs_per_block, float_type=float_type)
                    if flux_plan.invalid_reason() is None:
                        yield flux_plan

        return optimize_plan(
                generate_valid_plans,
                lambda plan: plan.elements_per_block()
                )




    def _partition_mesh(self, mesh, flux_plan):
        # search for mesh partition that matches plan
        from pymetis import part_graph
        orig_part_count = part_count = (
                len(mesh.elements)//flux_plan.elements_per_block()+1)
        while True:
            cuts, partition = part_graph(part_count,
                    mesh.element_adjacency_graph(),
                    vweights=[1000]*len(mesh.elements))

            # prepare a mapping:  block# -> # of external interfaces
            block2extifaces = dict((i, 0) for i in range(part_count))

            for (e1, f1), (e2, f2) in mesh.both_interfaces():
                b1 = partition[e1.id]
                b2 = partition[e2.id]

                if b1 != b2:
                    block2extifaces[b1] += 1

            for el, face_nbr in mesh.tag_to_boundary[hedge.mesh.TAG_ALL]:
                b1 = partition[el.id]
                block2extifaces[b1] += 1

            blocks = dict((i, []) for i in range(part_count))
            for el_id, block in enumerate(partition):
                blocks[block].append(el_id)
            block_elements = max(len(block_els) for block_els in blocks.itervalues())

            from hedge.cuda.plan import Parallelism
            actual_plan = flux_plan.copy(
                    max_ext_faces=max(block2extifaces.itervalues()),
                    max_faces=max(
                        len(blocks[b])*flux_plan.faces_per_el()
                        + block2extifaces[b]
                        for b in range(len(blocks))),
                    )
            assert actual_plan.max_faces % 2 == 0

            if (block_elements <= actual_plan.elements_per_block()
                    and (flux_plan.occupancy_record().occupancy -
                        actual_plan.occupancy_record().occupancy) < 1e-10):
                break

            part_count += min(5, int(part_count*0.01))

        print "blocks: theoretical:%d practical:%d" % (orig_part_count, part_count)

        if False:
            from matplotlib.pylab import hist, show
            print plan.get_extface_count()
            hist(block2extifaces.values())
            show()
            hist([len(block_els) for block_els in blocks.itervalues()])
            show()
            
        return actual_plan, partition




    def __init__(self, mesh, local_discretization=None, 
            order=None, flux_plan=None, init_cuda=True, debug=False, 
            dev=None, default_scalar_type=numpy.float32):
        ldis = self.get_local_discretization(mesh, local_discretization, order)

        if init_cuda:
            cuda.init()

        if dev is None:
            assert cuda.Device.count() >= 1
            dev = cuda.Device(0)
        if isinstance(dev, int):
            dev = cuda.Device(dev)
        if init_cuda:
            self.cuda_context = dev.make_context()

        from pycuda.tools import DeviceMemoryPool
        self.pool = DeviceMemoryPool()

        self.device = dev
        from hedge.cuda.tools import DeviceData
        self.devdata = DeviceData(dev)

        # make preliminary plan
        if flux_plan is None:
            flux_plan = self._make_plan(ldis, mesh, default_scalar_type)
        print "projected flux exec plan:", flux_plan

        # partition mesh, obtain updated plan
        self.flux_plan, self.partition = self._partition_mesh(mesh, flux_plan)
        del flux_plan
        print "actual flux exec plan:", self.flux_plan
        print "actual diff op exec plan:", self.flux_plan.diff_plan()
        print "actual flux local exec plan:", self.flux_plan.flux_lifting_plan()

        # initialize superclass
        hedge.discretization.Discretization.__init__(self, mesh, ldis, debug=debug,
                default_scalar_type=default_scalar_type)

        # build our own data structures
        self.blocks = self._build_blocks()
        self.face_storage_map = self._build_face_storage_map()

        # make a reference discretization
        from hedge.discr_precompiled import Discretization
        self.test_discr = Discretization(mesh, ldis)

    def _build_blocks(self):
        block_el_numbers = {}
        for el_id, block in enumerate(self.partition):
            block_el_numbers.setdefault(block, []).append(el_id)

        block_count = len(block_el_numbers)

        def make_block(block_num):
            fplan = self.flux_plan

            microblocks = []
            current_microblock = []
            el_offsets_list = []
            el_number_map = {}
            elements = [self.mesh.elements[ben] for ben in block_el_numbers[block_num]]
            for block_el_nr, el in enumerate(elements):
                el_offset = (
                        len(microblocks)*fplan.microblock.aligned_floats
                        + len(current_microblock)*fplan.dofs_per_el())
                el_number_map[el] = block_el_nr
                el_offsets_list.append(el_offset)

                current_microblock.append(el)
                if len(current_microblock) == fplan.microblock.elements:
                    microblocks.append(current_microblock)
                    current_microblock = []

            if current_microblock:
                microblocks.append(current_microblock)

            assert len(microblocks) <= fplan.microblocks_per_block()

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
        fil_registry = IndexListRegistry(self.debug)

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


        def narrow_ilist(in_el_ilist, native_el_ilist):
            return get_read_from_map_from_permutation

            el_dof_to_face_dof = dict(
                    (el_dof, i)
                    for i, el_dof in enumerate(native_el_ilist))
            return tuple(el_dof_to_face_dof[el_dof]
                    for el_dof in in_el_ilist)

        int_fg, = self.face_groups
        ldis = int_fg.ldis_loc
        assert ldis == int_fg.ldis_opp

        id_face_index_list_number = fil_registry.register(
                None, 
                lambda: tuple(xrange(ldis.face_node_count()))
                )
        assert id_face_index_list_number == 0

        from pytools import single_valued
        for fp in int_fg.face_pairs:
            face1 = make_int_face(fp.loc)
            face2 = make_int_face(fp.opp)
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
                    fp.loc.face_index_list_number])

            def face2_in_el_ilist(): 
                return tuple(int_fg.index_lists[
                    fp.opp.face_index_list_number])

            def opp_write_map(): 
                return tuple(
                        int_fg.index_lists[fp.opp_native_write_map])
                
            face1.global_int_flux_index_list_id = fil_registry.register(
                    (int_fg, fp.loc.face_index_list_number),
                    face1_in_el_ilist)
            face1.global_ext_flux_index_list_id = fil_registry.register(
                    (int_fg, fp.opp.face_index_list_number),
                    face2_in_el_ilist)

            face2.global_int_flux_index_list_id = fil_registry.register(
                    (int_fg, fp.opp_native_write_map,
                        fp.opp.face_index_list_number),
                    lambda: apply_write_map(
                        opp_write_map(), face2_in_el_ilist())
                    )
            face2.global_ext_flux_index_list_id = fil_registry.register(
                    (int_fg, fp.opp_native_write_map,
                        fp.loc.face_index_list_number),
                    lambda: apply_write_map(
                        opp_write_map(), face1_in_el_ilist())
                    )

            from pytools import get_write_to_map_from_permutation as gwtm
            #assert gwtm(face2_in_el_ilist, f_ind[fp.opp.face_id]) == opp_write_map
            face1.opp_write_index_list_id = fil_registry.register(
                    (int_fg, "wtm", fp.opp.face_index_list_number,
                        fp.opp.face_id),
                    lambda: gwtm(face2_in_el_ilist(), f_ind[fp.opp.face_id])
                    )
            face2.opp_write_index_list_id = fil_registry.register(
                    (int_fg, "wtm", 
                        fp.opp_native_write_map,
                        fp.loc.face_index_list_number,
                        fp.loc.face_id),
                    lambda: gwtm(
                        apply_write_map(opp_write_map(), face1_in_el_ilist()),
                        f_ind[fp.loc.face_id])
                    )

        self.aligned_boundary_floats = 0
        from hedge.mesh import TAG_ALL
        for bdry_fg in self.get_boundary(TAG_ALL).face_groups:
            assert ldis == bdry_fg.ldis_loc
            aligned_fnc = self.devdata.align_dtype(ldis.face_node_count(), 
                    self.flux_plan.float_size)
            for fp in bdry_fg.face_pairs:
                assert fp.opp.element_id == hedge._internal.INVALID_ELEMENT
                #assert (tuple(bdry_fg.index_lists[fp.opp.face_index_list_number]) 
                        #== id_face_index_list)

                face1 = make_int_face(fp.loc)
                face2 = GPUBoundaryFaceStorage(
                        fp.opp.el_base_index,
                        self.aligned_boundary_floats,
                        fp.opp
                        )
                self.aligned_boundary_floats += aligned_fnc
                face1.opposite = face2
                face2.opposite = face1

                face1.global_int_flux_index_list_id = fil_registry.register(
                        (bdry_fg,fp.loc.face_index_list_number),
                        lambda: tuple(bdry_fg.index_lists[
                            fp.loc.face_index_list_number])
                        )
                face1.global_ext_flux_index_list_id = fil_registry.register(
                        (bdry_fg, fp.opp.face_index_list_number),
                        lambda: tuple(bdry_fg.index_lists[
                            fp.opp.face_index_list_number])
                        )

        self.index_lists = fil_registry.index_lists
        return fsm




    def find_el_gpu_index(self, el):
        fplan = self.flux_plan
        block = self.blocks[self.partition[el.id]]

        mb_nr, in_mb_nr = divmod(block.el_number_map[el], fplan.microblock.elements)

        return (block.number * self.flux_plan.dofs_per_block() 
                + mb_nr*fplan.microblock.aligned_floats
                + in_mb_nr*fplan.dofs_per_el())

    def find_number_in_block(self, el):
        block = self.blocks[self.partition[el.id]]
        return block.el_number_map[el]

    def gpu_dof_count(self):
        from hedge.cuda.tools import int_ceiling

        fplan = self.flux_plan
        return int_ceiling(
                int_ceiling(
                    fplan.dofs_per_block() * len(self.blocks),     
                    fplan.diff_plan().dofs_per_macroblock()),
                fplan.flux_lifting_plan().dofs_per_macroblock())

    def volume_to_gpu(self, field):
        from hedge.tools import log_shape
        ls = log_shape(field)
        if ls != ():
            result = numpy.empty(ls, dtype=object)

            from pytools import indices_in_shape

            for i in indices_in_shape(ls):
                result[i] = self.volume_to_gpu(field[i])
            return result
        else:
            copy_vec = numpy.empty((self.gpu_dof_count(),), dtype=numpy.float32)

            block_offset = 0
            block_size = self.flux_plan.dofs_per_block()
            for block in self.blocks:
                face_length = block.local_discretization.face_node_count()
                el_length = block.local_discretization.node_count()

                for el_offset, cpu_slice in zip(
                        block.el_offsets_list, block.cpu_slices):
                    copy_vec[block_offset+el_offset:block_offset+el_offset+el_length] = \
                            field[cpu_slice]

                block_offset += block_size

            return gpuarray.to_gpu(copy_vec, allocator=self.pool.allocate)

    def volume_from_gpu(self, field):
        from hedge.tools import log_shape
        ls = log_shape(field)
        if ls != ():
            result = numpy.zeros(ls, dtype=object)

            from pytools import indices_in_shape

            for i in indices_in_shape(ls):
                result[i] = self.volume_from_gpu(field[i])
            return result
        else:
            copied_vec = field.get(pagelocked=True)
            result = numpy.empty(shape=(len(self),), dtype=copied_vec.dtype)

            block_offset = 0
            block_size = self.flux_plan.dofs_per_block()
            for block in self.blocks:
                el_length = block.local_discretization.node_count()

                for el_offset, cpu_slice in zip(
                        block.el_offsets_list, block.cpu_slices):
                    result[cpu_slice] = \
                            copied_vec[block_offset+el_offset
                                    :block_offset+el_offset+el_length]

                block_offset += block_size

            return result

    @memoize_method
    def gpu_boundary_embedding(self, tag):
        """Return an array of indices embedding a CPU boundary
        field for C{tag} into the GPU boundary field."""

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

    def boundary_to_gpu(self, tag, field):
        from hedge.tools import log_shape
        ls = log_shape(field)
        if ls != ():
            from pytools import indices_in_shape
            result = numpy.zeros(ls, dtype=object)
            for i in indices_in_shape(ls):
                result[i] = self.boundary_to_gpu(tag, field[i])
            return result
        else:
            result = cuda.pagelocked_empty(
                    (self.aligned_boundary_floats,),
                    dtype=field.dtype)

            # The boundary cannot be completely uninitialized,
            # because it might contain NaNs. If a certain part of the
            # boundary is to be ignored, it is simply multiplied by
            # zero in the kernel, which won't make the NaNs disappear.

            # Therefore, as a second best solution, fill the boundary
            # with a bogus value so that we can tell if it actually
            # enters the computation.

            result.fill(17) 
            result[self.gpu_boundary_embedding(tag)] = field
            return gpuarray.to_gpu(result, allocator=self.pool.allocate)

    def _empty_gpuarray(self, shape, dtype):
        return gpuarray.empty(shape, dtype=dtype,
                allocator=self.pool.allocate)

    def _zeros_gpuarray(self, shape, dtype):
        result = gpuarray.empty(shape, dtype=dtype,
                allocator=self.pool.allocate)
        result.fill(0)
        return result

    # vector construction -----------------------------------------------------
    def volume_empty(self, shape=(), dtype=None):
        if dtype is None:
            dtype = self.flux_plan.float_type

        return self._empty_gpuarray(shape+(self.gpu_dof_count(),), dtype=dtype)

    def volume_zeros(self, shape=()):
        result = self.volume_empty(shape)
        result.fill(0)
        return result

    def interpolate_volume_function(self, f, dtype=None):
        s = hedge.discretization.Discretization

        def tgt_factory(shape, dtype):
            return s.volume_empty(self, shape, dtype)

        return self.volume_to_gpu(
                s.interpolate_volume_function(self, f, tgt_factory))

    def _new_bdry(self, tag, shape, create_func, dtype):
        if dtype is None:
            dtype = self.default_scalar_type

        if shape == ():
            return create_func((self.aligned_boundary_floats,), dtype=dtype)

        result = numpy.empty(shape, dtype=object)
        from pytools import indices_in_shape
        bdry = self.get_boundary(TAG_ALL)
        for i in indices_in_shape(shape):
            result[i] = create_func((self.aligned_boundary_floats,), dtype=dtype)
        return result
    
    def boundary_empty(self, tag=hedge.mesh.TAG_ALL, shape=(), dtype=None):
        return self._new_bdry(tag, shape, self._empty_gpuarray, dtype)

    def boundary_zeros(self, tag=hedge.mesh.TAG_ALL, shape=(), dtype=None):
        return self._new_bdry(tag, shape, self._zeros_gpuarray, dtype)

    def interpolate_boundary_function(self, f, tag=hedge.mesh.TAG_ALL):
        s = hedge.discretization.Discretization
        def tgt_factory(tag, shape, dtype):
            return s.boundary_zeros(self, tag, shape, dtype)

        return self.boundary_to_gpu(tag,
                s.interpolate_boundary_function(self, f, tag, tgt_factory))

    def boundary_normals(self, tag=hedge.mesh.TAG_ALL):
        s = hedge.discretization.Discretization
        def tgt_factory(tag, shape, dtype):
            return s.boundary_zeros(self, tag, shape, dtype)

        return self.boundary_to_gpu(tag,
                s.boundary_normals(self, tag, tgt_factory))

    def volumize_boundary_field(self, bfield, tag=hedge.mesh.TAG_ALL):
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

        return (
                gpuarray.to_gpu(
                    numpy.array(from_indices, dtype=numpy.uint32)),
                gpuarray.to_gpu(
                    numpy.array(to_indices, dtype=numpy.uint32)),
                len(from_indices)
                )

        
    def boundarize_volume_field(self, field, tag=hedge.mesh.TAG_ALL):
        kernel, field_texref = _boundarize_kernel()

        from_indices, to_indices, idx_count = self._boundarize_info(tag)
        block_count, threads_per_block, elems_per_block = \
                gpuarray.splay(idx_count)

        def do_scalar(subfield):
            from hedge.mesh import TAG_ALL
            if tag != TAG_ALL:
                result = self.boundary_zeros(tag)
            else:
                result = self.boundary_empty(tag)

            if idx_count:
                subfield.bind_to_texref(field_texref)
                kernel(result, to_indices, from_indices,
                        numpy.uint32(idx_count),
                        block=(threads_per_block,1,1), grid=(block_count,1),
                        texrefs=[field_texref])
            return result

        from hedge.tools import log_shape
        ls = log_shape(field)

        if ls == ():
            return do_scalar(field)
        else:
            result = numpy.empty(ls, dtype=object)
            from pytools import indices_in_shape
            for i in indices_in_shape(ls):
                result[i] = do_scalar(field[i])

            return result

    # host vector construction ------------------------------------------------
    s = hedge.discretization.Discretization
    host_volume_empty = s.volume_empty
    host_volume_zeros = s.volume_zeros 
    del s

    # optemplate processing ---------------------------------------------------
    def compile(self, optemplate):
        from hedge.cuda.execute import OpTemplateWithEnvironment
        return OpTemplateWithEnvironment(self, optemplate)
