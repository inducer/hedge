"""Interface with Nvidia CUDA."""

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



import numpy
import numpy.linalg as la
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pytools import memoize_method
import hedge.discretization




def _ceiling(value, multiple_of=1):
    """Mimicks the Excel "ceiling" function."""

    from math import ceil
    return int(ceil(value/multiple_of))*multiple_of

def _floor(value, multiple_of=1):
    """Mimicks the Excel "floor" function."""

    from math import floor
    return int(floor(value/multiple_of))*multiple_of




class DeviceData:
    def __init__(self, dev):
        import pycuda.driver as drv

        self.max_threads = dev.get_attribute(drv.device_attribute.MAX_THREADS_PER_BLOCK)
        self.warp_size = dev.get_attribute(drv.device_attribute.WARP_SIZE)
        self.thread_blocks_per_mp = 8
        self.warps_per_mp = 24
        self.registers = dev.get_attribute(drv.device_attribute.REGISTERS_PER_BLOCK)
        self.shared_memory = dev.get_attribute(drv.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK)

    def align_bytes(self):
        return 16




class OccupancyRecord:
    def __init__(self, devdata, threads, shared_mem=0, registers=0):
        if threads > devdata.max_threads:
            raise ValueError("too many threads")

        alloc_warps = _ceiling(threads/devdata.warp_size)
        alloc_regs = _ceiling(alloc_warps*2, 4)*16*registers
        alloc_smem = _ceiling(shared_mem, 512)

        self.tb_per_mp_limits = [(devdata.thread_blocks_per_mp, "device"),
                (_floor(devdata.warps_per_mp/alloc_warps), "warps")
                ]
        if registers > 0:
            self.tb_per_mp_limits.append((_floor(devdata.registers/alloc_regs), "regs"))
        if shared_mem > 0:
            self.tb_per_mp_limits.append((_floor(devdata.shared_memory/alloc_smem), "smem"))

        self.tb_per_mp, self.limited_by = min(self.tb_per_mp_limits)

        self.warps_per_mp = self.tb_per_mp * alloc_warps
        self.occupancy = self.warps_per_mp / devdata.warps_per_mp




class Parallelism:
    def __init__(self, parallel, serial):
        self.p = parallel
        self.s = serial

    def total(self):
        return self.p*self.s

    def __str__(self):
        return "(p%d s%d)" % (self.p, self.s)




class ExecutionPlan:
    def __init__(self, devdata, ldis, flux_par, 
            extfaces=None, float_size=4, int_size=4):
        self.devdata = devdata
        self.ldis = ldis
        self.flux_par = flux_par
        self.extfaces = extfaces
        self.float_size = float_size
        self.int_size = int_size

    def copy(self, devdata=None, ldis=None, flux_par=None, 
            extfaces=None, float_size=None, int_size=None):
        return ExecutionPlan(
                devdata or self.devdata,
                ldis or self.ldis,
                flux_par or self.flux_par,
                extfaces or self.extfaces,
                float_size or self.float_size,
                int_size or self.int_size,
                )

    def dofs_per_el(self):
        return self.ldis.node_count()

    def dofs_per_face(self):
        return self.ldis.face_node_count()

    def faces_per_el(self):
        return len(self.ldis.face_indices())

    def block_el(self):
        return self.flux_par.total()

    @memoize_method
    def estimate_extface_count(self):
        d = self.ldis.dimensions

        # How many equivalent cubes would I need to tesselate the same space
        # as the elements in my thread block?
        from pytools import factorial
        equiv_cubes = self.block_el() / factorial(d)

        # If these cubes in turn formed a perfect macro-cube, how long would
        # its side be?
        macrocube_side = equiv_cubes ** (1/d)

        # What total face area does the macro-cube have?
        macrocube_face_area = 2*d * macrocube_side ** (d-1)

        # How many of my faces do I need to tesselate this face area?
        return macrocube_face_area * factorial(d-1)

    def get_extface_count(self):
        if self.extfaces is None:
            return _ceiling(self.estimate_extface_count())
        else:
            return self.extfaces

    def int_dof_smem(self):
        return _ceiling(self.block_el() * self.dofs_per_el() 
                * self.float_size, 
                self.devdata.align_bytes())

    @memoize_method
    def ext_dof_smem(self):
        return _ceiling(self.get_extface_count()
                * self.dofs_per_face() * self.float_size,
                self.devdata.align_bytes())

    @memoize_method
    def face_count(self):
        return (self.block_el() * self.faces_per_el() + 
                self.get_extface_count())

    def indexing_bytes_per_face_pair(self):
        # How much planning info per face pair?
        # h, order, face_jacobian, normal
        # a_base, b_base, a_ilist_number, b_ilist_number
        # bwrite_base, bwrite_ilist_number
        # Total: 3+d floats, 6 ints

        return (3+self.ldis.dimensions)*self.float_size + 6*self.int_size

    def indexing_smem(self):
        return _ceiling(
                self.int_size+ # number of active elements in block
                self.int_size+ # number of active faces in block
                self.indexing_bytes_per_face_pair()*self.facepair_count(),
                self.devdata.align_bytes())

    def facepair_count(self):
        return (self.face_count()+1) // 2

    @memoize_method
    def shared_mem_use(self):
        return (64 # for parameters
                + self.int_dof_smem() 
                + self.ext_dof_smem() 
                + self.indexing_smem())

    def threads(self):
        return self.flux_par.p*self.faces_per_el()*self.dofs_per_face()

    def invalid_reason(self):
        if self.threads() >= self.devdata.max_threads:
            return "too many threads"
        if self.shared_mem_use() >= self.devdata.shared_memory:
            return "too much shared memory"
        return None

    @memoize_method
    def occupancy_record(self):
        return OccupancyRecord(self.devdata,
                self.threads(), self.shared_mem_use())

    @memoize_method
    def find_localop_par(self):
        threads = self.threads()
        total_threads = self.block_el()*self.dofs_per_el()
        ser, rem = divmod(total_threads, threads)
        if rem == 0:
            return Parallelism(threads, ser)
        else:
            return Parallelism(threads, ser+1)

    def __str__(self):
            return "flux_par=%s threads=%d int_smem=%d ext_smem=%d ind_smem=%d smem=%d occ=%f" % (
                    self.flux_par, self.threads(), 
                    self.int_dof_smem(), 
                    self.ext_dof_smem(),
                    self.indexing_smem(),
                    self.shared_mem_use(), 
                    self.occupancy_record().occupancy)




# gpu data organization -------------------------------------------------------
class GPUBlock(object):
    """Describes what data is local to each thread block on the GPU.

    @ivar number: The global number of this block.
    @ivar local_discretization: The L{hedge.element.Element} instance used
      for elements in this block.
    @ivar cpu_slices: A list of slices describing the CPU-side
      storage locations for the block's elements.
    @ivar elements: A list of L{hedge.mesh.Element} instances representing the
      elements in this block.
    @ivar ext_faces: A list of C{GPUFaceStorage} instances.
    """
    __slots__ = ["number", "local_discretization", "cpu_slices", "elements", 
            "ext_faces"]

    def __init__(self, number, local_discretization, cpu_slices, elements):
        self.number = number
        self.local_discretization = local_discretization
        self.cpu_slices = cpu_slices
        self.elements = elements
        self.ext_faces = []

    def get_el_index(self, sought_el):
        from pytools import one
        return one(i for i, el in enumerate(self.elements)
                if el == sought_el)

    def register_ext_face(self, face):
        result = len(self.ext_faces)
        self.ext_faces.append(face)
        return result




class GPUFaceStorage(object):
    """Describes where the dofs of an element face are stored.

    @ivar el_face: a tuple (element, face_number).
    """

    def __init__(self):
        self.opposite = None

class GPUInteriorFaceStorage(GPUFaceStorage):
    """Describes storage locations for a face belongi

    @ivar el_face: a tuple C{(element, face_number)}.
    @ivar cpu_slice: the base index of the element in CPU numbering.
    @ivar native_index_list_id: 
    @ivar native_block: block in which element is to be found.
    @ivar native_block_el_num: number of this element in the C{native_block}.
    @ivar dup_block: 
    @ivar dup_ext_face_number:
    """
    __slots__ = [
            "opposite",
            "el_face", "cpu_slice", "native_index_list_id",
            "native_block", "native_block_el_num",
            "dup_block", "dup_ext_face_number", "dup_index_list_id"]

    def __init__(self, el_face, cpu_slice, native_index_list_id,
            native_block, native_block_el_num):
        GPUFaceStorage.__init__(self)
        self.el_face = el_face
        self.cpu_slice = cpu_slice
        self.native_index_list_id = native_index_list_id
        self.native_block = native_block
        self.native_block_el_num = native_block_el_num

class GPUBoundaryFaceStorage(GPUFaceStorage):
    """Describes storage locations for a face.

    @ivar bdry_index: this face's starting index in the global boundary array.
    @ivar dup_block: 
    @ivar dup_ext_face_number:
    """
    __slots__ = ["opposite", "bdry_index", "dup_block", "dup_ext_face_number"]

    def __init__(self, bdry_index):
        GPUFaceStorage.__init__(self)
        self.bdry_index = 0




# tools -----------------------------------------------------------------------
def vec_to_gpu(field):
    from hedge.tools import log_shape
    ls = log_shape(field)
    if ls != ():
        result = numpy.array(ls, dtype=object)

        from pytools import indices_in_shape

        for i in indices_in_shape(ls):
            result[i] = gpuarray.to_gpu(field[i])
        return result
    else:
        return gpuarray.to_gpu(field)





# GPU discretization ----------------------------------------------------------
class CudaDiscretization(hedge.discretization.Discretization):
    def _make_plan(self, ldis, mesh):
        def generate_valid_plans():
            for pe in range(2,32):
                for se in range(1,256):
                    flux_par = Parallelism(pe, se)
                    plan = ExecutionPlan(self.devdata, ldis, flux_par)
                    if plan.invalid_reason() is None:
                        yield plan

        plans = list(generate_valid_plans())

        if not plans:
            raise RuntimeError, "no valid CUDA execution plans found"

        max_occup = max(plan.occupancy_record().occupancy for plan in plans)
        good_plans = [p for p in generate_valid_plans()
                if p.occupancy_record().occupancy > max_occup - 1e-10]

        from pytools import argmax2
        return argmax2((p, p.block_el()) for p in good_plans)




    def _partition_mesh(self, mesh, plan):
        # search for mesh partition that matches plan
        from pymetis import part_graph
        part_count = len(mesh.elements)//plan.flux_par.total()+1
        while True:
            cuts, partition = part_graph(part_count,
                    mesh.element_adjacency_graph(),
                    vweights=[1000]*len(mesh.elements))

            # prepare a mapping:  block# -> # of interfaces
            block2extifaces = {}

            for elface1, elface2 in mesh.interfaces:
                e1, f1 = elface1
                e2, f2 = elface2
                r1 = partition[e1.id]
                r2 = partition[e2.id]

                if r1 != r2:
                    block2extifaces[r1] = block2extifaces.get(r1, 0) + 1
                    block2extifaces[r2] = block2extifaces.get(r2, 0) + 1

            blocks = {}
            for el_id, block in enumerate(partition):
                blocks.setdefault(block, []).append(el_id)

            block_elements = max(len(block_els) for block_els in blocks.itervalues())
            flux_par_s = _ceiling(block_elements/plan.flux_par.p)
            actual_plan = plan.copy(
                    extfaces=max(block2extifaces.itervalues()),
                    flux_par=Parallelism(plan.flux_par.p, flux_par_s))

            if (flux_par_s == plan.flux_par.s and
                    abs(plan.occupancy_record().occupancy -
                        actual_plan.occupancy_record().occupancy) < 1e-10):
                break

            part_count += 1

        if False:
            from matplotlib.pylab import hist, show
            print plan.get_extface_count()
            hist(block2extifaces.values())
            show()
            hist([len(block_els) for block_els in blocks.itervalues()])
            show()
            
        return actual_plan, partition




    def __init__(self, mesh, local_discretization=None, 
            order=None, plan=None, init_cuda=True, debug=False, 
            dev=None):
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

        self.device = dev
        self.devdata = DeviceData(dev)

        if plan is None:
            plan = self._make_plan(ldis, mesh)
            print "projected:", plan

        self.plan, self.partition = self._partition_mesh(mesh, plan)
        del plan
        print "actual:", self.plan

        # initialize superclass -----------------------------------------------
        hedge.discretization.Discretization.__init__(self, mesh, ldis, debug=debug)

        self.blocks = self._build_blocks()
        self.face_storage_map = self._build_face_storage_map()

        self.int_dof_floats = self.plan.int_dof_smem()//self.plan.float_size
        self.ext_dof_floats = self.plan.ext_dof_smem()//self.plan.float_size




    def _build_blocks(self):
        block_el_numbers = {}
        for el_id, block in enumerate(self.partition):
            block_el_numbers.setdefault(block, []).append(el_id)

        block_count = len(block_el_numbers)

        def make_block(block_num):
            elements = [self.mesh.elements[ben] for ben in block_el_numbers[block_num]]

            eg, = self.element_groups
            return GPUBlock(block_num, 
                    local_discretization=eg.local_discretization,
                    cpu_slices=[slice(*self.find_el_range(el.id)) for el in elements], 
                    elements=elements)

        return [make_block(block_num) for block_num in range(block_count)]

    


    def _build_face_storage_map(self):
        fsm = {}

        from hedge.mesh import TAG_ALL
        all_bdry = self.get_boundary(TAG_ALL)

        face_index_list_register = {}
        self.index_lists = []
        def get_face_index_list_number(il):
            il_tup = tuple(il)
            try:
                return face_index_list_register[il_tup]
            except KeyError:
                self.index_lists.append(numpy.array(il))
                return face_index_list_register.setdefault(
                        tuple(il), len(face_index_list_register))

        def make_int_face(flux_face, ilist_number):
            el = self.mesh.elements[flux_face.element_id]
            block = self.blocks[self.partition[el.id]]
            elface = (el, flux_face.face_id)
            iln = get_face_index_list_number(int_fg.index_lists[ilist_number])
            result = GPUInteriorFaceStorage(
                elface, 
                slice(*self.find_el_range(el.id)), 
                iln,
                block, 
                block.get_el_index(el), 
                )
            fsm[elface] = result
            return result

        (int_fg, fmm), = self.face_groups
        id_face_index_list = tuple(xrange(fmm.shape[0]))
        id_face_index_list_number = get_face_index_list_number(id_face_index_list)
        assert id_face_index_list_number == 0

        for fp in int_fg.face_pairs:
            face1 = make_int_face(
                    int_fg.flux_faces[fp.flux_face_index], 
                    fp.face_index_list_number)
            face2 = make_int_face(
                    int_fg.flux_faces[fp.opp_flux_face_index], 
                    fp.opp_face_index_list_number)
            face1.opposite = face2
            face2.opposite = face1

            if face1.native_block != face2.native_block:
                # allocate resources for duplicated face
                face1.dup_block = face2.native_block
                face2.dup_block = face1.native_block
                face1.dup_ext_face_number = face1.dup_block.register_ext_face(face1)
                face2.dup_ext_face_number = face2.dup_block.register_ext_face(face2)
            
        (bdry_fg, ldis), = all_bdry.face_groups_and_ldis
        for fp in bdry_fg.face_pairs:
            assert fp.opp_flux_face_index == fp.INVALID_INDEX
            assert (tuple(bdry_fg.index_lists[fp.opp_face_index_list_number]) 
                    == id_face_index_list)

            face1 = make_int_face(
                    bdry_fg.flux_faces[fp.flux_face_index], 
                    fp.face_index_list_number)
            face2 = GPUBoundaryFaceStorage(fp.opp_el_base_index)
            face1.opposite = face2
            face2.opposite = face1
            face2.dup_block = face1.native_block
            face2.dup_ext_face_number = face2.dup_block.register_ext_face(face2)

        return fsm




    def block_dof_count(self):
        return self.int_dof_floats + self.ext_dof_floats

    def gpu_dof_count(self):
        return self.block_dof_count() * len(self.blocks)

    def to_gpu(self, field):
        from hedge.tools import log_shape
        ls = log_shape(field)
        if ls != ():
            result = numpy.array(ls, dtype=object)

            from pytools import indices_in_shape

            for i in indices_in_shape(ls):
                result[i] = self.to_gpu(field[i])
            return result
        else:
            copy_vec = numpy.empty((self.gpu_dof_count(),), dtype=numpy.float32)

            block_dofs = self.block_dof_count()

            block_offset = 0
            for block in self.blocks:
                face_length = block.local_discretization.face_node_count()
                el_length = block.local_discretization.node_count()

                # write internal dofs
                el_offset = block_offset
                for cpu_slice in block.cpu_slices:
                    copy_vec[el_offset:el_offset+el_length] = field[cpu_slice]
                    el_offset += el_length

                ef_start = block_offset+self.int_dof_floats
                for i_ef, ext_face in enumerate(block.ext_faces):
                    if isinstance(ext_face, GPUInteriorFaceStorage):
                        f_start = ef_start+face_length*i_ef
                        il = ext_face.opposite.cpu_slice.start + \
                                self.index_lists[ext_face.native_index_list_id]
                        copy_vec[f_start:f_start+face_length] = field[il] 

                block_offset += block_dofs

            return gpuarray.to_gpu(copy_vec)

    def from_gpu(self, field, check=None):
        if check is None:
            check = self.debug

        from hedge.tools import log_shape
        ls = log_shape(field)
        if ls != ():
            result = numpy.array(ls, dtype=object)

            from pytools import indices_in_shape

            for i in indices_in_shape(ls):
                result[i] = self.from_gpu(field[i])
            return result
        else:
            copied_vec = field.get(pagelocked=True)
            result = numpy.empty(shape=(len(self),), dtype=copied_vec.dtype)

            block_dofs = self.block_dof_count()

            block_offset = 0
            for block in self.blocks:
                face_length = block.local_discretization.face_node_count()
                el_length = block.local_discretization.node_count()

                # write internal dofs
                el_offset = block_offset
                for cpu_slice in block.cpu_slices:
                    result[cpu_slice] = copied_vec[el_offset:el_offset+el_length]
                    el_offset += el_length

                if check:
                    ef_start = block_offset+self.int_dof_floats
                    for i_ef, ext_face in enumerate(block.ext_faces):
                        if isinstance(ext_face, GPUInteriorFaceStorage):
                            f_start = ef_start+face_length*i_ef
                            il = ext_face.opposite.cpu_slice.start + \
                                    self.index_lists[ext_face.native_index_list_id]
                            diff = result[il] - copied_vec[f_start:f_start+face_length]
                            assert la.norm(diff) < 1e-10 * la.norm(result)
                        
                block_offset += block_dofs

            return result

    @memoize_method
    def get_bdry_embedding(self, tag):
        bdry = self.get_boundary(tag)
        entire_bdry = self.get_boundary(hedge.mesh.TAG_ALL)
        if tag == hedge.mesh.TAG_ALL:
            return numpy.arange(len(entire_bdry.nodes))
        else:
            inverse_imap_entire = dict(
                    (gi, fi) for fi, gi in enumerate(entire_bdry.index_map))
            return numpy.array(
                    [inverse_imap_entire[gi] for gi in bdry.index_map])

    def to_full_bdry(self, tag, field):
        if tag == hedge.mesh.TAG_ALL:
            return field
        else:
            from hedge.tools import log_shape
            ls = log_shape(field)
            if ls != ():
                from pytools import indices_in_shape
                result = numpy.array(ls, dtype=object)
                for i in indices_in_shape(shape):
                    result[i] = self.to_full_bdry(field[i], tag)
                return result
            else:
                s = hedge.discretization.Discretization
                result = s.boundary_empty(self, hedge.mesh.TAG_ALL)
                result[self.get_bdry_embedding(tag)] = field
                return result

    # vector construction -----------------------------------------------------
    def volume_empty(self, shape=()):
        return gpuarray.empty(shape+(self.gpu_dof_count(),), dtype=numpy.float32)

    def volume_zeros(self, shape=()):
        result = self.volume_empty(shape)
        result.fill(0)
        return result

    def interpolate_volume_function(self, f):
        s = hedge.discretization.Discretization

        def tgt_factory(shape):
            return s.volume_empty(self, shape)

        return self.to_gpu(
                s.interpolate_volume_function(self, f, tgt_factory))

    def boundary_empty(self, tag=hedge.mesh.TAG_ALL, shape=()):
        result = numpy.empty(shape, dtype=object)
        from pytools import indices_in_shape
        bdry = self.get_boundary(TAG_ALL)
        for i in indices_in_shape(shape):
            result[i] = gpuarray.empty((len(bdry.nodes),), dtype=numpy.float32)
        return result

    def boundary_zeros(self, tag=hedge.mesh.TAG_ALL, shape=()):
        result = numpy.zeros(shape, dtype=object)
        bdry = self.get_boundary(TAG_ALL)
        from pytools import indices_in_shape
        for i in indices_in_shape(shape):
            result[i] = gpuarray.zeros((len(bdry.nodes),), dtype=numpy.float32)
        return result

    def interpolate_boundary_function(self, f, tag=hedge.mesh.TAG_ALL):
        s = hedge.discretization.Discretization
        def tgt_factory(shape, tag):
            return s.boundary_empty(self, shape, tag)

        return vec_to_gpu(
                self.to_full_bdry(tag,
                s.interpolate_boundary_function(self, f, tag, tgt_factory)))

    def boundary_normals(self, tag=hedge.mesh.TAG_ALL):
        raise NotImplementedError

    def volumize_boundary_field(self, bfield, tag=hedge.mesh.TAG_ALL):
        raise NotImplementedError

    def boundarize_volume_field(self, field, tag=hedge.mesh.TAG_ALL):
        raise NotImplementedError

    # host vector construction ------------------------------------------------
    s = hedge.discretization.Discretization
    host_volume_empty = s.volume_empty
    host_volume_zeros = s.volume_zeros 
    del s

    # optemplate processing ---------------------------------------------------
    def preprocess_optemplate(self, optemplate):
        print optemplate
        raise NotImplementedError

    def run_preprocessed_optemplate(self, pp_optemplate, vars):
        raise NotImplementedError












            





            






def _test_occupancy():
    for threads in range(32, 512, 16):
        for smem in range(1024, 16384+1, 1024):
            occ = Occupancy(threads, smem)
            print "t%d s%d: %f %s" % (threads, smem, occ.occupancy, occ.limited_by)




def _test_planner():
    from hedge.element import TetrahedralElement
    for order in [3]:
        for pe in range(2,16):
            for se in range(1,16):
                flux_par = Parallelism(pe, se)
                plan = ExecutionPlan(TetrahedralElement(order), flux_par)
                inv_reas = plan.invalid_reason()
                if inv_reas is None:
                    print "o%d %s: smem=%d extfacepairs=%d/%d occ=%f (%s) lop:%s" % (
                            order, flux_par,
                            plan.shared_mem_use(),
                            plan.estimate_extface_count(),
                            plan.face_count()//2,
                            plan.occupancy().occupancy,
                            plan.occupancy().limited_by,
                            plan.find_localop_par()
                            )
                else:
                    print "o%d p%d s%d: %s" % (order, pe, se, inv_reas)




if __name__ == "__main__":
    import pycuda.driver as drv
    drv.init()

    _test_planner()

