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




# tools -----------------------------------------------------------------------
def _exact_div(dividend, divisor):
    quot, rem = divmod(dividend, divisor)
    assert rem == 0
    return quot

def _ceiling(value, multiple_of=1):
    """Round C{value} up to be a C{multiple_of} something."""
    # Mimicks the Excel "floor" function (for code stolen from occupany calculator)

    from math import ceil
    return int(ceil(value/multiple_of))*multiple_of

def _floor(value, multiple_of=1):
    """Round C{value} down to be a C{multiple_of} something."""
    # Mimicks the Excel "floor" function (for code stolen from occupany calculator)

    from math import floor
    return int(floor(value/multiple_of))*multiple_of

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





# knowledge about hardware ----------------------------------------------------
class DeviceData:
    def __init__(self, dev):
        import pycuda.driver as drv

        self.max_threads = dev.get_attribute(drv.device_attribute.MAX_THREADS_PER_BLOCK)
        self.warp_size = dev.get_attribute(drv.device_attribute.WARP_SIZE)
        self.thread_blocks_per_mp = 8
        self.warps_per_mp = 24
        self.registers = dev.get_attribute(drv.device_attribute.REGISTERS_PER_BLOCK)
        self.shared_memory = dev.get_attribute(drv.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK)

    def align(self, bytes):
        return _ceiling(bytes, self.align_bytes())

    def align_dtype(self, elements, dtype_size):
        return _ceiling(elements, _exact_div(self.align_bytes(), dtype_size))

    def align_bytes(self):
        return 16




class OccupancyRecord:
    def __init__(self, devdata, threads, shared_mem=0, registers=0):
        if threads > devdata.max_threads:
            raise ValueError("too many threads")

        # copied literally from occupancy calculator
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




# C generation utilities ------------------------------------------------------
class StructField(object):
    def __init__(self, name, dtype):
        self.name = name
        self.dtype = numpy.dtype(dtype)

    def struct_format(self):
        return self.dtype.char

    @staticmethod
    def ctype(dtype):
        if dtype == numpy.int32:
            return "int"
        elif dtype == numpy.uint32:
            return "unsigned int"
        elif dtype == numpy.int16:
            return "short int"
        elif dtype == numpy.uint16:
            return "short unsigned int"
        elif dtype == numpy.int8:
            return "signed char"
        elif dtype == numpy.uint8:
            return "unsigned char"
        elif dtype == numpy.intp or dtype == numpy.uintp:
            return "void *"
        elif dtype == numpy.float32:
            return "float"
        elif dtype == numpy.float64:
            return "double"
        else:
            raise ValueError, "unable to map dtype '%s'" % dtype

    def cdecl(self):
        return "  %s %s;\n" % (self.ctype(self.dtype), self.name)
    
    def struct_format(self):
        return self.dtype.char

    def prepare(self, arg):
        return [arg]

    def __len__(self):
        return self.dtype.itemsize




class ArrayStructField(StructField):
    def __init__(self, name, dtype, count):
        StructField.__init__(self, name, dtype)
        self.count = count

    def struct_format(self):
        return "%d%s" % (self.count, StructField.struct_format(self))

    def cdecl(self):
        return "  %s %s[%d];\n" % (self.ctype(self.dtype), self.name, self.count)

    def prepare(self, arg):
        assert len(arg) == self.count
        return arg

    def __len__(self):
        return self.count * StructField.__len__(self)




class Struct(object):
    def __init__(self, fields):
        self.fields = fields

    def make(self, **kwargs):
        import struct
        data = []
        for f in self.fields:
            data.extend(f.prepare(kwargs[f.name]))
        return struct.pack(self.struct_format(), *data)

    @memoize_method
    def struct_format(self):
        return "".join(f.struct_format() for f in self.fields)

    @memoize_method
    def cdecl(self, name):
        return "struct %s\n{\n%s};\n" % (name, "".join(f.cdecl() for f in self.fields))

    def __len__(self):
        a = sum(len(f) for f in self.fields)
        from struct import calcsize
        b = calcsize(self.struct_format())
        assert a == b
        return a




# planning --------------------------------------------------------------------
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
            max_ext_faces=None, max_faces=None, float_size=4, int_size=4):
        self.devdata = devdata
        self.ldis = ldis
        self.flux_par = flux_par

        self.max_ext_faces = max_ext_faces
        self.max_faces = max_faces

        self.float_size = float_size
        self.int_size = int_size

    def copy(self, devdata=None, ldis=None, flux_par=None, 
            max_ext_faces=None, max_faces=None, float_size=None, int_size=None):
        return ExecutionPlan(
                devdata or self.devdata,
                ldis or self.ldis,
                flux_par or self.flux_par,
                max_ext_faces or self.max_ext_faces,
                max_faces or self.max_faces,
                float_size or self.float_size,
                int_size or self.int_size,
                )

    def dofs_per_el(self):
        return self.ldis.node_count()

    def dofs_per_face(self):
        return self.ldis.face_node_count()

    def faces_per_el(self):
        return self.ldis.face_count()

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
        if self.max_ext_faces is None:
            return _ceiling(self.estimate_extface_count())
        else:
            return self.max_ext_faces

    def int_dof_smem(self):
        return self.devdata.align(self.block_el() * self.dofs_per_el() 
                * self.float_size)

    @memoize_method
    def ext_dof_smem(self):
        return self.devdata.align(self.get_extface_count()*
                self.dofs_per_face() * self.float_size)

    @memoize_method
    def get_face_pair_struct(self):
        return Struct([
            StructField("h", numpy.float32),
            StructField("order", numpy.float32),
            StructField("face_jacobian", numpy.float32),
            ArrayStructField("normal", numpy.float32, self.ldis.dimensions),
            StructField("a_base", numpy.uint16),
            StructField("b_base", numpy.uint16),
            StructField("a_ilist_number", numpy.uint8),
            StructField("b_ilist_number", numpy.uint8),
            StructField("bdry_flux_number", numpy.uint8), # 0 if not on boundary
            StructField("reserved", numpy.uint8),
            StructField("b_global_base", numpy.uint32),

            # memory handling here deserves a comment.
            # Interior face (bdry_flux_number==0) dofs are duplicated if they cross
            # a block boundary. The flux results for these dofs are written out 
            # to b_global_base in addition to their local location.
            #
            # Boundary face (bdry_flux_number!=0) dofs are read from b_global_base
            # linearly (not (!) using b_global_ilist_number) into the extface
            # space at b_base. They are not written out again.
            ])

    @memoize_method
    def get_block_header_struct(self):
        return Struct([
            StructField("els_in_block", numpy.int16),
            StructField("face_pairs_in_block", numpy.int16),
            ])

    def indexing_smem(self):
        return self.devdata.align(
                len(self.get_block_header_struct())
                +len(self.get_face_pair_struct())*self.face_pair_count())

    @memoize_method
    def face_count(self):
        if self.max_faces is not None:
            return self.max_faces
        else:
            return (self.block_el() * self.faces_per_el() + 
                    self.get_extface_count())

    def face_pair_count(self):
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

    @ivar opposite: the L{GPUFacestorage} instance for the face
      oposite to this one.
    """

    def __init__(self):
        self.opposite = None

class GPUInteriorFaceStorage(GPUFaceStorage):
    """Describes storage locations for a face local to an element in a block.

    @ivar el_face: a tuple C{(element, face_number)}.
    @ivar cpu_slice: the base index of the element in CPU numbering.
    @ivar native_index_list_id: 
    @ivar native_block: block in which element is to be found.
    @ivar native_block_el_num: number of this element in the C{native_block}.
    @ivar flux_face:
    @ivar dup_block: 
    @ivar dup_ext_face_number:
    """
    __slots__ = [
            "opposite",
            "el_face", "cpu_slice", "native_index_list_id",
            "native_block", "native_block_el_num",
            "dup_block", "dup_ext_face_number", "dup_index_list_id"]

    def __init__(self, el_face, cpu_slice, native_index_list_id,
            native_block, native_block_el_num, flux_face):
        GPUFaceStorage.__init__(self)
        self.el_face = el_face
        self.cpu_slice = cpu_slice
        self.native_index_list_id = native_index_list_id
        self.native_block = native_block
        self.native_block_el_num = native_block_el_num
        self.flux_face = flux_face

class GPUBoundaryFaceStorage(GPUFaceStorage):
    """Describes storage locations for a boundary face.

    @ivar cpu_bdry_index_in_floats: this face's starting index 
      in the CPU-based TAG_ALL boundary array [floats].
    @ivar gpu_bdry_index_in_floats: this face's starting index 
      in the GPU-based TAG_ALL boundary array [floats].
    @ivar dup_block: 
    @ivar dup_ext_face_number:
    """
    __slots__ = ["opposite", 
            "cpu_bdry_index_in_floats", 
            "gpu_bdry_index_in_floats", 
            "dup_block", "dup_ext_face_number"]

    def __init__(self, 
            cpu_bdry_index_in_floats,
            gpu_bdry_index_in_floats,
            ):
        GPUFaceStorage.__init__(self)
        self.cpu_bdry_index_in_floats = cpu_bdry_index_in_floats
        self.gpu_bdry_index_in_floats = gpu_bdry_index_in_floats




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

            # prepare a mapping:  block# -> # of external interfaces
            block2extifaces = {}

            for (e1, f1), (e2, f2) in mesh.both_interfaces():
                b1 = partition[e1.id]
                b2 = partition[e2.id]

                if b1 != b2:
                    block2extifaces[b1] = block2extifaces.get(b1, 0) + 1

            for el, face_nbr in mesh.tag_to_boundary[hedge.mesh.TAG_ALL]:
                b1 = partition[el.id]
                block2extifaces[b1] = block2extifaces.get(b1, 0) + 1

            blocks = {}
            for el_id, block in enumerate(partition):
                blocks.setdefault(block, []).append(el_id)

            block_elements = max(len(block_els) for block_els in blocks.itervalues())
            flux_par_s = _ceiling(block_elements/plan.flux_par.p)
            actual_plan = plan.copy(
                    max_ext_faces=max(block2extifaces.itervalues()),
                    max_faces=max(
                        len(blocks[b])*plan.faces_per_el()
                        + block2extifaces[b]
                        for b in range(len(blocks))),
                    flux_par=Parallelism(plan.flux_par.p, flux_par_s))
            assert actual_plan.max_faces % 2 == 0

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

        # make preliminary plan
        if plan is None:
            plan = self._make_plan(ldis, mesh)
        print "projected:", plan

        # partition mesh, obtain updated plan
        self.plan, self.partition = self._partition_mesh(mesh, plan)
        del plan
        print "actual:", self.plan

        # initialize superclass
        hedge.discretization.Discretization.__init__(self, mesh, ldis, debug=debug)

        # build our own data structures
        self.blocks = self._build_blocks()
        self.face_storage_map = self._build_face_storage_map()

        self.int_dof_floats = self.plan.int_dof_smem()//self.plan.float_size
        self.ext_dof_floats = self.plan.ext_dof_smem()//self.plan.float_size

        # check the ext_dof_smem estimate
        assert (self.devdata.align(
            max(len(block.ext_faces)*self.plan.dofs_per_face()
                for block in self.blocks)*self.plan.float_size)
            == self.plan.ext_dof_smem())




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
        # Side effects:
        # - fill in GPUBlock.extfaces
        # - set self.aligned_boundary_floats
        fsm = {}

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
                cpu_slice=slice(*self.find_el_range(el.id)), 
                native_index_list_id=iln,
                native_block=block, 
                native_block_el_num=block.get_el_index(el), 
                flux_face=flux_face
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
            
        self.aligned_boundary_floats = 0
        from hedge.mesh import TAG_ALL
        for bdry_fg, ldis in self.get_boundary(TAG_ALL).face_groups_and_ldis:
            aligned_fnc = self.devdata.align_dtype(ldis.face_node_count(), 
                    self.plan.float_size)
            for fp in bdry_fg.face_pairs:
                assert fp.opp_flux_face_index == fp.INVALID_INDEX
                assert (tuple(bdry_fg.index_lists[fp.opp_face_index_list_number]) 
                        == id_face_index_list)

                face1 = make_int_face(
                        bdry_fg.flux_faces[fp.flux_face_index], 
                        fp.face_index_list_number)
                face2 = GPUBoundaryFaceStorage(
                        fp.opp_el_base_index,
                        self.aligned_boundary_floats
                        )
                self.aligned_boundary_floats += aligned_fnc
                face1.opposite = face2
                face2.opposite = face1
                face2.dup_block = face1.native_block
                face2.dup_ext_face_number = face2.dup_block.register_ext_face(face2)

        return fsm




    def block_dof_count(self):
        return self.int_dof_floats + self.ext_dof_floats

    def gpu_dof_count(self):
        return self.block_dof_count() * len(self.blocks)

    def volume_to_gpu(self, field):
        from hedge.tools import log_shape
        ls = log_shape(field)
        if ls != ():
            result = numpy.array(ls, dtype=object)

            from pytools import indices_in_shape

            for i in indices_in_shape(ls):
                result[i] = self.volume_to_gpu(field[i])
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

    def volume_from_gpu(self, field, check=None):
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
            result = numpy.array(ls, dtype=object)
            for i in indices_in_shape(shape):
                result[i] = self.boundary_to_gpu(field[i], tag)
            return result
        else:
            result = cuda.pagelocked_empty(
                    (self.aligned_boundary_floats,),
                    dtype=field.dtype)
            result[self.gpu_boundary_embedding(tag)] = field
            return cuda.to_device(result)

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

        return self.volume_to_gpu(
                s.interpolate_volume_function(self, f, tgt_factory))

    def _new_bdry(self, tag, shape, create_func):
        result = numpy.empty(shape, dtype=object)
        from pytools import indices_in_shape
        bdry = self.get_boundary(TAG_ALL)
        for i in indices_in_shape(shape):
            result[i] = create_func(
                    (self.aligned_boundary_floats), 
                    dtype=numpy.float32)
        return result
    
    def boundary_empty(self, tag=hedge.mesh.TAG_ALL, shape=(), host=False):
        return self._new_bdry(tag, shape, gpuarray.empty)

    def boundary_zeros(self, tag=hedge.mesh.TAG_ALL, shape=(), host=False):
        return self._new_bdry(tag, shape, gpuarray.zeros)

    def interpolate_boundary_function(self, f, tag=hedge.mesh.TAG_ALL):
        s = hedge.discretization.Discretization
        def tgt_factory(shape, tag):
            return s.boundary_empty(self, shape, tag)

        return self.boundary_to_gpu(tag,
                s.interpolate_boundary_function(self, f, tag, tgt_factory))

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
    @memoize_method
    def _assemble_indexing_info(self):
        result = ""
        block_len = self.plan.indexing_smem()
        block_dofs = self.int_dof_floats + self.ext_dof_floats

        INVALID_U8 = (1<<8) - 1
        INVALID_U16 = (1<<16) - 1
        INVALID_U32 = (1<<32) - 1

        block_lengths = []

        for block in self.blocks:
            ldis = block.local_discretization
            el_dofs = ldis.node_count()

            faces_todo = set((el,face_nbr)
                    for el in block.elements
                    for face_nbr in range(ldis.face_count()))
            fp_blocks = []

            bf = isame = idiff = 0
            while faces_todo:
                elface = faces_todo.pop()

                int_face = self.face_storage_map[elface]
                opp = int_face.opposite

                if isinstance(opp, GPUBoundaryFaceStorage):
                    # boundary face
                    b_base = INVALID_U16
                    bdry_flux_number = 1
                    b_global_base = opp.gpu_bdry_index_in_floats
                    b_ilist_number = INVALID_U8
                    bf += 1
                else:
                    # interior face
                    b_base = opp.native_block_el_num*el_dofs
                    bdry_flux_number = 0
                    if opp.native_block == int_face.native_block:
                        # same block
                        faces_todo.remove(opp.el_face)
                        b_global_base = INVALID_U32
                        b_ilist_number = opp.native_index_list_id
                        isame += 1
                    else:
                        # different block
                        b_global_base = (
                                opp.native_block_el_num*el_dofs
                                + block_dofs*opp.native_block.number)
                        b_ilist_number = INVALID_U8
                        idiff += 1

                fp_blocks.append(
                        self.plan.get_face_pair_struct().make(
                            h=int_face.flux_face.h,
                            order=int_face.flux_face.order,
                            face_jacobian=int_face.flux_face.face_jacobian,
                            normal=int_face.flux_face.normal,
                            a_base=int_face.native_block_el_num*el_dofs,
                            b_base=b_base,
                            a_ilist_number=int_face.native_index_list_id,
                            b_ilist_number=b_ilist_number,
                            bdry_flux_number=bdry_flux_number,
                            reserved=0,
                            b_global_base=b_global_base,
                            ))

            bheader = self.plan.get_block_header_struct().make(
                    els_in_block=len(block.elements),
                    face_pairs_in_block=len(fp_blocks)
                    )
            block_data = bheader + "".join(fp_blocks)

            # take care of alignment
            missing_bytes = block_len - len(block_data)
            assert missing_bytes >= 0
            block_data = block_data + "\x00"*missing_bytes
            block_lengths.append(len(block_data))

            result += block_data

        # make sure the indexing_smem estimate is achieved
        assert max(block_lengths) == self.plan.indexing_smem()
        assert len(result) == block_len*len(self.blocks)
        return cuda.to_device(result)
    
    def preprocess_optemplate(self, optemplate):
        ind_inf = self._assemble_indexing_info()
        print optemplate
        from hedge.mesh import TAG_ALL
        self.gpu_boundary_embedding(TAG_ALL)
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

