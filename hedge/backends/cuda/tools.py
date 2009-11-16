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




import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import numpy
from pytools import Record
from pytools.log import LogQuantity
import hedge.timestep.base




class FakeGPUArray(Record):
    def __init__(self):
        Record.__init__(self, gpudata=0)




# tools -----------------------------------------------------------------------
def exact_div(dividend, divisor):
    quot, rem = divmod(dividend, divisor)
    assert rem == 0
    return quot

def int_ceiling(value, multiple_of=1):
    """Round *value* up to be a *multiple_of* something."""
    # Mimicks the Excel "floor" function (for code stolen from occupany calculator)

    from math import ceil
    return int(ceil(value/multiple_of))*multiple_of

def int_floor(value, multiple_of=1):
    """Round *value* down to be a *multiple_of* something."""
    # Mimicks the Excel "floor" function (for code stolen from occupany calculator)

    from math import floor
    return int(floor(value/multiple_of))*multiple_of

def pad(s, block_size):
    missing_bytes = block_size - len(s)
    assert missing_bytes >= 0
    return s + "\x00"*missing_bytes

def pad_and_join(blocks, block_size):
    return "".join(pad(b, block_size) for b in blocks)

def make_blocks(devdata, data):
    from hedge.backends.cuda.tools import pad_and_join

    blocks = ["".join(b) for b in data]
    block_size = devdata.align(max(len(b) for b in blocks))

    class BlockedDataStructure(Record):
        pass

    return BlockedDataStructure(
            blocks=cuda.to_device(pad_and_join(blocks, block_size)),
            max_per_block=max(len(b) for b in data),
            block_size=block_size,
            )

def make_superblocks(devdata, struct_name, single_item, multi_item, extra_fields={}):
    from hedge.backends.cuda.tools import pad_and_join

    # single_item = [([ block1, block2, ... ], decl), ...]
    # multi_item = [([ [ item1, item2, ...], ... ], decl), ...]

    multi_blocks = [
            ["".join(s) for s in part_data]
            for part_data, part_decls in multi_item]
    block_sizes = [
            max(len(b) for b in part_blocks)
            for part_blocks in multi_blocks]

    from pytools import single_valued
    block_count = single_valued(
            len(si_part_blocks) for si_part_blocks, si_part_decl in single_item)

    from codepy.cgen import Struct, ArrayOf

    struct_members = []
    for part_data, part_decl in single_item:
        assert block_count == len(part_data)
        single_valued(len(block) for block in part_data)
        struct_members.append(part_decl)

    for part_data, part_decl in multi_item:
        struct_members.append(
                ArrayOf(part_decl, max(len(s) for s in part_data)))

    superblocks = []
    for superblock_num in range(block_count):
        data = ""
        for part_data, part_decl in single_item:
            data += part_data[superblock_num]

        for part_blocks, part_size in zip(multi_blocks, block_sizes):
            assert block_count == len(part_blocks)
            data += pad(part_blocks[superblock_num], part_size)

        superblocks.append(data)

    superblock_size = devdata.align(
            single_valued(len(sb) for sb in superblocks))

    data = pad_and_join(superblocks, superblock_size)
    assert len(data) == superblock_size*block_count

    class SuperblockedDataStructure(Record):
        pass

    return SuperblockedDataStructure(
            struct=Struct(struct_name, struct_members),
            device_memory=cuda.to_device(data),
            block_bytes=superblock_size,
            data=data,
            **extra_fields
            )




class CUDALinearCombiner:
    def __init__(self, result_dtype, scalar_dtype, sample_vec, arg_count):
        from pycuda.elementwise import get_linear_combination_kernel
        self.vector_dtype = sample_vec.dtype
        self.result_dtype = result_dtype
        self.shape = sample_vec.shape
        self.block = sample_vec._block
        self.grid = sample_vec._grid
        self.mem_size = sample_vec.mem_size

        self.kernel, _ = get_linear_combination_kernel(
                arg_count*((False, scalar_dtype, self.vector_dtype),),
                result_dtype)

    def __call__(self, *args):
        result = gpuarray.empty(self.shape, self.result_dtype)

        knl_args = []
        for fac, vec in args:
            if vec.dtype != self.vector_dtype:
                raise TypeError("unexpected vector type in CUDA linear combination")

            knl_args.append(fac)
            knl_args.append(vec.gpudata)

        knl_args.append(result.gpudata)
        knl_args.append(self.mem_size)

        self.kernel.set_block_shape(*self.block)
        self.kernel.prepared_async_call(self.grid, None, *knl_args)

        return result




# code generation -------------------------------------------------------------
def get_load_code(dest, base, bytes, word_type=numpy.uint32,
        descr=None):
    from codepy.cgen import \
            Pointer, POD, Value, \
            Comment, Block, Line, \
            Constant, For, Statement

    from codepy.cgen import dtype_to_ctype
    copy_dtype = numpy.dtype(word_type)
    copy_dtype_str = dtype_to_ctype(copy_dtype)

    code = []
    if descr is not None:
        code.append(Comment(descr))

    code.extend([
        Block([
            Constant(Pointer(POD(copy_dtype, "load_base")),
                ("(%s *) (%s)" % (copy_dtype_str, base))),
            For("unsigned word_nr = THREAD_NUM",
                "word_nr*sizeof(int) < (%s)" % bytes,
                "word_nr += COALESCING_THREAD_COUNT",
                Statement("((%s *) (%s))[word_nr] = load_base[word_nr]"
                    % (copy_dtype_str, dest))
                ),
            ]),
        Line(),
        ])

    return code




def unroll(body_gen, total_number, max_unroll=None, start=0):
    from codepy.cgen import For, Line, Block
    from pytools import flatten

    if max_unroll is None:
        max_unroll = total_number

    result = []

    if total_number > max_unroll:
        loop_items = (total_number // max_unroll) * max_unroll

        result.extend([
                For("unsigned j = 0",
                    "j < %d" % loop_items,
                    "j += %d" % max_unroll,
                    Block(list(flatten(
                        body_gen("(j+%d)" % i)
                        for i in range(max_unroll))))
                    ),
                Line()
                ])
        start += loop_items

    result.extend(flatten(
        body_gen(i) for i in range(start, total_number)))

    return result




# CUDA timer ------------------------------------------------------------------
class CUDAIntervalTimer(LogQuantity):
    """Records elapsed times."""

    class _SubTimer:
        def __init__(self, itimer, stream):
            self.itimer = itimer
            self.stream = stream
            self.start_event = cuda.Event().record(self.stream)

        def stop(self):
            self.stop_event = cuda.Event().record(self.stream)
            return self

        def submit(self):
            self.itimer.callables.append(self.elapsed)

        def elapsed(self):
            return 1e-3*(self.stop_event.time_since(self.start_event))

    def __init__(self, name, description=None, stream=None):
        LogQuantity.__init__(self, name, "s", description)
        self.stream = stream
        self.callables = []

    def start_sub_timer(self):
        return self._SubTimer(self, self.stream)

    def add_timer_callable(self, f):
        self.callables.append(f)

    def __call__(self):
        result = sum(clbl() for clbl in self.callables)
        self.callables = []
        return result




# MPI interaction -------------------------------------------------------------
class CudaDevInfo(Record):
    pass

def mpi_get_default_device(comm, dev_filter=lambda dev: True):
    from socket import gethostname
    cuda_devices = [cuda.Device(i) for i in range(cuda.Device.count())]
    cuda_devprops = [CudaDevInfo(
        index=i,
        name=d.name(),
        memory=d.total_memory(),
        multiprocessors=d.get_attribute(cuda.device_attribute.MULTIPROCESSOR_COUNT))
        for i, d in enumerate(cuda_devices)
        if dev_filter(d)]

    all_devprops = comm.gather((gethostname(), cuda_devprops), root=0)

    if comm.rank == 0:
        host_to_devs = {}
        rank_to_host = {}
        for rank, (hostname, cuda_devprops) in enumerate(all_devprops):
            for dev in cuda_devprops:
                if hostname in host_to_devs:
                    assert cuda_devprops == host_to_devs[hostname]
                else:
                    host_to_devs[hostname] = cuda_devprops

                rank_to_host[rank] = hostname

        def grab_device(rank):
            devs = host_to_devs[rank_to_host[rank]]
            if not devs:
                raise RuntimeError("No available CUDA device for rank %d (%s)"
                        % (rank, rank_to_host[rank]))
            else:
                return devs.pop(0)

        rank_to_device = [grab_device(rank) for rank in range(len(all_devprops))]
        for rank, dev_info in enumerate(rank_to_device):
            print "rank %d (%s) is using %s (idx: %d)" % (
                    rank, rank_to_host[rank], dev_info.name, dev_info.index)

    else:
        rank_to_device = None

    return cuda.Device(comm.scatter(rank_to_device, root=0).index)
