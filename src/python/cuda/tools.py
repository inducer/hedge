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




# tools -----------------------------------------------------------------------
def exact_div(dividend, divisor):
    quot, rem = divmod(dividend, divisor)
    assert rem == 0
    return quot

def int_ceiling(value, multiple_of=1):
    """Round C{value} up to be a C{multiple_of} something."""
    # Mimicks the Excel "floor" function (for code stolen from occupany calculator)

    from math import ceil
    return int(ceil(value/multiple_of))*multiple_of

def int_floor(value, multiple_of=1):
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

def pad(s, block_size):
    missing_bytes = block_size - len(s)
    assert missing_bytes >= 0
    return s + "\x00"*missing_bytes

def pad_and_join(blocks, block_size):
    return "".join(pad(b, block_size) for b in blocks)

def make_blocks(devdata, data):
    from pytools import Record
    from hedge.cuda.tools import pad_and_join
    from pytools import Record

    blocks = ["".join(b) for b in data]
    block_size = devdata.align(max(len(b) for b in blocks))
    return Record(
            blocks=cuda.to_device(pad_and_join(blocks, block_size)),
            max_per_block=max(len(b) for b in data),
            block_size=block_size,
            )

def make_superblocks(devdata, struct_name, single_item, multi_item):
    from pytools import Record
    from hedge.cuda.tools import pad_and_join
    from pytools import Record

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

    from hedge.cuda.cgen import Struct, Value, ArrayOf

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

    return Record(
            struct=Struct(struct_name, struct_members),
            device_memory=cuda.to_device(data),
            block_bytes=superblock_size,
            data=data,
            )



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
        return int_ceiling(bytes, self.align_bytes())

    def align_dtype(self, elements, dtype_size):
        return int_ceiling(elements, 
                self.align_words(dtype_size))

    def align_words(self, word_size):
        return exact_div(self.align_bytes(word_size), word_size)

    def align_bytes(self, wordsize=4):
        if wordsize == 4:
            return 64
        elif wordsize == 8:
            return 128
        elif wordsize == 16:
            return 128
        else:
            raise ValueError, "no alignment possible for fetches of size %d" % wordsize

    def coalesce(self, thread_count):
        return int_ceiling(thread_count, 16)

    @staticmethod
    def make_valid_tex_channel_count(size):
        valid_sizes = [1,2,4]
        for vs in valid_sizes:
            if size <= vs:
                return vs

        raise ValueError, "could not enlarge argument to valid channel count"




class OccupancyRecord:
    def __init__(self, devdata, threads, shared_mem=0, registers=0):
        if threads > devdata.max_threads:
            raise ValueError("too many threads")

        # copied literally from occupancy calculator
        alloc_warps = int_ceiling(threads/devdata.warp_size)
        alloc_regs = int_ceiling(alloc_warps*2, 4)*16*registers
        alloc_smem = int_ceiling(shared_mem, 512)

        self.tb_per_mp_limits = [(devdata.thread_blocks_per_mp, "device"),
                (int_floor(devdata.warps_per_mp/alloc_warps), "warps")
                ]
        if registers > 0:
            self.tb_per_mp_limits.append((int_floor(devdata.registers/alloc_regs), "regs"))
        if shared_mem > 0:
            self.tb_per_mp_limits.append((int_floor(devdata.shared_memory/alloc_smem), "smem"))

        self.tb_per_mp, self.limited_by = min(self.tb_per_mp_limits)

        self.warps_per_mp = self.tb_per_mp * alloc_warps
        self.occupancy = self.warps_per_mp / devdata.warps_per_mp




def _test_occupancy():
    for threads in range(32, 512, 16):
        for smem in range(1024, 16384+1, 1024):
            occ = Occupancy(threads, smem)
            print "t%d s%d: %f %s" % (threads, smem, occ.occupancy, occ.limited_by)




if __name__ == "__main__":
    import pycuda.driver as drv
    drv.init()

    _test_occupancy()
