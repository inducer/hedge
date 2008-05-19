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
        return int_ceiling(elements, exact_div(self.align_bytes(), dtype_size))

    def align_bytes(self):
        return 16




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
