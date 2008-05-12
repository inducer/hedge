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
    def __init__(self, dev=None):
        import pycuda.driver as drv

        if dev is None:
            assert drv.Device.count() >= 1
            dev = drv.Device(0)
        self.max_threads = dev.get_attribute(drv.device_attribute.MAX_THREADS_PER_BLOCK)
        self.warp_size = dev.get_attribute(drv.device_attribute.WARP_SIZE)
        self.thread_blocks_per_mp = 8
        self.warps_per_mp = 24
        self.registers = dev.get_attribute(drv.device_attribute.REGISTERS_PER_BLOCK)
        self.shared_memory = dev.get_attribute(drv.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK)




class OccupancyRecord:
    def __init__(self, threads, shared_mem=0, registers=0, devdata=None):
        if devdata is None:
            devdata = DeviceData()

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
    def __init__(self, ldis, flux_par, float_size=4, int_size=4):
        self.ldis = ldis
        self.flux_par = flux_par
        self.float_size = float_size
        self.int_size = int_size

    def dofs_per_el(self):
        return self.ldis.node_count()

    def dofs_per_face(self):
        return len(self.ldis.face_indices()[0])

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

    def int_dof_smem(self):
        return self.block_el() * self.dofs_per_el() * self.float_size

    @memoize_method
    def ext_dof_smem(self):
        return (self.estimate_extface_count()
                * self.dofs_per_face() * self.float_size)

    @memoize_method
    def face_count(self):
        from math import ceil
        return (self.block_el() * self.faces_per_el() + 
                int(ceil(self.estimate_extface_count()))
                )

    @memoize_method
    def shared_mem_use(self):
        facepair_count = self.face_count() // 2

        # How much planning info per face pair?
        # h, order, face_jacobian, normal
        # a_base, b_base, a_ilist_number, b_ilist_number
        # bwrite_base, bwrite_ilist_number
        # Total: 3+d floats, 6 ints

        planning_smem = (3+self.ldis.dimensions)*self.float_size + 6*self.int_size
        return self.int_dof_smem() + self.ext_dof_smem() + planning_smem

    def threads(self):
        return self.flux_par.p*self.faces_per_el()*self.dofs_per_face()

    def invalid_reason(self, devdata=None):
        if devdata is None:
            devdata = DeviceData()

        if self.threads() >= devdata.max_threads:
            return "too many threads"
        if self.shared_mem_use() >= devdata.shared_memory:
            return "too much shared memory"
        return None

    @memoize_method
    def occupancy_record(self):
        return OccupancyRecord(self.threads(), self.shared_mem_use())

    @memoize_method
    def find_localop_par(self):
        threads = self.threads()
        total_threads = self.block_el()*self.dofs_per_el()
        ser, rem = divmod(total_threads, threads)
        if rem == 0:
            return Parallelism(threads, ser)
        else:
            return Parallelism(threads, ser+1)




class CudaDiscretization(hedge.discretization.Discretization):
    def __init__(self, *args, **kwargs):
        # argument parsing ----------------------------------------------------

        if "plan" in kwargs:
            plan = kwargs["plan"]
            del kwargs["plan"]
        else:
            plan = None

        if "init_cuda" in kwargs:
            init_cuda = kwargs["init_cuda"]
            del kwargs["init_cuda"]
        else:
            init_cuda = True

        # cuda initialization -------------------------------------------------
        if init_cuda:
            import pycuda.driver as cuda
            cuda.init()

        hedge.discretization.Discretization.__init__(self, *args, **kwargs)

        # plan generation -----------------------------------------------------
        eg, = self.element_groups

        if plan is None:
            def generate_valid_plans():
                for pe in range(2,32):
                    for se in range(1,256):
                        flux_par = Parallelism(pe, se)
                        plan = ExecutionPlan(eg.local_discretization, flux_par)
                        if plan.invalid_reason() is None:
                            yield plan

            plans = list(generate_valid_plans())

            if not plans:
                raise RuntimeError, "no valid CUDA execution plans found"

            max_occup = max(plan.occupancy_record().occupancy for plan in plans)
            good_plans = [p for p in generate_valid_plans()
                    if p.occupancy_record().occupancy > max_occup - 1e-10]

            from pytools import argmax2
            self.plan = argmax2((p, p.block_el()) for p in good_plans)

            print "cuda plan: flux_par=%s threads=%d int_smem=%d ext_smem=%d smem=%d occ=%f" % (
                    self.plan.flux_par, self.plan.threads(), 
                    self.plan.int_dof_smem(), self.plan.ext_dof_smem(),
                    self.plan.shared_mem_use(), 
                    self.plan.occupancy_record().occupancy)
        else:
            self.plan = plan
            





            






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
