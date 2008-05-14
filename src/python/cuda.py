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
                self.indexing_bytes_per_face_pair()*self.facepair_count(),
                self.devdata.align_bytes())

    def facepair_count(self):
        return (self.face_count()+1) // 2

    @memoize_method
    def shared_mem_use(self):
        return (self.int_dof_smem() 
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



class CudaDiscretization(hedge.discretization.Discretization):
    def make_plan(self, ldis, mesh):
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




    def partition_mesh(self, mesh, plan):
        # search for mesh partition that matches plan
        from pymetis import part_graph
        part_count = len(mesh.elements)//plan.flux_par.total()+1
        while True:
            cuts, partition = part_graph(part_count,
                    mesh.element_adjacency_graph(),
                    vweights=[1000]*len(mesh.elements))

            # prepare a mapping from (el, face_nr) to the block
            # at the other end of the interface, if different from
            # current. concurrently, prepare a mapping 
            #  block# -> # of interfaces
            elface2block = {}
            block2extifaces = {}

            for elface1, elface2 in mesh.interfaces:
                e1, f1 = elface1
                e2, f2 = elface2
                r1 = partition[e1.id]
                r2 = partition[e2.id]

                if r1 != r2:
                    block2extifaces[r1] = block2extifaces.get(r1, 0) + 1
                    block2extifaces[r2] = block2extifaces.get(r2, 0) + 1

                    elface2block[elface1] = r2
                    elface2block[elface2] = r1

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
            
        return block_elements, actual_plan, blocks, partition




    def __init__(self, mesh, local_discretization=None, 
            order=None, plan=None, init_cuda=True, debug=False, 
            dev=None):
        ldis = self.get_local_discretization(mesh, local_discretization, order)

        import pycuda.driver as cuda
        if init_cuda:
            cuda.init()

        if dev is None:
            assert cuda.Device.count() >= 1
            dev = cuda.Device(0)

        self.device = dev
        self.devdata = DeviceData(dev)

        if plan is None:
            plan = self.make_plan(ldis, mesh)
            print "projected:", plan

        block_elements, plan, blocks, partition = self.partition_mesh(
                mesh, plan)
        self.plan = plan
        print "actual:", plan

        # initialize superclass -----------------------------------------------
        hedge.discretization.Discretization.__init__(self, mesh, ldis, debug=debug)











            





            






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

