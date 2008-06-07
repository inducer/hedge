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
from pytools import memoize, memoize_method




class Parallelism:
    """Defines how much of a task is accomplished sequentially vs. in parallel."""
    def __init__(self, parallel, serial):
        self.p = parallel
        self.s = serial

    def total(self):
        return self.p*self.s

    def __str__(self):
        return "(p%d s%d)" % (self.p, self.s)




class ExecutionPlan(object):
    def __init__(self, devdata):
        self.devdata = devdata

    def invalid_reason(self):
        if self.threads() >= self.devdata.max_threads:
            return "too many threads"

        if self.shared_mem_use() >= int(self.devdata.shared_memory): 
            return "too much shared memory"

        if self.threads()*self.registers() > self.devdata.registers:
            return "too many registers"
        return None

    @memoize_method
    def occupancy_record(self):
        from hedge.cuda.tools import OccupancyRecord
        return OccupancyRecord(self.devdata,
                self.threads(), self.shared_mem_use(),
                registers=self.registers())

    def __str__(self):
            return ("regs=%d par=%s threads=%d smem=%d occ=%f" % (
                self.registers(),
                self.parallelism, 
                self.threads(), 
                self.shared_mem_use(), 
                self.occupancy_record().occupancy,
                ))




@memoize
def find_microblock_size(devdata, dofs_per_el, float_size):
    from hedge.cuda.tools import exact_div, int_ceiling
    float_alignment = exact_div(devdata.align_bytes(float_size), float_size)

    for mb_align_chunks in range(1, 256):
        mb_aligned_floats = float_alignment * mb_align_chunks
        mb_elements = mb_aligned_floats // dofs_per_el
        mb_floats = dofs_per_el*mb_elements
        overhead = (mb_aligned_floats-mb_floats)/mb_aligned_floats
        if overhead <= 0.1:
            return mb_elements, mb_aligned_floats, mb_align_chunks

    assert False, "a valid microblock size was not found"




class FluxExecutionPlan(ExecutionPlan):
    def __init__(self, devdata, ldis, 
            parallelism, 
            max_ext_faces=None, max_faces=None, 
            float_type=numpy.float32, 
            ):
        ExecutionPlan.__init__(self, devdata)
        self.ldis = ldis
        self.parallelism = parallelism

        self.max_ext_faces = max_ext_faces
        self.max_faces = max_faces

        self.float_type = numpy.dtype(float_type)

        self.mb_elements, self.mb_aligned_floats, self.mb_chunks = find_microblock_size(
                self.devdata, 
                ldis.node_count(), 
                self.float_size)

    @property
    def float_size(self):
        return self.float_type.itemsize

    def copy(self, devdata=None, ldis=None, parallelism=None, 
            max_ext_faces=None, max_faces=None, float_type=None):
        return self.__class__(
                devdata or self.devdata,
                ldis or self.ldis,
                parallelism or self.parallelism,
                max_ext_faces or self.max_ext_faces,
                max_faces or self.max_faces,
                float_type or self.float_type,
                )

    def dofs_per_el(self):
        return self.ldis.node_count()

    def dofs_per_face(self):
        return self.ldis.face_node_count()

    def faces_per_el(self):
        return self.ldis.face_count()

    def microblocks_per_block(self):
        return self.parallelism.total()

    def elements_per_block(self):
        return self.microblocks_per_block()*self.mb_elements

    def dofs_per_block(self):
        return self.microblocks_per_block()*self.mb_aligned_floats

    @memoize_method
    def estimate_extface_count(self):
        d = self.ldis.dimensions

        # How many equivalent cubes would I need to tesselate the same space
        # as the elements in my thread block?
        from pytools import factorial
        equiv_cubes = self.elements_per_block() / factorial(d)

        # If these cubes in turn formed a perfect macro-cube, how long would
        # its side be?
        macrocube_side = equiv_cubes ** (1/d)

        # What total face area does the macro-cube have?
        macrocube_face_area = 2*d * macrocube_side ** (d-1)

        # How many of my faces do I need to tesselate this face area?
        return macrocube_face_area * factorial(d-1)

    def get_extface_count(self):
        from hedge.cuda.tools import int_ceiling

        if self.max_ext_faces is None:
            return int_ceiling(self.estimate_extface_count())
        else:
            return self.max_ext_faces

    @memoize_method
    def face_count(self):
        if self.max_faces is not None:
            return self.max_faces
        else:
            return (self.elements_per_block() * self.faces_per_el() + 
                    self.get_extface_count())

    def face_pair_count(self):
        return (self.face_count()+1) // 2

    @memoize_method
    def shared_mem_use(self):
        from hedge.cuda.execute import face_pair_struct
        d = self.ldis.dimensions

        if self.dofs_per_face() > 255:
            index_lists_entry_size = 2
        else:
            index_lists_entry_size = 1

        return (128 # parameters, block header, small extra stuff
                + self.elements_per_block()*self.faces_per_el()*self.dofs_per_face()*self.float_size
                + len(face_pair_struct(self.float_type, d))*self.face_pair_count()
                + index_lists_entry_size*20*self.dofs_per_face()
                )

    def threads(self):
        return self.parallelism.p*self.mb_aligned_floats

    def registers(self):
        return 12

    @memoize_method
    def localop_plan(self):
        def generate_valid_plans():
            from hedge.cuda.tools import int_ceiling

            chunk_sizes = range(16, self.mb_elements*self.dofs_per_el()+1, 16)

            if not chunk_sizes:
                chunk_sizes = [16]

            for pe in range(1,32):
                from hedge.cuda.tools import int_ceiling
                localop_par = Parallelism(pe, 8)
                for chunk_size in chunk_sizes:
                    plan = LocalOpExecutionPlan(self, localop_par, chunk_size)
                    if plan.invalid_reason() is None:
                        yield plan

        plans = list(generate_valid_plans())

        if not plans:
            raise RuntimeError, "no valid CUDA execution plans found"

        desired_occup = max(plan.occupancy_record().occupancy for plan in plans)
        if desired_occup > 0.66:
            # see http://forums.nvidia.com/lofiversion/index.php?t67766.html
            desired_occup = 0.66

        good_plans = [p for p in plans
                if p.occupancy_record().occupancy >= desired_occup - 1e-10
                ]

        # optimize for minimum waste
        from pytools import argmax2
        return argmax2((p, p.parallelism.total()) for p in good_plans)

    def __str__(self):
            return ("%s mb_elements=%d" % (
                ExecutionPlan.__str__(self),
                self.mb_elements,
                ))




class LocalOpExecutionPlan(ExecutionPlan):
    def __init__(self, flux_plan, parallelism, chunk_size):
        ExecutionPlan.__init__(self, flux_plan.devdata)
        self.flux_plan = flux_plan
        self.parallelism = parallelism
        self.chunk_size = chunk_size

    def max_elements_touched_by_chunk(self):
        fplan = self.flux_plan

        from hedge.cuda.tools import int_ceiling
        if fplan.dofs_per_el() > self.chunk_size:
            return 2
        else:
            return int_ceiling(self.chunk_size/fplan.dofs_per_el()) + 1

    def dofs_per_macroblock(self):
        return self.parallelism.total() * self.flux_plan.mb_aligned_floats

    @memoize_method
    def shared_mem_use(self):
        fplan = self.flux_plan
        
        return (64 # parameters, block header, small extra stuff
               + fplan.float_size * (
                   # chunk of the differentiation matrix
                   + self.chunk_size # this many rows
                   * fplan.dofs_per_el()
                   * fplan.ldis.dimensions # r,s,t

                   + self.parallelism.p
                   * self.max_elements_touched_by_chunk()
                   * fplan.dofs_per_el()
                   )
               )

    def threads(self):
        return self.parallelism.p*self.chunk_size

    def registers(self):
        return 18

    def __str__(self):
            return ("%s chunk_size=%d" % (
                ExecutionPlan.__str__(self),
                self.chunk_size,
                ))




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

