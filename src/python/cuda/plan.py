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




def optimize_plan(plan_generator, max_func):
    plans = list(p for p in plan_generator()
            if p.invalid_reason() is None)

    if not plans:
        raise RuntimeError, "no valid CUDA execution plans found"

    desired_occup = max(plan.occupancy_record().occupancy for plan in plans)
    #if desired_occup > 0.75:
        # see http://forums.nvidia.com/lofiversion/index.php?t67766.html
        #desired_occup = 0.75

    from pytools import argmax2
    return argmax2((p, max_func(p)) 
            for p in plans
            if p.occupancy_record().occupancy >= desired_occup - 1e-10
            )




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
            return ("regs=%d threads=%d smem=%d occ=%f" % (
                self.registers(),
                self.threads(), 
                self.shared_mem_use(), 
                self.occupancy_record().occupancy,
                ))




@memoize
def find_microblock_size(devdata, dofs_per_el, float_size):
    from hedge.cuda.tools import exact_div, int_ceiling
    align_size = exact_div(devdata.align_bytes(float_size), float_size)

    for mb_align_chunks in range(1, 256):
        mb_aligned_floats = align_size * mb_align_chunks
        mb_elements = mb_aligned_floats // dofs_per_el
        mb_floats = dofs_per_el*mb_elements
        overhead = (mb_aligned_floats-mb_floats)/mb_aligned_floats
        if overhead <= 0.05:
            from pytools import Record
            return Record(
                    align_size=align_size,
                    elements=mb_elements,
                    aligned_floats=mb_aligned_floats,
                    accesses=mb_align_chunks
                    )

    assert False, "a valid microblock size was not found"




class FluxExecutionPlan(ExecutionPlan):
    def __init__(self, devdata, ldis, 
            parallel_faces, mbs_per_block,
            max_ext_faces=None, max_faces=None, 
            float_type=numpy.float32, 
            diff_chunk=False,
            ):
        ExecutionPlan.__init__(self, devdata)
        self.ldis = ldis
        self.parallel_faces = parallel_faces
        self.mbs_per_block = mbs_per_block

        self.max_ext_faces = max_ext_faces
        self.max_faces = max_faces

        self.float_type = numpy.dtype(float_type)

        self.diff_chunk = diff_chunk

        self.microblock = find_microblock_size(
                self.devdata, ldis.node_count(), self.float_size)

    @property
    def float_size(self):
        return self.float_type.itemsize

    def copy(self, devdata=None, ldis=None, 
            parallel_faces=None, mbs_per_block=None,
            max_ext_faces=None, max_faces=None, float_type=None):
        return self.__class__(
                devdata or self.devdata,
                ldis or self.ldis,
                parallel_faces or self.parallel_faces,
                mbs_per_block or self.mbs_per_block,
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

    def face_dofs_per_el(self):
        return self.ldis.face_node_count()*self.faces_per_el()

    def microblocks_per_block(self):
        return self.mbs_per_block

    def elements_per_block(self):
        return self.microblocks_per_block()*self.microblock.elements

    def dofs_per_block(self):
        return self.microblocks_per_block()*self.microblock.aligned_floats

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

    def face_dofs_per_microblock(self):
        return self.microblock.elements*self.faces_per_el()*self.dofs_per_face()

    def aligned_face_dofs_per_microblock(self):
        return self.devdata.align_dtype(
                self.face_dofs_per_microblock(),
                self.float_size)

    @memoize_method
    def shared_mem_use(self):
        from hedge.cuda.execute import face_pair_struct
        d = self.ldis.dimensions

        if self.dofs_per_face() > 255:
            index_lists_entry_size = 2
        else:
            index_lists_entry_size = 1

        return (128 # parameters, block header, small extra stuff
                + self.aligned_face_dofs_per_microblock()
                * self.microblocks_per_block()
                * self.float_size
                + len(face_pair_struct(self.float_type, d))*self.face_pair_count()
                + index_lists_entry_size*20*self.dofs_per_face()
                )

    def threads(self):
        return self.parallel_faces*self.dofs_per_face()

    def registers(self):
        return 16

    @memoize_method
    def diff_plan(self):
        def generate_plans():
            from hedge.cuda.tools import int_ceiling

            chunk_sizes = range(self.microblock.align_size, 
                    self.microblock.elements*self.dofs_per_el()+1, 
                    self.microblock.align_size)

            for pe in range(1,32):
                from hedge.cuda.tools import int_ceiling
                localop_par = Parallelism(pe, 256//pe)
                for chunk_size in chunk_sizes:
                    yield ChunkedDiffExecutionPlan(self, localop_par, chunk_size)

        return optimize_plan(
                generate_plans,
                lambda plan: plan.parallelism.total()
                )

    @memoize_method
    def flux_lifting_plan(self):
        def generate_valid_plans():
            from hedge.cuda.tools import int_ceiling

            chunk_sizes = range(self.microblock.align_size, 
                    self.microblock.elements*self.dofs_per_el()+1, 
                    self.microblock.align_size)

            for pe in range(1,32):
                from hedge.cuda.tools import int_ceiling
                localop_par = Parallelism(pe, 256//pe)
                for chunk_size in chunk_sizes:
                    yield FluxLiftingExecutionPlan(self, localop_par, chunk_size)

        return optimize_plan(
                generate_valid_plans,
                lambda plan: plan.parallelism.total()
                )

    def __str__(self):
            return ("%s pfaces=%d mbs_per_block=%d mb_elements=%d" % (
                ExecutionPlan.__str__(self),
                self.parallel_faces,
                self.mbs_per_block,
                self.microblock.elements,
                ))




class ChunkedLocalOperatorExecutionPlan(ExecutionPlan):
    def __init__(self, flux_plan, parallelism, chunk_size):
        ExecutionPlan.__init__(self, flux_plan.devdata)
        self.flux_plan = flux_plan
        self.parallelism = parallelism
        self.chunk_size = chunk_size

    def chunks_per_microblock(self):
        from hedge.cuda.tools import int_ceiling
        return int_ceiling(
                self.flux_plan.microblock.aligned_floats/self.chunk_size)

    def dofs_per_macroblock(self):
        return self.parallelism.total() * self.flux_plan.microblock.aligned_floats

    def max_elements_touched_by_chunk(self):
        fplan = self.flux_plan

        from hedge.cuda.tools import int_ceiling
        if fplan.dofs_per_el() > self.chunk_size:
            return 2
        else:
            return int_ceiling(self.chunk_size/fplan.dofs_per_el()) + 1

    @memoize_method
    def shared_mem_use(self):
        fplan = self.flux_plan
        
        return (64 # parameters, block header, small extra stuff
               + fplan.float_size * (
                   # chunk of the differentiation matrix
                   + self.chunk_size # this many rows
                   * self.columns()
                   # fetch buffer for each chunk
                   + self.parallelism.p
                   * self.chunk_size
                   * self.fetch_buffer_chunks()
                   )
               )

    def threads(self):
        return self.parallelism.p*self.chunk_size

    def __str__(self):
            return ("%s par=%s chunk_size=%d" % (
                ExecutionPlan.__str__(self),
                self.parallelism,
                self.chunk_size,
                ))





class ChunkedDiffExecutionPlan(ChunkedLocalOperatorExecutionPlan):
    def columns(self):
        fplan = self.flux_plan
        return fplan.dofs_per_el() * fplan.ldis.dimensions # r,s,t

    def registers(self):
        return 17

    def fetch_buffer_chunks(self):
        return 0





class FluxLiftingExecutionPlan(ChunkedLocalOperatorExecutionPlan):
    def columns(self):
        return self.flux_plan.face_dofs_per_el()

    def registers(self):
        return 13

    def fetch_buffer_chunks(self):
        return 1



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

