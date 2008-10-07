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
    """Defines how much of a task is accomplished sequentially vs. in-line parallel
    vs. completely in parallel.

    To fix terminology: 
    
    - "parallel" means "in separate threads".
    - "inline" means "in the same thread, but sharing some data."
    - "serial" means "in the same thread, but in separate, data-independent stages."
    """
    def __init__(self, parallel, inline, serial):
        self.parallel = parallel
        self.inline = inline
        self.serial = serial

    def total(self):
        return self.parallel*self.inline*self.serial

    def __str__(self):
        return "(%s)" % (" ".join("%s%d" % (cat, count) for cat, count in [
            ("p", self.parallel), ("i", self.inline), ("s", self.serial)]
            if count != 1))




def optimize_plan(plan_generator, max_func, maximize, debug=False, occupancy_slack=0.5,
        log_filename=None):
    plans = list(p for p in plan_generator()
            if p.invalid_reason() is None)

    if not plans:
        raise RuntimeError, "no valid CUDA execution plans found"

    max_occup = max(plan.occupancy_record().occupancy for plan in plans)
    desired_occup = occupancy_slack*max_occup

    if log_filename is not None:
        from pytools import single_valued
        feature_columns = single_valued(p.feature_columns() for p in plans)
        feature_names = [fc.split()[0] for fc in feature_columns]

        try:
            import sqlite3 as sqlite
        except ImportError:
            from pysqlite2 import dbapi2 as sqlite

        db_conn = sqlite.connect("plan-%s.dat" % log_filename)

        db_conn.execute("""
              create table data (
                id integer primary key autoincrement,
                %s,
                value real)""" 
                % ", ".join(feature_columns))

    plan_values = []
    for p in plans:
        if p.occupancy_record().occupancy >= desired_occup - 1e-10:
            if debug:
                print "<---- trying %s:" % p

            value = max_func(p)
            if isinstance(value, tuple):
                extra_info = value[1:]
                value = value[0]

            if value is not None:
                if debug:
                    print "----> yielded %g" % (value)
                plan_values.append((p, value))

                if log_filename is not None:
                    db_conn.execute(
                            "insert into data (%s,value) values (%s)"
                            % (", ".join(feature_names), 
                                ",".join(["?"]*(1+len(feature_names)))),
                            p.features(*extra_info)+(value,))

    if log_filename is not None:
        db_conn.commit()

    from pytools import argmax2, argmin2
    if maximize:
        result = argmax2(plan_values)
    else:
        result = argmin2(plan_values)

    if debug:
        print "chosen: %s" % result

    return result




class ExecutionPlan(object):
    def __init__(self, given):
        self.given = given

    def invalid_reason(self):
        try:
            self.occupancy_record()
            return None
        except ValueError, ve:
            return str(ve)

        return None

    def max_registers(self):
        regs = self.registers()

        from pycuda.tools import OccupancyRecord
        while True:
            try:
                OccupancyRecord(self.given.devdata,
                        self.threads(), self.shared_mem_use(),
                        registers=regs+1)
            except ValueError:
                return regs

            regs += 1

    @memoize_method
    def occupancy_record(self):
        from pycuda.tools import OccupancyRecord
        return OccupancyRecord(self.given.devdata,
                self.threads(), self.shared_mem_use(),
                registers=self.registers())

    def __str__(self):
            return ("regs=%d(+%d) threads=%d smem=%d occ=%f" % (
                self.registers(),
                self.max_registers()-self.registers(),
                self.threads(), 
                self.shared_mem_use(), 
                self.occupancy_record().occupancy,
                ))




class PlanGivenData(object):
    def __init__(self, devdata, ldis, float_type=numpy.float32):
        self.devdata = devdata
        self.ldis = ldis
        self.float_type = numpy.dtype(float_type)

        self.microblock = self._find_microblock_size()

    def float_size(self):
        return self.float_type.itemsize

    def order(self):
        return self.ldis.order

    @memoize_method
    def dofs_per_el(self):
        return self.ldis.node_count()

    @memoize_method
    def dofs_per_face(self):
        return self.ldis.face_node_count()

    def faces_per_el(self):
        return self.ldis.face_count()

    def face_dofs_per_el(self):
        return self.ldis.face_node_count()*self.faces_per_el()

    def face_dofs_per_microblock(self):
        return self.microblock.elements*self.faces_per_el()*self.dofs_per_face()
        
    @memoize_method
    def aligned_face_dofs_per_microblock(self):
        return self.devdata.align_dtype(
                self.face_dofs_per_microblock(),
                self.float_size())

    def _find_microblock_size(self):
        from hedge.cuda.tools import exact_div, int_ceiling
        align_size = exact_div(self.devdata.align_bytes(self.float_size()), 
                self.float_size())

        for mb_align_chunks in range(1, 256):
            mb_aligned_floats = align_size * mb_align_chunks
            mb_elements = mb_aligned_floats // self.dofs_per_el()
            mb_floats = self.dofs_per_el()*mb_elements
            overhead = (mb_aligned_floats-mb_floats)/mb_aligned_floats
            if overhead <= 0.05:
                from pytools import Record
                class MicroblockInfo(Record): pass

                return MicroblockInfo(
                        align_size=align_size,
                        elements=mb_elements,
                        aligned_floats=mb_aligned_floats,
                        accesses=mb_align_chunks
                        )

        assert False, "a valid microblock size was not found"




class ChunkedMatrixLocalOpExecutionPlan(ExecutionPlan):
    def __init__(self, given, parallelism, chunk_size, max_unroll):
        ExecutionPlan.__init__(self, given.devdata)
        self.given = given
        self.parallelism = parallelism
        self.chunk_size = chunk_size
        self.max_unroll = max_unroll

    def chunks_per_microblock(self):
        from hedge.cuda.tools import int_ceiling
        return int_ceiling(
                self.given.microblock.aligned_floats/self.chunk_size)

    def dofs_per_macroblock(self):
        return self.parallelism.total() * self.given.microblock.aligned_floats

    def max_elements_touched_by_chunk(self):
        given = self.given

        from hedge.cuda.tools import int_ceiling
        if given.dofs_per_el() > self.chunk_size:
            return 2
        else:
            return int_ceiling(self.chunk_size/given.dofs_per_el()) + 1

    @memoize_method
    def shared_mem_use(self):
        given = self.given
        
        return (128 # parameters, block header, small extra stuff
               + given.float_size() * (
                   # chunk of the local op matrix
                   + self.chunk_size # this many rows
                   * self.columns()
                   # fetch buffer for each chunk
                   + self.parallelism.parallel*self.parallelism.inline
                   * self.chunk_size
                   * self.fetch_buffer_chunks()
                   )
               )

    def threads(self):
        return self.parallelism.parallel*self.chunk_size

    def __str__(self):
            return ("%s par=%s chunk_size=%d unroll=%d" % (
                ExecutionPlan.__str__(self),
                self.parallelism,
                self.chunk_size,
                self.max_unroll))




class SMemFieldLocalOpExecutionPlan(ExecutionPlan):
    def __init__(self, given, parallelism, max_unroll):
        ExecutionPlan.__init__(self, given.devdata)
        self.given = given
        self.parallelism = parallelism
        self.max_unroll = max_unroll

    def dofs_per_macroblock(self):
        return self.parallelism.total() * self.given.microblock.aligned_floats

    @memoize_method
    def shared_mem_use(self):
        given = self.given
        
        return (64 # parameters, block header, small extra stuff
               + given.float_size() * (
                   self.parallelism.parallel 
                   * self.parallelism.inline
                   * self.given.microblock.aligned_floats))

    def threads(self):
        return self.parallelism.parallel * self.given.microblock.aligned_floats

    def __str__(self):
            return "smem_field %s par=%s unroll=%d" % (
                ExecutionPlan.__str__(self),
                self.parallelism,
                self.max_unroll)





def _test_planner():
    from hedge.element import TetrahedralElement
    for order in [3]:
        for pe in range(2,16):
            for se in range(1,16):
                flux_par = Parallelism(pe, 1, se)
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

