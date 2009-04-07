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




def optimize_plan(opt_name, plan_generator, target_func, maximize, debug_flags=set(), occupancy_slack=0.5,
        log_filename=None):
    plans = list(p for p in plan_generator()
            if p.invalid_reason() is None)

    debug = "cuda_%s_plan" % opt_name in debug_flags
    show_progress = ("cuda_plan_no_progress" not in debug_flags) and not debug

    if "cuda_plan_log" not in debug_flags:
        log_filename = None

    if not plans:
        raise RuntimeError, "no valid CUDA execution plans found"

    if "cuda_no_plan" in debug_flags:
        from pytools import argmax2
        return argmax2((plan, plan.occupancy_record().occupancy)
                for plan in plans), 0

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

        try:
            db_conn.execute("""
                  create table data (
                    id integer primary key autoincrement,
                    %s,
                    value real)""" 
                    % ", ".join(feature_columns))
        except sqlite.OperationalError:
            pass

    if show_progress:
        from pytools import ProgressBar
        pbar = ProgressBar("plan "+opt_name, len(plans))
    try:
        plan_values = []
        for p in plans:
            if show_progress:
                pbar.progress()

            if p.occupancy_record().occupancy >= desired_occup - 1e-10:
                if debug:
                    print "<---- trying %s:" % p

                value = target_func(p)
                if isinstance(value, tuple):
                    extra_info = value[1:]
                    value = value[0]
                else:
                    extra_info = None

                if value is not None:
                    if debug:
                        print "----> yielded %g" % (value)
                    plan_values.append(((len(plan_values), p), value))

                    if log_filename is not None:
                        db_conn.execute(
                                "insert into data (%s,value) values (%s)"
                                % (", ".join(feature_names), 
                                    ",".join(["?"]*(1+len(feature_names)))),
                                p.features(*extra_info)+(value,))
    finally:
        if show_progress:
            pbar.finished()

    if log_filename is not None:
        db_conn.commit()

    from pytools import argmax2, argmin2
    if maximize:
        num_plan, plan = argmax2(plan_values)
    else:
        num_plan, plan = argmin2(plan_values)

    plan_value = plan_values[num_plan][1]

    if debug:
        print "----------------------------------------------"
        print "chosen: %s" % plan
        print "value: %g" % plan_value
        print "----------------------------------------------"

    return plan, plan_value





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
    def __init__(self, devdata, ldis, allow_microblocking, float_type=numpy.float32):
        self.devdata = devdata
        self.ldis = ldis
        self.float_type = numpy.dtype(float_type)

        self.microblock = self._find_microblock_size(allow_microblocking)

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

    def _find_microblock_size(self, allow_microblocking):
        from hedge.backends.cuda.tools import exact_div, int_ceiling
        align_size = self.devdata.align_words(self.float_size())

        from pytools import Record
        class MicroblockInfo(Record): pass

        if not allow_microblocking:
            return MicroblockInfo(
                    align_size=align_size,
                    elements=1,
                    aligned_floats=int_ceiling(self.dofs_per_el(), align_size)
                    )

        for mb_align_chunks in range(1, 256):
            mb_aligned_floats = align_size * mb_align_chunks
            mb_elements = mb_aligned_floats // self.dofs_per_el()
            mb_floats = self.dofs_per_el()*mb_elements
            overhead = (mb_aligned_floats-mb_floats)/mb_aligned_floats
            if overhead <= 0.05:
                return MicroblockInfo(
                        align_size=align_size,
                        elements=mb_elements,
                        aligned_floats=mb_aligned_floats,
                        )

        assert False, "a valid microblock size was not found"

    def post_decomposition(self, block_count, microblocks_per_block):
        self.block_count = block_count
        self.microblocks_per_block = microblocks_per_block

    # below methods are available after decomposition has posted
    def matmul_preimage_shape(self, matmul_plan):
        from hedge.backends.cuda.tools import int_ceiling
        fof_dofs = (
            self.block_count
            * self.microblocks_per_block
            * matmul_plan.aligned_preimage_dofs_per_microblock)
        fof_dofs = int_ceiling(fof_dofs, matmul_plan.preimage_dofs_per_macroblock())

        return (fof_dofs,)

    def elements_per_block(self):
        return self.microblocks_per_block * self.microblock.elements

    def dofs_per_block(self):
        return self.microblock.aligned_floats * self.microblocks_per_block

    def total_dofs(self):
        return self.block_count * self.dofs_per_block()





class SegmentedMatrixLocalOpExecutionPlan(ExecutionPlan):
    def __init__(self, given, parallelism, segment_size, max_unroll):
        ExecutionPlan.__init__(self, given.devdata)
        self.given = given
        self.parallelism = parallelism
        self.segment_size = segment_size
        self.max_unroll = max_unroll

    def segments_per_microblock(self):
        from hedge.backends.cuda.tools import int_ceiling
        return int_ceiling(
                self.given.microblock.aligned_floats/self.segment_size)

    def dofs_per_macroblock(self):
        return self.parallelism.total() * self.given.microblock.aligned_floats

    def preimage_dofs_per_macroblock(self):
        return self.parallelism.total() * self.aligned_preimage_dofs_per_microblock

    def max_elements_touched_by_segment(self):
        given = self.given

        from hedge.backends.cuda.tools import int_ceiling
        if given.dofs_per_el() > self.segment_size:
            return 2
        else:
            return int_ceiling(self.segment_size/given.dofs_per_el()) + 1

    @memoize_method
    def shared_mem_use(self):
        given = self.given
        
        return (128 # parameters, block header, small extra stuff
               + given.float_size() * (
                   # segment of the local op matrix
                   + self.segment_size # this many rows
                   * self.columns()
                   # fetch buffer for each segment
                   + self.parallelism.parallel*self.parallelism.inline
                   * self.segment_size
                   * self.fetch_buffer_segments()
                   )
               )

    def threads(self):
        return self.parallelism.parallel*self.segment_size

    def __str__(self):
            return ("seg_matrix %s par=%s segment_size=%d unroll=%d" % (
                ExecutionPlan.__str__(self),
                self.parallelism,
                self.segment_size,
                self.max_unroll))




class SMemFieldLocalOpExecutionPlan(ExecutionPlan):
    def __init__(self, given, parallelism, max_unroll):
        ExecutionPlan.__init__(self, given.devdata)
        self.given = given
        self.parallelism = parallelism
        self.max_unroll = max_unroll

    def dofs_per_macroblock(self):
        return self.parallelism.total() * self.given.microblock.aligned_floats

    def preimage_dofs_per_macroblock(self):
        return (self.parallelism.total() 
                * self.aligned_preimage_dofs_per_microblock)

    def threads(self):
        return self.parallelism.parallel * self.given.microblock.aligned_floats

    def __str__(self):
            return "smem_field %s par=%s unroll=%d" % (
                ExecutionPlan.__str__(self),
                self.parallelism,
                self.max_unroll)





MAX_INLINE = 6




def make_diff_plan(discr, given):
    def generate_plans():
        segment_sizes = range(given.microblock.align_size, 
                given.microblock.elements*given.dofs_per_el()+1, 
                given.microblock.align_size)

        from hedge.backends.cuda.diff_shared_segmat import ExecutionPlan as SSegPlan

        if "cuda_no_smem_matrix" not in discr.debug:
            for pe in range(1,32+1):
                for inline in range(1, MAX_INLINE+1):
                    for seq in range(1, 4):
                        for segment_size in segment_sizes:
                            yield SSegPlan(
                                    given, Parallelism(pe, inline, seq), 
                                    segment_size, 
                                    max_unroll=given.dofs_per_el())

        from hedge.backends.cuda.diff_shared_fld import ExecutionPlan as SFieldPlan

        for pe in range(1,32+1):
            for inline in range(1, MAX_INLINE+1):
                yield SFieldPlan(given, Parallelism(pe, inline, 1), 
                        max_unroll=given.dofs_per_el())

    def target_func(plan):
        return plan.make_kernel(discr).benchmark()

    from hedge.backends.cuda.plan import optimize_plan
    return optimize_plan("diff", generate_plans, target_func, maximize=False,
            debug_flags=discr.debug,
            log_filename="diff-%d" % given.order())




def make_lift_plan(discr, given):
    def generate_plans():
        if "cuda_no_smem_matrix" not in discr.debug:
            from hedge.backends.cuda.el_local_shared_segmat import ExecutionPlan as SSegPlan

            for use_prefetch_branch in [True]:
            #for use_prefetch_branch in [True, False]:
                segment_sizes = range(given.microblock.align_size, 
                        given.microblock.elements*given.dofs_per_el()+1, 
                        given.microblock.align_size)

                for pe in range(1,32+1):
                    for inline in range(1, MAX_INLINE+1):
                        for seq in range(1, 4+1):
                            for segment_size in segment_sizes:
                                yield SSegPlan(given, 
                                        Parallelism(pe, inline, seq),
                                        segment_size,
                                        max_unroll=given.face_dofs_per_el(),
                                        use_prefetch_branch=use_prefetch_branch,

                                        debug_name="cuda_lift",
                                        aligned_preimage_dofs_per_microblock=
                                            given.aligned_face_dofs_per_microblock(),
                                        preimage_dofs_per_el=given.face_dofs_per_el())

        from hedge.backends.cuda.el_local_shared_fld import ExecutionPlan as SFieldPlan

        for pe in range(1,32+1):
            for inline in range(1, MAX_INLINE):
                yield SFieldPlan(given, Parallelism(pe, inline, 1), 
                        max_unroll=given.face_dofs_per_el(),

                        debug_name="cuda_lift",
                        aligned_preimage_dofs_per_microblock=
                            given.aligned_face_dofs_per_microblock(),
                        preimage_dofs_per_el=given.face_dofs_per_el())

    def target_func(plan):
        return plan.make_kernel(discr).benchmark()

    from hedge.backends.cuda.plan import optimize_plan
    return optimize_plan(
            "lift", generate_plans, target_func, maximize=False,
            debug_flags=discr.debug,
            log_filename="lift-%d" % given.order())
