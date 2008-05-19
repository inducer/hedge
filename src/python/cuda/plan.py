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
from pytools import memoize_method




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
        from hedge.cuda.tools import int_ceiling

        if self.max_ext_faces is None:
            return int_ceiling(self.estimate_extface_count())
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
        from hedge.cuda.cgen import Struct, StructField, ArrayStructField

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
        from hedge.cuda.cgen import Struct, StructField

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
        from hedge.cuda.tools import OccupancyRecord
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

