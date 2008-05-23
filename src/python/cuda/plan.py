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

    def int_dofs(self):
        return self.devdata.align_dtype(self.block_el() * self.dofs_per_el(),
                self.float_size)

    def int_dof_smem(self):
        return self.int_dofs() * self.float_size

    @memoize_method
    def ext_dof_smem(self):
        return self.devdata.align(self.get_extface_count()*
                self.dofs_per_face() * self.float_size)

    def flux_indexing_smem(self):
        from hedge.cuda.execute import flux_face_pair_struct
        return self.devdata.align(
                len(flux_face_pair_struct(self.ldis.dimensions))
                *self.face_pair_count())

    @memoize_method
    def localop_indexing_smem(self):
        from hedge.cuda.execute import localop_facedup_struct
        return self.devdata.align(
                len(localop_facedup_struct())
                # mildly off: should be ext_faces_from_me, add 256 bytes of fudge
                * self.get_extface_count() + 128
                )

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
    def flux_shared_mem_use(self):
        return (128 # parameters, block header, small extra stuff
                + self.int_dof_smem() 
                + self.ext_dof_smem() 
                + self.flux_indexing_smem())

    @memoize_method
    def localop_shared_mem_use(self):
        return (128 # parameters, block header, small extra stuff
                + self.int_dof_smem() 
                + self.localop_indexing_smem())

    def flux_threads(self):
        return self.flux_par.p*self.faces_per_el()*self.dofs_per_face()

    def invalid_reason(self):
        if self.flux_threads() >= self.devdata.max_threads:
            return "too many threads"
        if self.flux_shared_mem_use() >= self.devdata.shared_memory:
            return "too much shared memory"
        return None

    @memoize_method
    def flux_occupancy_record(self):
        from hedge.cuda.tools import OccupancyRecord
        return OccupancyRecord(self.devdata,
                self.flux_threads(), self.flux_shared_mem_use())
    
    @memoize_method
    def find_localop_par(self):
        from hedge.cuda.tools import int_ceiling, OccupancyRecord

        els_per_block = self.flux_par.total()
        localop_plans = []
        for par in range(1,els_per_block):
            els_per_block
            ser = int_ceiling(els_per_block/par)
            threads = par * self.dofs_per_el()

            localop_par = Parallelism(par, ser)
            if threads >= self.devdata.max_threads:
                continue
            occ = OccupancyRecord(self.devdata, threads, 
                    shared_mem=self.localop_shared_mem_use())
            localop_plans.append((occ, localop_par))

        max_occup = max(plan[0].occupancy for plan in localop_plans)
        good_plans = [p for p in localop_plans
                if p[0].occupancy > max_occup - 1e-10]

        # minimum parallelism is better to smooth out inefficiencies due to
        # rounding up the number of serial steps
        from pytools import argmin2
        return argmin2((p[1], p[1].p) for p in good_plans)

    def __str__(self):
            return (
                    "flux: par=%s threads=%d int_smem=%d ext_smem=%d "
                    "ind_smem=%d smem=%d occ=%f\n"
                    "localop: par=%s threads=%d ind_smem=%d smem=%d" % (
                    self.flux_par, 
                    self.flux_threads(), 
                    self.int_dof_smem(), 
                    self.ext_dof_smem(),
                    self.flux_indexing_smem(),
                    self.flux_shared_mem_use(), 
                    self.flux_occupancy_record().occupancy,

                    self.find_localop_par(), 
                    self.find_localop_par().p*self.dofs_per_el(), 
                    self.localop_indexing_smem(), 
                    self.localop_shared_mem_use(),
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

