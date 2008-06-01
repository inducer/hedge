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
    """Defines how much of a task is accomplished in serial vs. parallel."""
    def __init__(self, parallel, serial):
        self.p = parallel
        self.s = serial

    def total(self):
        return self.p*self.s

    def __str__(self):
        return "(p%d s%d)" % (self.p, self.s)




class ExecutionPlan:
    def __init__(self, devdata, ldis, flux_par, 
            max_ext_faces=None, max_faces=None, 
            float_type=numpy.float32, 
            ):
        self.devdata = devdata
        self.ldis = ldis
        self.flux_par = flux_par

        self.max_ext_faces = max_ext_faces
        self.max_faces = max_faces

        self.float_type = numpy.dtype(float_type)

    @property
    def float_size(self):
        return self.float_type.itemsize

    def copy(self, devdata=None, ldis=None, flux_par=None, 
            max_ext_faces=None, max_faces=None, float_type=None):
        return ExecutionPlan(
                devdata or self.devdata,
                ldis or self.ldis,
                flux_par or self.flux_par,
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

    def elements_per_block(self):
        return self.flux_par.total()

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

    def int_dofs(self):
        return self.devdata.align_dtype(self.elements_per_block() * self.dofs_per_el(),
                self.float_size)

    def int_dof_smem(self):
        return self.int_dofs() * self.float_size

    @memoize_method
    def ext_dof_smem(self):
        return self.devdata.align(self.get_extface_count()*
                self.dofs_per_face() * self.float_size)

    def flux_location_smem(self):
        from hedge.cuda.execute import flux_face_location_struct
        return self.devdata.align(
                    len(flux_face_location_struct())
                    * self.elements_per_block() 
                    * self.faces_per_el())

    def flux_properties_smem(self):
        from hedge.cuda.execute import flux_face_properties_struct
        return self.devdata.align(
                len(flux_face_properties_struct(self.float_type, self.ldis.dimensions))
                * self.face_pair_count()) 

    def flux_aux_smem(self):
        return (self.flux_location_smem() 
                + self.flux_properties_smem() 
                + self.facedup_smem()
                )

    @memoize_method
    def facedup_smem(self):
        from hedge.cuda.execute import facedup_read_struct
        return self.devdata.align(
                len(facedup_read_struct())
                # number of dup'd faces:
                # face duplication is symmetric, except at the 
                # boundary, where a duplicated write is not
                # necessary, so there are strictly fewer
                # dup faces than ext_faces_to.
                * self.get_extface_count() 
                )

    def localop_indexing_smem(self):
        return self.facedup_smem()

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
    def flux_shared_mem_use(self):
        return (128 # parameters, block header, small extra stuff
                + self.int_dof_smem() 
                + self.ext_dof_smem() 
                + self.flux_aux_smem()
                + self.facedup_smem()
                )

    @memoize_method
    def localop_shared_mem_use(self):
        return (128 # parameters, block header, small extra stuff
                + self.int_dof_smem() 
                + self.localop_indexing_smem()
                # rst2xyz coefficients
                + (self.elements_per_block()
                    * self.ldis.dimensions
                    * self.ldis.dimensions
                    * self.float_size)
                )

    def flux_threads(self):
        return self.flux_par.p*self.dofs_per_el()

    def flux_registers(self):
        return 18

    def invalid_reason(self):
        if self.flux_threads() >= self.devdata.max_threads:
            return "too many threads"

        if self.flux_shared_mem_use() >= int(self.devdata.shared_memory): 
            return "too much shared memory"

        if self.flux_threads()*self.flux_registers() > self.devdata.registers:
            return "too many registers"
        return None

    @memoize_method
    def flux_occupancy_record(self):
        from hedge.cuda.tools import OccupancyRecord
        return OccupancyRecord(self.devdata,
                self.flux_threads(), self.flux_shared_mem_use(),
                registers=self.flux_registers())
    
    @memoize_method
    def find_localop_par(self):
        return self.flux_par

    def __str__(self):
            return (
                    "flux: par=%s threads=%d int_smem=%d ext_smem=%d "
                    "aux_smem=%d smem=%d occ=%f\n"
                    "localop: par=%s threads=%d ind_smem=%d smem=%d" % (
                    self.flux_par, 
                    self.flux_threads(), 
                    self.int_dof_smem(), 
                    self.ext_dof_smem(),
                    self.flux_aux_smem(),
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

