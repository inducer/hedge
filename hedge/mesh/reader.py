"""Mesh topology/geometry representation."""

from __future__ import division

__copyright__ = "Copyright (C) 2009 Xueyu Zhu, Andreas Kloeckner"

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
from pytools import memoize


# gmsh nodes-per-elements
elm_type =(2,3,4,4,8,6,5,3,6,9,10,27,18,14,1,8,20,15,13,9,10,12,15,15,21,12,15,15,18,20,35,56)

class GmshContainer:
    """
    Data structure to store the information from gmsh file
    """
    def __init__(self):
        self.Nodes    = []
        self.Elements = []
        self.phy_tags = []
        self.element_type_map = self._get_gmsh_element_type_to_info_map()

    @memoize
    def _get_gmsh_element_type_to_info_map(self):
        from pytools import Record
        class ElementTypeInfo(Record):
            __slots__ = ['ele_type']
        from hedge.element import TriangularElement,TetrahedralElement
        return {
                1:  "2-nodes Line",
                2:  ElementTypeInfo(ele_type = TriangularElement(1)),
                3:  "4-nodes quadrangle",
                4:  ElementTypeInfo(ele_type = TetrahedralElement(1)),
                5:  "8-nodes hexahedron",
                6:  "6-nodes prism",
                7:  "5-nodes pyramid",
                8:  "2nd order Line",
                9:  ElementTypeInfo(ele_type = TriangularElement(2)),
                10: "9-node second order quadrangle",
                11: ElementTypeInfo(ele_type = TetrahedralElement(2)),
                12: "27-node second order hexahedron",
                13: "18-node second order prism",
                14: "14-node second order pyramid",
                15: "1-node point",
                16: "8-node second order quadrangle",
                17: "20-node second order hexahedron",
                18: "15-node second order prism",
                19: "13-node second order pyramid",
                20: ElementTypeInfo(ele_type = TetrahedralElement(3)),
                21: ElementTypeInfo(ele_type = TetrahedralElement(3)),
                22: ElementTypeInfo(ele_type = TetrahedralElement(4)),
                23: ElementTypeInfo(ele_type = TetrahedralElement(4)),
                24: ElementTypeInfo(ele_type = TetrahedralElement(5)),
                25: ElementTypeInfo(ele_type = TetrahedralElement(5)),
                26: ElementTypeInfo(ele_type = TetrahedralElement(4)),
                27: ElementTypeInfo(ele_type = TetrahedralElement(4)), 
                28: ElementTypeInfo(ele_type = TetrahedralElement(4)), 
                29: ElementTypeInfo(ele_type = TetrahedralElement(3)),
                30: ElementTypeInfo(ele_type = TetrahedralElement(4)),
                31: ElementTypeInfo(ele_type = TetrahedralElement(5)),
                }
   


def read_gmsh(filename):
    """
    mesh reader for gmsh file
    """

    # initialize Mesh class
    readed_mesh = GmshContainer()
    
    # open target file
    lines = open(filename, 'r').readlines()
    i = 0
    while i < len(lines):
        l = lines[i].strip()
        i += 1  
        if l == "$MeshFormat" :
            while True:
                i+=1
                l = lines[i].strip()
                if l == "$EndMeshFormat":
                    break 
            i+=1
        elif l == "$Nodes":
            l = lines[i].strip()
            Nv = numpy.int(l)
            Nodes = []
            while True:
                i +=1
                l = lines[i].strip()
                lvalue = l.split()
                if l == "$EndNodes":
                    break
                readed_mesh.Nodes.append({lvalue[0]: [float(x) for x in lvalue[1:]] })
            i+=1
        elif l == "$Elements":
            l = lines[i].strip()
            K = numpy.int(l)
            EToV = []
            while True:
                i +=1
                l = lines[i].strip() 
                if l == "$EndElements":
                    break
                l_str = l.split()    
                lvalue = [int(x) for x in l_str] 
                type = lvalue[1]
                readed_mesh.Elements.append(
                                            dict(
                                                ele_indices = lvalue[0],
                                                el_type     = readed_mesh.element_type_map[type],
                                                ele_number_of_tags =lvalue[2],
                                                el_tags = lvalue[3:3+lvalue[2]],
                                                nodes   = lvalue[-elm_type[type-1]:]
                                                )
                                            )
                #readed_mesh.Elements.append(temp)
            i+=1
        elif l == "$PhysicalNames":
            l = lines[i].strip()
            no_tags = numpy.int(l)
            phy_tags = []
            while True:
                i +=1
                l = lines[i].strip() 
                if l == "$EndPhysicalNames":
                    break
                l_str = l.split()    
                lvalue = [int(x) for x in l_str[:-1]] 
                readed_mesh.phy_tags.append(
                                            dict(
                                                 phy_dimension = lvalue[0],
                                                 tag_index = lvalue[1],
                                                 tag_name  = l_str[-1].replace('\"',' ')
                                                 )
                                            )                                
            i+=1
        elif l == "$NodeData":
            l = lines[i].strip()
            while True:
                i +=1
                l = lines[i].strip()
                if l == "$EndNodeData":
                    break 
            i+=1
        elif l == "$ElementData":
            l = lines[i].strip()
            while True:
                i +=1
                l = lines[i].strip()
                if l == "$EndElementData":
                    break  
            i+=1
        elif l == "ElementNodeData":
            l = lines[i].strip()
            while True:
                i +=1
                l = lines[i].strip()
                if l == "$EndElementNodeData":
                    break
            i+=1
        else: 
            # unrecognized section, skip to end
            while True:
                i=len(lines)-1
                if i>=len(lines)-1:
                    break

    #close the file explicitly
    open(filename, 'r').close
    return readed_mesh





    
