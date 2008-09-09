// Hedge - the Hybrid'n'Easy DG Environment
// Copyright (C) 2007 Andreas Kloeckner
// 
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.




#include <boost/foreach.hpp>
#include <pyublas/numpy.hpp>
#include "wrap_helpers.hpp"
#include "base.hpp"




namespace py = boost::python;
using namespace pyublas;
using namespace hedge;




namespace 
{
  py::tuple tetrahedron_fj_and_normal(
      double el_orientation,
      py::list face_vertex_numbers, 
      py::list vertices_py)
  {
    static const int face_orientations[] = { -1,1,-1,1 };

    py::list normals, jacobians;

    COPY_PY_LIST(py_vector, vertices);

    unsigned face_nr = 0;

    PYTHON_FOREACH(py::tuple, single_face_vertex_numbers, face_vertex_numbers)
    {
      unsigned sfvn[] = {
        py::extract<unsigned>(single_face_vertex_numbers[0]),
        py::extract<unsigned>(single_face_vertex_numbers[1]),
        py::extract<unsigned>(single_face_vertex_numbers[2])
      };

      py_vector normal = cross<bounded_vector, bounded_vector>(
          vertices[sfvn[1]]-vertices[sfvn[0]],
          vertices[sfvn[2]]-vertices[sfvn[0]]);

      double n_length = norm_2(normal);

      // ||n_length|| is the area of the parallelogram spanned by the two
      // vectors above. Half of that is the area of the triangle we're interested
      // in. Next, the area of the unit triangle is two, so divide by two again.
      normals.append(
          py_vector(
            el_orientation
            * face_orientations[face_nr++]
            * normal
            / n_length));
      jacobians.append(n_length/4);
    }

    return py::make_tuple(normals, jacobians);
  }
}




void hedge_expose_mesh()
{
  DEF_SIMPLE_FUNCTION(tetrahedron_fj_and_normal);
}
