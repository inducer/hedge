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




#include "base.hpp"
#include "wrap_helpers.hpp"




using namespace hedge;
using namespace boost::python;




void hedge_expose_base()
{
  {
    typedef affine_map cl;
    class_<cl>("AffineMap", init<const matrix &, const vector &, const double &>())
      .add_property("matrix", 
          make_function(&cl::matrix, 
            return_internal_reference<>()))
      .add_property("vector", 
          make_function(&cl::vector, 
            return_internal_reference<>()))
      .add_property("jacobian", &cl::jacobian)
      .def("__call__", &affine_map::operator())

      .enable_pickling()
      ;
  }
}
