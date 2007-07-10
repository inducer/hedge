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




#include <boost/python.hpp>
#include "op_target.hpp"
#include "wrap_helpers.hpp"




using namespace boost::python;
using namespace hedge;
namespace ublas = boost::numeric::ublas;




namespace
{
  template <class cl>
  void expose_op_target(class_<cl> &wrapper)
  {
    wrapper
      .def("begin", &cl::begin)
      .def("finalize", &cl::finalize)
      .def("add_coefficient", &cl::add_coefficient)
      .def("add_coefficients", 
          (void (cl::*)(unsigned, unsigned, unsigned, unsigned, const matrix &) const)
          &cl::add_coefficients)
      ;
  }

}




void hedge_expose_op_target()
{
  {
    typedef vector_target cl;
    class_<cl> wrapper("VectorTarget", 
        init<const vector &, vector &>()
        [with_custodian_and_ward<1,2, with_custodian_and_ward<1,3> >()]
        );
    expose_op_target(wrapper);
  }

  {
    typedef coord_matrix_target cl;
    class_<cl> wrapper("MatrixTarget", 
        init<cl::matrix_type &>()
        [with_custodian_and_ward<1,2>()]
        );
    expose_op_target(wrapper);
  }

}
