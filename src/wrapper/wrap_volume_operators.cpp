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
#include <boost/numeric/ublas/io.hpp>
#include "op_target.hpp"
#include "volume_operators.hpp"
#include "wrap_helpers.hpp"




using namespace boost::python;
using namespace hedge;
namespace ublas = boost::numeric::ublas;




namespace
{
  template <class ER>
  tuple element_ranges_getitem(const ER &er, int i)
  {
    if (i < 0)
      i += er.size();
    if (i < 0 || i >= int(er.size()))
      PYTHON_ERROR(IndexError, "element_ranges index out of bounds");

    const element_range &erng = er[i];
    return make_tuple(erng.first, erng.second);
  }
}




void hedge_expose_volume_operators()
{
  {
    typedef nonuniform_element_ranges cl;
    class_<cl>("NonuniformElementRanges")
      .def("__len__", &cl::size)
      .def("clear", &cl::clear)
      .def("append_range", &cl::append_range)
      .def("__getitem__", element_ranges_getitem<cl>)
      ;
  }

  {
    typedef uniform_element_ranges cl;
    class_<cl>("UniformElementRanges", init<int, int, int>())
      .def("__len__", &cl::size)
      .def("__getitem__", element_ranges_getitem<cl>)
      .add_property("start", &cl::start)
      .add_property("el_size", &cl::el_size)
      ;
  }

#define ARG_TYPES \
  const uniform_element_ranges &, \
  const uniform_element_ranges &, \
  const py_matrix &,
  DEF_FOR_EACH_OP_TARGET(perform_elwise_operator, ARG_TYPES);
#undef ARG_TYPES
#define ARG_TYPES \
  const uniform_element_ranges &, \
  const uniform_element_ranges &, \
  const py_vector &, \
  const py_matrix &,
  DEF_FOR_EACH_OP_TARGET(perform_elwise_scaled_operator, ARG_TYPES);
#undef ARG_TYPES

  def("perform_elwise_scale", 
      perform_elwise_scale<uniform_element_ranges>,
      (arg("ers"), arg("scale_factors"), arg("target")));
}
