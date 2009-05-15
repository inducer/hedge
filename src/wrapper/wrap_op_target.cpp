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
#include <pyublas/python_helpers.hpp>
#include <hedge/op_target.hpp>
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
          (void (cl::*)(unsigned, unsigned, const numpy_matrix<typename cl::scalar_type> &))
          &cl::add_coefficients,
          (arg("i_start"), arg("j_start"), arg("submat")))
      .def("add_scaled_coefficients", 
          (void (cl::*)(unsigned, unsigned, typename cl::scalar_type, 
                        const numpy_matrix<typename cl::scalar_type> &))
          &cl::add_scaled_coefficients,
          (arg("i_start"), arg("j_start"), arg("factor"), arg("submat")))
      ;
  }

  

  template <class Vector>
  vector_target<Vector> *make_vector_target(const Vector argument, Vector result)
  {
    return new vector_target<Vector>(argument, result);
  }

  template <class Vector>
  null_target<Vector> *make_null_target(int, Vector result)
  {
    return new null_target<Vector>;
  }


  template <class ValueType>
  void hedge_expose_op_target_for_type()
  {
    typedef numpy_vector<ValueType> vector_t;

    {
      typedef null_target<vector_t> cl;
      class_<cl> wrapper("NullTarget");
      expose_op_target(wrapper);
    }

    {
      typedef vector_target<vector_t> cl;
      class_<cl> wrapper("VectorTarget", init<const vector_t, vector_t>());
      expose_op_target(wrapper);
    }

    def("make_vector_target", make_vector_target<vector_t>, 
        return_value_policy<manage_new_object>());
    def("make_vector_target", make_null_target<vector_t>,
        return_value_policy<manage_new_object>());

  }

}




void hedge_expose_op_target()
{
  hedge_expose_op_target_for_type<float>();
  hedge_expose_op_target_for_type<double>();

  {
    typedef coord_matrix_target cl;
    class_<cl> wrapper("MatrixTarget", 
        init<cl::matrix_type &, optional<cl::index_type, cl::index_type> >()
        [with_custodian_and_ward<1,2>()]
        )
      ;
    wrapper
      .DEF_SIMPLE_METHOD(rebased_target)
      .add_property("row_offset", &cl::row_offset)
      .add_property("column_offset", &cl::column_offset)
      ;
    expose_op_target(wrapper);
  }
}
