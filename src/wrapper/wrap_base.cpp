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




#include <hedge/base.hpp>
#include "wrap_helpers.hpp"
#include <boost/foreach.hpp>
#include <boost/scoped_array.hpp>
#include <boost/python/stl_iterator.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/bindings/traits/traits.hpp>
#include <boost/numeric/bindings/traits/ublas_matrix.hpp>




using namespace hedge;
using namespace boost::python;
using namespace boost::numeric::bindings;
namespace ublas = boost::numeric::ublas;




namespace
{
  template <class T>
  std::auto_ptr<std::vector<T> > construct_vector(object iterable)
  {
    std::auto_ptr<std::vector<T> > result(new std::vector<T>());
    copy(
        stl_input_iterator<T>(iterable),
        stl_input_iterator<T>(),
        back_inserter(*result));
    return result;
  }




  // affine map ---------------------------------------------------------------
  template <class Scalar>
  affine_map<Scalar> *get_simplex_map_unit_to_global(const int dimensions, object vertices)
  {
    typedef numpy_vector<Scalar> vec_t;
    numpy_matrix<Scalar> mat(dimensions, dimensions);

    const vec_t &vertex0 = extract<vec_t >(vertices[0]);
    vec_t vsum = ublas::zero_vector<typename vec_t::value_type>(dimensions);
    for (int i = 0; i < dimensions; i++)
    {
      const vec_t &vertex = extract<vec_t>(vertices[i+1]);
      vsum += vertex;
      column(mat, i) = 0.5*(vertex-vertex0);
    }

    return new affine_map<double>(mat, 0.5*vsum - 0.5*(dimensions-2)*vertex0);
  }




  template <class Scalar>
  void map_element_nodes(numpy_vector<Scalar> all_nodes, const unsigned el_start, 
      const affine_map<Scalar> &map, const numpy_vector<Scalar> &unit_nodes, const unsigned dim)
  {
    // vectors, even if they're copied, have reference semantics
    for (unsigned nstart = 0; nstart < unit_nodes.size(); nstart += dim)
      subrange(all_nodes, el_start+nstart, el_start+nstart+dim) = 
        map.template apply<numpy_vector<Scalar> >(
            subrange(unit_nodes, nstart, nstart+dim));
  }
}




void hedge_expose_base()
{
  scope().attr("INVALID_ELEMENT") = INVALID_ELEMENT;
  scope().attr("INVALID_VERTEX") = INVALID_VERTEX;
  scope().attr("INVALID_NODE") = INVALID_NODE;

  {
    typedef std::vector<int> cl;
    class_<cl>("IntVector")
      .def("__init__", make_constructor(construct_vector<int>))
      .def("reserve", &cl::reserve, arg("advised_size"))
      .def(vector_indexing_suite<cl> ())
      ;
  }

  {
    typedef affine_map<double> cl;
    class_<cl>("AffineMap", init<const cl::matrix_t &, const cl::vector_t &>())
      .add_property("matrix", 
          make_function(&cl::matrix, return_value_policy<return_by_value>()))
      .add_property("vector", 
          make_function(&cl::vector, return_value_policy<return_by_value>()))
      .def("__call__", 
          &cl::apply<cl::vector_t, cl::vector_t>)
      .DEF_SIMPLE_METHOD(inverted)
      .DEF_SIMPLE_METHOD(jacobian)

      .enable_pickling()
      ;
  }

  def("map_element_nodes", map_element_nodes<double>,
      (arg("all_nodes"), arg("el_start"), arg("map"), arg("unit_nodes"), arg("dim")));
  def("get_simplex_map_unit_to_global",
      get_simplex_map_unit_to_global<double>,
      (arg("dimensions"), arg("vertices")),
      return_value_policy<manage_new_object>());

  {
    typedef ublas::zero_vector<double> cl;
    class_<cl>("ZeroVector");
  }
}
