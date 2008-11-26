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
  affine_map *get_simplex_map_unit_to_global(const int dimensions, object vertices)
  {
    py_matrix mat(dimensions, dimensions);

    const py_vector &vertex0 = extract<py_vector>(vertices[0]);
    py_vector vsum = ublas::zero_vector<py_vector::value_type>(dimensions);
    for (int i = 0; i < dimensions; i++)
    {
      const py_vector &vertex = extract<py_vector>(vertices[i+1]);
      vsum += vertex;
      column(mat, i) = 0.5*(vertex-vertex0);
    }

    return new affine_map(mat, 0.5*vsum - 0.5*(dimensions-2)*vertex0);
  }




  void map_element_nodes(py_vector all_nodes, const unsigned el_start, 
      const affine_map &map, const py_vector &unit_nodes, const unsigned dim)
  {
    // vectors, even if they're copied, have reference semantics
    for (unsigned nstart = 0; nstart < unit_nodes.size(); nstart += dim)
      subrange(all_nodes, el_start+nstart, el_start+nstart+dim) = 
        map.apply<py_vector>(subrange(unit_nodes, nstart, nstart+dim));
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
    typedef affine_map cl;
    class_<cl>("AffineMap", init<const py_matrix &, const py_vector &>())
      .add_property("matrix", 
          make_function(&cl::matrix, return_value_policy<return_by_value>()))
      .add_property("vector", 
          make_function(&cl::vector, return_value_policy<return_by_value>()))
      .def("__call__", 
          (const py_vector (cl::*)(const py_vector &) const) 
          &affine_map::operator())
      .DEF_SIMPLE_METHOD(inverted)
      .DEF_SIMPLE_METHOD(jacobian)

      .enable_pickling()
      ;

    def("map_element_nodes", map_element_nodes,
        (arg("all_nodes"), arg("el_start"), arg("map"), arg("unit_nodes"), arg("dim")));
    def("get_simplex_map_unit_to_global",
        get_simplex_map_unit_to_global,
        (arg("dimensions"), arg("vertices")),
        return_value_policy<manage_new_object>());
  }

  {
    typedef ublas::zero_vector<double> cl;
    class_<cl>("ZeroVector");
  }
}
