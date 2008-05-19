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




  // binary buffers -----------------------------------------------------------
  template <class T>
  PyObject *bufferize_sequence(object iterable)
  {
    std::vector<T> v;
    std::copy(
        stl_input_iterator<unsigned>(iterable), 
        stl_input_iterator<unsigned>(),
        back_inserter(v));
    return PyString_FromStringAndSize(
        reinterpret_cast<const char *>(&v.front()), v.size()*sizeof(T));
  }




  PyObject *bufferize_vector(const py_vector &v)
  {
    return PyString_FromStringAndSize(
        reinterpret_cast<const char *>(traits::vector_storage(v)), 
        v.size()*sizeof(py_vector::value_type));
  }




  PyObject *bufferize_list_of_vectors(object &vec_list, unsigned component_count)
  {
    int vec_count = len(vec_list);
    unsigned data_size = component_count*vec_count;
    boost::scoped_array<py_vector::value_type> result(
        new py_vector::value_type[data_size]);

    unsigned vec_num = 0;
    BOOST_FOREACH(const py_vector &v, make_pair(
        stl_input_iterator<const py_vector &>(vec_list), 
        stl_input_iterator<const py_vector &>()))
    {
      unsigned i = (vec_num++)*component_count;
      unsigned start = i;

      BOOST_FOREACH(const py_vector::value_type x, v)
        result[i++] = x;
      while (i < start + component_count)
        result[i++] = 0;
    }

    return PyString_FromStringAndSize(
        reinterpret_cast<const char *>(result.get()), 
        data_size*sizeof(py_vector::value_type));
  }




  PyObject *bufferize_list_of_components(object &vec_list, unsigned vec_count)
  {
    int component_count = len(vec_list);
    unsigned data_size = component_count*vec_count;
    boost::scoped_array<py_vector::value_type> result(
        new py_vector::value_type[data_size]);

    unsigned component_num = 0;
    BOOST_FOREACH(object o, make_pair(
        stl_input_iterator<object>(vec_list), 
        stl_input_iterator<object>()))
    {

      unsigned i = component_num++;

      if (o == object())
      {
        for (unsigned j = 0; j<vec_count; j++)
        {
          result[i] = 0;
          i += component_count;
        }
      }
      else 
      {
        py_vector v(handle<>(borrowed(o.ptr())));
        BOOST_FOREACH(const py_vector::value_type x, v)
        {
          result[i] = x;
          i += component_count;
        }
      }
    }

    return PyString_FromStringAndSize(
        reinterpret_cast<const char *>(result.get()), 
        data_size*sizeof(py_vector::value_type));
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
    typedef affine_map cl;
    class_<cl>("AffineMap", init<const py_matrix &, const py_vector &>())
      .add_property("matrix", 
          make_function(&cl::matrix, return_value_policy<return_by_value>()))
      .add_property("vector", 
          make_function(&cl::vector, return_value_policy<return_by_value>()))
      .def("__call__", 
          (const py_vector (cl::*)(const py_vector &) const) 
          &affine_map::operator())

      .enable_pickling()
      ;

    def("map_element_nodes", map_element_nodes,
        (arg("all_nodes"), arg("el_start"), arg("map"), arg("unit_nodes"), arg("dim")));
    def("get_simplex_map_unit_to_global",
        get_simplex_map_unit_to_global,
        (arg("dimensions"), arg("vertices")),
        return_value_policy<manage_new_object>());
  }

  // FIXME: pretty crude, but covers 32- and 64-bit machines
  if (sizeof(int) == 4)
    def("bufferize_int32", bufferize_sequence<int>);
  if (sizeof(short int) == 4)
    def("bufferize_int32", bufferize_sequence<short int>);
  def("bufferize_uint8", bufferize_sequence<unsigned char>);

  DEF_SIMPLE_FUNCTION(bufferize_vector);
  DEF_SIMPLE_FUNCTION(bufferize_list_of_vectors);
  DEF_SIMPLE_FUNCTION(bufferize_list_of_components);

  {
    typedef ublas::zero_vector<double> cl;
    class_<cl>("ZeroVector");
  }
}
