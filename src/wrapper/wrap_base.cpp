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
#include <boost/numeric/bindings/traits/traits.hpp>
#include <boost/numeric/bindings/traits/ublas_matrix.hpp>




using namespace hedge;
using namespace boost::python;
using namespace boost::numeric::bindings;




namespace
{
  template <class T>
  PyObject *bufferize_sequence(object iterable)
  {
    std::vector<T> v;
    std::copy(
        stl_input_iterator<unsigned>(iterable), 
        stl_input_iterator<unsigned>(),
        back_inserter(v));
    return PyString_FromStringAndSize(
        reinterpret_cast<const char *>(v.data()), v.size()*sizeof(T));
  }

  PyObject *bufferize_vector(const vector &v)
  {
    return PyString_FromStringAndSize(
        reinterpret_cast<const char *>(traits::vector_storage(v)), 
        v.size()*sizeof(vector::value_type));
  }

  PyObject *bufferize_list_of_vectors(object &vec_list, unsigned component_count)
  {
    int vec_count = len(vec_list);
    unsigned data_size = component_count*vec_count;
    boost::scoped_array<vector::value_type> result(
        new vector::value_type[data_size]);

    unsigned vec_num = 0;
    BOOST_FOREACH(const vector &v, make_pair(
        stl_input_iterator<const vector &>(vec_list), 
        stl_input_iterator<const vector &>()))
    {
      unsigned i = (vec_num++)*component_count;
      unsigned start = i;

      BOOST_FOREACH(const vector::value_type x, v)
        result[i++] = x;
      while (i < start + component_count)
        result[i++] = 0;
    }

    return PyString_FromStringAndSize(
        reinterpret_cast<const char *>(result.get()), 
        data_size*sizeof(vector::value_type));
  }

  PyObject *bufferize_list_of_components(object &vec_list, unsigned vec_count)
  {
    int component_count = len(vec_list);
    unsigned data_size = component_count*vec_count;
    boost::scoped_array<vector::value_type> result(
        new vector::value_type[data_size]);

    unsigned component_num = 0;
    BOOST_FOREACH(object o, make_pair(
        stl_input_iterator<object>(vec_list), 
        stl_input_iterator<object>()))
    {

      unsigned i = component_num++;

      extract<const vector &> extract_vec(o);
      if (extract_vec.check())
      {
        const vector &v = extract_vec();
        BOOST_FOREACH(const vector::value_type x, v)
        {
          result[i] = x;
          i += component_count;
        }
      }
      else if (o == object())
      {
        for (unsigned j = 0; j<vec_count; j++)
        {
          result[i] = 0;
          i += component_count;
        }
      }
      else
        PYTHON_ERROR(ValueError, "vec_list must consist of vectors or Nones.");
    }

    return PyString_FromStringAndSize(
        reinterpret_cast<const char *>(result.get()), 
        data_size*sizeof(vector::value_type));
  }
}




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

  // FIXME: pretty crude, but covers 32- and 64-bit machines
  if (sizeof(int) == 4)
    def("bufferize_int32", bufferize_sequence<int>);
  if (sizeof(short int) == 4)
    def("bufferize_int32", bufferize_sequence<short int>);
  def("bufferize_uint8", bufferize_sequence<unsigned char>);

  DEF_SIMPLE_FUNCTION(bufferize_vector);
  DEF_SIMPLE_FUNCTION(bufferize_list_of_vectors);
  DEF_SIMPLE_FUNCTION(bufferize_list_of_components);
}
