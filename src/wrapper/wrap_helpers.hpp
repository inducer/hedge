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




#ifndef _UTHHFF_HEDGE_WRAP_HELPERS_HPP_INCLUDED
#define _UTHHFF_HEDGE_WRAP_HELPERS_HPP_INCLUDED




#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>




#define PYTHON_ERROR(TYPE, REASON) \
{ \
  PyErr_SetString(PyExc_##TYPE, REASON); \
  throw boost::python::error_already_set(); \
}




#define DEF_FOR_EACH_OP_TARGET(NAME, SCALAR, ARG_TYPES) \
  def(#NAME, (void (*)(ARG_TYPES null_target<numpy_vector<SCALAR> >)) NAME); \
  def(#NAME, (void (*)(ARG_TYPES matrix_target<coordinate_matrix<SCALAR, column_major> >)) NAME); \
  def(#NAME, (void (*)(ARG_TYPES vector_target<numpy_vector<SCALAR> >)) NAME);

#define ENUM_VALUE(NAME) \
  value(#NAME, NAME)

#define DEF_SIMPLE_METHOD(NAME) \
  def(#NAME, &cl::NAME)

#define DEF_SIMPLE_FUNCTION(NAME) \
  boost::python::def(#NAME, &NAME)

#define DEF_SIMPLE_RO_MEMBER(NAME) \
  def_readonly(#NAME, &cl::NAME)

#define DEF_SIMPLE_RW_MEMBER(NAME) \
  def_readwrite(#NAME, &cl::NAME)

#define DEF_BYVAL_RO_MEMBER(NAME) \
  def(pyublas::by_value_ro_member(#NAME, &cl::NAME))

#define DEF_BYVAL_RW_MEMBER(NAME) \
  def(pyublas::by_value_rw_member(#NAME, &cl::NAME))


template <class T>
class no_compare_indexing_suite :
  public boost::python::vector_indexing_suite<T, false, no_compare_indexing_suite<T> >
{
  public:
    static bool contains(T &container, typename T::value_type const &key)
    { PYTHON_ERROR(NotImplementedError, "containment checking not supported on this container"); }
};




#define PYTHON_FOREACH(TYPE, NAME, ITERABLE) \
  BOOST_FOREACH(TYPE NAME, \
      std::make_pair( \
        boost::python::stl_input_iterator<TYPE>(ITERABLE), \
        boost::python::stl_input_iterator<TYPE>()))

#define COPY_PY_LIST(TYPE, NAME) \
  std::vector<TYPE> NAME; \
  std::copy( \
      boost::python::stl_input_iterator<TYPE>(NAME##_py), \
      boost::python::stl_input_iterator<TYPE>(), \
      std::back_inserter(NAME)); \



#endif
