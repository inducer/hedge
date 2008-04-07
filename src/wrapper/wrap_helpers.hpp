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




#define DEF_FOR_EACH_OP_TARGET(NAME, ARG_TYPES) \
  def(#NAME, (void (*)(ARG_TYPES null_target)) NAME); \
  def(#NAME, (void (*)(ARG_TYPES coord_matrix_target)) NAME); \
  def(#NAME, (void (*)(ARG_TYPES vector_target)) NAME);

#define EXPOSE_FLUX_PERFORM(FLUX_CLASS) \
  def("perform_flux", (void (*)(\
          const hedge::face_group &, const py_matrix &, \
          FLUX_CLASS, hedge::null_target, FLUX_CLASS, hedge::null_target)) \
      hedge::perform_flux_detailed); \
  def("perform_flux", (void (*)(\
          const hedge::face_group &, const py_matrix &, \
          FLUX_CLASS, hedge::vector_target, FLUX_CLASS, hedge::null_target)) \
      hedge::perform_flux_detailed); \
  def("perform_flux", (void (*)(\
          const hedge::face_group &, const py_matrix &, \
          FLUX_CLASS, hedge::null_target, FLUX_CLASS, hedge::vector_target)) \
      hedge::perform_flux_detailed); \
  def("perform_flux", (void (*)(\
          const hedge::face_group &, const py_matrix &, \
          FLUX_CLASS, hedge::vector_target, FLUX_CLASS, hedge::vector_target)) \
      hedge::perform_flux_detailed); \
  def("perform_flux", (void (*)(\
          const hedge::face_group &, const py_matrix &, \
          FLUX_CLASS, hedge::coord_matrix_target, FLUX_CLASS, hedge::coord_matrix_target)) \
      hedge::perform_flux_detailed); \
  def("perform_flux", (void (*)(\
          const hedge::face_group &, const py_matrix &, \
          FLUX_CLASS, hedge::coord_matrix_target, FLUX_CLASS, hedge::null_target)) \
      hedge::perform_flux_detailed); \
  def("perform_flux_on_one_target", (void (*)(\
          const hedge::face_group &, const py_matrix &, \
          FLUX_CLASS, FLUX_CLASS, hedge::vector_target)) \
      hedge::perform_flux_on_one_target); \
  def("perform_flux_on_one_target", (void (*)(\
          const hedge::face_group &, const py_matrix &, \
          FLUX_CLASS, FLUX_CLASS, hedge::coord_matrix_target)) \
      hedge::perform_flux_on_one_target); \

#define ENUM_VALUE(NAME) \
  value(#NAME, NAME)

#define DEF_SIMPLE_METHOD(NAME) \
  def(#NAME, &cl::NAME)

#define DEF_SIMPLE_FUNCTION(NAME) \
  def(#NAME, &NAME)

#define DEF_SIMPLE_RO_MEMBER(NAME) \
  def_readonly(#NAME, &cl::NAME)

#define DEF_SIMPLE_RW_MEMBER(NAME) \
  def_readwrite(#NAME, &cl::NAME)



template <class T>
class no_compare_indexing_suite :
  public boost::python::vector_indexing_suite<T, false, no_compare_indexing_suite<T> >
{
  public:
    static bool contains(T &container, typename T::value_type const &key)
    { PYTHON_ERROR(NotImplementedError, "containment checking not supported on this container"); }
};




#endif
