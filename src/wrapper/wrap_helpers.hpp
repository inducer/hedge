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




#define PYTHON_ERROR(TYPE, REASON) \
{ \
  PyErr_SetString(PyExc_##TYPE, REASON); \
  throw boost::python::error_already_set(); \
}




#define DEF_FOR_EACH_OP_TARGET(NAME, TEMPLATE_ARGS) \
  def(#NAME, NAME<TEMPLATE_ARGS vector_target>); \
  def(#NAME, NAME<TEMPLATE_ARGS coord_matrix_target>);
#define ENUM_VALUE(NAME) \
  value(#NAME, NAME)
#define DEF_SIMPLE_METHOD(NAME) \
  def(#NAME, &cl::NAME)
#define DEF_SIMPLE_FUNCTION(NAME) \
  def(#NAME, &NAME)




#endif
