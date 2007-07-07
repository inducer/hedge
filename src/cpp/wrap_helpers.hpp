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
