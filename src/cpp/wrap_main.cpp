#include <boost/python.hpp>




void hedge_expose_fluxes();
void hedge_expose_op_target();
void hedge_expose_volume_operators();
void hedge_expose_face_operators();
void hedge_expose_polynomial();
void hedge_expose_index_subset();




BOOST_PYTHON_MODULE(_internal)
{
  hedge_expose_fluxes();
  hedge_expose_op_target();
  hedge_expose_volume_operators();
  hedge_expose_face_operators();
  hedge_expose_polynomial();
  hedge_expose_index_subset();
}
