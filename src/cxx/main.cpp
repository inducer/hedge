#include <boost/python.hpp>




void hedge_expose_fluxes();
void hedge_expose_dg();
void hedge_expose_polynomial();




BOOST_PYTHON_MODULE(_internal)
{
  hedge_expose_fluxes();
  hedge_expose_dg();
  hedge_expose_polynomial();
}
