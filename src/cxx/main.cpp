#include <boost/python.hpp>




void hedge_expose_fluxes();
void hedge_expose_dg();




BOOST_PYTHON_MODULE(_internal)
{
  hedge_expose_fluxes();
  hedge_expose_dg();
}
