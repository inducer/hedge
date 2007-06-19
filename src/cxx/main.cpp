#include <boost/python.hpp>




void hedge_expose_fluxes();




BOOST_PYTHON_MODULE(_internal)
{
  hedge_expose_fluxes();
}
