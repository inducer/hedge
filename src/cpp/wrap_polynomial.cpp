#include <boost/python.hpp>
#include "polynomial.hpp"




using namespace boost::python;
using namespace hedge;




void hedge_expose_polynomial()
{
  {
    typedef jacobi_polynomial cl;
    class_<cl>("JacobiPolynomial", init<double, double, unsigned>())
      .def("__call__", &cl::operator())
      ;
  }
}
