#include <boost/python.hpp>
#include "special_function.hpp"
#include <boost/math/special_functions/bessel.hpp>




using namespace boost::python;
using namespace hedge;




#define EXPOSE_BESSEL_INT_AND_FLOAT(name) \
  def(#name, name<int, double>);

  //def(#name "_float", name<double, double>);
void hedge_expose_polynomial()
{
  {
    typedef jacobi_polynomial cl;
    class_<cl>("JacobiPolynomial", init<double, double, unsigned>())
      .def("__call__", &cl::operator())
      ;
  }

  using namespace boost::math;

  EXPOSE_BESSEL_INT_AND_FLOAT(cyl_bessel_j);
  EXPOSE_BESSEL_INT_AND_FLOAT(cyl_neumann);
  //EXPOSE_BESSEL_INT_AND_FLOAT(cyl_bessel_i);
  //EXPOSE_BESSEL_INT_AND_FLOAT(cyl_bessel_k);
  //def("sph_bessel", sph_bessel<double>);
  //def("sph_neumann", sph_neumann<double>);
}
