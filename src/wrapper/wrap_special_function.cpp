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
