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




namespace python = boost::python;
using namespace hedge;




#define EXPOSE_BESSEL_INT_AND_FLOAT(name) \
  python::def(#name, name<int, double, boost::math::policies::policy<> >);

  //def(#name "_float", name<double, double>);




#define EXPOSE_SPECIAL_FUNCTION(NAME, PY_NAME, CONSTRUCTOR_ARGS) \
{ \
  typedef NAME cl; \
  python::class_<cl, boost::noncopyable>(#PY_NAME, python::init<CONSTRUCTOR_ARGS>()) \
      .def("__call__", &cl::operator()) \
      ; \
}


namespace 
{
  python::object tuple_to_python(boost::tuples::null_type)
  {
    return python::tuple();
  }

  template <class H, class T>
    python::object tuple_to_python(boost::tuples::cons<H,T> const& x)
    {
      return python::make_tuple(x.get_head()) + tuple_to_python(x.get_tail());
    }

  template <class T>
    struct tupleconverter
    {
      static PyObject* convert(T const& x)
      {
        return python::incref(tuple_to_python(x).ptr());
      }
    };
}

  BOOST_PYTHON_MODULE(whatever)
  {
  }




void hedge_expose_polynomial()
{
  typedef boost::tuple<double, double> two_doubles;
  typedef boost::tuple<double, double, double> three_doubles;

  python::to_python_converter<two_doubles, tupleconverter<two_doubles> >();
  python::to_python_converter<three_doubles, tupleconverter<three_doubles> >();

#define CONSTR_ARGS_JACOBI double, double, unsigned
#define CONSTR_ARGS_TRIANGLE unsigned, unsigned
#define CONSTR_ARGS_TET unsigned, unsigned, unsigned

  EXPOSE_SPECIAL_FUNCTION(jacobi_polynomial, JacobiPolynomial, 
      CONSTR_ARGS_JACOBI);
  EXPOSE_SPECIAL_FUNCTION(diff_jacobi_polynomial, DiffJacobiPolynomial, 
      CONSTR_ARGS_JACOBI);
  EXPOSE_SPECIAL_FUNCTION(triangle_basis_function, TriangleBasisFunction, 
      CONSTR_ARGS_TRIANGLE);
  EXPOSE_SPECIAL_FUNCTION(grad_triangle_basis_function, GradTriangleBasisFunction, 
      CONSTR_ARGS_TRIANGLE);
  EXPOSE_SPECIAL_FUNCTION(tetrahedron_basis_function, TetrahedronBasisFunction, 
      CONSTR_ARGS_TET);
  EXPOSE_SPECIAL_FUNCTION(grad_tetrahedron_basis_function, GradTetrahedronBasisFunction, 
      CONSTR_ARGS_TET);

  using namespace boost::math;

  EXPOSE_BESSEL_INT_AND_FLOAT(cyl_bessel_j);
  EXPOSE_BESSEL_INT_AND_FLOAT(cyl_neumann);
  EXPOSE_BESSEL_INT_AND_FLOAT(cyl_bessel_i);
  EXPOSE_BESSEL_INT_AND_FLOAT(cyl_bessel_k);
  python::def("sph_bessel", sph_bessel<double, boost::math::policies::policy<> >);
  python::def("sph_neumann", sph_neumann<double, boost::math::policies::policy<> >);
}
