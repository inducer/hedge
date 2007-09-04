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




#include <vector>
#include <iostream>
#include <boost/python.hpp>
#include <boost/mpl/for_each.hpp>
#include <boost/mpl/clear.hpp>
#include "flux.hpp"




using namespace boost::python;
using namespace hedge::flux;
namespace mpl = boost::mpl;




namespace {
  struct flux_wrap : flux, wrapper<flux>
  {
    double local_coeff(const face &local) const
    {
      return this->get_override("local_coeff")(boost::ref(local));
    }
    double neighbor_coeff(const face &local, const face *neighbor) const
    {
      return this->get_override("neighbor_coeff")(boost::ref(local), boost::ref(neighbor));
    }
  };




  template<class Flux>
  std::auto_ptr<Flux> new_unary_op_flux(const flux &op1, const flux &op2)
  {
    return std::auto_ptr<Flux>(new Flux(chained_flux(op1)));
  }




  template<class Operation> 
  void expose_unary_operator(Operation, const std::string &name)
  {
    typedef unary_operator<Operation, chained_flux> cl;
    class_<cl, bases<flux>, boost::noncopyable>(name.c_str(), no_init);
    def(("make_"+name).c_str(), 
        new_unary_op_flux<cl>,
        with_custodian_and_ward_postcall<0, 1>())
      ;
  }




  template<class Flux>
  std::auto_ptr<Flux> new_binary_op_flux(const flux &op1, const flux &op2)
  {
    return std::auto_ptr<Flux>(new Flux(chained_flux(op1), chained_flux(op2)));
  }




  template<class Operation> 
  void expose_binary_operator(Operation, const std::string &name)
  {
    typedef binary_operator<Operation, chained_flux, chained_flux> cl;
    class_<cl, bases<flux>, boost::noncopyable>(name.c_str(), no_init);
    def(("make_"+name).c_str(), 
        new_binary_op_flux<cl>,
        with_custodian_and_ward_postcall<0, 1, 
        with_custodian_and_ward_postcall<0, 2> >())
      ;
  }




  template<class Flux>
  std::auto_ptr<Flux> new_binary_constant_op_flux(const flux &op1, double c)
  {
    return std::auto_ptr<Flux>(new Flux(chained_flux(op1), constant(c)));
  }




  template<class Operation> 
  void expose_binary_constant_operator(Operation, const std::string &name)
  {
    typedef binary_operator<Operation, chained_flux, constant> cl;
    class_<cl, bases<flux>, boost::noncopyable>(name.c_str(), no_init);
    def(("make_"+name).c_str(), 
        new_binary_constant_op_flux<cl>,
        with_custodian_and_ward_postcall<0, 1>())
      ;
  }
}




void hedge_expose_fluxes()
{
  {
    typedef face cl;
    class_<face>("Face")
      .def_readwrite("h", &cl::h)
      .def_readwrite("face_jacobian", &cl::face_jacobian)
      .def_readwrite("element_id", &cl::element_id)
      .def_readwrite("face_id", &cl::face_id)
      .def_readwrite("order", &cl::order)
      .def_readwrite("normal", &cl::normal)
      ;
  }
  {
    typedef flux cl;
    class_<flux_wrap, boost::noncopyable>("Flux")
      .def("neighbor_coeff", pure_virtual(&flux::neighbor_coeff))
      .def("local_coeff", pure_virtual(&flux::local_coeff))
      ;
  }

  {
    typedef chained_flux cl;
    class_<cl, bases<flux> >("ChainedFlux", 
        init<flux &>()
        [with_custodian_and_ward<1, 2>()]
        )
      ;
  }

  class_<constant, bases<flux> >(
      "ConstantFlux", 
      init<double, double>(
        (arg("local"), arg("neighbor"))
        ))
      .def(self + self)
      .def(self - self)
      .def(- self)
      .def(self * double())
      ;

  class_<normal>("NormalFlux", init<int>( (arg("axis"))) );

  class_<penalty_term, bases<flux> >("PenaltyTermFlux",
      init<double, double>(
        (arg("coefficient"), arg("power"))
        )
      );


  expose_binary_operator(std::plus<double>(), "SumFlux");
  expose_binary_operator(std::minus<double>(), "DifferenceFlux");
  expose_binary_operator(std::multiplies<double>(), "ProductFlux");
  expose_binary_constant_operator(std::multiplies<double>(), 
      "ProductWithConstantFlux");

  expose_unary_operator(std::negate<double>(), "NegativeFlux");

  -normal(1) * 3 * (constant(5) + normal(0));

}

