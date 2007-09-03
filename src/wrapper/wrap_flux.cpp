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




  struct call_without_arguments
  {
    template <class Exposer>
    void operator()(Exposer exp)
    { exp(); }
  };

  struct printit
  {
    template <class T>
    void operator()(T t)
    { std::cout << T::value << std::endl; }
  };
}




namespace hedge { namespace python {

  template <class ConstantFlux>
  ConstantFlux add_constant_fluxes(
      const ConstantFlux &fl1,
      const ConstantFlux &fl2
      )
  {
    return ConstantFlux(
        fl1.local_constant() + fl2.local_constant(),
        fl1.neighbor_constant() + fl2.neighbor_constant()
        );
  }




  template <class ConstantFlux>
  ConstantFlux subtract_constant_fluxes(
      const ConstantFlux &fl1,
      const ConstantFlux &fl2
      )
  {
    return ConstantFlux(
        fl1.local_constant() - fl2.local_constant(),
        fl1.neighbor_constant() - fl2.neighbor_constant()
        );
  }




  template <class ConstantFlux>
  ConstantFlux multiply_constant_flux_by_constant(
      const ConstantFlux &fl1,
      double c
      )
  {
    return ConstantFlux(
        fl1.local_constant() * c,
        fl1.neighbor_constant() * c
        );
  }




  template <class Flux>
  struct expose_default_constructible_flux
  {
    void operator()()
    {
      boost::python::class_
        <Flux, boost::python::bases<flux::flux> >
        (Flux::name().c_str());
    }
  };




  template <class Flux>
  struct expose_constant_flux
  {
    void operator()()
    {
      boost::python::class_
        <Flux, boost::python::bases<flux::flux> >
        (Flux::name().c_str(), 
         init<double, double>(
           (arg("local"), arg("neighbor"))
           )
        )
        .add_property("local_constant", &Flux::local_constant)
        .add_property("neighbor_constant", &Flux::neighbor_constant)
        ;
      def("add_fluxes", add_constant_fluxes<Flux>);
      def("subtract_fluxes", subtract_constant_fluxes<Flux>);
      def("multiply_fluxes", multiply_constant_flux_by_constant<Flux>);
    }
  };




  template <class Flux>
  struct expose_normal_multipliable_constant_flux
  {
    void operator()()
    {
      expose_constant_flux<Flux>()();
      def("multiply_fluxes", multiply_constant_flux_by_normal<Flux>);
    }
  };
} }




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

  class_<zero, bases<flux> >("ZeroFlux");
  class_<constant, bases<flux> >(
      "ConstantFlux", 
      init<double, double>(
        (arg("local"), arg("neighbor"))
        )
      );

  mpl::for_each<
    normal_fluxes, 
    hedge::python::expose_default_constructible_flux<mpl::_1> 
      > (call_without_arguments());

  mpl::for_each<
    constant_times_normal_fluxes, 
    hedge::python::expose_constant_flux<mpl::_1> 
      > (call_without_arguments());

  mpl::for_each<
    constant_times_2normal_fluxes, 
    hedge::python::expose_constant_flux<mpl::_1> 
      > (call_without_arguments());

  class_<penalty_term, bases<flux> >("PenaltyTermFlux",
      init<double, double>(
        (arg("coefficient"), arg("power"))
        )
      );

  {
    typedef runtime_binary_operator<std::plus<double> > cl;
    class_<cl, bases<flux>, boost::noncopyable>("SumFlux", 
        init<flux &, flux &>()
        [with_custodian_and_ward<1, 2, with_custodian_and_ward<1, 3> >()]
        )
      ;
  }

  {
    typedef runtime_binary_operator<std::minus<double> > cl;
    class_<cl, bases<flux>, boost::noncopyable>("DifferenceFlux", 
        init<flux &, flux &>()
        [with_custodian_and_ward<1, 2, with_custodian_and_ward<1, 3> >()]
        )
      ;
  }

  {
    typedef runtime_binary_operator<std::multiplies<double> > cl;
    class_<cl, bases<flux>, boost::noncopyable >("ProductFlux", 
        init<flux &, flux &>()
        [with_custodian_and_ward<1, 2, with_custodian_and_ward<1, 3> >()]
        )
      ;
  }
  
  {
    typedef runtime_binary_operator_with_constant<std::multiplies<double> > cl;
    class_<cl, bases<flux>, boost::noncopyable>("ConstantProductFlux", 
        init<flux &, double>()
        [with_custodian_and_ward<1, 2>()]
        )
      ;
  }
  
  {
    typedef runtime_unary_operator<std::negate<double> > cl;
    class_<cl, bases<flux>, boost::noncopyable >("NegativeFlux", 
        init<flux &>()
        [with_custodian_and_ward<1, 2>()]
        )
      ;
  }

}

