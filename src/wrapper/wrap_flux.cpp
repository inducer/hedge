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




#include <iostream>
#include <boost/python.hpp>
#include "flux.hpp"




using namespace boost::python;
using namespace hedge::flux;




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
  class_<normal_x, bases<flux> >("NormalXFlux");
  class_<normal_y, bases<flux> >("NormalYFlux");
  class_<normal_z, bases<flux> >("NormalZFlux");
  class_<jump_x, bases<flux> >("JumpXFlux");
  class_<jump_y, bases<flux> >("JumpYFlux");
  class_<jump_z, bases<flux> >("JumpZFlux");
  class_<zero, bases<flux> >("ZeroFlux");
  class_<local, bases<flux> >("LocalFlux");
  class_<neighbor, bases<flux> >("NeighborFlux");
  class_<average, bases<flux> >("AverageFlux");
  class_<trace_sign, bases<flux> >("TraceSignFlux");
  class_<neg_trace_sign, bases<flux> >("NegativeTraceSignFlux");
  class_<penalty_term, bases<flux> >("PenaltyTermFlux",
      init<double, double>()
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

