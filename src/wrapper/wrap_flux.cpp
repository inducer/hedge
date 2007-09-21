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
#include <boost/tuple/tuple.hpp>
#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include "flux.hpp"
#include "face_operators.hpp"
#include "op_target.hpp"
#include "wrap_helpers.hpp"




using namespace boost::python;
using namespace hedge;
namespace mpl = boost::mpl;
namespace ublas = boost::numeric::ublas;




namespace {
  // flux-related -------------------------------------------------------------
  struct flux_wrap : fluxes::flux, wrapper<fluxes::flux>
  {
    double local_coeff(const fluxes::face &local) const
    {
      return this->get_override("local_coeff")(boost::ref(local));
    }
    double neighbor_coeff(
        const fluxes::face &local, 
        const fluxes::face *neighbor) const
    {
      return this->get_override("neighbor_coeff")(boost::ref(local), boost::ref(neighbor));
    }
  };




  template<class Flux>
  Flux *new_unary_op_flux(const fluxes::flux &op1)
  {
    return new Flux(fluxes::chained_flux(op1));
  }




  template<class Operation> 
  void expose_unary_operator(Operation, const std::string &name)
  {
    typedef fluxes::unary_operator<Operation, 
            fluxes::chained_flux> cl;

    class_<cl, bases<fluxes::flux>, boost::noncopyable>
      (name.c_str(), no_init)
      EXPOSE_FLUX_PERFORM(cl)
      .add_property("operand", 
          make_function(&cl::operand, return_internal_reference<>()))
      ;
    def(("make_"+name).c_str(), 
        new_unary_op_flux<cl>,
        with_custodian_and_ward_postcall<0, 1,
        return_value_policy<manage_new_object> >());
  }




  template<class Flux>
  Flux *new_binary_op_flux(
      const fluxes::flux &op1, const fluxes::flux &op2)
  {
    return new Flux(
          fluxes::chained_flux(op1), fluxes::chained_flux(op2));
  }




  template<class Operation> 
  void expose_binary_operator(Operation, const std::string &name)
  {
    typedef fluxes::binary_operator<Operation, 
            fluxes::chained_flux, fluxes::chained_flux> cl;

    class_<cl, bases<fluxes::flux>, boost::noncopyable>
      (name.c_str(), no_init)
      EXPOSE_FLUX_PERFORM(cl)
      .add_property("operand1", 
          make_function(&cl::operand1, return_internal_reference<>()))
      .add_property("operand2", 
          make_function(&cl::operand2, return_internal_reference<>()))
      ;

    def(("make_"+name).c_str(), 
        new_binary_op_flux<cl>,
        with_custodian_and_ward_postcall<0, 1, 
        with_custodian_and_ward_postcall<0, 2,
        return_value_policy<manage_new_object> > >());
  }




  template<class Flux>
  Flux *new_binary_constant_op_flux(
      const fluxes::flux &op1, double c)
  {
    return new Flux(fluxes::chained_flux(op1), fluxes::constant(c));
  }




  template<class Operation> 
  void expose_binary_constant_operator(Operation, const std::string &name)
  {
    typedef fluxes::binary_operator<Operation, 
            fluxes::chained_flux, fluxes::constant> cl;
    class_<cl, bases<fluxes::flux>, boost::noncopyable>
      (name.c_str(), no_init)
      EXPOSE_FLUX_PERFORM(cl)
      .add_property("operand1", 
          make_function(&cl::operand1, return_internal_reference<>()))
      .add_property("operand2", 
          make_function(&cl::operand2, return_internal_reference<>()))
      ;

    def(("make_"+name).c_str(), 
        new_binary_constant_op_flux<cl>,
        with_custodian_and_ward_postcall<0, 1,
        return_value_policy<manage_new_object> >());
  }




  // face_group ---------------------------------------------------------------
  class face_group_indexing_suite :
    public vector_indexing_suite<face_group, false, face_group_indexing_suite>
  {
    public:
      static bool contains(face_group &container, face_pair const &key)
      { PYTHON_ERROR(NotImplementedError, "face pairs are not comparable"); }
  };




  void face_group_connect_faces(face_group &fg, object &cnx_list_py)
  {
    BOOST_FOREACH(tuple tp, std::make_pair(
          stl_input_iterator<tuple>(cnx_list_py),
          stl_input_iterator<tuple>()))
      fg[extract<unsigned>(tp[0])].opp_flux_face = &fg[extract<unsigned>(tp[0])].flux_face;
  }
}




void hedge_expose_fluxes()
{
  enum_<which_faces>("which_faces")
    .ENUM_VALUE(BOTH)
    .ENUM_VALUE(LOCAL)
    .ENUM_VALUE(NEIGHBOR)
    ;


  {
    typedef fluxes::flux cl;
    class_<flux_wrap, boost::noncopyable>("Flux")
      .def("neighbor_coeff", pure_virtual(&cl::neighbor_coeff))
      .def("local_coeff", pure_virtual(&cl::local_coeff))
      ;
  }

  {
    typedef fluxes::chained_flux cl;
    class_<cl, bases<fluxes::flux> >("ChainedFlux", 
        init<fluxes::flux &>()
        [with_custodian_and_ward<1, 2>()]
        )
      EXPOSE_FLUX_PERFORM(cl)
      .add_property("child", 
          make_function(&cl::child, return_internal_reference<>()))
      ;
  }

  {
    typedef fluxes::constant cl;

    class_<cl, bases<fluxes::flux> >(
        "ConstantFlux", 
        init<double, double>(
          (arg("local"), arg("neighbor"))
          ))
      .def(init<double>(arg("both")))
      .def(self + self)
      .def(self - self)
      .def(- self)
      .def(self * double())
      .add_property("local_c", &cl::local_constant)
      .add_property("neighbor_c", &cl::neighbor_constant)
      EXPOSE_FLUX_PERFORM(cl)
      ;
  }

  {
    typedef fluxes::normal cl;
    class_<cl, bases<fluxes::flux> >("NormalFlux", init<int>(arg("axis")))
      EXPOSE_FLUX_PERFORM(cl)
      .add_property("axis", &cl::axis)
      ;
  }

  {
    typedef fluxes::penalty_term cl;
    class_<cl, bases<fluxes::flux> >("PenaltyFlux",
        init<double>(arg("power"))
        )
      EXPOSE_FLUX_PERFORM(cl)
      .add_property("power", &cl::power)
      ;
  }

  expose_binary_operator(std::plus<double>(), "SumFlux");
  expose_binary_operator(std::minus<double>(), "DifferenceFlux");
  expose_binary_operator(std::multiplies<double>(), "ProductFlux");
  expose_binary_constant_operator(std::multiplies<double>(), 
      "ProductWithConstantFlux");

  expose_unary_operator(std::negate<double>(), "NegativeFlux");




  // face information ---------------------------------------------------------
  {
    typedef fluxes::face cl;
    class_<cl>("Face")
      .DEF_SIMPLE_RW_MEMBER(h)
      .DEF_SIMPLE_RW_MEMBER(face_jacobian)
      .DEF_SIMPLE_RW_MEMBER(element_id)
      .DEF_SIMPLE_RW_MEMBER(face_id)
      .DEF_SIMPLE_RW_MEMBER(order)
      .DEF_SIMPLE_RW_MEMBER(normal)
      ;
  }
  {
    typedef face_pair cl;
    class_<cl>("FacePair")
      .DEF_SIMPLE_RW_MEMBER(face_indices)
      .DEF_SIMPLE_RW_MEMBER(opposite_indices)
      .DEF_SIMPLE_RW_MEMBER(flux_face)
      .DEF_SIMPLE_RW_MEMBER(opp_flux_face)
      ;
  }

  {
    typedef face_group cl;
    class_<cl>("FaceGroup")
      .def(face_group_indexing_suite())
      .DEF_SIMPLE_METHOD(clear)
      .def("connect_faces", face_group_connect_faces)
      ;
  }
}

