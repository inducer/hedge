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
  template <class A, class B>
  std::pair<A, B> extract_pair(const object &obj)
  {
    return std::pair<A, B>(extract<A>(obj[0]), extract<B>(obj[1]));
  }




  void face_group_add_face(face_group &fg, 
      object &my_ind_py, object &opp_ind_py, 
      const fluxes::face &face)
  {
    face_group::index_list my_ind, opp_ind;

    for (unsigned i = 0; i < unsigned(len(my_ind_py)); i++)
      my_ind.push_back(extract<unsigned>(my_ind_py[i]));
    for (unsigned i = 0; i < unsigned(len(my_ind_py)); i++)
      opp_ind.push_back(extract<unsigned>(opp_ind_py[i]));

    fg.add_face(my_ind, opp_ind, face);
  }




  void face_group_connect_faces(face_group &fg, object &cnx_list_py)
  {
    face_group::connection_list cnx_list;

    for (unsigned i = 0; i < unsigned(len(cnx_list_py)); i++)
      cnx_list.push_back(extract_pair<unsigned, unsigned>(cnx_list_py[i]));

    fg.connect_faces(cnx_list);
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
      .def_readwrite("h", &cl::h)
      .def_readwrite("face_jacobian", &cl::face_jacobian)
      .def_readwrite("element_id", &cl::element_id)
      .def_readwrite("face_id", &cl::face_id)
      .def_readwrite("order", &cl::order)
      .def_readwrite("normal", &cl::normal)
      ;
  }

  {
    typedef face_group cl;
    class_<cl>("FaceGroup")
      .def("__len__", &cl::size)
      .def("clear", &cl::clear)
      .def("add_face", face_group_add_face)
      .def("connect_faces", face_group_connect_faces)
      ;
  }
}

