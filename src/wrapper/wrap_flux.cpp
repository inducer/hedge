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
#include <boost/shared_ptr.hpp>
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
    double operator()(
        const fluxes::face &local, 
        const fluxes::face *neighbor) const
    {
      return this->get_override("__call__")(boost::ref(local), boost::ref(neighbor));
    }
  };




  template<class Operation> 
  void expose_unary_operator(Operation, const std::string &name)
  {
    typedef fluxes::unary_operator<Operation, 
            fluxes::chained_flux> cl;

    class_<cl, bases<fluxes::flux>, boost::noncopyable>
      (name.c_str(), 
       init<fluxes::chained_flux>(args("operand"))[
       with_custodian_and_ward<1, 2>()
       ])
      .add_property("operand", 
          make_function(&cl::operand, return_internal_reference<>()))
      ;
  }




  template<class Operation> 
  void expose_binary_operator(Operation, const std::string &name)
  {
    typedef fluxes::binary_operator<Operation, 
            fluxes::chained_flux, fluxes::chained_flux> cl;

    class_<cl, bases<fluxes::flux>, boost::noncopyable>
      (name.c_str(), 
       init<
       const fluxes::chained_flux &, 
       const fluxes::chained_flux &>
       (args("operand1", "operand2"))
       [with_custodian_and_ward<1, 2, 
        with_custodian_and_ward<1, 3> >()])
      .add_property("operand1", 
          make_function(&cl::operand1, return_internal_reference<>()))
      .add_property("operand2", 
          make_function(&cl::operand2, return_internal_reference<>()))
      ;
  }




  template<class Operation> 
  void expose_binary_constant_operator(Operation, const std::string &name)
  {
    typedef fluxes::binary_operator<Operation, 
            fluxes::chained_flux, fluxes::constant> cl;
    class_<cl, bases<fluxes::flux>, boost::noncopyable>
      (name.c_str(), 
       init<const fluxes::chained_flux &, double>
       (args("operand1", "constant"))[with_custodian_and_ward<1, 2>()])
      .add_property("operand1", 
          make_function(&cl::operand1, return_internal_reference<>()))
      .add_property("operand2", 
          make_function(&cl::operand2, return_internal_reference<>()))
      ;
  }




  double_sided_flux_info<fluxes::chained_flux, fluxes::chained_flux> 
    parse_dsfi(object tup)
  {
    typedef fluxes::chained_flux cf;

    if (len(tup) != 3)
      PYTHON_ERROR(ValueError, "flux descriptor tuple must have three entries");

    return double_sided_flux_info<cf, cf>(
        extract<cf>(tup[0]),
        extract<cf>(tup[1]),
        extract<hedge::py_vector>(tup[2]));
  }




  void wrap_perform_multiple_double_sided_fluxes_on_single_operand(
      const face_group &fg,
      const hedge::py_matrix &fmm,
      object fluxes,
      const hedge::py_vector &operand)
  {
    typedef fluxes::chained_flux cf;
    typedef double_sided_flux_info<cf, cf> dsfi_t;

    unsigned i = 0;
    while (unsigned(len(fluxes)) >= i+4)
    {
      dsfi_t flux_info[4] = {
        parse_dsfi(fluxes[i+0]),
        parse_dsfi(fluxes[i+1]),
        parse_dsfi(fluxes[i+2]),
        parse_dsfi(fluxes[i+3])
      };
      perform_multiple_double_sided_fluxes_on_single_operand<4>(
          fg, fmm, flux_info, operand);
      i += 4;
    }

    if (unsigned(len(fluxes)) == i+3)
    {
      dsfi_t flux_info[3] = {
        parse_dsfi(fluxes[i+0]),
        parse_dsfi(fluxes[i+1]),
        parse_dsfi(fluxes[i+2])
      };
      perform_multiple_double_sided_fluxes_on_single_operand<3>(
          fg, fmm, flux_info, operand);
    }
    else if (unsigned(len(fluxes)) == i+2)
    {
      dsfi_t flux_info[2] = {
        parse_dsfi(fluxes[i+0]),
        parse_dsfi(fluxes[i+1])
      };
      perform_multiple_double_sided_fluxes_on_single_operand<2>(
          fg, fmm, flux_info, operand);
    }
    else if (unsigned(len(fluxes)) == i+1)
    {
      dsfi_t flux_info[1] = {
        parse_dsfi(fluxes[i+0]),
      };
      perform_multiple_double_sided_fluxes_on_single_operand<1>(
          fg, fmm, flux_info, operand);
    }
  }
}




void hedge_expose_fluxes()
{
  {
    typedef fluxes::flux cl;
    class_<flux_wrap, boost::noncopyable>("Flux")
      .def("__call__", pure_virtual(&cl::operator()))
      ;
  }

  {
    typedef fluxes::chained_flux cl;
    class_<cl, bases<fluxes::flux> >("ChainedFlux", 
        init<fluxes::flux &>()
        [with_custodian_and_ward<1, 2>()]
        )
      .add_property("child", 
          make_function(&cl::child, return_internal_reference<>()))
      ;
    EXPOSE_FLUX_PERFORM(cl)
  }

  {
    typedef fluxes::constant cl;

    class_<cl, bases<fluxes::flux> >(
        "ConstantFlux", 
        init<double>(arg("value")))
      .def(init<double>(arg("both")))
      .def(self + self)
      .def(self - self)
      .def(- self)
      .def(self * double())
      .add_property("value", &cl::value)
      ;
  }

  {
    typedef fluxes::normal cl;
    class_<cl, bases<fluxes::flux> >("NormalFlux", init<int>(arg("axis")))
      .add_property("axis", &cl::axis)
      ;
  }

  {
    typedef fluxes::penalty_term cl;
    class_<cl, bases<fluxes::flux> >("PenaltyFlux",
        init<double>(arg("power"))
        )
      .add_property("power", &cl::power)
      ;
  }

  {
    typedef fluxes::if_positive<fluxes::chained_flux, 
            fluxes::chained_flux, fluxes::chained_flux> cl;

    class_<cl, bases<fluxes::flux>, boost::noncopyable>
      ("IfPositiveFlux", 
       init<
         const fluxes::chained_flux &,
         const fluxes::chained_flux &,
         const fluxes::chained_flux &
         >(args("criterion", "then_part", "else_part"))[
           with_custodian_and_ward<1, 2, 
           with_custodian_and_ward<1, 3,
           with_custodian_and_ward<1, 4> > >()]
       )
      .add_property("criterion", 
          make_function(&cl::criterion, return_internal_reference<>()))
      .add_property("then_part", 
          make_function(&cl::then_part, return_internal_reference<>()))
      .add_property("else_part", 
          make_function(&cl::else_part, return_internal_reference<>()))
      ;
  }

  expose_binary_operator(std::plus<double>(), "SumFlux");
  expose_binary_operator(std::minus<double>(), "DifferenceFlux");
  expose_binary_operator(std::multiplies<double>(), "ProductFlux");
  expose_binary_constant_operator(std::multiplies<double>(), 
      "ProductWithConstantFlux");
  expose_binary_constant_operator(std::multiplies<double>(), 
      "ProductWithConstantFlux");

  expose_unary_operator(std::negate<double>(), "NegativeFlux");




  // face information ---------------------------------------------------------
  {
    typedef fluxes::face cl;
    class_<cl>("FluxFace")
      .DEF_SIMPLE_RW_MEMBER(h)
      .DEF_SIMPLE_RW_MEMBER(face_jacobian)
      .DEF_SIMPLE_RW_MEMBER(element_id)
      .DEF_SIMPLE_RW_MEMBER(face_id)
      .DEF_SIMPLE_RW_MEMBER(order)
      .def(pyublas::by_value_rw_member("normal", &cl::normal))
      ;
  }
  {
    typedef face_pair cl;
    class_<cl>("FacePair")
      .add_static_property("INVALID_INDEX", &cl::get_INVALID_INDEX)
      .DEF_SIMPLE_RW_MEMBER(el_base_index)
      .DEF_SIMPLE_RW_MEMBER(opp_el_base_index)
      .DEF_SIMPLE_RW_MEMBER(face_index_list_number)
      .DEF_SIMPLE_RW_MEMBER(opp_face_index_list_number)
      .DEF_SIMPLE_RW_MEMBER(flux_face_index)
      .DEF_SIMPLE_RW_MEMBER(opp_flux_face_index)
      ;
  }

  {
    typedef face_group::face_pair_vector cl;
    class_<cl>("FacePairVector")
      .def(no_compare_indexing_suite<cl>())
      ;
  }

  {
    typedef face_group::flux_face_vector cl;
    class_<cl>("FluxFaceVector")
      .def(no_compare_indexing_suite<cl>())
      ;
  }

  {
    typedef face_group::index_list_vector cl;
    class_<cl>("IndexListVector")
      .def(no_compare_indexing_suite<cl>())
      ;
  }

  {
    typedef face_group cl;
    class_<cl, boost::shared_ptr<cl> >("FaceGroup", init<bool>(arg("double_sided")))
      .DEF_SIMPLE_RW_MEMBER(face_pairs)
      .DEF_SIMPLE_RW_MEMBER(flux_faces)
      .DEF_SIMPLE_RW_MEMBER(index_lists)
      ;
  }

  def("perform_multiple_double_sided_fluxes_on_single_operand",
      wrap_perform_multiple_double_sided_fluxes_on_single_operand);
}

