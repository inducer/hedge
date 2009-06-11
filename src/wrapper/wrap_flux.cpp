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
#include <hedge/flux.hpp>
#include <hedge/face_operators.hpp>
#include <hedge/op_target.hpp>
#include "wrap_helpers.hpp"




using namespace boost::python;
using namespace hedge;
namespace mpl = boost::mpl;
namespace ublas = boost::numeric::ublas;




namespace
{
  template <class Scalar>
  void expose_lift()
  {
    def("lift_flux", lift_flux<Scalar>,
        args("fg", "mat", "elwise_post_scaling", "fluxes_on_faces", "result")
       );
  }
}




void hedge_expose_fluxes()
{
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
    typedef face_pair::side cl;
    class_<cl, bases<fluxes::face> >("FacePairSide")
      .DEF_SIMPLE_RW_MEMBER(el_base_index)
      .DEF_SIMPLE_RW_MEMBER(face_index_list_number)
      .DEF_SIMPLE_RW_MEMBER(local_el_number)
      ;
  }
  {
    typedef face_pair cl;
    class_<cl>("FacePair")
      .add_static_property("INVALID_INDEX", &cl::get_INVALID_INDEX)
      .DEF_SIMPLE_RW_MEMBER(loc)
      .DEF_SIMPLE_RW_MEMBER(opp)
      .DEF_SIMPLE_RW_MEMBER(opp_native_write_map)
      ;
  }

  {
    typedef face_group::face_pair_vector cl;
    class_<cl>("FacePairVector")
      .def(no_compare_indexing_suite<cl>())
      ;
  }

  {
    typedef face_group cl;
    class_<cl, boost::shared_ptr<cl> >("FaceGroup", init<bool>(args("double_sided")))
      .DEF_SIMPLE_RW_MEMBER(face_pairs)
      .DEF_SIMPLE_RW_MEMBER(face_count)
      .DEF_BYVAL_RW_MEMBER(local_el_to_global_el_base)
      .DEF_BYVAL_RW_MEMBER(index_lists)
      .DEF_SIMPLE_METHOD(element_count)
      .DEF_SIMPLE_METHOD(face_length)
      ;
  }

  expose_lift<float>();
  expose_lift<double>();
}

