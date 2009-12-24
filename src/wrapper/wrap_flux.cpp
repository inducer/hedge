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
#include <hedge/face_operators.hpp>
#include "wrap_helpers.hpp"




using namespace boost::python;
using namespace hedge;
namespace mpl = boost::mpl;
namespace ublas = boost::numeric::ublas;




namespace
{
#define MAKE_LIFT_EXPOSER(NAME) \
  template <class MatrixScalar, class FieldScalar> \
  void expose_##NAME() \
  { \
    def("lift_flux", NAME<MatrixScalar, FieldScalar>, \
        args("fg", "mat", "elwise_post_scaling", "fluxes_on_faces", "result") \
       ); \
  }

  MAKE_LIFT_EXPOSER(lift_flux);
  MAKE_LIFT_EXPOSER(lift_flux_without_blas);
}




template <class FaceType>
void expose_face_pair_side(std::string const &face_type_name)
{
  {
    typedef face_pair_side<FaceType> cl;
    class_<cl, bases<FaceType> >((face_type_name + "FacePairSide" ).c_str())
      .DEF_SIMPLE_RW_MEMBER(el_base_index)
      .DEF_SIMPLE_RW_MEMBER(face_index_list_number)
      .DEF_SIMPLE_RW_MEMBER(local_el_number)
      ;
  }
}




template <class IntFaceType, class ExtFaceType>
void expose_face_pair(std::string const &face_pair_type_name)
{
  typedef face_pair<IntFaceType, ExtFaceType> face_pair_type;
  typedef face_group<face_pair_type> face_group_type;
  
  // FaceGroup name is composed here from type of face pair
  // and "FaceGroup" suffix.
  class_<face_group_type, boost::shared_ptr<face_group_type> > fg_wrap(
      (face_pair_type_name + "FaceGroup" ).c_str(),
      init<bool>(args("double_sided")));

  scope fg_scope = fg_wrap;

  {
    typedef face_pair_type cl;
    class_<cl>("FacePair")
      .DEF_SIMPLE_RW_MEMBER(int_side)
      .DEF_SIMPLE_RW_MEMBER(ext_side)
      .DEF_SIMPLE_RW_MEMBER(ext_native_write_map)
      ;
  }

  {
    typedef typename face_group_type::face_pair_vector cl;
    class_<cl>("FacePairVector")
      .def(no_compare_indexing_suite<cl>())
      ;
  }

  {
    typedef face_group_type cl;
    fg_wrap
      .DEF_SIMPLE_RW_MEMBER(face_pairs)
      .DEF_SIMPLE_RW_MEMBER(face_count)
      .DEF_BYVAL_RW_MEMBER(local_el_to_global_el_base)
      .DEF_BYVAL_RW_MEMBER(index_lists)
      .DEF_SIMPLE_METHOD(element_count)
      .DEF_SIMPLE_METHOD(face_length)
      ;
  }
}




void hedge_expose_fluxes()
{
  {
    typedef face_base cl;
    class_<cl>("FaceBase")
      .DEF_SIMPLE_RW_MEMBER(h)
      .DEF_SIMPLE_RW_MEMBER(element_id)
      .DEF_SIMPLE_RW_MEMBER(face_id)
      .DEF_SIMPLE_RW_MEMBER(order)
      ;
  }

  {
    typedef straight_face cl;
    class_<cl, bases<face_base> >("StraightFace")
      .DEF_SIMPLE_RW_MEMBER(face_jacobian)
      .DEF_SIMPLE_RW_MEMBER(element_jacobian)
      .def(pyublas::by_value_rw_member("normal", &cl::normal))
      ;
  }

  {
    typedef curved_face cl;
    class_<cl, bases<face_base> >("CurvedFace")
      ;
  }

  expose_face_pair_side<straight_face>("Straight");
  expose_face_pair_side<curved_face>("Curved");

  expose_face_pair<straight_face, straight_face>("Straight");
  expose_face_pair<straight_face, curved_face>("StraightCurved");
  expose_face_pair<curved_face, curved_face>("Curved");

  expose_lift_flux<float, float>();
  expose_lift_flux<double, double>();
  expose_lift_flux_without_blas<float, std::complex<float> >();
  expose_lift_flux_without_blas<double, std::complex<double> >();
}
