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




#ifndef _ASFAHDALSU_HEDGE_FACE_OPERATORS_HPP_INCLUDED
#define _ASFAHDALSU_HEDGE_FACE_OPERATORS_HPP_INCLUDED




#include <boost/foreach.hpp>
#include <boost/numeric/bindings/traits/ublas_matrix.hpp>
#include <boost/numeric/bindings/blas/blas3.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <vector>
#include <utility>
#include "base.hpp"



namespace hedge 
{
  struct face_base
  {
    double h;
    element_number_t element_id;
    face_number_t face_id;
    unsigned order;

    face_base()
      : h(0), 
      element_id(INVALID_ELEMENT), face_id(INVALID_FACE),
      order(0)
    { }
  };



  struct straight_face : face_base
  {
    double face_jacobian;
    bounded_vector<double, max_dims> normal;

    straight_face()
      : face_jacobian(0)
    { }
  };




  struct curved_face : face_base
  {
  };





  typedef numpy_vector<index_t> index_lists_t;
  typedef unsigned index_list_number_t;

  template <class FaceType>
  struct face_pair_side : FaceType
  {
    node_number_t el_base_index;
    index_list_number_t face_index_list_number;

    /** An element number local to this face group. */
    unsigned local_el_number;

    face_pair_side()
      : el_base_index(INVALID_NODE),
      face_index_list_number(INVALID_INDEX),
      local_el_number(INVALID_INDEX)
    { }
  };





  template <class IntFaceType, class ExtFaceType = IntFaceType>
  struct face_pair
  {
    typedef IntFaceType int_face_type;
    typedef ExtFaceType ext_face_type;

    face_pair_side<int_face_type> int_side;
    face_pair_side<ext_face_type> ext_side;

    index_list_number_t ext_native_write_map;

    face_pair()
      : ext_native_write_map(INVALID_INDEX)
    { }
  };




  template <class FacePairType>
  struct face_group
  {
    typedef FacePairType face_pair_type;
    typedef std::vector<face_pair_type> face_pair_vector;

    face_pair_vector face_pairs;
    index_lists_t index_lists;

    const bool double_sided;
    /** The number of elements touched by this face group.
     * Used for sizing a temporary.
     */
    unsigned face_count;
    numpy_vector<npy_uint> local_el_to_global_el_base;

    face_group(bool d_sided)
      : double_sided(d_sided), 
      face_count(0)
    { }

    unsigned element_count() const
    { return local_el_to_global_el_base.size(); }

    unsigned face_length() const
    { return index_lists.dims()[1]; }

    index_lists_t::const_iterator index_list(index_list_number_t number) const
    { 
      return index_lists.begin() + face_length()*number;
    }
  };




  template <class MatrixScalar, class FieldScalar>
  inline
  void lift_flux_without_blas(
      const face_group<face_pair<straight_face> > &fg,
      const numpy_matrix<MatrixScalar> &matrix, 
      const pyublas::invalid_ok<numpy_vector<double> > &elwise_post_scaling,
      numpy_vector<FieldScalar> fluxes_on_faces,
      numpy_vector<FieldScalar> result)
  {
    const unsigned el_length_result = matrix.size1();
    const unsigned el_length_temp = fg.face_count*fg.face_length();

    if (el_length_temp != matrix.size2())
      throw std::runtime_error("matrix size mismatch in finish_flux");

    if (elwise_post_scaling->is_valid())
    {
      numpy_vector<double>::const_iterator el_scale_it = elwise_post_scaling->begin();
      for (unsigned i_loc_el = 0; i_loc_el < fg.element_count(); ++i_loc_el)
        noalias(
            subrange(result,
              fg.local_el_to_global_el_base[i_loc_el],
              fg.local_el_to_global_el_base[i_loc_el]+el_length_result))
          += MatrixScalar(*el_scale_it++) * prod(matrix,
              subrange(fluxes_on_faces,
                el_length_temp*i_loc_el,
                el_length_temp*(i_loc_el+1))
              );
    }
    else
    {
      for (unsigned i_loc_el = 0; i_loc_el < fg.element_count(); ++i_loc_el)
        noalias(
            subrange(result,
              fg.local_el_to_global_el_base[i_loc_el],
              fg.local_el_to_global_el_base[i_loc_el]+el_length_result))
          += prod(matrix,
              subrange(fluxes_on_faces,
                el_length_temp*i_loc_el,
                el_length_temp*(i_loc_el+1))
              );
    }
  }
  template <class MatrixScalar, class FieldScalar>
  inline
  void lift_flux(
      const face_group<face_pair<straight_face> > &fg,
      const numpy_matrix<MatrixScalar> &matrix, 
      const pyublas::invalid_ok<numpy_vector<double> > &elwise_post_scaling,
      numpy_vector<FieldScalar> fluxes_on_faces,
      numpy_vector<FieldScalar> result)
#ifdef USE_BLAS
  {
    using namespace boost::numeric::bindings;
    using blas::detail::gemm;

    const unsigned el_length_result = matrix.size1();
    const unsigned el_length_temp = fg.face_count*fg.face_length();

    if (el_length_temp != matrix.size2())
      throw std::runtime_error("matrix size mismatch in finish_flux");

    vector<FieldScalar> result_temp(el_length_result*fg.element_count());
    result_temp.clear();
    gemm(
        'T', // "matrix" is row-major
        'N', // a contiguous array of vectors is column-major
        matrix.size1(),
        fg.element_count(),
        matrix.size2(),
        /*alpha*/ 1,
        /*a*/ traits::matrix_storage(matrix.as_ublas()), 
        /*lda*/ matrix.size2(),
        /*b*/ traits::vector_storage(fluxes_on_faces), 
        /*ldb*/ el_length_temp,
        /*beta*/ 0,
        /*c*/ traits::vector_storage(result_temp), 
        /*ldc*/ el_length_result
        );

    if (elwise_post_scaling->is_valid())
    {
      numpy_vector<double>::const_iterator el_scale_it = elwise_post_scaling->begin();
      for (unsigned i_loc_el = 0; i_loc_el < fg.element_count(); ++i_loc_el)
        noalias(
            subrange(result,
              fg.local_el_to_global_el_base[i_loc_el],
              fg.local_el_to_global_el_base[i_loc_el]+el_length_result))
          += MatrixScalar(*el_scale_it++) * subrange(result_temp,
              el_length_result*i_loc_el,
              el_length_result*(i_loc_el+1));
    }
    else
    {
      for (unsigned i_loc_el = 0; i_loc_el < fg.element_count(); ++i_loc_el)
        noalias(
            subrange(result,
              fg.local_el_to_global_el_base[i_loc_el],
              fg.local_el_to_global_el_base[i_loc_el]+el_length_result))
          += subrange(result_temp,
              el_length_result*i_loc_el,
              el_length_result*(i_loc_el+1));
    }
  }
#else
  {
    lift_flux_without_blas(fg, matrix, elwise_post_scaling,
        fluxes_on_faces, result);
  }
#endif
}




#endif
