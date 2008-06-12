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
#include "flux.hpp"




namespace hedge 
{
  typedef py_uint_vector index_lists_t;
  typedef unsigned index_list_number_t;

  struct face_pair
  {
    static const unsigned INVALID_INDEX = UINT_MAX;

    static unsigned get_INVALID_INDEX()
    { return INVALID_INDEX; }

    struct side : public fluxes::face
    {
      node_number_t el_base_index;
      index_list_number_t face_index_list_number;

      /** An element number local to this face group. */
      unsigned local_el_number;

      side()
        : el_base_index(INVALID_NODE),
        face_index_list_number(INVALID_INDEX),
        local_el_number(INVALID_INDEX)
      { }
    };

    side loc, opp;
    index_list_number_t opp_native_write_map;

    face_pair()
      : opp_native_write_map(INVALID_INDEX)
    { }
  };

  struct face_group
  {
    public:
      typedef std::vector<face_pair> face_pair_vector;

      face_pair_vector face_pairs;
      index_lists_t index_lists;

      const bool double_sided;
      /** The number of elements touched by this face group.
       * Used for sizing a temporary.
       */
      unsigned face_count;
      py_uint_vector local_el_to_global_el_base;

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




  inline
  void lift_flux(
      const face_group &fg,
      const py_matrix &matrix, 
      const pyublas::invalid_ok<py_vector> &elwise_post_scaling,
      py_vector fluxes_on_faces,
      py_vector result)
#ifdef USE_BLAS
  {
    using namespace boost::numeric::bindings;
    using blas::detail::gemm;

    const unsigned el_length_result = matrix.size1();
    const unsigned el_length_temp = fg.face_count*fg.face_length();

    if (el_length_temp != matrix.size2())
      throw std::runtime_error("matrix size mismatch in finish_flux");

    dyn_vector result_temp(el_length_result*fg.element_count());
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
      py_vector::const_iterator el_scale_it = elwise_post_scaling->begin();
      for (unsigned i_loc_el = 0; i_loc_el < fg.element_count(); ++i_loc_el)
        noalias(
            subrange(result,
              fg.local_el_to_global_el_base[i_loc_el],
              fg.local_el_to_global_el_base[i_loc_el]+el_length_result))
          += *el_scale_it++ * subrange(result_temp,
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
    const unsigned el_length_result = matrix.size1();
    const unsigned el_length_temp = fg.face_count*fg.face_length();

    if (el_length_temp != matrix.size2())
      throw std::runtime_error("matrix size mismatch in finish_flux");

    if (elwise_post_scaling->is_valid())
    {
      py_vector::const_iterator el_scale_it = elwise_post_scaling->begin();
      for (unsigned i_loc_el = 0; i_loc_el < fg.element_count(); ++i_loc_el)
        noalias(
            subrange(result,
              fg.local_el_to_global_el_base[i_loc_el],
              fg.local_el_to_global_el_base[i_loc_el]+el_length_result))
          += *el_scale_it++ * prod(matrix,
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
#endif




  inline 
  py_vector::value_type subscript(
      const py_vector::const_iterator it, unsigned index)
  {
    return it[index];
  }

  inline 
  double subscript(
      boost::numeric::ublas::zero_vector<double>::const_iterator it,
      unsigned)
  {
    return 0;
  }




  template <class IntFlux, class ExtFlux, class LocOperand, class OppOperand>
  inline
  void perform_single_sided_flux(const face_group &fg, 
      IntFlux int_flux, ExtFlux ext_flux, 
      const LocOperand loc_operand,
      const OppOperand opp_operand,
      py_vector fluxes_on_faces
      )
  {
    const typename LocOperand::const_iterator loc_op_it = loc_operand.begin();
    const typename OppOperand::const_iterator opp_op_it = opp_operand.begin();
    const py_vector::iterator fof_it = fluxes_on_faces.begin();

    BOOST_FOREACH(const face_pair &fp, fg.face_pairs)
    {
      const double int_coeff_here = 
        fp.loc.face_jacobian*int_flux(
            fp.loc, &fp.opp);
      const double ext_coeff_here = 
        fp.loc.face_jacobian*ext_flux(
            fp.loc, &fp.opp);

      index_lists_t::const_iterator loc_idx_list = 
        fg.index_list(fp.loc.face_index_list_number);
      index_lists_t::const_iterator opp_idx_list = 
        fg.index_list(fp.opp.face_index_list_number);

      const int lebi = fp.loc.el_base_index;
      const int oebi = fp.opp.el_base_index;

      const unsigned loc_tempbase = 
        fg.face_length()*(
            fp.loc.local_el_number*fg.face_count 
            + fp.loc.face_id);

      for (unsigned i = 0; i < fg.face_length(); i++)
      {
        const double loc_val = subscript(loc_op_it, lebi+loc_idx_list[i]);
        const double opp_val = subscript(opp_op_it, oebi+opp_idx_list[i]);
        fof_it[loc_tempbase+i] = 
          int_coeff_here*loc_val + ext_coeff_here*opp_val;
      }
    }
  }




  template <class IntFlux, class ExtFlux>
  inline
  void perform_double_sided_flux(const face_group &fg, 
      IntFlux int_flux, ExtFlux ext_flux, 
      const py_vector &operand,
      py_vector fluxes_on_faces
      )
  {
    const py_vector::const_iterator op_it = operand.begin();
    const py_vector::iterator fof_it = fluxes_on_faces.begin();

    BOOST_FOREACH(const face_pair &fp, fg.face_pairs)
    {
      const double int_coeff_here = 
        fp.loc.face_jacobian*int_flux(
            fp.loc, &fp.opp);
      const double ext_coeff_here = 
        fp.loc.face_jacobian*ext_flux(
            fp.loc, &fp.opp);
      const double int_coeff_opp = 
        fp.loc.face_jacobian*int_flux(
            fp.opp, &fp.loc);
      const double ext_coeff_opp = 
        fp.loc.face_jacobian*ext_flux(
            fp.opp, &fp.loc);

      index_lists_t::const_iterator loc_idx_list = 
        fg.index_list(fp.loc.face_index_list_number);
      index_lists_t::const_iterator opp_idx_list = 
        fg.index_list(fp.opp.face_index_list_number);

      const int lebi = fp.loc.el_base_index;
      const int oebi = fp.opp.el_base_index;

      index_lists_t::const_iterator opp_write_map = 
        fg.index_list(fp.opp_native_write_map);
      const unsigned loc_tempbase = 
        fg.face_length()*(
            fp.loc.local_el_number*fg.face_count 
            + fp.loc.face_id);
      const unsigned opp_tempbase = 
        fg.face_length()*(
            fp.opp.local_el_number*fg.face_count 
            + fp.opp.face_id);

      for (unsigned i = 0; i < fg.face_length(); i++)
      {
        const double loc_val = op_it[lebi+loc_idx_list[i]];
        const double opp_val = op_it[oebi+opp_idx_list[i]];
        fof_it[loc_tempbase+i] = 
          int_coeff_here*loc_val + ext_coeff_here*opp_val;
        fof_it[opp_tempbase+opp_write_map[i]] = 
          int_coeff_opp*opp_val + ext_coeff_opp*loc_val;
      }
    }
  }
}




#endif
