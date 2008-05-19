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
#include <vector>
#include <utility>
#include "base.hpp"
#include "flux.hpp"
#include "op_target.hpp"




namespace hedge 
{
  typedef py_uint_vector index_lists_t;

  struct face_pair
  {
    static const unsigned INVALID_INDEX = UINT_MAX;

    static unsigned get_INVALID_INDEX()
    { return INVALID_INDEX; }

    struct side : public fluxes::face
    {
      node_number_t el_base_index;
      unsigned face_index_list_number;

      /** An element number local to this face group. */
      unsigned local_el_number;

      side()
        : el_base_index(INVALID_NODE),
        face_index_list_number(INVALID_INDEX),
        local_el_number(INVALID_INDEX)
      { }
    };

    side loc, opp;
    unsigned opp_native_write_map;

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
      unsigned element_count;
      unsigned face_count;
      bool el_number_local_is_global;
      py_uint_vector local_el_indices_in_global_vec;

      face_group(bool d_sided)
        : double_sided(d_sided), 
        element_count(0), 
        face_count(0)
      { }

      unsigned face_length() const
      { return index_lists.dims()[1]; }
      index_lists_t::const_iterator index_list(unsigned number) const
      { 
        return index_lists.begin() + face_length()*number;
      }
  };




  template <class LFlux, class LTarget, class NFlux, class NTarget>
  struct flux_target_data
  {
    typedef LFlux local_flux_t;
    typedef NFlux neighbor_flux_t;
    typedef LTarget local_target_t;
    typedef NTarget neighbor_target_t;

    local_flux_t local_flux;
    local_target_t local_target;
    
    neighbor_flux_t neighbor_flux;
    neighbor_target_t neighbor_target;

    flux_target_data(LFlux lflux, LTarget ltarget, NFlux nflux, NTarget ntarget)
      : local_flux(lflux), local_target(ltarget), 
      neighbor_flux(nflux), neighbor_target(ntarget)
    { }
  };




  template <class LFlux, class LTarget, class NFlux, class NTarget>
  flux_target_data<LFlux, LTarget, NFlux, NTarget> make_flux_data(
      LFlux lflux, LTarget ltarget, NFlux nflux, NTarget ntarget)
  {
    return flux_target_data<LFlux, LTarget, NFlux, NTarget>(lflux, ltarget, nflux, ntarget);
  }




  template <class Mat, class FData>
  void perform_single_sided_flux(const face_group &fg, const Mat &mat, FData fdata)
  {
    BOOST_FOREACH(const face_pair &fp, fg.face_pairs)
    {
      const double local_coeff = 
        fp.loc.face_jacobian*fdata.local_flux(fp.loc, 0);
      const double neighbor_coeff = 
        fp.loc.face_jacobian*fdata.neighbor_flux(fp.loc, 0);

      index_lists_t::const_iterator loc_idx_list = 
        fg.index_list(fp.loc.face_index_list_number);
      index_lists_t::const_iterator opp_idx_list = 
        fg.index_list(fp.opp.face_index_list_number);

      for (unsigned i = 0; i < fg.face_length(); i++)
      {
        const int lili = fp.loc.el_base_index+loc_idx_list[i];
        
        for (unsigned j = 0; j < fg.face_length(); j++)
        {
          const typename Mat::value_type mat_entry = mat(i, j);

          fdata.local_target.add_coefficient(
              lili, fp.loc.el_base_index+loc_idx_list[j], 
              local_coeff*mat_entry);
          fdata.neighbor_target.add_coefficient(
              lili, fp.opp.el_base_index+opp_idx_list[j], 
              neighbor_coeff*mat_entry);
        }
      }
    }
  }




  template <class Mat, class FData>
  void perform_double_sided_flux(const face_group &fg, const Mat &mat, FData fdata)
  {
    BOOST_FOREACH(const face_pair &fp, fg.face_pairs)
    {
      const double local_coeff_here = 
        fp.loc.face_jacobian*fdata.local_flux(
            fp.loc, &fp.opp);
      const double neighbor_coeff_here = 
        fp.loc.face_jacobian*fdata.neighbor_flux(
            fp.loc, &fp.opp);
      const double local_coeff_opp = 
        fp.loc.face_jacobian*fdata.local_flux(
            fp.opp, &fp.loc);
      const double neighbor_coeff_opp = 
        fp.loc.face_jacobian*fdata.neighbor_flux(
            fp.opp, &fp.loc);

      index_lists_t::const_iterator loc_idx_list = 
        fg.index_list(fp.loc.face_index_list_number);
      index_lists_t::const_iterator opp_idx_list = 
        fg.index_list(fp.opp.face_index_list_number);

      for (unsigned i = 0; i < fg.face_length(); i++)
      {
        const int lili = fp.loc.el_base_index+loc_idx_list[i];
        const int oili = fp.opp.el_base_index+opp_idx_list[i];

        index_lists_t::const_iterator lilj_iterator = loc_idx_list;
        index_lists_t::const_iterator oilj_iterator = opp_idx_list;

        unsigned j;

        for (j = 0; j < fg.face_length(); j++)
        {
          const typename Mat::value_type mat_entry = mat(i, j);

          const int lilj = fp.loc.el_base_index+*lilj_iterator++;
          const int oilj = fp.opp.el_base_index+*oilj_iterator++;

          fdata.local_target.add_coefficient(
              lili, lilj, 
              local_coeff_here*mat_entry);
          fdata.neighbor_target.add_coefficient(
              lili, oilj, 
              neighbor_coeff_here*mat_entry);

          fdata.local_target.add_coefficient(
              oili, oilj, 
              local_coeff_opp*mat_entry);
          fdata.neighbor_target.add_coefficient(
              oili, lilj, 
              neighbor_coeff_opp*mat_entry);
        }
      }
    }
  }




  template <class LFlux, class NFlux>
  void perform_double_sided_flux_on_single_vector_target(const face_group &fg, 
      const py_matrix &mat, LFlux local_flux, NFlux neighbor_flux, vector_target target,
      bool newflux
      )
  {
    const py_vector::const_iterator op_it = target.m_operand.begin();
    const py_vector::iterator result_it = target.m_result.begin();

    dyn_vector flux_temp(
        fg.face_count*fg.face_length()*fg.element_count);
    flux_temp.clear();

    BOOST_FOREACH(const face_pair &fp, fg.face_pairs)
    {
      const double local_coeff_here = 
        fp.loc.face_jacobian*local_flux(
            fp.loc, &fp.opp);
      const double neighbor_coeff_here = 
        fp.loc.face_jacobian*neighbor_flux(
            fp.loc, &fp.opp);
      const double local_coeff_opp = 
        fp.loc.face_jacobian*local_flux(
            fp.opp, &fp.loc);
      const double neighbor_coeff_opp = 
        fp.loc.face_jacobian*neighbor_flux(
            fp.opp, &fp.loc);

      index_lists_t::const_iterator loc_idx_list = 
        fg.index_list(fp.loc.face_index_list_number);
      index_lists_t::const_iterator opp_idx_list = 
        fg.index_list(fp.opp.face_index_list_number);

      const int lebi = fp.loc.el_base_index;
      const int oebi = fp.opp.el_base_index;

      if (newflux)
      {
        index_lists_t::const_iterator opp_write_map = 
          fg.index_list(fp.opp_native_write_map);
        const unsigned loc_tempbase = 
          fg.face_length()*(
              fp.loc.element_id*fg.face_count 
              + fp.loc.face_id);
        const unsigned opp_tempbase = 
          fg.face_length()*(
              fp.opp.element_id*fg.face_count 
              + fp.opp.face_id);

        for (unsigned i = 0; i < fg.face_length(); i++)
        {
          const double loc_val = op_it[lebi+loc_idx_list[i]];
          const double opp_val = op_it[oebi+opp_idx_list[i]];
          flux_temp[loc_tempbase+i] = 
            local_coeff_here*loc_val + neighbor_coeff_here*opp_val;
          flux_temp[opp_tempbase+opp_write_map[i]] = 
            local_coeff_opp*opp_val + neighbor_coeff_opp*loc_val;
        }

      }
      else
      for (unsigned i = 0; i < fg.face_length(); i++)
      {
        const int lili = lebi+loc_idx_list[i];
        const int oili = oebi+opp_idx_list[i];

        index_lists_t::const_iterator lilj_iterator = loc_idx_list;
        index_lists_t::const_iterator oilj_iterator = opp_idx_list;

        py_vector::value_type res_lili_addition = 0;
        py_vector::value_type res_oili_addition = 0;

        for (unsigned j = 0; j < fg.face_length(); j++)
        {
          double mat_entry = mat(i, j);

          const double matop_lilj = op_it[lebi+*lilj_iterator++]*mat_entry;
          const double matop_oilj = op_it[oebi+*oilj_iterator++]*mat_entry;

          /*
          __builtin_prefetch(&operand[ebi+ilj_iterator[0]], 0, 1);
          __builtin_prefetch(&operand[oebi+oilj_iterator[0]], 0, 1);
          */

          res_lili_addition += 
             matop_lilj*local_coeff_here
            +matop_oilj*neighbor_coeff_here;
          res_oili_addition += 
            matop_oilj*local_coeff_opp 
            +matop_lilj*neighbor_coeff_opp;
        }

        result_it[lili] += res_lili_addition;
        result_it[oili] += res_oili_addition;
      }
    }

    if (newflux)
    {
      using namespace boost::numeric::bindings;
      using blas::detail::gemm;

      const py_matrix &matrix = mat;

      gemm(
          'T', // "matrix" is row-major
          'N', // a contiguous array of vectors is column-major
          matrix.size1(),
          fg.element_count,
          matrix.size2(),
          /*alpha*/ 1,
          /*a*/ traits::matrix_storage(matrix.as_ublas()), 
          /*lda*/ matrix.size2(),
          /*b*/ traits::vector_storage(flux_temp), 
          /*ldb*/ fg.face_count*fg.face_length(),
          /*beta*/ 1,
          /*c*/ traits::vector_storage(target.m_result), 
          /*ldc*/ matrix.size1()
          );
    }
  }




  template <class LFlux, class NFlux>
  struct double_sided_flux_info
  {
    const LFlux local_flux;
    const NFlux neighbor_flux;
    hedge::py_vector result;

    double_sided_flux_info(
        const LFlux lflux,
        const NFlux nflux,
        hedge::py_vector res)
      : local_flux(lflux), neighbor_flux(nflux), result(res)
    { }
  };




  template <unsigned flux_count, class DSFluxInfoType>
  struct multiple_flux_coeffs
  {
    double local_coeff_here[flux_count];
    double local_coeff_opp[flux_count];
    double neighbor_coeff_here[flux_count];
    double neighbor_coeff_opp[flux_count];

    multiple_flux_coeffs(
        const face_pair &fp,
        const DSFluxInfoType flux_info[flux_count]
        )
    {
      const double fj = fp.loc.face_jacobian;

      for (unsigned i_flux = 0; i_flux < flux_count; ++i_flux)
      {
        local_coeff_here[i_flux] = 
          fj*flux_info[i_flux].local_flux(fp.loc, &fp.opp);
        neighbor_coeff_here[i_flux] = 
          fj*flux_info[i_flux].neighbor_flux(fp.loc, &fp.opp);
        local_coeff_opp[i_flux] = 
          fj*flux_info[i_flux].local_flux(fp.opp, &fp.loc);
        neighbor_coeff_opp[i_flux] = 
          fj*flux_info[i_flux].neighbor_flux(fp.opp, &fp.loc);
      }
    }
  };




  template <unsigned flux_count, class Mat, class DSFluxInfoType>
  void perform_multiple_double_sided_fluxes_on_single_operand(const face_group &fg, 
      const Mat &mat, DSFluxInfoType flux_info[flux_count],
      const hedge::py_vector &operand
      )
  {
    hedge::py_vector::const_iterator op_it = operand.begin();

    BOOST_FOREACH(const face_pair &fp, fg.face_pairs)
    {
      index_lists_t::const_iterator loc_idx_list = 
        fg.index_list(fp.loc.face_index_list_number);
      index_lists_t::const_iterator opp_idx_list = 
        fg.index_list(fp.opp.face_index_list_number);

      const int lebi = fp.loc.el_base_index;
      const int oebi = fp.opp.el_base_index;

      const multiple_flux_coeffs<flux_count, DSFluxInfoType> coeffs(
          fp, flux_info);

      for (unsigned i = 0; i < fg.face_length(); i++)
      {
        const int lili = lebi+loc_idx_list[i];
        const int oili = oebi+opp_idx_list[i];

        index_lists_t::const_iterator lilj_iterator = loc_idx_list;
        index_lists_t::const_iterator oilj_iterator = opp_idx_list;

        py_vector::value_type res_lili_additions[flux_count];
        py_vector::value_type res_oili_additions[flux_count];

        for (unsigned i_flux = 0; i_flux < flux_count; ++i_flux)
        {
          res_lili_additions[i_flux] = 0;
          res_oili_additions[i_flux] = 0;
        }

        for (unsigned j = 0; j < fg.face_length(); j++)
        {
          const typename Mat::value_type mat_entry = mat(i, j);

          const double matop_lilj = op_it[lebi+*lilj_iterator++]*mat_entry;
          const double matop_oilj = op_it[oebi+*oilj_iterator++]*mat_entry;

          for (unsigned i_flux = 0; i_flux < flux_count; ++i_flux)
          {
            res_lili_additions[i_flux] += 
              matop_lilj*coeffs.local_coeff_here[i_flux]
              +matop_oilj*coeffs.neighbor_coeff_here[i_flux];
            res_oili_additions[i_flux] += 
              matop_oilj*coeffs.local_coeff_opp[i_flux]
              +matop_lilj*coeffs.neighbor_coeff_opp[i_flux];
          }
        }

        for (unsigned i_flux = 0; i_flux < flux_count; ++i_flux)
        {
          flux_info[i_flux].result[lili] += res_lili_additions[i_flux];
          flux_info[i_flux].result[oili] += res_oili_additions[i_flux];
        }
      }
    }
  }




  template <class Mat, class LFlux, class NFlux>
  void perform_flux_on_one_target(const face_group &fg, const Mat& mat,
      LFlux lflux, NFlux nflux, vector_target target, bool newflux)
  {
    if (fg.double_sided)
      perform_double_sided_flux_on_single_vector_target(fg, mat, lflux, nflux, target, newflux);
    else
      perform_single_sided_flux(fg, mat, make_flux_data(lflux, target, nflux, target));
  }




  template <class Mat, class LFlux, class NFlux, class Target>
  void perform_flux_on_one_target(const face_group &fg, const Mat& mat,
      LFlux lflux, NFlux nflux, Target target, bool newflux)
  {
    if (fg.double_sided)
      perform_double_sided_flux(fg, mat, make_flux_data(lflux, target, nflux, target));
    else
      perform_single_sided_flux(fg, mat, make_flux_data(lflux, target, nflux, target));
  }




  template <class Mat, class LFlux, class LTarget, class NFlux, class NTarget>
  void perform_flux_detailed(const face_group &fg, const Mat& mat,
      LFlux lflux, LTarget ltarget, NFlux nflux, NTarget ntarget)
  {
    if (fg.double_sided)
      perform_double_sided_flux(fg, mat, make_flux_data(lflux, ltarget, nflux, ntarget));
    else
      perform_single_sided_flux(fg, mat, make_flux_data(lflux, ltarget, nflux, ntarget));
  }
}




#endif
