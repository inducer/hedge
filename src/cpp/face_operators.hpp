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
#include <vector>
#include <utility>
#include "base.hpp"
#include "flux.hpp"
#include "op_target.hpp"




namespace hedge 
{
  typedef pyublas::numpy_vector<unsigned> index_lists_t;

  struct face_pair
  {
    static const unsigned INVALID_INDEX = UINT_MAX;

    static unsigned get_INVALID_INDEX()
    { return INVALID_INDEX; }

    struct side
    {
      node_index el_base_index;
      unsigned face_index_list_number;
      unsigned flux_face_index;
      unsigned local_el_number;

      side()
        : el_base_index(INVALID_NODE),
        face_index_list_number(INVALID_INDEX),
        flux_face_index(INVALID_INDEX)
      { }
    };

    side loc, opp;
  };

  struct face_group
  {
    public:
      typedef std::vector<face_pair> face_pair_vector;
      typedef std::vector<fluxes::face> flux_face_vector;

      face_pair_vector face_pairs;
      flux_face_vector flux_faces;
      index_lists_t index_lists;

      const bool double_sided;
      /** The number of elements touched by this face group.
       * Used for sizing a temporary.
       */
      unsigned local_element_count;

      face_group(bool d_sided)
        : double_sided(d_sided), local_element_count(0)
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
  void perform_single_sided_flux(const face_group &fg, const Mat &fmm, FData fdata)
  {
    BOOST_FOREACH(const face_pair &fp, fg.face_pairs)
    {
      const fluxes::face &flux_face = fg.flux_faces[fp.loc.flux_face_index];

      const double local_coeff = 
        flux_face.face_jacobian*fdata.local_flux(flux_face, 0);
      const double neighbor_coeff = 
        flux_face.face_jacobian*fdata.neighbor_flux(flux_face, 0);

      index_lists_t::const_iterator loc_idx_list = 
        fg.index_list(fp.loc.face_index_list_number);
      index_lists_t::const_iterator opp_idx_list = 
        fg.index_list(fp.opp.face_index_list_number);

      for (unsigned i = 0; i < fg.face_length(); i++)
      {
        const int lili = fp.loc.el_base_index+loc_idx_list[i];
        
        for (unsigned j = 0; j < fg.face_length(); j++)
        {
          const typename Mat::value_type fmm_entry = fmm(i, j);

          fdata.local_target.add_coefficient(
              lili, fp.loc.el_base_index+loc_idx_list[j], 
              local_coeff*fmm_entry);
          fdata.neighbor_target.add_coefficient(
              lili, fp.opp.el_base_index+opp_idx_list[j], 
              neighbor_coeff*fmm_entry);
        }
      }
    }
  }




  template <class Mat, class FData>
  void perform_double_sided_flux(const face_group &fg, const Mat &fmm, FData fdata)
  {
    BOOST_FOREACH(const face_pair &fp, fg.face_pairs)
    {
      const fluxes::face &loc_flux_face = fg.flux_faces[fp.loc.flux_face_index];
      const fluxes::face &opp_flux_face = fg.flux_faces[fp.opp.flux_face_index];

      const double local_coeff_here = 
        loc_flux_face.face_jacobian*fdata.local_flux(
            loc_flux_face, &opp_flux_face);
      const double neighbor_coeff_here = 
        loc_flux_face.face_jacobian*fdata.neighbor_flux(
            loc_flux_face, &opp_flux_face);
      const double local_coeff_opp = 
        loc_flux_face.face_jacobian*fdata.local_flux(
            opp_flux_face, &loc_flux_face);
      const double neighbor_coeff_opp = 
        loc_flux_face.face_jacobian*fdata.neighbor_flux(
            opp_flux_face, &loc_flux_face);

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
          const typename Mat::value_type fmm_entry = fmm(i, j);

          const int lilj = fp.loc.el_base_index+*lilj_iterator++;
          const int oilj = fp.opp.el_base_index+*oilj_iterator++;

          fdata.local_target.add_coefficient(
              lili, lilj, 
              local_coeff_here*fmm_entry);
          fdata.neighbor_target.add_coefficient(
              lili, oilj, 
              neighbor_coeff_here*fmm_entry);

          fdata.local_target.add_coefficient(
              oili, oilj, 
              local_coeff_opp*fmm_entry);
          fdata.neighbor_target.add_coefficient(
              oili, lilj, 
              neighbor_coeff_opp*fmm_entry);
        }
      }
    }
  }




  template <class Mat, class LFlux, class NFlux>
  void perform_double_sided_flux_on_single_vector_target(const face_group &fg, 
      const Mat &fmm, LFlux local_flux, NFlux neighbor_flux, vector_target target
      )
  {
    const py_vector::const_iterator op_it = target.m_operand.begin();
    const py_vector::iterator result_it = target.m_result.begin();

    bool multi_face = fmm.size2() > fg.face_length();

    BOOST_FOREACH(const face_pair &fp, fg.face_pairs)
    {
      const fluxes::face &loc_flux_face = fg.flux_faces[fp.loc.flux_face_index];
      const fluxes::face &opp_flux_face = fg.flux_faces[fp.opp.flux_face_index];

      const double local_coeff_here = 
        loc_flux_face.face_jacobian*local_flux(
            loc_flux_face, &opp_flux_face);
      const double neighbor_coeff_here = 
        loc_flux_face.face_jacobian*neighbor_flux(
            loc_flux_face, &opp_flux_face);
      const double local_coeff_opp = 
        loc_flux_face.face_jacobian*local_flux(
            opp_flux_face, &loc_flux_face);
      const double neighbor_coeff_opp = 
        loc_flux_face.face_jacobian*neighbor_flux(
            opp_flux_face, &loc_flux_face);

      index_lists_t::const_iterator loc_idx_list = 
        fg.index_list(fp.loc.face_index_list_number);
      index_lists_t::const_iterator opp_idx_list = 
        fg.index_list(fp.opp.face_index_list_number);

      const int lebi = fp.loc.el_base_index;
      const int oebi = fp.opp.el_base_index;

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
          const typename Mat::value_type fmm_entry = fmm(i, j);

          const double fmmop_lilj = op_it[lebi+*lilj_iterator++]*fmm_entry;
          const double fmmop_oilj = op_it[oebi+*oilj_iterator++]*fmm_entry;

          /*
          __builtin_prefetch(&operand[ebi+ilj_iterator[0]], 0, 1);
          __builtin_prefetch(&operand[oebi+oilj_iterator[0]], 0, 1);
          */

          res_lili_addition += 
             fmmop_lilj*local_coeff_here
            +fmmop_oilj*neighbor_coeff_here;
          res_oili_addition += 
            fmmop_oilj*local_coeff_opp 
            +fmmop_lilj*neighbor_coeff_opp;
        }

        result_it[lili] += res_lili_addition;
        result_it[oili] += res_oili_addition;
      }
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
        const fluxes::face &loc_flux_face,
        const fluxes::face &opp_flux_face,
        const DSFluxInfoType flux_info[flux_count]
        )
    {
      const double fj = loc_flux_face.face_jacobian;

      for (unsigned i_flux = 0; i_flux < flux_count; ++i_flux)
      {
        local_coeff_here[i_flux] = 
          fj*flux_info[i_flux].local_flux(loc_flux_face, &opp_flux_face);
        neighbor_coeff_here[i_flux] = 
          fj*flux_info[i_flux].neighbor_flux(loc_flux_face, &opp_flux_face);
        local_coeff_opp[i_flux] = 
          fj*flux_info[i_flux].local_flux(opp_flux_face, &loc_flux_face);
        neighbor_coeff_opp[i_flux] = 
          fj*flux_info[i_flux].neighbor_flux(opp_flux_face, &loc_flux_face);
      }
    }
  };




  template <unsigned flux_count, class Mat, class DSFluxInfoType>
  void perform_multiple_double_sided_fluxes_on_single_operand(const face_group &fg, 
      const Mat &fmm, DSFluxInfoType flux_info[flux_count],
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
          fg.flux_faces[fp.loc.flux_face_index],
          fg.flux_faces[fp.opp.flux_face_index],
          flux_info);

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
          const typename Mat::value_type fmm_entry = fmm(i, j);

          const double fmmop_lilj = op_it[lebi+*lilj_iterator++]*fmm_entry;
          const double fmmop_oilj = op_it[oebi+*oilj_iterator++]*fmm_entry;

          for (unsigned i_flux = 0; i_flux < flux_count; ++i_flux)
          {
            res_lili_additions[i_flux] += 
              fmmop_lilj*coeffs.local_coeff_here[i_flux]
              +fmmop_oilj*coeffs.neighbor_coeff_here[i_flux];
            res_oili_additions[i_flux] += 
              fmmop_oilj*coeffs.local_coeff_opp[i_flux]
              +fmmop_lilj*coeffs.neighbor_coeff_opp[i_flux];
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
  void perform_flux_on_one_target(const face_group &fg, const Mat& fmm,
      LFlux lflux, NFlux nflux, vector_target target)
  {
    if (fg.double_sided)
      perform_double_sided_flux_on_single_vector_target(fg, fmm, lflux, nflux, target);
    else
      perform_single_sided_flux(fg, fmm, make_flux_data(lflux, target, nflux, target));
  }




  template <class Mat, class LFlux, class NFlux, class Target>
  void perform_flux_on_one_target(const face_group &fg, const Mat& fmm,
      LFlux lflux, NFlux nflux, Target target)
  {
    if (fg.double_sided)
      perform_double_sided_flux(fg, fmm, make_flux_data(lflux, target, nflux, target));
    else
      perform_single_sided_flux(fg, fmm, make_flux_data(lflux, target, nflux, target));
  }




  template <class Mat, class LFlux, class LTarget, class NFlux, class NTarget>
  void perform_flux_detailed(const face_group &fg, const Mat& fmm,
      LFlux lflux, LTarget ltarget, NFlux nflux, NTarget ntarget)
  {
    if (fg.double_sided)
      perform_double_sided_flux(fg, fmm, make_flux_data(lflux, ltarget, nflux, ntarget));
    else
      perform_single_sided_flux(fg, fmm, make_flux_data(lflux, ltarget, nflux, ntarget));
  }
}




#endif
