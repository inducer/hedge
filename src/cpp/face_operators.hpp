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
  typedef std::vector<int> index_list;

  struct face_pair
  {
    face_pair()
      : el_base_index(-1),
      opp_el_base_index(-1),
      face_index_list_number(-1),
      opp_face_index_list_number(-1),
      flux_face_index(-1), 
      opp_flux_face_index(-1)
    { }

    node_index el_base_index;
    node_index opp_el_base_index;

    unsigned face_index_list_number;
    unsigned opp_face_index_list_number;

    unsigned flux_face_index;
    unsigned opp_flux_face_index;
  };

  struct face_group
  {
    public:
      typedef std::vector<face_pair> face_pair_vector;
      typedef std::vector<fluxes::face> flux_face_vector;
      typedef std::vector<index_list> index_list_vector;

      face_pair_vector face_pairs;
      flux_face_vector flux_faces;
      index_list_vector index_lists;

      const bool        double_sided;

      face_group(bool d_sided)
        : double_sided(d_sided)
      { }
  };




  template <class LFlux, class LTarget, class NFlux, class NTarget>
  struct flux_data
  {
    typedef LFlux local_flux_t;
    typedef NFlux neighbor_flux_t;
    typedef LTarget local_target_t;
    typedef NTarget neighbor_target_t;

    local_flux_t local_flux;
    local_target_t local_target;
    
    neighbor_flux_t neighbor_flux;
    neighbor_target_t neighbor_target;

    flux_data(LFlux lflux, LTarget ltarget, NFlux nflux, NTarget ntarget)
      : local_flux(lflux), local_target(ltarget), 
      neighbor_flux(nflux), neighbor_target(ntarget)
    { }
  };




  template <class LFlux, class LTarget, class NFlux, class NTarget>
  flux_data<LFlux, LTarget, NFlux, NTarget> make_flux_data(
      LFlux lflux, LTarget ltarget, NFlux nflux, NTarget ntarget)
  {
    return flux_data<LFlux, LTarget, NFlux, NTarget>(lflux, ltarget, nflux, ntarget);
  }




  template <class Mat, class FData>
  void perform_single_sided_flux(const face_group &fg, const Mat &fmm, FData fdata)
  {
    const unsigned face_length = fmm.size1();

    assert(fmm.size1() == fmm.size2());

    BOOST_FOREACH(const face_pair &fp, fg.face_pairs)
    {
      const fluxes::face &flux_face = fg.flux_faces[fp.flux_face_index];

      const double local_coeff = 
        flux_face.face_jacobian*fdata.local_flux(flux_face, 0);
      const double neighbor_coeff = 
        flux_face.face_jacobian*fdata.neighbor_flux(flux_face, 0);

      const index_list &idx_list = fg.index_lists[fp.face_index_list_number];
      const index_list &opp_idx_list = fg.index_lists[fp.opp_face_index_list_number];

      assert(face_length == index_list.size());
      assert(face_length == opp_index_list.size());

      for (unsigned i = 0; i < face_length; i++)
      {
        const int ili = fp.el_base_index+idx_list[i];
        
        for (unsigned j = 0; j < face_length; j++)
        {
          const typename Mat::value_type fmm_entry = fmm(i, j);

          fdata.local_target.add_coefficient(
              ili, fp.el_base_index+idx_list[j], 
              local_coeff*fmm_entry);
          fdata.neighbor_target.add_coefficient(
              ili, fp.opp_el_base_index+opp_idx_list[j], 
              neighbor_coeff*fmm_entry);
        }
      }
    }
  }




  template <class Mat, class FData>
  void perform_double_sided_flux(const face_group &fg, const Mat &fmm, FData fdata)
  {
    const unsigned face_length = fmm.size1();

    assert(fmm.size1() == fmm.size2());

    BOOST_FOREACH(const face_pair &fp, fg.face_pairs)
    {
      const fluxes::face &flux_face = fg.flux_faces[fp.flux_face_index];
      const fluxes::face &opp_flux_face = fg.flux_faces[fp.opp_flux_face_index];

      const double local_coeff_here = 
        flux_face.face_jacobian*fdata.local_flux(flux_face, &opp_flux_face);
      const double neighbor_coeff_here = 
        flux_face.face_jacobian*fdata.neighbor_flux(flux_face, &opp_flux_face);
      const double local_coeff_opp = 
        flux_face.face_jacobian*fdata.local_flux(opp_flux_face, &flux_face);
      const double neighbor_coeff_opp = 
        flux_face.face_jacobian*fdata.neighbor_flux(opp_flux_face, &flux_face);

      const index_list &idx_list = fg.index_lists[fp.face_index_list_number];
      const index_list &opp_idx_list = fg.index_lists[fp.opp_face_index_list_number];

      assert(face_length == index_list.size());
      assert(face_length == opp_index_list.size());

      for (unsigned i = 0; i < face_length; i++)
      {
        const int ili = fp.el_base_index+idx_list[i];
        const int oili = fp.opp_el_base_index+opp_idx_list[i];

        index_list::const_iterator ilj_iterator = idx_list.begin();
        index_list::const_iterator oilj_iterator = opp_idx_list.begin();

        unsigned j;

        for (j = 0; j < face_length; j++)
        {
          const typename Mat::value_type fmm_entry = fmm(i, j);

          const int ilj = fp.el_base_index+*ilj_iterator++;
          const int oilj = fp.opp_el_base_index+*oilj_iterator++;

          fdata.local_target.add_coefficient(
              ili, ilj, 
              local_coeff_here*fmm_entry);
          fdata.neighbor_target.add_coefficient(
              ili, oilj, 
              neighbor_coeff_here*fmm_entry);

          fdata.local_target.add_coefficient(
              oili, oilj, 
              local_coeff_opp*fmm_entry);
          fdata.neighbor_target.add_coefficient(
              oili, ilj, 
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
    const unsigned face_length = fmm.size1();

    assert(fmm.size1() == fmm.size2());

    BOOST_FOREACH(const face_pair &fp, fg.face_pairs)
    {
      const fluxes::face &flux_face = fg.flux_faces[fp.flux_face_index];
      const fluxes::face &opp_flux_face = fg.flux_faces[fp.opp_flux_face_index];

      const double local_coeff_here = 
        flux_face.face_jacobian*local_flux(flux_face, &opp_flux_face);
      const double neighbor_coeff_here = 
        flux_face.face_jacobian*neighbor_flux(flux_face, &opp_flux_face);
      const double local_coeff_opp = 
        flux_face.face_jacobian*local_flux(opp_flux_face, &flux_face);
      const double neighbor_coeff_opp = 
        flux_face.face_jacobian*neighbor_flux(opp_flux_face, &flux_face);

      const index_list &idx_list = fg.index_lists[fp.face_index_list_number];
      const index_list &opp_idx_list = fg.index_lists[fp.opp_face_index_list_number];

      assert(face_length == index_list.size());
      assert(face_length == opp_index_list.size());

      const vector &operand = target.m_operand;
      vector &result = target.m_result;

      const int ebi = fp.el_base_index;
      const int oebi = fp.opp_el_base_index;

      for (unsigned i = 0; i < face_length; i++)
      {
        const int ili = fp.el_base_index+idx_list[i];
        const int oili = fp.opp_el_base_index+opp_idx_list[i];

        index_list::const_iterator ilj_iterator = idx_list.begin();
        index_list::const_iterator oilj_iterator = opp_idx_list.begin();

        unsigned j;

        vector::value_type res_ili_addition = 0;
        vector::value_type res_oili_addition = 0;

        for (j = 0; j < face_length; j++)
        {
          const typename Mat::value_type fmm_entry = fmm(i, j);

          const int ilj = ebi+*ilj_iterator++;
          const int oilj = oebi+*oilj_iterator++;

          /*
          __builtin_prefetch(&operand[ebi+ilj_iterator[0]], 0, 1);
          __builtin_prefetch(&operand[oebi+oilj_iterator[0]], 0, 1);
          */

          res_ili_addition += 
            operand[ilj]*local_coeff_here*fmm_entry
            +operand[oilj]*neighbor_coeff_here*fmm_entry;
          res_oili_addition += 
            operand[oilj]*local_coeff_opp*fmm_entry
            +operand[ilj]*neighbor_coeff_opp*fmm_entry;
        }

        result[ili] += res_ili_addition;
        result[oili] += res_oili_addition;
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
