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




namespace hedge 
{
  typedef std::vector<unsigned> index_list;

  struct face_pair
  {
    face_pair()
      : flux_face(0), opp_flux_face(0)
    { }

    index_list face_indices;
    index_list opposite_indices;
    fluxes::face *flux_face;
    fluxes::face *opp_flux_face;
  };

  struct face_group
  {
    public:
      typedef std::vector<face_pair> face_pair_vector;
      typedef std::vector<fluxes::face> flux_face_vector;

      face_pair_vector face_pairs;
      flux_face_vector flux_faces;

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
    unsigned face_length = fmm.size1();

    assert(fmm.size1() == fmm.size2());

    BOOST_FOREACH(const face_pair &fp, fg.face_pairs)
    {
      const double local_coeff = 
        fp.flux_face->face_jacobian*fdata.local_flux(*fp.flux_face, fp.opp_flux_face);
      const double neighbor_coeff = 
        fp.flux_face->face_jacobian*fdata.neighbor_flux(*fp.flux_face, fp.opp_flux_face);

      assert(fmm.size1() == fp.face_indices.size());
      assert(fmm.size1() == fp.opp_indices.size());

      for (unsigned i = 0; i < face_length; i++)
        for (unsigned j = 0; j < face_length; j++)
        {
          const typename Mat::value_type fmm_entry = fmm(i, j);
          fdata.local_target.add_coefficient(fp.face_indices[i], fp.face_indices[j],
              local_coeff*fmm_entry);
          fdata.neighbor_target.add_coefficient(fp.face_indices[i], fp.opposite_indices[j],
              neighbor_coeff*fmm_entry);
        }
    }
  }




  template <class Mat, class FData>
  void perform_double_sided_flux(const face_group &fg, const Mat &fmm, FData fdata)
  {
    unsigned face_length = fmm.size1();

    assert(fmm.size1() == fmm.size2());

    BOOST_FOREACH(const face_pair &fp, fg.face_pairs)
    {
      const double local_coeff_here = 
        fp.flux_face->face_jacobian*fdata.local_flux(*fp.flux_face, fp.opp_flux_face);
      const double neighbor_coeff_here = 
        fp.flux_face->face_jacobian*fdata.neighbor_flux(*fp.flux_face, fp.opp_flux_face);
      const double local_coeff_opp = 
        fp.flux_face->face_jacobian*fdata.local_flux(*fp.opp_flux_face, fp.flux_face);
      const double neighbor_coeff_opp = 
        fp.flux_face->face_jacobian*fdata.neighbor_flux(*fp.opp_flux_face, fp.flux_face);

      assert(fmm.size1() == fp.face_indices.size());
      assert(fmm.size1() == fp.opp_indices.size());

      for (unsigned i = 0; i < face_length; i++)
        for (unsigned j = 0; j < face_length; j++)
        {
          const typename Mat::value_type fmm_entry = fmm(i, j);

          fdata.local_target.add_coefficient(
              fp.face_indices[i], fp.face_indices[j],
              local_coeff_here*fmm_entry);
          fdata.neighbor_target.add_coefficient(
              fp.face_indices[i], fp.opposite_indices[j],
              neighbor_coeff_here*fmm_entry);

          fdata.local_target.add_coefficient(
              fp.opposite_indices[i], fp.opposite_indices[j],
              local_coeff_opp*fmm_entry);
          fdata.neighbor_target.add_coefficient(
              fp.opposite_indices[i], fp.face_indices[j],
              neighbor_coeff_opp*fmm_entry);
        }
    }
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
