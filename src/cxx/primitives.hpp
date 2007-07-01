#ifndef _ASFAHDALSU_HEDGE_PRIMITIVES_HPP_INCLUDED
#define _ASFAHDALSU_HEDGE_PRIMITIVES_HPP_INCLUDED




#include <boost/foreach.hpp>
#include <vector>
#include <utility>
#include "base.hpp"
#include "flux.hpp"




namespace hedge {
  struct element_ranges 
  {
    public:
      typedef std::pair<unsigned, unsigned> element_range;
      typedef std::vector<element_range> container;

      element_ranges(unsigned first_element)
        : m_first_element(first_element)
      { }

      unsigned size() const
      { return m_element_ranges.size(); }
      void clear()
      { m_element_ranges.clear(); }
      void append_range(unsigned start, unsigned end)
      { m_element_ranges.push_back(std::make_pair(start, end)); }
      const element_range &operator[](unsigned i) const
      { return m_element_ranges[i-m_first_element]; }

      // non-public interface
      container m_element_ranges;
      unsigned m_first_element;
  };

  template <class Mat, class OT>
  inline
  void perform_elwise_operator(const element_ranges &eg, const Mat &matrix, OT target)
  {
    BOOST_FOREACH(const element_ranges::element_range &r, eg.m_element_ranges)
      target.add_coefficients(r.first, r.second, r.first, r.second, matrix);
  }

  template <class Mat, class OT>
  inline
  void perform_elwise_scaled_operator(const element_ranges &eg, 
      const Mat &matrix, vector &scale_factors, OT target)
  {
    unsigned i = 0;
    BOOST_FOREACH(const element_ranges::element_range &r, eg.m_element_ranges)
      target.add_coefficients(r.first, r.second, r.first, r.second, 
          scale_factors[i++]*matrix);
  }




  struct face_group 
  {
    typedef std::vector<unsigned> index_list;
    struct face_info
    {
      index_list face_indices;
      index_list opposite_indices;
      flux::face flux_face;
    };

    std::vector<face_info> m_face_infos;

    void clear()
    {
      m_face_infos.clear();
    }

    void add_face(const index_list &my_ind, const index_list &opp_ind, 
        const flux::face &face)
    {
      face_info fi;
      fi.face_indices = my_ind;
      fi.opposite_indices = opp_ind;
      fi.flux_face = face;
      m_face_infos.push_back(fi);
    }
  };

  template <class Mat, class Flux, class OT>
  inline
  void perform_both_fluxes_operator(const face_group &fg, 
      const Mat &fmm, Flux flux, OT target)
  {
    unsigned face_length = fmm.size1();

    assert(fmm.size1() == fmm.size2());

    BOOST_FOREACH(const face_group::face_info &fi, fg.m_face_infos)
    {
      double local_coeff = flux.local_coeff(fi.flux_face);
      double neighbor_coeff = flux.neighbor_coeff(fi.flux_face);

      assert(fmm.size1() == fi.face_indices.size());
      assert(fmm.size1() == fi.opp_indices.size());

      for (unsigned i = 0; i < face_length; i++)
        for (unsigned j = 0; j < face_length; j++)
        {
          target.add_coefficient(fi.face_indices[i], fi.face_indices[j],
              fi.flux_face.face_jacobian*local_coeff*fmm(i, j));
          target.add_coefficient(fi.face_indices[i], fi.opposite_indices[j],
              fi.flux_face.face_jacobian*neighbor_coeff*fmm(i, j));
        }
    }
  }




  template <class Mat, class Flux, class OT>
  inline
  void perform_local_flux_operator(const face_group &fg, 
      const Mat &fmm, Flux flux, OT target)
  {
    unsigned face_length = fmm.size1();

    assert(fmm.size1() == fmm.size2());

    BOOST_FOREACH(const face_group::face_info &fi, fg.m_face_infos)
    {
      double local_coeff = flux.local_coeff(fi.flux_face);

      assert(fmm.size1() == fi.face_indices.size());

      for (unsigned i = 0; i < face_length; i++)
        for (unsigned j = 0; j < face_length; j++)
        {
          target.add_coefficient(fi.face_indices[i], fi.face_indices[j],
              fi.flux_face.face_jacobian*local_coeff*fmm(i, j));
        }
    }
  }




  template <class Mat, class Flux, class OT>
  inline
  void perform_neighbor_flux_operator(const face_group &fg, 
      const Mat &fmm, Flux flux, OT target)
  {
    unsigned face_length = fmm.size1();

    assert(fmm.size1() == fmm.size2());

    BOOST_FOREACH(const face_group::face_info &fi, fg.m_face_infos)
    {
      double neighbor_coeff = flux.neighbor_coeff(fi.flux_face);

      assert(fmm.size1() == fi.opp_indices.size());

      for (unsigned i = 0; i < face_length; i++)
        for (unsigned j = 0; j < face_length; j++)
        {
          target.add_coefficient(fi.face_indices[i], fi.opposite_indices[j],
              fi.flux_face.face_jacobian*neighbor_coeff*fmm(i, j));
        }
    }
  }
}




#endif
