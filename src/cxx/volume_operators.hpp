#ifndef _AHFYHAT_HEDGE_VOLUME_OPERATORS_HPP_INCLUDED
#define _AHFYHAT_HEDGE_VOLUME_OPERATORS_HPP_INCLUDED




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
      { }

      unsigned size() const
      { return m_element_ranges.size(); }
      void clear()
      { m_element_ranges.clear(); }
      void append_range(unsigned start, unsigned end)
      { m_element_ranges.push_back(std::make_pair(start, end)); }
      const element_range &operator[](unsigned i) const
      { return m_element_ranges[i]; }

      // non-public interface
      container m_element_ranges;
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
      target.add_scaled_coefficients(r.first, r.second, r.first, r.second, 
          scale_factors[i++], matrix);
  }
}




#endif

