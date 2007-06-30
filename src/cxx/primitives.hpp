#ifndef _ASFAHDALSU_HEDGE_PRIMITIVES_HPP_INCLUDED
#define _ASFAHDALSU_HEDGE_PRIMITIVES_HPP_INCLUDED




#include <boost/foreach.hpp>
#include <vector>
#include <utility>
#include "base.hpp"




namespace hedge {
  struct element_ranges {
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

  template <class OT, class Mat>
  inline
  void apply_elwise_matrix(const element_ranges &eg, OT target, 
      const Mat &matrix)
  {
    BOOST_FOREACH(const element_ranges::element_range &r, eg.m_element_ranges)
      target.add_coefficients(r.first, r.second, r.first, r.second, matrix);
  }

  template <class OT, class Mat>
  inline
  void apply_elwise_scaled_matrix(const element_ranges &eg, OT target, 
      const Mat &matrix, vector &scale_factors)
  {
    unsigned i = 0;
    BOOST_FOREACH(const element_ranges::element_range &r, eg.m_element_ranges)
      target.add_coefficients(r.first, r.second, r.first, r.second, 
          scale_factors[i++]*matrix);
  }
}




#endif
