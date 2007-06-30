#ifndef _ASFAHDALSU_HEDGE_ELEMENTGROUP_HPP_INCLUDED
#define _ASFAHDALSU_HEDGE_ELEMENTGROUP_HPP_INCLUDED




#include <boost/foreach.hpp>
#include <vector>
#include <utility>
#include "base.hpp"




namespace hedge {
  struct element_group {
    public:
      typedef std::pair<unsigned, unsigned> element_range;
      typedef std::vector<element_range> element_ranges;

      unsigned size()
      { return m_element_ranges.size(); }
      void clear()
      { m_element_ranges.clear(); }
      void add_range(unsigned start, unsigned end)
      { m_element_ranges.push_back(std::make_pair(start, end)); }

      // non-public interface
      element_ranges m_element_ranges;
  };

  template <class OT, class Mat>
  inline
  void apply_elwise_matrix(const element_group &eg, const OT &target, 
      const Mat &matrix)
  {
    BOOST_FOREACH(const element_group::element_range &r, eg.m_element_ranges)
      target.add_coefficients(r.first, r.second, r.first, r.second, matrix);
  }
}




#endif
