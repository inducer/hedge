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
      vector &scale_factors, const Mat &matrix, OT target)
  {
    unsigned i = 0;
    BOOST_FOREACH(const element_ranges::element_range &r, eg.m_element_ranges)
      target.add_scaled_coefficients(r.first, r.second, r.first, r.second, 
          scale_factors[i++], matrix);
  }
}




#endif

