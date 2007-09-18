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




#ifndef _AFAYTHF_HEDGE_INDEX_MAP_HPP_INCLUDED
#define _AFAYTHF_HEDGE_INDEX_MAP_HPP_INCLUDED




#include <boost/foreach.hpp>
#include <vector>
#include <stdexcept>
#include "base.hpp"




namespace hedge {
  struct from_index_map
  {
    std::vector<unsigned> m_map;
    unsigned m_from_length, m_to_length;

    from_index_map(unsigned from_length, unsigned to_length)
      : m_from_length(from_length), m_to_length(to_length)
    { }
  };

  template <class OT>
  inline
  void perform_index_map(const from_index_map &fim, OT target)
  {
    target.begin(fim.m_to_length, fim.m_from_length);
    unsigned i = 0;
    BOOST_FOREACH(unsigned from_index, fim.m_map)
      target.add_coefficient(i++, from_index, 1);
    target.finalize();
  }

  template <class OT>
  inline
  void perform_inverse_index_map(const from_index_map &fim, OT target)
  {
    target.begin(fim.m_from_length, fim.m_to_length);
    unsigned i = 0;
    BOOST_FOREACH(unsigned from_index, fim.m_map)
      target.add_coefficient(from_index, i++, 1);
    target.finalize();
  }
}




#endif
